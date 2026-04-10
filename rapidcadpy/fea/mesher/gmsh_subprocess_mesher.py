"""
GMSH-based mesher using subprocess calls.

This mesher runs GMSH as a subprocess, avoiding library conflicts with OCP/OCC.
It requires GMSH to be installed and available in PATH.
"""

import subprocess
import tempfile
import logging
import time
from pathlib import Path
from typing import Literal, Tuple
import torch
import os
import meshio

logger = logging.getLogger(__name__)

from .base import MesherBase


class GmshSubprocessMesher(MesherBase):
    """
    GMSH mesher that runs as a subprocess.

    This implementation avoids library conflicts by running GMSH as an external
    process rather than using Python bindings.
    """

    def __init__(
        self,
        num_threads: int = 0,
        mesh_timeout: float = 120.0,
        tmp_dir: str | None = None,
    ):
        """
        Initialize GMSH subprocess mesher.

        Args:
            num_threads: Number of threads for parallel meshing (0 = auto-detect)
            mesh_timeout: Maximum wall-clock seconds allowed for a single GMSH
                subprocess call.  Defaults to 120 s — well below the typical
                NCCL watchdog limit (900 s) so a pathological geometry cannot
                stall distributed training.  Set to 0 to disable the timeout.
            tmp_dir: Directory for temporary .msh files produced during meshing.
                Defaults to None, which uses the OS default temp directory
                (typically /tmp).  Set to the run directory to keep all
                intermediary files co-located with training artifacts.

        The GMSH executable is resolved from the ``GMSH_EXECUTABLE_PATH``
        environment variable, falling back to ``gmsh`` (expected in PATH).
        """
        super().__init__(num_threads)
        self.mesh_timeout: float | None = float(mesh_timeout) if mesh_timeout else None
        self.tmp_dir: str | None = tmp_dir
        configured = os.environ.get("GMSH_EXECUTABLE_PATH", "gmsh")
        # If the configured path is an absolute path that does not exist on this
        # machine (e.g. the Docker cadruntime path stored in .env), fall back to
        # looking up "gmsh" on PATH so local runs work without editing .env.
        if os.path.isabs(configured) and not os.path.isfile(configured):
            import shutil

            fallback = shutil.which("gmsh") or "gmsh"
            logger.warning(
                f"GMSH_EXECUTABLE_PATH='{configured}' not found; falling back to '{fallback}'"
            )
            self.gmsh_path = fallback
        else:
            self.gmsh_path = configured

    @classmethod
    def is_available(cls) -> bool:
        """Check if GMSH is available."""
        try:
            gmsh = os.environ.get("GMSH_EXECUTABLE_PATH", "gmsh")
            result = subprocess.run(
                [gmsh, "--version"], capture_output=True, text=True, timeout=10
            )
            return result.returncode == 0
        except (subprocess.SubprocessError, FileNotFoundError):
            return False

    @classmethod
    def get_name(cls) -> str:
        """Get the name of this mesher."""
        return "GMSH (subprocess)"

    def get_supported_element_types(self) -> list[str]:
        """Get list of element types supported by this mesher."""
        return ["tet4", "tet10"]

    def get_supported_formats(self) -> list[str]:
        """Get list of geometry file formats supported by this mesher."""
        return [".step", ".stp", ".brep", ".iges", ".igs", ".stl"]

    def generate_mesh(
        self,
        filename: str,
        mesh_size: float = 1.0,
        element_type: Literal["tet4", "tet10", "hex8", "hex20"] = "tet4",
        dim: int = 3,
        verbose: bool = True,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate mesh from geometry file using GMSH subprocess.

        Args:
            filename: Path to geometry file (.step, .stp, etc.)
            mesh_size: Maximum element size
            element_type: Type of elements to generate ('tet4' or 'tet10')
            dim: Spatial dimension (must be 3)
            verbose: Print meshing progress
            **kwargs: Additional GMSH-specific parameters

        Returns:
            Tuple of (nodes, elements):
                - nodes: Tensor of shape (n_nodes, 3) with nodal coordinates
                - elements: Tensor of shape (n_elements, nodes_per_element) with connectivity

        Raises:
            FileNotFoundError: If geometry file doesn't exist
            ValueError: If unsupported element type or file format
            RuntimeError: If meshing fails
        """
        # Validate inputs
        self.validate_inputs(filename, element_type, dim)

        if dim != 3:
            raise ValueError(
                "GMSH subprocess mesher currently only supports 3D meshing"
            )

        # Create temporary output file.
        # Use a PID-based prefix so that DDP workers forked from the same
        # parent process do not race on the same path: tempfile internally
        # uses os.urandom(), which may return identical bytes on all ranks
        # before the OS re-seeds the entropy pool after fork.
        # When tmp_dir is set (e.g. to the training run directory) all mesh
        # intermediaries land there instead of /tmp, keeping artifacts
        # co-located with training outputs and avoiding /tmp exhaustion.
        if self.tmp_dir:
            import pathlib

            pathlib.Path(self.tmp_dir).mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            prefix=f"gmsh_pid{os.getpid():05x}_",
            suffix=".msh",
            delete=False,
            dir=self.tmp_dir or None,
        ) as tmp:
            msh_path = tmp.name
        # print(f"  Meshing {filename} with mesh size {mesh_size}...")
        # print(f"  Temporary mesh file: {msh_path}")
        try:
            # Build GMSH command with robustness options
            cmd = [
                self.gmsh_path,
                str(filename),
                "-3",  # 3D meshing
                "-format",
                "msh2",
                "-o",
                str(msh_path),
                "-clmax",
                str(mesh_size),  # Maximum element size
                # Robustness options
                # "-algo",
                # "del3d",  # Delaunay 3D (generally more robust for complex 3D shapes)
                # "-check",  # Check for mesh errors
            ]

            # Add element order (1 for linear, 2 for quadratic)
            if element_type == "tet4":
                cmd.extend(["-order", "1"])
            elif element_type == "tet10":
                cmd.extend(
                    ["-order", "2", "-optimize_ho"]
                )  # Optimize high-order meshes

            # Run GMSH
            _timeout = self.mesh_timeout if self.mesh_timeout else None
            mesh_start = time.perf_counter()
            if verbose and os.getenv("RCADPY_VERBOSE") == "1":
                # print(f"  Running GMSH mesher: {' '.join(cmd)}")
                result = subprocess.run(
                    cmd, check=True, capture_output=True, text=True, timeout=_timeout
                )
                if result.stdout:
                    print(result.stdout)
            else:
                result = subprocess.run(
                    cmd,
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=_timeout,
                )
            mesh_elapsed = time.perf_counter() - mesh_start
            # print(f"  GMSH meshing time: {mesh_elapsed:.3f}s")

            # Parse the MSH file using meshio
            # meshio cell type names: "tetra" = tet4, "tetra10" = tet10
            meshio_type_map = {"tet4": "tetra", "tet10": "tetra10"}
            meshio_type = meshio_type_map[element_type]
            mesh = meshio.read(msh_path, file_format="gmsh")
            nodes = torch.tensor(mesh.points, dtype=torch.float32)

            # print(f"  Mesh file read: {nodes.shape[0]} nodes, {len(mesh.cells)} cell blocks")

            cells = None
            for cell_block in mesh.cells:
                if cell_block.type == meshio_type:
                    cells = cell_block.data
                    break

            if cells is None:
                raise RuntimeError(
                    f"No {element_type} ({meshio_type}) elements found in mesh. "
                    f"Available cell types: {[cb.type for cb in mesh.cells]}"
                )

            elements = torch.tensor(cells, dtype=torch.int64)
            return nodes, elements

        except subprocess.TimeoutExpired as e:
            raise RuntimeError(
                f"GMSH meshing timed out after {self.mesh_timeout:.0f} s.\n"
                f"Command: {' '.join(cmd)}\n"
                "Consider increasing mesh_timeout or using a larger mesh_size."
            ) from e
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"GMSH meshing failed with exit code {e.returncode}.\n"
                f"Command: {' '.join(cmd)}\n"
                f"Error: {e.stderr if e.stderr else 'No error output'}"
            )
        except SystemExit as exc:
            raise RuntimeError(
                f"meshio exited unexpectedly while reading mesh file (SystemExit {exc}). "
                "The mesh file may be corrupted or empty."
            ) from exc
        finally:
            # Clean up temporary file
            try:
                Path(msh_path).unlink()
            except Exception:
                pass
