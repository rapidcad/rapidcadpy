"""
GMSH-based mesher using subprocess calls.

This mesher runs GMSH as a subprocess, avoiding library conflicts with OCP/OCC.
It requires GMSH to be installed and available in PATH.
"""

import subprocess
import tempfile
from pathlib import Path
from typing import Literal, Tuple
import torch
import os

from rapidcadpy.fea.mesher.base import MesherBase


class GmshSubprocessMesher(MesherBase):
    """
    GMSH mesher that runs as a subprocess.

    This implementation avoids library conflicts by running GMSH as an external
    process rather than using Python bindings.
    """

    def __init__(self, num_threads: int = 0, gmsh_path: str = "gmsh"):
        """
        Initialize GMSH subprocess mesher.

        Args:
            num_threads: Number of threads for parallel meshing (0 = auto-detect)
            gmsh_path: Path to gmsh executable (default: 'gmsh' from PATH)
        """
        super().__init__(num_threads)
        self.gmsh_path = gmsh_path

    @classmethod
    def is_available(cls) -> bool:
        """Check if GMSH is available in PATH."""
        try:
            result = subprocess.run(
                ["gmsh", "--version"], capture_output=True, text=True, timeout=5
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

        # Create temporary output file
        with tempfile.NamedTemporaryFile(suffix=".msh", delete=False) as tmp:
            msh_path = tmp.name

        try:
            # Build GMSH command with robustness options
            cmd = [
                self.gmsh_path,
                str(filename),
                "-3",  # 3D meshing
                "-format",
                "msh4",  # MSH 4.1 format
                "-o",
                str(msh_path),
                "-clmax",
                str(mesh_size),  # Maximum element size
                # Robustness options
                "-algo",
                "del3d",  # Delaunay 3D (generally more robust for complex 3D shapes)
                # "-algo", "front3d", # Alternative: Frontal 3D (try if Delaunay fails)
                "-optimize_netgen",  # Optimize mesh quality using Netgen algorithm
                "-check",  # Check for mesh errors
            ]

            # Add element order (1 for linear, 2 for quadratic)
            if element_type == "tet4":
                cmd.extend(["-order", "1"])
            elif element_type == "tet10":
                cmd.extend(
                    ["-order", "2", "-optimize_ho"]
                )  # Optimize high-order meshes

            # Run GMSH
            if verbose and os.getenv("RCADPY_VERBOSE") == "1":
                print(f"  Running GMSH mesher: {' '.join(cmd)}")
                result = subprocess.run(cmd, check=True, capture_output=True, text=True)
                if result.stdout:
                    print(result.stdout)
            else:
                result = subprocess.run(
                    cmd,
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    text=True,
                )

            # Parse the MSH file
            nodes, elements = self._parse_msh4(msh_path, element_type)

            if verbose and os.getenv("RCADPY_VERBOSE") == "1":
                print(
                    f"  ✓ Mesh generated: {nodes.shape[0]} nodes, {elements.shape[0]} elements"
                )

            return nodes, elements

        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"GMSH meshing failed with exit code {e.returncode}.\n"
                f"Command: {' '.join(cmd)}\n"
                f"Error: {e.stderr if e.stderr else 'No error output'}"
            )
        finally:
            # Clean up temporary file
            try:
                Path(msh_path).unlink()
            except Exception:
                pass

    def _parse_msh4(
        self, msh_path: str, element_type: str
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parse GMSH MSH 4.1 format file.

        Args:
            msh_path: Path to .msh file
            element_type: Expected element type

        Returns:
            Tuple of (nodes, elements) tensors
        """
        nodes_list = []
        elements_list = []
        node_tag_map = {}  # Map GMSH node tag -> local index (0-based)

        # MSH 4.1 element type codes
        # 4 = 4-node tetrahedron (tet4)
        # 11 = 10-node tetrahedron (tet10)
        element_type_map = {
            "tet4": 4,
            "tet10": 11,
        }

        target_elem_code = element_type_map.get(element_type)
        if target_elem_code is None:
            raise ValueError(f"Unsupported element type for parsing: {element_type}")

        with open(msh_path, "r") as f:
            lines = f.readlines()

        i = 0
        while i < len(lines):
            line = lines[i].strip()

            # Parse nodes section
            if line == "$Nodes":
                i += 1
                # MSH 4.1 format: numEntityBlocks numNodes minNodeTag maxNodeTag
                header = lines[i].strip().split()
                if len(header) < 2:
                    i += 1
                    continue
                num_entity_blocks = int(header[0])
                num_nodes = int(header[1])
                i += 1

                # Read each entity block
                for _ in range(num_entity_blocks):
                    # Entity block header: entityDim entityTag parametric numNodesInBlock
                    block_header = lines[i].strip().split()
                    num_nodes_in_block = int(block_header[3])
                    i += 1

                    # Read node tags
                    node_tags = []
                    for _ in range(num_nodes_in_block):
                        node_tags.append(int(lines[i].strip()))
                        i += 1

                    # Read node coordinates and populate map
                    for j in range(num_nodes_in_block):
                        coords = [float(x) for x in lines[i].strip().split()]

                        # Store standard 3D coords (ignore parametric if any)
                        nodes_list.append(coords[:3])

                        # Map tag to current index
                        current_idx = len(nodes_list) - 1
                        node_tag_map[node_tags[j]] = current_idx

                        i += 1

                # Skip $EndNodes
                i += 1

            # Parse elements section
            elif line == "$Elements":
                i += 1
                # MSH 4.1 format: numEntityBlocks numElements minElementTag maxElementTag
                header = lines[i].strip().split()
                if len(header) < 2:
                    i += 1
                    continue
                num_entity_blocks = int(header[0])
                i += 1

                # Read each entity block
                for _ in range(num_entity_blocks):
                    # Entity block header: entityDim entityTag elementType numElementsInBlock
                    block_header = lines[i].strip().split()
                    elem_type_code = int(block_header[2])
                    num_elems_in_block = int(block_header[3])
                    i += 1

                    # Only process elements of the target type
                    if elem_type_code == target_elem_code:
                        for _ in range(num_elems_in_block):
                            parts = lines[i].strip().split()
                            # First value is element tag, rest are node tags
                            # Use map to get correct local index
                            try:
                                elem_nodes = [node_tag_map[int(x)] for x in parts[1:]]
                                elements_list.append(elem_nodes)
                            except KeyError:
                                # This can happen if GMSH drops nodes or we missed some.
                                # Should generally not happen in a valid mesh.
                                pass
                            i += 1
                    else:
                        # Skip elements of other types
                        i += num_elems_in_block

                # Skip $EndElements
                i += 1

            else:
                i += 1

        # Convert to tensors
        nodes = torch.tensor(nodes_list, dtype=torch.float32)
        elements = torch.tensor(elements_list, dtype=torch.int64)

        return nodes, elements
