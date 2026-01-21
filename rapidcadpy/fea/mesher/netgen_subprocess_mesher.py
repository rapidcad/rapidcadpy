"""
Netgen-based mesher using subprocess calls.

This mesher runs Netgen as a subprocess, avoiding library conflicts with OCP/OCC.
It requires Netgen to be installed and available in PATH.
"""

import subprocess
import tempfile
from pathlib import Path
from typing import Literal, Tuple
import torch
import os

from rapidcadpy.fea.mesher.base import MesherBase


class NetgenSubprocessMesher(MesherBase):
    """
    Netgen mesher that runs as a subprocess.
    
    This implementation avoids library conflicts by running Netgen as an external
    process rather than using Python bindings.
    """

    def __init__(self, num_threads: int = 0, netgen_path: str = "netgen"):
        """
        Initialize Netgen subprocess mesher.

        Args:
            num_threads: Number of threads for parallel meshing (0 = auto-detect)
            netgen_path: Path to netgen executable (default: 'netgen' from PATH)
        """
        super().__init__(num_threads)
        self.netgen_path = netgen_path

    @classmethod
    def is_available(cls) -> bool:
        """Check if Netgen is available in PATH."""
        try:
            result = subprocess.run(
                ["netgen", "-h"],
                capture_output=True,
                text=True,
                timeout=5
            )
            return result.returncode == 0
        except (subprocess.SubprocessError, FileNotFoundError):
            return False

    @classmethod
    def get_name(cls) -> str:
        """Get the name of this mesher."""
        return "Netgen (subprocess)"

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
        Generate mesh from geometry file using Netgen subprocess.

        Args:
            filename: Path to geometry file (.step, .stp, etc.)
            mesh_size: Maximum element size
            element_type: Type of elements to generate ('tet4' or 'tet10')
            dim: Spatial dimension (must be 3)
            verbose: Print meshing progress
            **kwargs: Additional Netgen-specific parameters

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
            raise ValueError("Netgen subprocess mesher currently only supports 3D meshing")

        # Create temporary output files
        with tempfile.NamedTemporaryFile(suffix=".vol", delete=False) as tmp_vol:
            vol_path = tmp_vol.name
        
        with tempfile.NamedTemporaryFile(suffix=".msh", delete=False) as tmp_msh:
            msh_path = tmp_msh.name

        try:
            # Build Netgen command to generate .vol file
            cmd_vol = [
                self.netgen_path,
                str(filename),
                "-meshfile", str(vol_path),
                "-maxh", str(mesh_size),
                "-meshfiletype", "Netgen Mesh",
                "-batchmode",
            ]
            
            # Set element order
            if element_type == "tet10":
                cmd_vol.append("-secondorder")

            # Run Netgen to create .vol file
            if verbose and os.getenv("RCADPY_VERBOSE") == "1":
                print(f"  Running Netgen mesher: {' '.join(cmd_vol)}")
                result = subprocess.run(cmd_vol, check=True, capture_output=True, text=True)
                if result.stdout:
                    print(result.stdout)
            else:
                result = subprocess.run(
                    cmd_vol, 
                    check=True, 
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )

            # Convert .vol to .msh format for easier parsing
            cmd_export = [
                self.netgen_path,
                vol_path,
                "-meshfile", str(msh_path),
                "-meshfiletype", "GMSH Format",
                "-batchmode",
            ]
            
            subprocess.run(
                cmd_export,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )

            # Parse the MSH file (Netgen exports GMSH 2.0 format)
            nodes, elements = self._parse_gmsh2(msh_path, element_type)

            if verbose and os.getenv("RCADPY_VERBOSE") == "1":
                print(f"  ✓ Mesh generated: {nodes.shape[0]} nodes, {elements.shape[0]} elements")

            return nodes, elements

        except subprocess.CalledProcessError as e:
            # Determine which command failed
            cmd_str = ' '.join(e.cmd) if e.cmd else 'unknown'
            raise RuntimeError(
                f"Netgen meshing failed with exit code {e.returncode}.\n"
                f"Command: {cmd_str}\n"
                f"Error: {e.stderr if e.stderr else 'No error output'}"
            )
        finally:
            # Clean up temporary files
            for path in [vol_path, msh_path]:
                try:
                    Path(path).unlink()
                except Exception:
                    pass

    def _parse_gmsh2(
        self, msh_path: str, element_type: str
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parse GMSH 2.0 format file (exported by Netgen).

        Args:
            msh_path: Path to .msh file
            element_type: Expected element type

        Returns:
            Tuple of (nodes, elements) tensors
        """
        nodes_list = []
        elements_list = []

        # GMSH 2.0 element type codes
        # 4 = 4-node tetrahedron (tet4)
        # 11 = 10-node tetrahedron (tet10)
        element_type_map = {
            "tet4": 4,
            "tet10": 11,
        }
        
        target_elem_code = element_type_map.get(element_type)
        if target_elem_code is None:
            raise ValueError(f"Unsupported element type for parsing: {element_type}")

        with open(msh_path, 'r') as f:
            lines = f.readlines()

        i = 0
        while i < len(lines):
            line = lines[i].strip()

            # Parse nodes section
            if line == "$Nodes":
                i += 1
                num_nodes = int(lines[i].strip())
                i += 1

                # Read nodes: nodeTag x y z
                for _ in range(num_nodes):
                    parts = lines[i].strip().split()
                    # Skip node tag (first value), use x, y, z
                    coords = [float(parts[j]) for j in range(1, 4)]
                    nodes_list.append(coords)
                    i += 1

                # Skip $EndNodes
                i += 1

            # Parse elements section
            elif line == "$Elements":
                i += 1
                num_elements = int(lines[i].strip())
                i += 1

                # Read elements
                for _ in range(num_elements):
                    parts = lines[i].strip().split()
                    # GMSH 2.0 format: elmNumber elmType numTags tags... nodeIndices...
                    elem_type_code = int(parts[1])
                    num_tags = int(parts[2])
                    
                    # Only process elements of the target type
                    if elem_type_code == target_elem_code:
                        # Node indices start after: elmNumber, elmType, numTags, and tags
                        node_start = 3 + num_tags
                        elem_nodes = [int(parts[j]) - 1 for j in range(node_start, len(parts))]  # Convert to 0-based
                        elements_list.append(elem_nodes)
                    
                    i += 1

                # Skip $EndElements
                i += 1

            else:
                i += 1

        # Convert to tensors
        nodes = torch.tensor(nodes_list, dtype=torch.float32)
        elements = torch.tensor(elements_list, dtype=torch.int64)

        return nodes, elements