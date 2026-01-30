"""
Isolated GMSH mesher that runs in a separate Python process.

This mesher completely isolates GMSH execution to avoid library conflicts
with OCP/OCC by running it in a fresh Python subprocess without OCP loaded.
"""

import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Literal, Tuple
import torch
import os

from tools.fea.mesher.base import MesherBase


# Script template that will run in isolated subprocess
GMSH_SCRIPT_TEMPLATE = '''
import sys
import tempfile

def run_gmsh():
    """Run GMSH in isolated process."""
    try:
        import gmsh
        import pickle
    except ImportError as e:
        print("ERROR: " + str(e), file=sys.stderr)
        sys.exit(1)

    # Parse arguments
    input_file = r"{input_file}"
    output_file = r"{output_file}"
    mesh_size = {mesh_size}
    element_type = "{element_type}"
    verbose = {verbose}

    try:
        # Initialize GMSH
        gmsh.initialize()

        if not verbose:
            gmsh.option.setNumber("General.Terminal", 0)

        gmsh.option.setNumber("General.NumThreads", {num_threads})
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", mesh_size)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", mesh_size * 0.1)

        # Import geometry
        gmsh.model.add("mesh")
        gmsh.model.occ.importShapes(input_file)
        gmsh.model.occ.synchronize()

        # Set meshing parameters
        gmsh.option.setNumber("Mesh.Algorithm3D", 10)  # HXT
        mesh_order = 1 if element_type == "tet4" else 2
        gmsh.option.setNumber("Mesh.ElementOrder", mesh_order)
        gmsh.option.setNumber("Mesh.Optimize", 1)
        gmsh.option.setNumber("Mesh.OptimizeNetgen", 1)

        # Generate mesh
        gmsh.model.mesh.generate(3)

        if element_type == "tet10":
            gmsh.model.mesh.setOrder(2)

        # Get mesh data directly (more reliable than file I/O)
        node_tags, node_coords, _ = gmsh.model.mesh.getNodes()

        # Reshape node coordinates
        nodes = node_coords.reshape(-1, 3)

        # Get elements (type 4 = tet4, type 11 = tet10)
        elem_type_code = 4 if element_type == "tet4" else 11
        elem_tags, elem_node_tags = gmsh.model.mesh.getElementsByType(elem_type_code)

        nodes_per_elem = 4 if element_type == "tet4" else 10
        elements = elem_node_tags.reshape(-1, nodes_per_elem)

        # Create node tag mapping (GMSH tags may not be sequential starting from 1)
        node_tag_map = {{tag: i for i, tag in enumerate(node_tags)}}

        # Remap element node tags to 0-based indices
        elements_remapped = [[node_tag_map[int(tag)] for tag in elem]
                            for elem in elements]

        # Save results as pickle
        result = {{
            "nodes": nodes.tolist(),
            "elements": elements_remapped,
            "success": True,
            "n_nodes": len(nodes),
            "n_elements": len(elements)
        }}

        with open(output_file, "wb") as f:
            pickle.dump(result, f)

        if verbose:
            print("Success: " + str(len(nodes)) + " nodes, " + str(len(elements)) + " elements")

        gmsh.finalize()

    except Exception as e:
        # Save error
        result = {{
            "success": False,
            "error": str(e)
        }}
        try:
            import pickle
            with open(output_file, "wb") as f:
                pickle.dump(result, f)
        except:
            pass

        print("ERROR: " + str(e), file=sys.stderr)
        try:
            gmsh.finalize()
        except:
            pass
        sys.exit(1)

if __name__ == "__main__":
    run_gmsh()
'''


class IsolatedGmshMesher(MesherBase):
    """
    GMSH mesher that runs in a completely isolated Python subprocess.

    This avoids all library conflicts with OCP/OCC by running GMSH
    in a fresh Python process without any OCP imports.
    """

    def __init__(self, num_threads: int = 0):
        """
        Initialize Isolated GMSH mesher.

        Args:
            num_threads: Number of threads for parallel meshing (0 = auto-detect)
        """
        super().__init__(num_threads)

    @classmethod
    def is_available(cls) -> bool:
        """Check if GMSH Python API is available."""
        # Test in subprocess to avoid loading gmsh in main process
        try:
            result = subprocess.run(
                [sys.executable, "-c", "import gmsh; print('OK')"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            return result.returncode == 0 and "OK" in result.stdout
        except (subprocess.SubprocessError, FileNotFoundError):
            return False

    @classmethod
    def get_name(cls) -> str:
        """Get the name of this mesher."""
        return "GMSH (isolated)"

    def get_supported_element_types(self) -> list[str]:
        """Get list of element types supported by this mesher."""
        return ["tet4", "tet10"]

    def get_supported_formats(self) -> list[str]:
        """Get list of geometry file formats supported by this mesher."""
        return [".step", ".stp", ".brep", ".iges", ".igs"]

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
        Generate mesh from geometry file using isolated GMSH process.

        Args:
            filename: Path to geometry file (.step, .stp, etc.)
            mesh_size: Maximum element size
            element_type: Type of elements to generate ('tet4' or 'tet10')
            dim: Spatial dimension (must be 3)
            verbose: Print meshing progress
            **kwargs: Additional parameters

        Returns:
            Tuple of (nodes, elements):
                - nodes: Tensor of shape (n_nodes, 3) with nodal coordinates
                - elements: Tensor of shape (n_elements, nodes_per_element) with connectivity

        Raises:
            FileNotFoundError: If geometry file doesn't exist
            ValueError: If unsupported element type or file format
            RuntimeError: If meshing fails
        """
        import pickle

        # Validate inputs
        self.validate_inputs(filename, element_type, dim)

        if dim != 3:
            raise ValueError("IsolatedGmshMesher currently only supports 3D meshing")

        # Create temporary files for script and output
        with tempfile.NamedTemporaryFile(
            suffix=".py", delete=False, mode="w"
        ) as tmp_py:
            script_path = tmp_py.name

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp_out:
            output_path = tmp_out.name

        try:
            # Create the isolated script
            script_content = GMSH_SCRIPT_TEMPLATE.format(
                input_file=Path(filename).resolve(),
                output_file=output_path,
                mesh_size=mesh_size,
                element_type=element_type,
                verbose="True" if verbose else "False",
                num_threads=self.num_threads,
            )

            with open(script_path, "w") as f:
                f.write(script_content)

            if verbose:
                print(f"  Running isolated GMSH process...")

            # Run in completely fresh Python subprocess
            # Don't inherit any environment that might have OCP loaded
            result = subprocess.run(
                [sys.executable, script_path],
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout
            )

            # Check for errors
            if result.returncode != 0:
                error_msg = result.stderr if result.stderr else "Unknown error"
                raise RuntimeError(
                    f"Isolated GMSH process failed with exit code {result.returncode}.\n"
                    f"Error: {error_msg}"
                )

            # Load results
            if not Path(output_path).exists():
                raise RuntimeError("Meshing process did not produce output file")

            with open(output_path, "rb") as f:
                result_data = pickle.load(f)

            if not result_data.get("success", False):
                error = result_data.get("error", "Unknown error")
                raise RuntimeError(f"Meshing failed: {error}")

            # Convert to tensors
            nodes = torch.tensor(result_data["nodes"], dtype=torch.float32)
            elements = torch.tensor(result_data["elements"], dtype=torch.int64)

            if elements.shape[0] == 0:
                raise RuntimeError(
                    f"No volume elements generated. The geometry "
                    f"could not be meshed as a 3D solid. Check that your STEP file "
                    f"contains valid closed volumes."
                )

            if verbose:
                print(
                    f"  ✓ Mesh generated: {nodes.shape[0]} nodes, {elements.shape[0]} elements"
                )

            return nodes, elements

        finally:
            # Clean up temporary files
            for path in [script_path, output_path]:
                try:
                    Path(path).unlink()
                except Exception:
                    pass
