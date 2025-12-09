import numpy as np
import torch
from typing import Literal, Tuple
from pathlib import Path
import os

# Note: Netgen imports are done lazily inside functions to avoid conflicts
# with OCP/OCC bindings. Both libraries try to register the same OpenCASCADE
# types (like gp_Pnt), causing "generic_type: type is already registered!" errors.

# PyVista for visualization (lazy import)
_pv = None


def _get_pyvista():
    """Lazy import of PyVista to avoid import-time side effects."""
    global _pv
    if _pv is None:
        try:
            import pyvista as pv

            pv.set_plot_theme("document")
            _pv = pv
        except ImportError:
            raise ImportError("PyVista not found. Please install pyvista.")
    return _pv


def import_geometry_netgen(
    filename: str,
    mesh_size: float = 1.0,
    element_type: Literal["tet4", "tet10"] = "tet4",
    dim: int = 3,
    verbose: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Import geometry and generate mesh using Netgen.

    Note: Netgen is imported lazily inside this function to avoid conflicts
    with OCP/OCC bindings at module import time.
    """
    # Lazy import of Netgen to avoid conflicts with OCP/OCC
    from netgen.occ import OCCGeometry
    from netgen.stl import STLGeometry

    # --- Helper to handle Netgen Index Types ---
    def get_node_index(v):
        """
        Safely extract integer index from Netgen PointId/PointIndex.
        Netgen uses 1-based indexing, so we subtract 1.
        """
        try:
            # Try direct cast (works in some versions)
            val = int(v)
        except TypeError:
            # Fallback: Try string conversion (often '123' or similar)
            try:
                val = int(str(v))
            except ValueError:
                # Fallback: Check for common attributes if string fails
                if hasattr(v, "nr"):
                    val = v.nr
                elif hasattr(v, "id"):
                    val = v.id
                else:
                    raise RuntimeError(f"Could not extract index from type: {type(v)}")

        return val - 1  # Convert 1-based (Netgen) to 0-based (Python)

    # -------------------------------------------

    if element_type not in ["tet4", "tet10"]:
        raise NotImplementedError("Netgen only supports 'tet4' and 'tet10'.")

    filepath = Path(filename)
    if not filepath.exists():
        raise FileNotFoundError(f"Geometry file not found: {filename}")

    if verbose:
        print(f"--- Generating Mesh from {filepath.name} ---")

    # 1. Load Geometry and Generate Mesh
    suffix = filepath.suffix.lower()
    try:
        if suffix in [".step", ".stp"]:
            geo = OCCGeometry(str(filepath))
            mesh = geo.GenerateMesh(maxh=mesh_size)
        elif suffix == ".stl":
            geo = STLGeometry(str(filepath))
            mesh = geo.GenerateMesh(maxh=mesh_size)
        else:
            raise ValueError("Unsupported format.")

        # 2. Handle Element Order
        if element_type == "tet10":
            if verbose:
                print("Refining to Quadratic (Tet10)...")
            mesh.SecondOrder()

        # 3. Extract Nodes
        if verbose:
            print("Extracting nodes...")
        # mesh.Points() contains Point objects, we cast to list of coords
        node_coords = np.array([list(p) for p in mesh.Points()])

        # 4. Extract Elements
        if verbose:
            print("Extracting connectivity...")
        elements_list = []

        # Define expected node counts
        expected_nodes = 10 if element_type == "tet10" else 4

        if dim == 3:
            for el in mesh.Elements3D():
                # Extract indices using the helper
                indices = [get_node_index(v) for v in el.vertices]

                # Filter valid elements
                if len(indices) == expected_nodes:
                    elements_list.append(indices)
        else:
            raise ValueError("Only dim=3 supported for this example.")

        if not elements_list:
            raise ValueError("No compatible elements found.")

        elements = np.array(elements_list)

        if verbose:
            print(
                f"Success: {len(node_coords)} nodes, {len(elements)} {element_type} elements."
            )

    except Exception as e:
        raise RuntimeError(f"Netgen meshing failed: {str(e)}")

    # 5. Convert to PyTorch
    nodes_tensor = torch.tensor(node_coords[:, :dim], dtype=torch.float32)
    elements_tensor = torch.tensor(elements, dtype=torch.int64)

    return nodes_tensor, elements_tensor
