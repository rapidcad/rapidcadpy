"""
Netgen-based mesh generator implementation.

This module provides a Netgen mesher that implements the MesherBase interface.
Netgen imports are done lazily to avoid conflicts with OCP/OCC bindings.
"""

import numpy as np
import torch
from typing import Literal, Tuple
from pathlib import Path
import os

from .base import MesherBase


class NetgenMesher(MesherBase):
    """
    Netgen-based mesh generator.
    
    Supports tetrahedral meshing from STEP/STP and STL files.
    Uses lazy imports to avoid conflicts with OCP/OCC bindings.
    
    Note: Netgen imports are done lazily inside methods to avoid conflicts
    with OCP/OCC bindings. Both libraries try to register the same OpenCASCADE
    types (like gp_Pnt), causing "generic_type: type is already registered!" errors.
    """
    
    @classmethod
    def get_name(cls) -> str:
        """Get the name of this mesher."""
        return "Netgen"
    
    @classmethod
    def is_available(cls) -> bool:
        """Check if Netgen is available."""
        try:
            import netgen.occ
            import netgen.meshing
            return True
        except ImportError:
            return False
    
    def get_supported_element_types(self) -> list[str]:
        """Get supported element types."""
        return ["tet4", "tet10"]
    
    def get_supported_formats(self) -> list[str]:
        """Get supported file formats."""
        return [".step", ".stp", ".stl"]
    
    def generate_mesh(
        self,
        filename: str,
        mesh_size: float = 1.0,
        element_type: Literal["tet4", "tet10", "hex8", "hex20"] = "tet4",
        dim: int = 3,
        verbose: bool = True,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate mesh from geometry file using Netgen.
        
        Args:
            filename: Path to geometry file (.step, .stp, or .stl)
            mesh_size: Maximum element size
            element_type: Element type ('tet4' or 'tet10')
            dim: Spatial dimension (only 3 supported)
            verbose: Print meshing progress
            **kwargs: Additional Netgen-specific parameters (ignored)

        Returns:
            Tuple of (nodes, elements) as PyTorch tensors
            
        Raises:
            FileNotFoundError: If geometry file doesn't exist
            ValueError: If unsupported element type or file format
            RuntimeError: If meshing fails
        """
        # Validate inputs
        self.validate_inputs(filename, element_type, dim)
        
        # Lazy import of Netgen to avoid conflicts with OCP/OCC
        from netgen.occ import OCCGeometry
        from netgen.stl import STLGeometry
        import netgen.meshing as ngmesh

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
                        raise RuntimeError(
                            f"Could not extract index from type: {type(v)}"
                        )

            return val - 1  # Convert 1-based (Netgen) to 0-based (Python)

        # -------------------------------------------

        filepath = Path(filename)

        if verbose:
            print(f"--- Generating Mesh from {filepath.name} ---")

        # Configure Netgen to use multiple threads (if available)
        try:
            # Try the newer API first
            if hasattr(ngmesh, 'SetNumThreads'):
                ngmesh.SetNumThreads(self.num_threads)
            # Try alternative API
            elif hasattr(ngmesh, 'nthreads'):
                ngmesh.nthreads = self.num_threads
            # Set environment variable as fallback
            else:
                os.environ['OMP_NUM_THREADS'] = str(self.num_threads)
                os.environ['NETGEN_NUM_THREADS'] = str(self.num_threads)
            
            if verbose:
                print(f"Using {self.num_threads} threads for meshing")
        except Exception as e:
            if verbose:
                print(f"Note: Could not set thread count ({e}), using default")

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
                raise ValueError(f"Unsupported format: {suffix}")

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
                raise ValueError("Only dim=3 supported.")

            if not elements_list:
                raise ValueError("No compatible elements found.")

            elements = np.array(elements_list)

            if verbose:
                print(
                    f"Success: {len(node_coords)} nodes, "
                    f"{len(elements)} {element_type} elements."
                )

        except Exception as e:
            raise RuntimeError(f"Netgen meshing failed: {str(e)}")

        # 5. Convert to PyTorch
        nodes_tensor = torch.tensor(node_coords[:, :dim], dtype=torch.float32)
        elements_tensor = torch.tensor(elements, dtype=torch.int64)

        return nodes_tensor, elements_tensor


# Backward compatibility: keep the old function interface
def import_geometry_netgen(
    filename: str,
    mesh_size: float = 1.0,
    element_type: Literal["tet4", "tet10"] = "tet4",
    dim: int = 3,
    verbose: bool = True,
    num_threads: int = 0,  # 0 = auto-detect CPU cores
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Import geometry and generate mesh using Netgen.
    
    This is a backward-compatible wrapper around NetgenMesher.
    New code should use NetgenMesher directly for better flexibility.
    
    Args:
        filename: Path to geometry file (.step, .stp, or .stl)
        mesh_size: Maximum element size
        element_type: Element type ('tet4' or 'tet10')
        dim: Spatial dimension (only 3 supported)
        verbose: Print meshing progress
        num_threads: Number of threads for parallel meshing (0 = auto-detect)

    Returns:
        Tuple of (nodes, elements) as PyTorch tensors
        
    Example:
        >>> nodes, elements = import_geometry_netgen(
        ...     "part.step",
        ...     mesh_size=0.5,
        ...     element_type="tet4",
        ...     num_threads=4
        ... )
    """
    mesher = NetgenMesher(num_threads=num_threads)
    return mesher.generate_mesh(
        filename=filename,
        mesh_size=mesh_size,
        element_type=element_type,
        dim=dim,
        verbose=verbose
    )
