"""
Utility functions for FEA analysis.

This module provides helper functions for mesh generation, geometry import,
and result export.
"""

import torch
import numpy as np
from typing import Literal, Tuple, Optional, Dict, Any
from pathlib import Path


def find_nodes_in_box(
    nodes: torch.Tensor,
    xmin: Optional[float] = None,
    xmax: Optional[float] = None,
    ymin: Optional[float] = None,
    ymax: Optional[float] = None,
    zmin: Optional[float] = None,
    zmax: Optional[float] = None,
    tolerance: float = 0.1,
) -> torch.Tensor:
    """
    Find nodes within a bounding box with tolerance.

    Args:
        nodes: Node coordinates tensor (N x 3)
        xmin, xmax: X bounds (None = no constraint)
        ymin, ymax: Y bounds (None = no constraint)
        zmin, zmax: Z bounds (None = no constraint)
        tolerance: Distance tolerance for matching

    Returns:
        Tensor of node indices within the box
    """
    mask = torch.ones(nodes.shape[0], dtype=torch.bool)

    if xmin is not None:
        mask &= nodes[:, 0] >= (xmin - tolerance)
    if xmax is not None:
        mask &= nodes[:, 0] <= (xmax + tolerance)

    if ymin is not None:
        mask &= nodes[:, 1] >= (ymin - tolerance)
    if ymax is not None:
        mask &= nodes[:, 1] <= (ymax + tolerance)

    if zmin is not None:
        mask &= nodes[:, 2] >= (zmin - tolerance)
    if zmax is not None:
        mask &= nodes[:, 2] <= (zmax + tolerance)

    return torch.where(mask)[0]


def get_geometry_info(nodes: torch.Tensor) -> Dict[str, Any]:
    """
    Extract geometry information from mesh nodes.

    Args:
        nodes: Node coordinates tensor (N x 3)

    Returns:
        Dict with bounding box, lengths, and span direction
    """
    xmin, xmax = nodes[:, 0].min().item(), nodes[:, 0].max().item()
    ymin, ymax = nodes[:, 1].min().item(), nodes[:, 1].max().item()
    zmin, zmax = nodes[:, 2].min().item(), nodes[:, 2].max().item()

    lengths = {"x": xmax - xmin, "y": ymax - ymin, "z": zmax - zmin}

    span_direction = max(lengths, key=lengths.get)

    return {
        "bounding_box": {
            "xmin": xmin,
            "xmax": xmax,
            "ymin": ymin,
            "ymax": ymax,
            "zmin": zmin,
            "zmax": zmax,
        },
        "lengths": lengths,
        "span_direction": span_direction,
        "nodes": nodes,
    }


def calculate_von_mises(sigma: torch.Tensor) -> torch.Tensor:
    """
    Calculate von Mises stress from stress tensor.

    Args:
        sigma: Stress tensor (N x 3 x 3)

    Returns:
        Von Mises stress tensor (N,)
    """
    sigma_xx = sigma[:, 0, 0]
    sigma_yy = sigma[:, 1, 1]
    sigma_zz = sigma[:, 2, 2]
    sigma_xy = sigma[:, 0, 1]
    sigma_xz = sigma[:, 0, 2]
    sigma_yz = sigma[:, 1, 2]

    von_mises = torch.sqrt(
        0.5
        * (
            (sigma_xx - sigma_yy) ** 2
            + (sigma_yy - sigma_zz) ** 2
            + (sigma_zz - sigma_xx) ** 2
        )
        + 3.0 * (sigma_xy**2 + sigma_xz**2 + sigma_yz**2)
    )

    return von_mises


def get_geometry_properties(
    step_file_path: str, density: float = 1.0
) -> Dict[str, float]:
    """
    Calculate geometry properties (volume, mass) from STEP file.

    Tries to use the OCP namespace first (pythonocc-core >=7 packaging),
    and falls back to the OCC namespace if OCP isn't available.
    If neither is installed, returns placeholder values.

    Args:
        step_file_path: Path to STEP file
        density: Material density (g/cm³)

    Returns:
        Dict with volume_mm3, mass_kg, density_g_cm3
    """
    volume_props_func = None

    # Try OCP (newer packaging), then OCC (older packaging)
    try:
        from OCP.STEPControl import STEPControl_Reader
        from OCP.GProp import GProp_GProps
        from OCP.BRepGProp import BRepGProp
        from OCP.IFSelect import IFSelect_RetDone

        def volume_props_func(shape, props):
            BRepGProp.VolumeProperties_s(shape, props)

    except Exception:
        try:
            from OCC.Core.STEPControl import STEPControl_Reader
            from OCC.Core.GProp import GProp_GProps
            from OCC.Core.BRepGProp import brepgprop_VolumeProperties
            from OCC.Core.IFSelect import IFSelect_RetDone

            def volume_props_func(shape, props):
                brepgprop_VolumeProperties(shape, props)

        except Exception:
            # If neither OCP nor OCC is available, return placeholder values
            return {"volume_mm3": -1.0, "mass_kg": -1.0, "density_g_cm3": density}

    # Read STEP file
    reader = STEPControl_Reader()
    status = reader.ReadFile(str(step_file_path))

    if status != IFSelect_RetDone:
        raise RuntimeError(f"Failed to read STEP file: {step_file_path}")

    reader.TransferRoots()
    shape = reader.OneShape()

    # Calculate volume
    props = GProp_GProps()
    volume_props_func(shape, props)
    volume_mm3 = props.Mass()  # Expected in mm³

    # Calculate mass: density (g/cm³) * volume (mm³) * (1 cm³ / 1000 mm³) * (1 kg / 1000 g)
    mass_kg = density * volume_mm3 / 1e6

    return {"volume_mm3": volume_mm3, "mass_kg": mass_kg, "density_g_cm3": density}


def export_to_vtk(
    filename: str,
    nodes: torch.Tensor,
    elements: torch.Tensor,
    element_type: str,
    point_data: Optional[Dict[str, torch.Tensor]] = None,
    cell_data: Optional[Dict[str, torch.Tensor]] = None,
) -> None:
    """
    Export FEA results to VTK format for visualization.

    Args:
        filename: Output VTK file path
        nodes: Node coordinates (N x 3)
        elements: Element connectivity
        element_type: Element type ('tet4', 'tet10', 'hex8', 'hex20')
        point_data: Dict of nodal data arrays
        cell_data: Dict of element data arrays
    """
    try:
        import pyvista as pv
    except ImportError:
        raise ImportError(
            "PyVista is required for VTK export. "
            "Install with: pip install rapidcadpy[fea]"
        )

    # Convert to numpy
    points = nodes.cpu().numpy()
    cells = elements.cpu().numpy()

    # Determine cell type
    cell_type_map = {
        "tet4": pv.CellType.TETRA,
        "tet10": pv.CellType.QUADRATIC_TETRA,
        "hex8": pv.CellType.HEXAHEDRON,
        "hex20": pv.CellType.QUADRATIC_HEXAHEDRON,
    }

    nodes_per_elem_map = {"tet4": 4, "tet10": 10, "hex8": 8, "hex20": 20}

    cell_type = cell_type_map[element_type]
    nodes_per_elem = nodes_per_elem_map[element_type]

    # Create VTK cells
    vtk_cells = np.column_stack([np.full(len(cells), nodes_per_elem), cells]).ravel()
    celltypes = np.full(len(cells), cell_type)

    # Create mesh
    mesh = pv.UnstructuredGrid(vtk_cells, celltypes, points)

    # Add point data
    if point_data:
        for name, data in point_data.items():
            mesh.point_data[name] = data.cpu().numpy()

    # Add cell data
    if cell_data:
        for name, data in cell_data.items():
            mesh.cell_data[name] = data.cpu().numpy()

    # Save
    mesh.save(filename)


def import_geometry(
    filename: str,
    mesh_size: float = 1.0,
    element_type: Literal["tet4", "tet10", "hex8", "hex20"] = "tet4",
    dim: int = 3,
    algorithm: int = 1,
    verbose: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Import STEP or STL geometry file and generate mesh for torch-fem.

    Args:
        filename: Path to STEP (.step, .stp) or STL (.stl) file
        mesh_size: Target mesh element size
        element_type: Type of elements to generate
            - 'tet4': 4-node linear tetrahedron (default)
            - 'tet10': 10-node quadratic tetrahedron
            - 'hex8': 8-node linear hexahedron (requires structured mesh)
            - 'hex20': 20-node quadratic hexahedron
        dim: Spatial dimension (2 for planar, 3 for solid)
        algorithm: Meshing algorithm (1=MeshAdapt, 2=Automatic, 5=Delaunay, 6=Frontal)
        verbose: Print meshing information

    Returns:
        nodes: Tensor of shape (n_nodes, dim) with nodal coordinates
        elements: Tensor of shape (n_elements, nodes_per_element) with element connectivity

    Raises:
        ImportError: If gmsh is not installed
        FileNotFoundError: If geometry file doesn't exist
        ValueError: If unsupported element type or file format
    """
    try:
        import gmsh
    except ImportError:
        raise ImportError(
            "gmsh is required for geometry import. Install with: pip install gmsh"
        )

    filepath = Path(filename)
    if not filepath.exists():
        raise FileNotFoundError(f"Geometry file not found: {filename}")

    # Initialize gmsh
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1 if verbose else 0)

    try:
        # Load geometry
        suffix = filepath.suffix.lower()
        if suffix in [".step", ".stp"]:
            gmsh.model.occ.importShapes(str(filepath))
        elif suffix == ".stl":
            gmsh.merge(str(filepath))
        else:
            raise ValueError(
                f"Unsupported file format: {suffix}. Use .step, .stp, or .stl"
            )

        gmsh.model.occ.synchronize()

        # Set mesh size
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", mesh_size * 0.5)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", mesh_size * 1.5)
        gmsh.option.setNumber("Mesh.Algorithm", algorithm)

        # Set element order based on element type
        if element_type in ["tet10", "hex20"]:
            gmsh.option.setNumber("Mesh.ElementOrder", 2)
        else:
            gmsh.option.setNumber("Mesh.ElementOrder", 1)

        # Generate mesh
        gmsh.model.mesh.generate(dim)

        # Recombine to hexahedra if requested
        if element_type in ["hex8", "hex20"]:
            gmsh.model.mesh.recombine()
            gmsh.model.mesh.generate(dim)  # Regenerate with hex elements

        # Extract nodes
        node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
        node_coords = np.array(node_coords).reshape(-1, 3)

        # Create node ID mapping (gmsh node tags may not be sequential)
        node_map = {tag: idx for idx, tag in enumerate(node_tags)}

        # Determine gmsh element type ID
        element_type_map = {
            "tet4": 4,  # 4-node tetrahedron
            "tet10": 11,  # 10-node tetrahedron
            "hex8": 5,  # 8-node hexahedron
            "hex20": 17,  # 20-node hexahedron
        }

        if element_type not in element_type_map:
            raise ValueError(f"Unsupported element type: {element_type}")

        gmsh_elem_type = element_type_map[element_type]

        # Extract elements of the specified type
        elem_types, elem_tags, elem_node_tags = gmsh.model.mesh.getElements(dim)

        elements_list = []
        for i, et in enumerate(elem_types):
            if et == gmsh_elem_type:
                # Get connectivity for this element type
                n_nodes_per_elem = len(elem_node_tags[i]) // len(elem_tags[i])
                connectivity = np.array(elem_node_tags[i]).reshape(-1, n_nodes_per_elem)

                # Map gmsh node tags to sequential indices
                connectivity_mapped = np.vectorize(node_map.get)(connectivity)
                elements_list.append(connectivity_mapped)

        if not elements_list:
            raise ValueError(
                f"No elements of type '{element_type}' found in mesh. "
                f"Try a different element_type or check the geometry."
            )

        elements = np.vstack(elements_list)

        if verbose:
            print(f"\n{'='*60}")
            print(f"Mesh Import Summary:")
            print(f"{'='*60}")
            print(f"File: {filepath.name}")
            print(f"Element type: {element_type}")
            print(f"Nodes: {len(node_tags)}")
            print(f"Elements: {len(elements)}")
            print(f"Mesh size: {mesh_size}")
            print(f"{'='*60}\n")

    finally:
        gmsh.finalize()

    # Convert to torch tensors
    nodes_tensor = torch.tensor(node_coords[:, :dim], dtype=torch.get_default_dtype())
    elements_tensor = torch.tensor(elements, dtype=torch.int64)

    return nodes_tensor, elements_tensor
