#!/usr/bin/env python3
"""
Convert DeepCAD STEP parts into CalculiX .inp files for FEA analysis.

This script analyzes STEP files to:
- Extract face geometry and properties
- Select constraint and load faces based on heuristics
- Generate tetrahedral mesh
- Identify nodes within constraint/load regions
- Output complete .inp files that can be run directly with CalculiX (ccx)

Usage:
    python deepcad_to_inp.py \
        --step_file shaft.step \
        --out_dir ./inp_files \
        --force_newtons 1000 \
        --mesh_size 2.0 \
        --face_tol 0.05 \
        --box_pad 0.1
    
    # Run with CalculiX:
    ccx shaft
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Set
import csv
import math
import random
import numpy as np

# Add parent directory to path for relative imports to work
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

# Import face selection logic from deepcad_to_loadcase
from deepcad_to_loadcase import (
    FaceProperties,
    load_step_and_extract_faces,
    compute_shape_bbox,
    select_constraint_face,
    select_load_faces,
    select_cylindrical_faces,
    expand_bbox_to_selector,
    determine_load_direction,
    determine_cylinder_axis,
    is_inner_cylinder,
)

# CadQuery and meshing imports
try:
    import cadquery as cq
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopAbs import TopAbs_SOLID
    from OCP.TopoDS import TopoDS
    import pyvista as pv
except ImportError as e:
    print(f"Error: Required dependencies not found: {e}", file=sys.stderr)
    print("Please install: pip install cadquery pyvista", file=sys.stderr)
    sys.exit(1)


def mesh_step_file(
    step_path: Path, mesh_size: float = 2.0, mesher: str = "netgen"
) -> Optional[pv.PolyData]:
    """
    Generate tetrahedral mesh from STEP file using gmsh or netgen.

    Args:
        step_path: Path to STEP file
        mesh_size: Target element size in mm
        mesher: Mesher to use ('gmsh' or 'netgen')

    Returns:
        PyVista mesh or None if meshing fails
    """
    try:
        # Import meshing utilities
        from rapidcadpy.fea.mesher.netgen_mesher import NetgenMesher
        from rapidcadpy.fea.mesher.gmsh_subprocess_mesher import GmshSubprocessMesher
        import torch

        # Generate mesh using mesher class
        if mesher.lower() == "gmsh":
            mesher_obj = GmshSubprocessMesher()
        else:
            mesher_obj = NetgenMesher()

        # Mesh the STEP file directly
        nodes_tensor, elements_tensor = mesher_obj.generate_mesh(
            str(step_path), mesh_size=mesh_size, element_type="tet4"
        )

        # Convert to PyVista mesh
        # nodes: (N, 3) tensor
        # elements: (M, 4) tensor for tet4
        nodes = nodes_tensor.numpy()
        elements = elements_tensor.numpy()

        # PyVista format: [n, i1, i2, ..., n, i1, i2, ...]
        cells = []
        for elem in elements:
            cells.append(4)  # Number of vertices
            cells.extend(elem)

        cells = np.array(cells)

        # Cell types: 10 = VTK_TETRA
        cell_types = np.full(len(elements), 10, dtype=np.uint8)

        # Create PyVista mesh
        mesh = pv.UnstructuredGrid(cells, cell_types, nodes)

        return mesh

    except Exception as e:
        print(f"Error meshing STEP file {step_path}: {e}", file=sys.stderr)
        import traceback
        print(traceback.format_exc(), file=sys.stderr)
        return None


def select_nodes_in_box(
    mesh: pv.PolyData, bbox: Dict[str, float], pad_factor: float = 0.0
) -> Set[int]:
    """
    Select nodes within a bounding box.

    Args:
        mesh: PyVista mesh
        bbox: Bounding box with x_min, x_max, y_min, y_max, z_min, z_max
        pad_factor: Additional padding factor (0.0 = exact box)

    Returns:
        Set of node indices (0-based)
    """
    points = mesh.points
    x, y, z = points[:, 0], points[:, 1], points[:, 2]

    # Apply padding
    if pad_factor > 0:
        dx = bbox["x_max"] - bbox["x_min"]
        dy = bbox["y_max"] - bbox["y_min"]
        dz = bbox["z_max"] - bbox["z_min"]
        pad = pad_factor * max(dx, dy, dz)

        x_min = bbox["x_min"] - pad
        x_max = bbox["x_max"] + pad
        y_min = bbox["y_min"] - pad
        y_max = bbox["y_max"] + pad
        z_min = bbox["z_min"] - pad
        z_max = bbox["z_max"] + pad
    else:
        x_min, x_max = bbox["x_min"], bbox["x_max"]
        y_min, y_max = bbox["y_min"], bbox["y_max"]
        z_min, z_max = bbox["z_min"], bbox["z_max"]

    # Select nodes within box
    mask = (
        (x >= x_min)
        & (x <= x_max)
        & (y >= y_min)
        & (y <= y_max)
        & (z >= z_min)
        & (z <= z_max)
    )

    return set(np.where(mask)[0])


def select_nodes_on_cylinder(
    mesh: pv.PolyData,
    center: Tuple[float, float, float],
    radius: float,
    axis: str = "z",
    tolerance: float = 0.5,
) -> Set[int]:
    """
    Select nodes on or near a cylindrical surface.

    Args:
        mesh: PyVista mesh
        center: Cylinder center point
        radius: Cylinder radius
        axis: Cylinder axis ('x', 'y', or 'z')
        tolerance: Distance tolerance in mm

    Returns:
        Set of node indices (0-based)
    """
    points = mesh.points
    cx, cy, cz = center

    # Calculate radial distance based on cylinder axis
    if axis == "x":
        radial_dist = np.sqrt((points[:, 1] - cy) ** 2 + (points[:, 2] - cz) ** 2)
    elif axis == "y":
        radial_dist = np.sqrt((points[:, 0] - cx) ** 2 + (points[:, 2] - cz) ** 2)
    else:  # z
        radial_dist = np.sqrt((points[:, 0] - cx) ** 2 + (points[:, 1] - cy) ** 2)

    # Select nodes near cylinder surface
    mask = np.abs(radial_dist - radius) <= tolerance

    return set(np.where(mask)[0])


def write_calculix_inp(
    inp_path: Path,
    mesh: pv.PolyData,
    constraint_node_sets: List[Tuple[str, Set[int]]],
    load_node_sets: List[Tuple[str, Set[int], str, float]],
    pressure_loads: List[Tuple[str, Set[int], str, float]],
    material: Dict[str, Any],
    step_file_name: str,
) -> None:
    """
    Write complete CalculiX .inp file.

    Args:
        inp_path: Output .inp file path
        mesh: PyVista mesh with nodes and elements
        constraint_node_sets: List of (name, node_set) for fixed constraints
        load_node_sets: List of (name, node_set, direction, magnitude_N) for point loads
        pressure_loads: List of (name, node_set, direction, pressure_MPa) for pressure loads
        material: Material properties dict
        step_file_name: Original STEP file name for header
    """
    with open(inp_path, "w") as f:
        # Header
        f.write("*HEADING\n")
        f.write(f"CalculiX FEA model generated from: {step_file_name}\n")
        f.write(f"Generated by: deepcad_to_inp.py\n")
        f.write("\n")

        # Nodes (CalculiX uses 1-based indexing)
        f.write("*NODE, NSET=ALL_NODES\n")
        points = mesh.points
        for i, (x, y, z) in enumerate(points, start=1):
            f.write(f"{i}, {x:.6f}, {y:.6f}, {z:.6f}\n")
        f.write("\n")

        # Elements - Check if we have tetrahedral or other element types
        cells = mesh.cells
        cell_types = mesh.celltypes

        # CalculiX element type mapping
        # VTK_TETRA (10) -> C3D4 (4-node tetrahedron)
        # VTK_QUADRATIC_TETRA (24) -> C3D10 (10-node tetrahedron)
        # VTK_HEXAHEDRON (12) -> C3D8 (8-node hexahedron)
        element_type_map = {
            10: ("C3D4", 4),  # Linear tetrahedron
            24: ("C3D10", 10),  # Quadratic tetrahedron
            12: ("C3D8", 8),  # Linear hexahedron
        }

        # Find dominant element type
        unique_types = np.unique(cell_types)
        if len(unique_types) == 0:
            print("Error: No elements in mesh", file=sys.stderr)
            return

        dominant_type = unique_types[0]
        if dominant_type not in element_type_map:
            print(
                f"Warning: Unknown element type {dominant_type}, assuming C3D4",
                file=sys.stderr,
            )
            ccx_elem_type, nodes_per_elem = "C3D4", 4
        else:
            ccx_elem_type, nodes_per_elem = element_type_map[dominant_type]

        f.write(f"*ELEMENT, TYPE={ccx_elem_type}, ELSET=ALL_ELEMENTS\n")

        # Parse cells array (VTK format: [n, node1, node2, ..., n, node1, ...])
        elem_id = 1
        i = 0
        while i < len(cells):
            n_nodes = cells[i]
            if n_nodes != nodes_per_elem:
                # Skip elements with wrong node count
                i += n_nodes + 1
                continue

            # Get node IDs (add 1 for 1-based indexing)
            node_ids = [cells[i + j + 1] + 1 for j in range(n_nodes)]

            # Write element
            f.write(f"{elem_id}, " + ", ".join(map(str, node_ids)) + "\n")
            elem_id += 1
            i += n_nodes + 1

        f.write("\n")

        # Node sets for constraints
        for set_name, node_set in constraint_node_sets:
            if not node_set:
                continue

            f.write(f"*NSET, NSET={set_name.upper()}\n")
            # Convert to 1-based indexing and write in blocks of 16
            node_list = sorted([n + 1 for n in node_set])
            for i in range(0, len(node_list), 16):
                chunk = node_list[i : i + 16]
                f.write(", ".join(map(str, chunk)) + "\n")
            f.write("\n")

        # Node sets for loads
        for set_name, node_set, _, _ in load_node_sets:
            if not node_set:
                continue

            f.write(f"*NSET, NSET={set_name.upper()}\n")
            node_list = sorted([n + 1 for n in node_set])
            for i in range(0, len(node_list), 16):
                chunk = node_list[i : i + 16]
                f.write(", ".join(map(str, chunk)) + "\n")
            f.write("\n")

        # Node sets for pressure loads
        for set_name, node_set, _, _ in pressure_loads:
            if not node_set:
                continue

            f.write(f"*NSET, NSET={set_name.upper()}\n")
            node_list = sorted([n + 1 for n in node_set])
            for i in range(0, len(node_list), 16):
                chunk = node_list[i : i + 16]
                f.write(", ".join(map(str, chunk)) + "\n")
            f.write("\n")

        # Material definition
        mat_name = material.get("name", "MATERIAL").upper()
        E = material.get("elastic_modulus_mpa", 69000)  # MPa
        nu = material.get("poissons_ratio", 0.33)

        f.write(f"*MATERIAL, NAME={mat_name}\n")
        f.write("*ELASTIC\n")
        f.write(f"{E:.1f}, {nu:.3f}\n")
        f.write("\n")

        # Solid section
        f.write(f"*SOLID SECTION, ELSET=ALL_ELEMENTS, MATERIAL={mat_name}\n")
        f.write("\n")

        # Boundary conditions (fixed constraints)
        for set_name, node_set in constraint_node_sets:
            if not node_set:
                continue

            f.write("*BOUNDARY\n")
            f.write(f"{set_name.upper()}, 1, 3\n")
            f.write("\n")

        # Analysis step
        f.write("*STEP, NLGEOM=NO\n")
        f.write("*STATIC\n")
        f.write("0.1, 1.0\n")
        f.write("\n")

        # Concentrated loads (distributed over nodes)
        for set_name, node_set, direction, magnitude in load_node_sets:
            if not node_set:
                continue

            # Distribute load evenly across nodes
            num_nodes = len(node_set)
            load_per_node = magnitude / num_nodes if num_nodes > 0 else 0

            # Direction mapping: +x=1, +y=2, +z=3, -x=1 (negative mag), etc.
            dir_map = {"+x": 1, "-x": 1, "+y": 2, "-y": 2, "+z": 3, "-z": 3}
            dof = dir_map.get(direction, 3)
            sign = -1.0 if direction.startswith("-") else 1.0

            f.write("*CLOAD\n")
            f.write(f"{set_name.upper()}, {dof}, {sign * load_per_node:.6f}\n")
            f.write("\n")

        # Pressure loads (as distributed surface loads)
        for set_name, node_set, direction, pressure_mpa in pressure_loads:
            if not node_set:
                continue

            # For pressure loads, we need to apply them as DLOAD on element faces
            # For simplicity, we'll apply as equivalent nodal forces
            # More sophisticated: identify surface elements and use *DLOAD

            # Estimate surface area and convert pressure to force
            # Rough approximation: area per node from node density
            num_nodes = len(node_set)
            if num_nodes == 0:
                continue

            # Calculate average node spacing as proxy for area per node
            node_coords = mesh.points[[n for n in node_set]]
            mesh_density = num_nodes / (
                np.max(node_coords, axis=0) - np.min(node_coords, axis=0) + 1e-9
            ).prod()
            area_per_node = 1.0 / (mesh_density ** (2 / 3) + 1e-9)

            # Force per node = pressure * area_per_node
            force_per_node = pressure_mpa * area_per_node

            # Direction for pressure (inward vs outward)
            # "inward" means compression (negative normal)
            # "outward" means tension (positive normal)
            sign = -1.0 if direction == "inward" else 1.0

            # Apply as concentrated loads on nodes
            # Note: This is simplified - proper implementation would use *DLOAD
            f.write("*CLOAD\n")
            f.write(
                f"** Pressure load: {pressure_mpa:.2f} MPa ({direction}), ~{force_per_node:.3f} N/node\n"
            )
            # For now, apply in dominant axis direction (simplified)
            f.write(f"{set_name.upper()}, 3, {sign * force_per_node:.6f}\n")
            f.write("\n")

        # Output requests
        f.write("*NODE FILE\n")
        f.write("U, RF\n")
        f.write("*EL FILE\n")
        f.write("S, E\n")
        f.write("\n")

        f.write("*END STEP\n")


def write_debug_visualization(
    vtu_path: Path,
    mesh: pv.PolyData,
    constraint_node_sets: List[Tuple[str, Set[int]]],
    load_node_sets: List[Tuple[str, Set[int], str, float]],
    pressure_loads: List[Tuple[str, Set[int], str, float]],
) -> None:
    """
    Write debug visualization file showing constraint and load nodes.

    Creates a VTU file with scalar fields:
    - node_type: 0=free, 1=constraint, 2=load, 3=pressure
    - Individual fields for each constraint/load set
    """
    num_nodes = mesh.n_points

    # Create node type array
    node_type = np.zeros(num_nodes, dtype=np.int32)
    
    # Mark constraint nodes (type 1)
    for _, nodes in constraint_node_sets:
        for node_id in nodes:
            node_type[node_id] = 1

    # Mark load nodes (type 2)
    for _, nodes, _, _ in load_node_sets:
        for node_id in nodes:
            node_type[node_id] = 2

    # Mark pressure nodes (type 3)
    for _, nodes, _, _ in pressure_loads:
        for node_id in nodes:
            node_type[node_id] = 3

    # Add node type as scalar field
    mesh["node_type"] = node_type
    
    # Add individual constraint sets
    for set_name, nodes in constraint_node_sets:
        field = np.zeros(num_nodes, dtype=np.int32)
        for node_id in nodes:
            field[node_id] = 1
        mesh[f"{set_name}_constraint"] = field

    # Add individual load sets
    for set_name, nodes, direction, magnitude in load_node_sets:
        field = np.zeros(num_nodes, dtype=np.float32)
        for node_id in nodes:
            field[node_id] = magnitude
        mesh[f"{set_name}_load_{direction}"] = field

    # Add individual pressure sets
    for set_name, nodes, direction, pressure in pressure_loads:
        field = np.zeros(num_nodes, dtype=np.float32)
        for node_id in nodes:
            field[node_id] = pressure
        mesh[f"{set_name}_pressure_{direction}"] = field

    # Save to VTU
    mesh.save(str(vtu_path))


def process_step_file_to_inp(
    step_path: Path,
    out_dir: Path,
    force_newtons: float,
    pressure_mpa: float,
    mesh_size: float,
    face_tol: float,
    box_pad: float,
    prefer_opposing: float,
    mesher: str = "netgen",
) -> Dict[str, Any]:
    """
    Process a single STEP file and generate CalculiX .inp file.

    Returns:
        Dictionary with processing results and statistics
    """
    result = {
        "step_file": step_path.name,
        "success": False,
        "error": None,
        "num_nodes": 0,
        "num_elements": 0,
        "num_constraint_nodes": 0,
        "num_load_nodes": 0,
        "num_constraints": 0,
        "num_loads": 0,
        "mesh_size": mesh_size,
    }

    try:
        # Load STEP and extract faces
        shape, face_props_list = load_step_and_extract_faces(step_path)

        if not shape or not face_props_list:
            result["error"] = "No faces found or failed to load"
            return result

        # Compute part bounding box
        part_bbox = compute_shape_bbox(shape)
        if not part_bbox:
            result["error"] = "Failed to compute part bbox"
            return result

        # Select constraint face(s)
        constraint_faces = select_constraint_face(face_props_list)
        if not constraint_faces:
            result["error"] = "Failed to select constraint face"
            return result

        # Select load face(s)
        load_faces = select_load_faces(
            constraint_faces, face_props_list, part_bbox, prefer_opposing
        )
        if not load_faces:
            result["error"] = "Failed to select load face"
            return result

        # Select cylindrical faces for pressure loads
        cylindrical_faces = select_cylindrical_faces(
            face_props_list,
            part_bbox,
            min_radius=0.1,
            max_radius=100.0,
            prefer_inner=True,
        )

        # Generate mesh
        print(f"  Meshing with {mesher} (size={mesh_size:.2f}mm)...", end=" ")
        mesh = mesh_step_file(step_path, mesh_size=mesh_size, mesher=mesher)

        if mesh is None:
            result["error"] = "Meshing failed"
            return result

        result["num_nodes"] = mesh.n_points
        result["num_elements"] = mesh.n_cells
        print(f"{mesh.n_points} nodes, {mesh.n_cells} elements")

        # Generate spatial selectors for constraint faces
        constraint_node_sets = []
        for i, constraint_face in enumerate(constraint_faces, 1):
            constraint_selector = expand_bbox_to_selector(
                constraint_face, box_pad, face_tol, part_bbox, enforce_thickness=True
            )

            # Select nodes in constraint region
            constraint_nodes = select_nodes_in_box(mesh, constraint_selector)

            if not constraint_nodes:
                print(
                    f"  Warning: No nodes found in constraint region {i}",
                    file=sys.stderr,
                )
                continue

            set_name = (
                f"constraint_{i}" if len(constraint_faces) > 1 else "constraint"
            )
            constraint_node_sets.append((set_name, constraint_nodes))

        if not constraint_node_sets:
            result["error"] = "No constraint nodes found"
            return result

        # Calculate target number of loads
        total_faces = (
            len(load_faces) + len(cylindrical_faces) + len(constraint_faces)
        )
        if total_faces >= 40:
            target_total_loads = 3
        elif total_faces >= 20:
            target_total_loads = 2
        else:
            target_total_loads = 1

        # Generate pressure loads for cylindrical faces
        pressure_loads = []
        num_pressure_loads = min(len(cylindrical_faces), target_total_loads)

        print(f"  Found {len(cylindrical_faces)} cylindrical faces", file=sys.stderr)
        for idx, cf in enumerate(cylindrical_faces[:3], 1):
            is_inner = is_inner_cylinder(cf, part_bbox)
            print(
                f"    Cyl{idx}: r={cf.cylinder_radius:.2f}mm, center={tuple(round(x, 2) for x in cf.cylinder_location)}, "
                f"axis={determine_cylinder_axis(cf.cylinder_axis)}, {'INNER' if is_inner else 'outer'}",
                file=sys.stderr,
            )

        for i, cyl_face in enumerate(cylindrical_faces[:num_pressure_loads], 1):
            # Select nodes on cylinder surface
            cyl_axis = determine_cylinder_axis(cyl_face.cylinder_axis)
            
            # Use tighter tolerance - fraction of radius, not just mesh size
            tolerance = min(mesh_size * 0.3, cyl_face.cylinder_radius * 0.15)
            
            cyl_nodes = select_nodes_on_cylinder(
                mesh,
                cyl_face.cylinder_location,
                cyl_face.cylinder_radius,
                axis=cyl_axis,
                tolerance=tolerance,
            )

            print(
                f"    Cyl{i}: selected {len(cyl_nodes)} nodes (r={cyl_face.cylinder_radius:.2f}mm, tol={tolerance:.3f}mm)",
                file=sys.stderr,
            )

            if not cyl_nodes:
                print(
                    f"  Warning: No nodes found on cylinder {i}", file=sys.stderr
                )
                continue

            set_name = (
                f"pressure_{i}" if len(cylindrical_faces) > 1 else "pressure"
            )
            direction = "inward" if random.random() < 0.7 else "outward"
            pressure_loads.append((set_name, cyl_nodes, direction, pressure_mpa))

        # Generate load node sets for remaining load faces
        load_node_sets = []
        num_distributed_loads = max(0, target_total_loads - len(pressure_loads))

        for i, load_face in enumerate(load_faces[:num_distributed_loads], 1):
            load_selector = expand_bbox_to_selector(
                load_face, box_pad, face_tol, part_bbox, enforce_thickness=True
            )

            # Select nodes in load region
            load_nodes = select_nodes_in_box(mesh, load_selector)

            if not load_nodes:
                print(
                    f"  Warning: No nodes found in load region {i}", file=sys.stderr
                )
                continue

            set_name = f"load_{i}" if len(load_faces) > 1 else "load"
            load_direction = determine_load_direction(load_face.normal, randomize=True)

            # Split force among distributed loads
            force_per_load = (
                force_newtons / num_distributed_loads
                if num_distributed_loads > 0
                else force_newtons
            )

            load_node_sets.append(
                (set_name, load_nodes, load_direction, force_per_load)
            )

        # Update statistics
        result["num_constraint_nodes"] = sum(
            len(nodes) for _, nodes in constraint_node_sets
        )
        result["num_load_nodes"] = sum(len(nodes) for _, nodes, _, _ in load_node_sets)
        result["num_pressure_nodes"] = sum(
            len(nodes) for _, nodes, _, _ in pressure_loads
        )
        result["num_constraints"] = len(constraint_node_sets)
        result["num_loads"] = len(load_node_sets) + len(pressure_loads)

        # Material properties
        material = {
            "name": "Aluminium_6061",
            "elastic_modulus_mpa": 69000,
            "poissons_ratio": 0.33,
            "density_g_cm3": 2.70,
        }

        # Write .inp file
        inp_file = out_dir / f"{step_path.stem}.inp"
        write_calculix_inp(
            inp_file,
            mesh,
            constraint_node_sets,
            load_node_sets,
            pressure_loads,
            material,
            step_path.name,
        )

        # Write debug visualization file
        vtu_file = out_dir / f"{step_path.stem}_debug.vtu"
        write_debug_visualization(
            vtu_file,
            mesh,
            constraint_node_sets,
            load_node_sets,
            pressure_loads,
        )

        result["success"] = True
        result["inp_file"] = str(inp_file)
        result["vtu_file"] = str(vtu_file)

    except KeyboardInterrupt:
        raise
    except Exception as e:
        result["error"] = str(e)
        import traceback

        print(f"\nError details: {traceback.format_exc()}", file=sys.stderr)

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Convert DeepCAD STEP parts into CalculiX .inp files."
    )

    # Create mutually exclusive group for input (either file or directory)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--step_file", type=Path, help="Single STEP file to convert"
    )
    input_group.add_argument(
        "--step_dir", type=Path, help="Directory containing STEP files"
    )

    parser.add_argument(
        "--out_dir", type=Path, required=True, help="Output directory for .inp files"
    )
    parser.add_argument(
        "--force_newtons",
        type=float,
        default=1000.0,
        help="Total force magnitude in Newtons (default: 1000)",
    )
    parser.add_argument(
        "--pressure_mpa",
        type=float,
        default=5.0,
        help="Pressure magnitude in MPa for cylindrical loads (default: 5.0)",
    )
    parser.add_argument(
        "--mesh_size",
        type=float,
        default=2.0,
        help="Target mesh element size in mm (default: 2.0)",
    )
    parser.add_argument(
        "--mesher",
        type=str,
        default="netgen",
        choices=["netgen", "gmsh"],
        help="Mesher to use (default: netgen)",
    )
    parser.add_argument(
        "--face_tol",
        type=float,
        default=0.05,
        help="Tolerance for face thickness as fraction of part size (default: 0.05 = 5%%)",
    )
    parser.add_argument(
        "--box_pad",
        type=float,
        default=0.1,
        help="Padding for selector boxes as fraction of part size (default: 0.1 = 10%%)",
    )
    parser.add_argument(
        "--prefer_opposing",
        type=float,
        default=0.85,
        help="Threshold for opposing face selection (default: 0.85)",
    )
    parser.add_argument(
        "--summary_csv",
        type=Path,
        default=None,
        help="Path to write summary CSV (default: out_dir/summary.csv)",
    )

    args = parser.parse_args()

    # Create output directory
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Find all STEP files
    step_files = []

    if args.step_file:
        # Single file mode
        if not args.step_file.exists():
            print(f"Error: STEP file not found: {args.step_file}", file=sys.stderr)
            sys.exit(1)
        step_files = [args.step_file]
    else:
        # Directory mode
        if not args.step_dir.exists():
            print(f"Error: STEP directory not found: {args.step_dir}", file=sys.stderr)
            sys.exit(1)

        for pattern in ["*.step", "*.stp", "*.STEP", "*.STP"]:
            step_files.extend(args.step_dir.rglob(pattern))

        step_files = sorted(set(step_files))

    if not step_files:
        print(f"Warning: No STEP files found", file=sys.stderr)
        return

    print(f"Found {len(step_files)} STEP files to process")
    print(f"Mesher: {args.mesher}, Mesh size: {args.mesh_size}mm\n")

    # Process each STEP file
    results = []
    for i, step_path in enumerate(step_files, 1):
        print(f"[{i}/{len(step_files)}] Processing {step_path.name}...")

        result = process_step_file_to_inp(
            step_path,
            args.out_dir,
            args.force_newtons,
            args.pressure_mpa,
            args.mesh_size,
            args.face_tol,
            args.box_pad,
            args.prefer_opposing,
            args.mesher,
        )

        results.append(result)

        if result["success"]:
            print(
                f"  ✓ Written: {Path(result['inp_file']).name} + debug VTU"
            )
            print(
                f"    Mesh: {result['num_nodes']} nodes, {result['num_elements']} elements"
            )
            print(
                f"    BCs: {result['num_constraint_nodes']} constraint, "
                f"{result['num_load_nodes']} load, "
                f"{result.get('num_pressure_nodes', 0)} pressure nodes\n"
            )
        else:
            print(f"  ✗ Failed: {result['error']}\n")

    # Write summary CSV
    csv_path = args.summary_csv or (args.out_dir / "summary.csv")
    with open(csv_path, "w", newline="") as csvfile:
        fieldnames = [
            "step_file",
            "success",
            "num_nodes",
            "num_elements",
            "num_constraint_nodes",
            "num_load_nodes",
            "num_pressure_nodes",
            "num_constraints",
            "num_loads",
            "mesh_size",
            "error",
            "inp_file",
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for result in results:
            row = {k: result.get(k, "") for k in fieldnames}
            writer.writerow(row)

    # Print summary
    successful = sum(1 for r in results if r["success"])
    failed = len(results) - successful

    print(f"{'='*60}")
    print(f"Processing complete:")
    print(f"  Successful: {successful}/{len(results)}")
    print(f"  Failed: {failed}/{len(results)}")
    print(f"  .inp files: {args.out_dir}")
    print(f"  Summary CSV: {csv_path}")
    print(f"\nTo run with CalculiX:")
    print(f"  cd {args.out_dir}")
    print(f"  ccx <filename_without_extension>")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
