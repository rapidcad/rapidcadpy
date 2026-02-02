#!/usr/bin/env python3
"""
Convert DeepCAD STEP parts into FEA load-case JSON files.

This script analyzes STEP files to:
- Extract face geometry and properties (area, AABB, normals)
- Select constraint and load faces based on heuristics
- Generate spatial selector boxes (AABBs)
- Output JSON load-case definitions for FEA simulations

Usage:
    python deepcad_to_loadcase.py \
        --step_dir /path/to/steps \
        --out_dir /path/to/loadcases \
        --force_newtons 1000 \
        --face_tol 0.5 \
        --box_pad 1.0 \
        --prefer_opposing 1
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import csv
import math
import random

# Import load case parser for visualization
sys.path.insert(0, str(Path(__file__).parent))
from load_case_parser import parse_load_case

# CadQuery and OCP imports
try:
    import cadquery as cq
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopAbs import TopAbs_FACE, TopAbs_FORWARD
    from OCP.TopoDS import TopoDS_Face, TopoDS
    from OCP.GProp import GProp_GProps
    from OCP.BRepGProp import BRepGProp
    from OCP.Bnd import Bnd_Box
    from OCP.BRepBndLib import BRepBndLib
    from OCP.BRepAdaptor import BRepAdaptor_Surface
    from OCP.GeomAbs import GeomAbs_Plane, GeomAbs_Cylinder
    from OCP.BRepLProp import BRepLProp_SLProps
    from OCP.gp import gp_Pnt, gp_Vec
    from OCP.GeomAPI import GeomAPI_ProjectPointOnSurf
    from OCP.BRep import BRep_Tool
except ImportError as e:
    print(f"Error: Required dependencies not found: {e}", file=sys.stderr)
    print("Please install: pip install cadquery", file=sys.stderr)
    sys.exit(1)


class FaceProperties:
    """Container for computed face properties."""

    def __init__(self):
        self.face = None
        self.area: float = 0.0
        self.bbox_min: Tuple[float, float, float] = (0, 0, 0)
        self.bbox_max: Tuple[float, float, float] = (0, 0, 0)
        self.center: Tuple[float, float, float] = (0, 0, 0)
        self.normal: Tuple[float, float, float] = (0, 0, 1)
        self.is_planar: bool = False
        self.is_cylindrical: bool = False
        self.cylinder_radius: float = 0.0
        self.cylinder_axis: Tuple[float, float, float] = (0, 0, 1)
        self.cylinder_location: Tuple[float, float, float] = (0, 0, 0)

    @property
    def bbox_dict(self) -> Dict[str, float]:
        """Return bbox as dict."""
        return {
            "x_min": self.bbox_min[0],
            "y_min": self.bbox_min[1],
            "z_min": self.bbox_min[2],
            "x_max": self.bbox_max[0],
            "y_max": self.bbox_max[1],
            "z_max": self.bbox_max[2],
        }


def compute_face_properties(face) -> Optional[FaceProperties]:
    """
    Compute geometric properties for a TopoDS_Face.

    Returns:
        FaceProperties object or None if computation fails
    """
    props = FaceProperties()
    props.face = face

    try:
        # 1. Compute area
        g_props = GProp_GProps()
        BRepGProp.SurfaceProperties_s(face, g_props)
        props.area = g_props.Mass()

        if props.area <= 0:
            return None

        # 2. Compute AABB
        bbox = Bnd_Box()
        BRepBndLib.Add_s(face, bbox)

        if bbox.IsVoid():
            return None

        xmin, ymin, zmin, xmax, ymax, zmax = bbox.Get()
        props.bbox_min = (xmin, ymin, zmin)
        props.bbox_max = (xmax, ymax, zmax)

        # 3. Compute center (AABB center as approximation)
        props.center = (
            (xmin + xmax) / 2.0,
            (ymin + ymax) / 2.0,
            (zmin + zmax) / 2.0,
        )

        # 4. Check surface type and compute properties
        surface = BRepAdaptor_Surface(face)
        surface_type = surface.GetType()
        props.is_planar = surface_type == GeomAbs_Plane
        props.is_cylindrical = surface_type == GeomAbs_Cylinder

        if props.is_cylindrical:
            # Extract cylinder properties
            cylinder = surface.Cylinder()
            props.cylinder_radius = cylinder.Radius()

            # Get cylinder axis
            axis = cylinder.Axis()
            direction = axis.Direction()
            location = axis.Location()

            props.cylinder_axis = (direction.X(), direction.Y(), direction.Z())
            props.cylinder_location = (location.X(), location.Y(), location.Z())

            # For cylinder, use axis direction as "normal"
            props.normal = props.cylinder_axis

        elif props.is_planar:
            # Get plane and its normal
            plane = surface.Plane()
            axis = plane.Axis()
            direction = axis.Direction()

            # Check face orientation
            face_orientation = face.Orientation()
            sign = 1.0 if face_orientation == TopAbs_FORWARD else -1.0

            props.normal = (
                sign * direction.X(),
                sign * direction.Y(),
                sign * direction.Z(),
            )
        else:
            # Sample at mid-parameter for non-planar faces
            u_min, u_max, v_min, v_max = (
                surface.FirstUParameter(),
                surface.LastUParameter(),
                surface.FirstVParameter(),
                surface.LastVParameter(),
            )

            # Handle infinite parameters
            if math.isinf(u_min) or math.isinf(u_max):
                u_min, u_max = 0.0, 1.0
            if math.isinf(v_min) or math.isinf(v_max):
                v_min, v_max = 0.0, 1.0

            u_mid = (u_min + u_max) / 2.0
            v_mid = (v_min + v_max) / 2.0

            try:
                sl_props = BRepLProp_SLProps(surface, u_mid, v_mid, 1, 1e-6)
                if sl_props.IsNormalDefined():
                    normal_vec = sl_props.Normal()
                    props.normal = (normal_vec.X(), normal_vec.Y(), normal_vec.Z())
                else:
                    # Fallback to Z-axis
                    props.normal = (0, 0, 1)
            except:
                props.normal = (0, 0, 1)

        # Normalize the normal vector
        nx, ny, nz = props.normal
        norm = math.sqrt(nx * nx + ny * ny + nz * nz)
        if norm > 1e-9:
            props.normal = (nx / norm, ny / norm, nz / norm)

        return props

    except Exception as e:
        print(f"Warning: Failed to compute face properties: {e}", file=sys.stderr)
        return None


def load_step_and_extract_faces(
    step_path: Path,
) -> Tuple[Optional[Any], List[FaceProperties]]:
    """
    Load STEP file and extract all faces with their properties.

    Returns:
        Tuple of (shape, list of FaceProperties)
    """
    try:
        # Load STEP file
        shape = cq.importers.importStep(str(step_path))

        # Get underlying TopoDS_Shape
        if hasattr(shape, "val"):
            # It's a Workplane
            topo_shape = shape.val().wrapped
        elif hasattr(shape, "wrapped"):
            # It's a Shape object
            topo_shape = shape.wrapped
        elif hasattr(shape, "objects"):
            # Workplane with objects
            if shape.objects:
                topo_shape = (
                    shape.objects[0].wrapped
                    if hasattr(shape.objects[0], "wrapped")
                    else shape.objects[0]
                )
            else:
                topo_shape = None
        else:
            # Assume it's already a TopoDS_Shape
            topo_shape = shape

        if topo_shape is None:
            return None, []

        # Extract faces
        face_props_list = []
        explorer = TopExp_Explorer(topo_shape, TopAbs_FACE)

        while explorer.More():
            face = TopoDS.Face_s(explorer.Current())
            props = compute_face_properties(face)
            if props:
                face_props_list.append(props)
            explorer.Next()

        return topo_shape, face_props_list

    except Exception as e:
        print(f"Error loading STEP file {step_path}: {e}", file=sys.stderr)
        return None, []


def compute_shape_bbox(shape) -> Optional[Dict[str, float]]:
    """Compute axis-aligned bounding box for entire shape."""
    try:
        bbox = Bnd_Box()
        BRepBndLib.Add_s(shape, bbox)

        if bbox.IsVoid():
            return None

        xmin, ymin, zmin, xmax, ymax, zmax = bbox.Get()

        return {
            "x_min": xmin,
            "y_min": ymin,
            "z_min": zmin,
            "x_max": xmax,
            "y_max": ymax,
            "z_max": zmax,
        }
    except:
        return None


def select_constraint_face(
    face_props_list: List[FaceProperties],
) -> List[FaceProperties]:
    """
    Select constraint face(s): prefer planar, horizontal (bottom) faces.

    Priority:
    1. Planar faces with normal pointing down (-Z direction)
    2. Planar horizontal faces (normal in XY plane)
    3. Largest planar face
    4. Largest face overall

    Returns multiple constraint faces if part is complex:
    - 1 face if total faces <= 10
    - 2 faces if 10 < total faces <= 30
    - 3 faces if total faces > 30
    """
    if not face_props_list:
        return []

    # Determine how many constraint faces to select
    total_faces = len(face_props_list)
    if total_faces > 30:
        num_constraints = 3
    elif total_faces > 10:
        num_constraints = 2
    else:
        num_constraints = 1

    # Filter planar faces
    planar_faces = [fp for fp in face_props_list if fp.is_planar]

    if planar_faces:
        # Score faces based on:
        # - How horizontal they are (|nz| close to 1)
        # - Preference for bottom faces (nz < 0)
        # - Area
        scored_faces = []
        for fp in planar_faces:
            nx, ny, nz = fp.normal

            # Horizontal score: how close to purely vertical normal
            horizontal_score = abs(nz)

            # Bottom preference: bonus if pointing down
            bottom_bonus = 2.0 if nz < 0 else 1.0

            # Area score (normalized, assume max area is at most 10x min area)
            areas = [f.area for f in planar_faces]
            max_area = max(areas)
            area_score = fp.area / max_area if max_area > 0 else 0

            # Combined score
            score = horizontal_score * bottom_bonus * (0.5 + 0.5 * area_score)
            scored_faces.append((score, fp))

        # Return top N faces with highest scores
        scored_faces.sort(key=lambda x: x[0], reverse=True)
        return [fp for score, fp in scored_faces[:num_constraints]]
    else:
        # No planar faces - choose largest faces overall
        sorted_faces = sorted(face_props_list, key=lambda fp: fp.area, reverse=True)
        return sorted_faces[:num_constraints]


def dot_product(
    v1: Tuple[float, float, float], v2: Tuple[float, float, float]
) -> float:
    """Compute dot product of two 3D vectors."""
    return v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2]


def distance_3d(
    p1: Tuple[float, float, float], p2: Tuple[float, float, float]
) -> float:
    """Compute Euclidean distance between two 3D points."""
    dx, dy, dz = p1[0] - p2[0], p1[1] - p2[1], p1[2] - p2[2]
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def is_inner_cylinder(
    face_prop: FaceProperties, part_bbox: Dict[str, float], threshold: float = 0.15
) -> bool:
    """
    Determine if a cylinder is an inner hole vs outer shaft.

    Inner cylinders (holes) have their axis location away from part boundaries.
    Outer cylinders (shafts) have their axis location near part boundaries.

    Args:
        face_prop: Cylindrical face properties
        part_bbox: Bounding box of entire part
        threshold: Distance threshold as fraction of part size (0.15 = 15%)

    Returns:
        True if cylinder is likely an inner hole, False if outer shaft
    """
    if not face_prop.is_cylindrical:
        return False

    cx, cy, cz = face_prop.cylinder_location

    # Calculate part dimensions
    dx = part_bbox["x_max"] - part_bbox["x_min"]
    dy = part_bbox["y_max"] - part_bbox["y_min"]
    dz = part_bbox["z_max"] - part_bbox["z_min"]

    # Calculate normalized distance from each boundary (0 = at boundary, 0.5 = center)
    dist_x = (
        min(abs(cx - part_bbox["x_min"]), abs(cx - part_bbox["x_max"])) / dx
        if dx > 0
        else 0
    )
    dist_y = (
        min(abs(cy - part_bbox["y_min"]), abs(cy - part_bbox["y_max"])) / dy
        if dy > 0
        else 0
    )
    dist_z = (
        min(abs(cz - part_bbox["z_min"]), abs(cz - part_bbox["z_max"])) / dz
        if dz > 0
        else 0
    )

    # Inner cylinders should be away from at least 2 boundaries
    distances = [dist_x, dist_y, dist_z]
    distances_above_threshold = sum(1 for d in distances if d > threshold)

    # Consider it inner if it's away from boundaries in at least 2 dimensions
    return distances_above_threshold >= 2


def select_cylindrical_faces(
    face_props_list: List[FaceProperties],
    part_bbox: Dict[str, float],
    min_radius: float = 2.0,
    max_radius: float = 100.0,
    prefer_inner: bool = True,
) -> List[FaceProperties]:
    """
    Select cylindrical faces suitable for pressure loads.

    Filters cylinders by radius and prioritizes inner holes over outer shafts.

    Args:
        face_props_list: All face properties
        part_bbox: Bounding box of entire part
        min_radius: Minimum cylinder radius to consider (mm)
        max_radius: Maximum cylinder radius to consider (mm)
        prefer_inner: If True, prioritize inner holes over outer shafts

    Returns:
        List of cylindrical faces sorted by priority (inner holes first, then by radius)
    """
    # Filter by radius
    cylindrical_faces = [
        fp
        for fp in face_props_list
        if fp.is_cylindrical and min_radius <= fp.cylinder_radius <= max_radius
    ]

    if not cylindrical_faces:
        return []

    if prefer_inner:
        # Classify as inner or outer
        inner_cylinders = [
            fp for fp in cylindrical_faces if is_inner_cylinder(fp, part_bbox)
        ]
        outer_cylinders = [
            fp for fp in cylindrical_faces if not is_inner_cylinder(fp, part_bbox)
        ]

        # Sort each group by radius (smallest first)
        inner_cylinders.sort(key=lambda fp: fp.cylinder_radius)
        outer_cylinders.sort(key=lambda fp: fp.cylinder_radius)

        # Prioritize inner cylinders (holes) over outer cylinders (shafts)
        return inner_cylinders + outer_cylinders
    else:
        # Just sort by radius
        cylindrical_faces.sort(key=lambda fp: fp.cylinder_radius)
        return cylindrical_faces


def select_load_faces(
    constraint_faces: List[FaceProperties],
    face_props_list: List[FaceProperties],
    part_bbox: Dict[str, float],
    opposing_threshold: float = 0.85,
    num_loads: Optional[int] = None,
) -> List[FaceProperties]:
    """
    Select load face(s): prefer opposing face, then perpendicular faces.
    Uses the first (highest-scoring) constraint face as reference.

    Returns multiple load faces if part is complex:
    - 1 face if total faces < 20 (default)
    - 2 faces if 20 <= total faces < 40
    - 3 faces if total faces >= 40
    Or uses num_loads parameter if provided (to account for pressure loads)

    Scoring:
    - opposing score: s_opp = max(0, -dot(n_load, n_fix))
    - perpendicular score: s_perp = 1 - abs(dot)
    - distance score: normalized center distance
    - area score: face area
    """
    if not face_props_list or len(face_props_list) < 2 or not constraint_faces:
        return []

    # Determine how many load faces to select
    if num_loads is None:
        total_faces = len(face_props_list)
        if total_faces >= 40:
            num_loads = 3
        elif total_faces >= 20:
            num_loads = 2
        else:
            num_loads = 1

    # Use first (primary) constraint face as reference
    constraint_face = constraint_faces[0]
    n_fix = constraint_face.normal
    c_fix = constraint_face.center

    # Compute diagonal of part bbox for normalization
    dx = part_bbox["x_max"] - part_bbox["x_min"]
    dy = part_bbox["y_max"] - part_bbox["y_min"]
    dz = part_bbox["z_max"] - part_bbox["z_min"]
    diag = math.sqrt(dx * dx + dy * dy + dz * dz)

    if diag < 1e-9:
        diag = 1.0

    # Evaluate candidates (exclude all constraint faces)
    candidates = []

    for fp in face_props_list:
        if fp in constraint_faces:
            continue

        dot_val = dot_product(fp.normal, n_fix)

        # Opposing score (higher when normals point in opposite directions)
        s_opp = max(0.0, -dot_val)

        # Perpendicular score (higher when normals are perpendicular)
        s_perp = 1.0 - abs(dot_val)

        # Distance score (normalized)
        dist = distance_3d(fp.center, c_fix)
        d_norm = dist / diag

        # Area score (normalized by max area)
        a_norm = fp.area / constraint_face.area

        candidates.append(
            {
                "face": fp,
                "s_opp": s_opp,
                "s_perp": s_perp,
                "d_norm": d_norm,
                "a_norm": a_norm,
            }
        )

    if not candidates:
        return []

    # Check for opposing faces
    opposing_candidates = [c for c in candidates if c["s_opp"] >= opposing_threshold]

    if opposing_candidates:
        # Choose opposing faces with best scores
        opposing_candidates.sort(
            key=lambda c: c["s_opp"] * c["d_norm"] * c["a_norm"], reverse=True
        )
        return [c["face"] for c in opposing_candidates[:num_loads]]
    else:
        # Choose perpendicular faces with best scores
        candidates.sort(
            key=lambda c: c["s_perp"] * c["d_norm"] * c["a_norm"], reverse=True
        )
        return [c["face"] for c in candidates[:num_loads]]


def expand_bbox_to_selector(
    face_props: FaceProperties,
    box_pad: float,
    face_tol: float,
    part_bbox: Dict[str, float],
    enforce_thickness: bool = True,
) -> Dict[str, float]:
    """
    Expand face AABB into spatial selector box.

    Args:
        face_props: Face properties with bbox
        box_pad: Padding fraction (0.1 = 10% of part size)
        face_tol: Thickness tolerance fraction (0.05 = 5% of part size)
        part_bbox: Bounding box of entire part (for validation)
        enforce_thickness: If True, clamp thickness along face normal direction
    """
    x_min, y_min, z_min = face_props.bbox_min
    x_max, y_max, z_max = face_props.bbox_max
    cx, cy, cz = face_props.center
    nx, ny, nz = face_props.normal

    # Calculate part dimensions
    part_dx = part_bbox["x_max"] - part_bbox["x_min"]
    part_dy = part_bbox["y_max"] - part_bbox["y_min"]
    part_dz = part_bbox["z_max"] - part_bbox["z_min"]
    max_part_dim = max(part_dx, part_dy, part_dz)

    # Calculate absolute padding from percentage
    absolute_pad = box_pad * max_part_dim

    # Apply padding
    x_min -= absolute_pad
    x_max += absolute_pad
    y_min -= absolute_pad
    y_max += absolute_pad
    z_min -= absolute_pad
    z_max += absolute_pad

    # Enforce thickness if requested and face is planar
    if enforce_thickness and face_props.is_planar:
        # Find dominant axis of normal
        abs_nx, abs_ny, abs_nz = abs(nx), abs(ny), abs(nz)

        # Calculate absolute tolerance from percentage
        absolute_tol = face_tol * max_part_dim

        if abs_nx > abs_ny and abs_nx > abs_nz:
            # Normal mostly along X
            x_min = cx - absolute_tol
            x_max = cx + absolute_tol
        elif abs_ny > abs_nz:
            # Normal mostly along Y
            y_min = cy - absolute_tol
            y_max = cy + absolute_tol
        else:
            # Normal mostly along Z
            z_min = cz - absolute_tol
            z_max = cz + absolute_tol

    # Validate: ensure min < max for all dimensions
    if x_min >= x_max:
        cx_mid = (face_props.bbox_min[0] + face_props.bbox_max[0]) / 2
        x_min = cx_mid - max_part_dim * 0.01
        x_max = cx_mid + max_part_dim * 0.01
    if y_min >= y_max:
        cy_mid = (face_props.bbox_min[1] + face_props.bbox_max[1]) / 2
        y_min = cy_mid - max_part_dim * 0.01
        y_max = cy_mid + max_part_dim * 0.01
    if z_min >= z_max:
        cz_mid = (face_props.bbox_min[2] + face_props.bbox_max[2]) / 2
        z_min = cz_mid - max_part_dim * 0.01
        z_max = cz_mid + max_part_dim * 0.01

    return {
        "x_min": x_min,
        "y_min": y_min,
        "z_min": z_min,
        "x_max": x_max,
        "y_max": y_max,
        "z_max": z_max,
    }


def determine_load_direction(
    normal: Tuple[float, float, float], randomize: bool = True
) -> str:
    """
    Determine dominant axis direction string from normal vector.

    Args:
        normal: Face normal vector
        randomize: If True, randomly vary direction to include lateral loads

    Returns direction string like '-x', '+y', '-z', etc.
    """
    nx, ny, nz = normal
    abs_nx, abs_ny, abs_nz = abs(nx), abs(ny), abs(nz)

    if randomize:
        # Randomly choose between:
        # 1. Normal direction (into surface) - 50%
        # 2. Perpendicular direction (lateral) - 30%
        # 3. Opposite to normal (away from surface) - 20%
        choice = random.random()

        if choice < 0.5:
            # Apply load opposite to face normal (into the surface)
            nx, ny, nz = -nx, -ny, -nz
        elif choice < 0.8:
            # Apply lateral load perpendicular to normal
            # Find perpendicular vector
            if abs_nz < abs_nx:
                perp = (nz, 0, -nx)  # perpendicular in XZ plane
            else:
                perp = (0, nz, -ny)  # perpendicular in YZ plane
            nx, ny, nz = perp
            # Randomly flip direction
            if random.random() < 0.5:
                nx, ny, nz = -nx, -ny, -nz
        else:
            # Apply load in direction of normal (away from surface)
            pass  # Use normal as-is
    else:
        # Apply load opposite to face normal (into the surface)
        nx, ny, nz = -nx, -ny, -nz

    # Determine dominant axis
    abs_nx, abs_ny, abs_nz = abs(nx), abs(ny), abs(nz)

    if abs_nx > abs_ny and abs_nx > abs_nz:
        return "-x" if nx < 0 else "+x"
    elif abs_ny > abs_nz:
        return "-y" if ny < 0 else "+y"
    else:
        return "-z" if nz < 0 else "+z"


def determine_cylinder_axis(cylinder_axis: Tuple[float, float, float]) -> str:
    """
    Determine dominant axis direction for cylinder.

    Returns:
        'x', 'y', or 'z'
    """
    ax, ay, az = [abs(x) for x in cylinder_axis]

    if ax > ay and ax > az:
        return "x"
    elif ay > az:
        return "y"
    else:
        return "z"


def generate_loadcase_json(
    step_path: Path,
    shape,
    constraint_faces: List[FaceProperties],
    load_faces: List[FaceProperties],
    cylindrical_faces: List[FaceProperties],
    part_bbox: Dict[str, float],
    force_newtons: float,
    pressure_mpa: float,
    face_tol: float,
    box_pad: float,
) -> Dict[str, Any]:
    """
    Generate complete load-case JSON structure.
    Supports multiple constraint faces and multiple load faces with varied directions.
    Prioritizes pressure loads - if pressure loads exist, reduces number of distributed loads.
    """
    # Generate spatial selectors for all constraint faces
    spatial_selectors = []
    boundary_conditions = []

    for i, constraint_face in enumerate(constraint_faces, 1):
        constraint_selector = expand_bbox_to_selector(
            constraint_face, box_pad, face_tol, part_bbox, enforce_thickness=True
        )

        selector_id = (
            f"constraint_region_{i}"
            if len(constraint_faces) > 1
            else "constraint_region"
        )
        bc_name = (
            f"fix_constraint_{i}" if len(constraint_faces) > 1 else "fix_constraint"
        )

        spatial_selectors.append(
            {"id": selector_id, "type": "box_3d", "query": constraint_selector}
        )

        boundary_conditions.append(
            {
                "name": bc_name,
                "region_id": selector_id,
                "type": "fixed_displacement",
                "dof_lock": {"x": True, "y": True, "z": True},
            }
        )

    # Initialize loads list
    loads = []

    # Calculate target number of total loads based on part complexity
    total_faces = len(load_faces) + len(cylindrical_faces) + len(constraint_faces)
    if total_faces >= 40:
        target_total_loads = 3
    elif total_faces >= 20:
        target_total_loads = 2
    else:
        target_total_loads = 1

    # Generate pressure loads for cylindrical faces (holes/pins)
    # These are prioritized as they're more interesting for FEA
    num_pressure_loads = min(len(cylindrical_faces), target_total_loads)

    for i, cyl_face in enumerate(cylindrical_faces[:num_pressure_loads], 1):
        load_name = (
            f"pressure_cylinder_{i}"
            if len(cylindrical_faces) > 1
            else "pressure_cylinder"
        )

        # Determine cylinder axis orientation
        normal_axis = determine_cylinder_axis(cyl_face.cylinder_axis)

        # Random choice: inward (compression, like bearing pressure) or outward (expansion)
        direction = "inward" if random.random() < 0.7 else "outward"

        loads.append(
            {
                "name": load_name,
                "type": "pressure",
                "center": list(cyl_face.cylinder_location),
                "radius": cyl_face.cylinder_radius,
                "pressure_mpa": pressure_mpa,
                "direction": direction,
                "normal_axis": normal_axis,
            }
        )

    # Generate load selectors for remaining load faces with varied directions
    # Reduce count if we already have pressure loads (they take priority)
    num_distributed_loads = max(0, target_total_loads - len(loads))

    for i, load_face in enumerate(load_faces[:num_distributed_loads], 1):
        load_selector = expand_bbox_to_selector(
            load_face, box_pad, face_tol, part_bbox, enforce_thickness=True
        )

        selector_id = f"load_region_{i}" if len(load_faces) > 1 else "load_region"
        load_name = f"applied_load_{i}" if len(load_faces) > 1 else "applied_load"

        spatial_selectors.append(
            {"id": selector_id, "type": "box_3d", "query": load_selector}
        )

        # Determine load direction with randomization
        load_direction = determine_load_direction(load_face.normal, randomize=True)

        # Split force among distributed loads being added
        force_per_load = (
            force_newtons / num_distributed_loads
            if num_distributed_loads > 0
            else force_newtons
        )

        loads.append(
            {
                "name": load_name,
                "region_id": selector_id,
                "type": "distributed_force",
                "direction": load_direction,
                "magnitude_newtons": force_per_load,
            }
        )

    # Build JSON structure
    loadcase = {
        "meta": {
            "problem_id": "DEEPCAD_AUTO",
            "description": "Auto-generated load case from DeepCAD STEP.",
            "analysis_type": "3d",
            "source_step": str(step_path.absolute()),
        },
        "design_domain": {"units": "mm", "bounds": part_bbox},
        "spatial_selectors": spatial_selectors,
        "boundary_conditions": boundary_conditions,
        "loads": loads,
        "material": {
            "type": "isotropic",
            "name": "Aluminium_6061",
            "elastic_modulus_mpa": 69000,
            "poissons_ratio": 0.33,
            "density_g_cm3": 2.70,
        },
    }

    return loadcase


def process_step_file(
    step_path: Path,
    out_dir: Path,
    force_newtons: float,
    face_tol: float,
    box_pad: float,
    prefer_opposing: float,
) -> Dict[str, Any]:
    """
    Process a single STEP file and generate load-case JSON.

    Returns:
        Dictionary with processing results and statistics
    """
    result = {
        "step_file": step_path.name,
        "success": False,
        "error": None,
        "constraint_area": 0.0,
        "load_area": 0.0,
        "load_direction": None,
        "constraint_volume": 0.0,
        "load_volume": 0.0,
        "viz_error": None,
        "png_file": None,
        "num_constraints": 0,
        "num_loads": 0,
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

        # Select cylindrical faces for pressure loads (prioritize inner holes)
        # Lower min_radius to 0.1mm to catch even small chamfers/fillets that indicate holes
        cylindrical_faces = select_cylindrical_faces(
            face_props_list,
            part_bbox,
            min_radius=0.1,
            max_radius=100.0,
            prefer_inner=True,
        )

        # Debug: Log cylindrical faces found
        if cylindrical_faces:
            for idx, cf in enumerate(cylindrical_faces, 1):
                is_inner = is_inner_cylinder(cf, part_bbox)
                hole_type = "INNER_HOLE" if is_inner else "outer_shaft"
                print(
                    f"  Cyl{idx}: r={cf.cylinder_radius:.2f}mm at {tuple(round(x, 2) for x in cf.cylinder_location)} [{hole_type}]",
                    file=sys.stderr,
                )

        # Generate JSON
        loadcase = generate_loadcase_json(
            step_path,
            shape,
            constraint_faces,
            load_faces,
            cylindrical_faces,
            part_bbox,
            force_newtons,
            pressure_mpa=5.0,
            face_tol=face_tol,
            box_pad=box_pad,
        )

        # Write JSON file
        out_file = out_dir / f"{step_path.stem}.json"
        with open(out_file, "w") as f:
            json.dump(loadcase, f, indent=2)

        # Render visualization to PNG
        png_file = out_dir / f"{step_path.stem}.png"
        try:
            load_case = parse_load_case(out_file)
            fea = load_case.get_fea_analyzer(mesher="netgen")
            fea.show(filename=str(png_file), interactive=False, camera_position="iso")
            result["png_file"] = str(png_file)
        except KeyboardInterrupt:
            raise  # Allow user to stop the script
        except Exception as viz_error:
            result["viz_error"] = str(viz_error)
            # Log but continue processing
            print(f"Could not show debug mesh: {viz_error}", file=sys.stderr)

        # Compute volumes for statistics
        # Sum volumes from all constraint selectors
        constraint_vol = 0.0
        num_constraints = len(constraint_faces)
        for i in range(num_constraints):
            cs = loadcase["spatial_selectors"][i]["query"]
            constraint_vol += (
                (cs["x_max"] - cs["x_min"])
                * (cs["y_max"] - cs["y_min"])
                * (cs["z_max"] - cs["z_min"])
            )

        # Sum volumes from all load selectors (after constraint selectors)
        # Only count distributed loads that have spatial selectors
        distributed_loads = [
            load for load in loadcase["loads"] if load.get("region_id")
        ]
        num_distributed = len(distributed_loads)
        load_vol = 0.0
        for i in range(num_distributed):
            ls = loadcase["spatial_selectors"][num_constraints + i]["query"]
            load_vol += (
                (ls["x_max"] - ls["x_min"])
                * (ls["y_max"] - ls["y_min"])
                * (ls["z_max"] - ls["z_min"])
            )

        # Collect load directions
        load_directions = [load["direction"] for load in loadcase["loads"]]

        # Update result
        result["success"] = True
        result["constraint_area"] = sum(cf.area for cf in constraint_faces)
        result["load_area"] = sum(lf.area for lf in load_faces)
        result["load_direction"] = ",".join(load_directions)
        result["constraint_volume"] = constraint_vol
        result["load_volume"] = load_vol
        result["json_file"] = str(out_file)
        result["num_constraints"] = num_constraints
        result["num_loads"] = len(
            loadcase["loads"]
        )  # Total loads (pressure + distributed)
        result["num_cylinders"] = len(cylindrical_faces)
        result["cylinder_radii"] = ",".join(
            [f"{cf.cylinder_radius:.2f}" for cf in cylindrical_faces]
        )

    except Exception as e:
        result["error"] = str(e)

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Convert DeepCAD STEP parts into FEA load-case JSON files."
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
        "--out_dir", type=Path, required=True, help="Output directory for JSON files"
    )
    parser.add_argument(
        "--force_newtons",
        type=float,
        default=1000.0,
        help="Total force magnitude in Newtons (default: 1000)",
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

    # Process each STEP file
    results = []
    for i, step_path in enumerate(step_files, 1):
        print(f"[{i}/{len(step_files)}] Processing {step_path.name}...", end=" ")

        result = process_step_file(
            step_path,
            args.out_dir,
            args.force_newtons,
            args.face_tol,
            args.box_pad,
            args.prefer_opposing,
        )

        results.append(result)

        if result["success"]:
            viz_status = "✓viz" if result.get("png_file") else "✗viz"
            num_c = result.get("num_constraints", 1)
            num_l = result.get("num_loads", 1)
            num_cyl = result.get("num_cylinders", 0)
            cyl_info = (
                f", {num_cyl}×Cyl={result.get('cylinder_radii', '')}mm"
                if num_cyl > 0
                else ""
            )
            print(
                f"✓ {viz_status} ({num_c}×C={result['constraint_area']:.1f}mm², {num_l}×L={result['load_area']:.1f}mm², dir={result['load_direction']}{cyl_info})"
            )
        else:
            print(f"✗ ({result['error']})")

    # Write summary CSV
    csv_path = args.summary_csv or (args.out_dir / "summary.csv")
    with open(csv_path, "w", newline="") as csvfile:
        fieldnames = [
            "step_file",
            "success",
            "num_constraints",
            "num_loads",
            "num_cylinders",
            "constraint_area",
            "load_area",
            "cylinder_radii",
            "load_direction",
            "constraint_volume",
            "load_volume",
            "error",
            "png_file",
            "viz_error",
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for result in results:
            row = {k: result.get(k, "") for k in fieldnames}
            writer.writerow(row)

    # Print summary
    successful = sum(1 for r in results if r["success"])
    failed = len(results) - successful

    print(f"\n{'='*60}")
    print(f"Processing complete:")
    print(f"  Successful: {successful}/{len(results)}")
    print(f"  Failed: {failed}/{len(results)}")
    print(f"  JSON files: {args.out_dir}")
    print(f"  Summary CSV: {csv_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
