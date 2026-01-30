"""
Load Case Parser for FEA

Parses JSON load case files into RapidCadPy constraint and load objects.

Example JSON format:
{
  "meta": {
    "problem_id": "CANTILEVER_2D_BENCHMARK",
    "description": "Standard 100x20mm beam fixed at left, load at top right.",
    "analysis_type": "plane_stress",
    "thickness_mm": 5.0
  },
  "design_domain": {...},
  "spatial_selectors": [...],
  "boundary_conditions": [...],
  "loads": [...],
  "material": {...}
}
"""

import json
import logging
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

from rapidcadpy.fea.boundary_conditions import (
    DistributedLoad,
    PointLoad,
    ConcentratedLoad,
    LinearDistributedLoad,
    PressureLoad,
    FixedConstraint,
    CylindricalConstraint,
)
from rapidcadpy.fea.materials import Material

logger = logging.getLogger(__name__)


class LoadCaseValidationError(Exception):
    """Raised when load case JSON validation fails."""

    pass


def validate_load_case_data(data: Dict[str, Any]) -> List[str]:
    """
    Validate load case data structure and return list of warnings/errors.

    Args:
        data: Parsed JSON data dictionary

    Returns:
        List of validation warning messages (empty if valid)

    Raises:
        LoadCaseValidationError: If critical validation fails
    """
    warnings = []
    errors = []

    # Check required sections
    if "material" not in data:
        errors.append("Missing required 'material' section")

    if "boundary_conditions" not in data or not data["boundary_conditions"]:
        errors.append("Missing or empty 'boundary_conditions' section")

    if "loads" not in data or not data["loads"]:
        errors.append("Missing or empty 'loads' section")

    # Validate material section
    if "material" in data:
        mat = data["material"]
        required_mat_fields = ["elastic_modulus_mpa", "poissons_ratio", "density_g_cm3"]
        for field in required_mat_fields:
            if field not in mat:
                errors.append(f"Missing required material field: '{field}'")

        # Validate material values
        if mat.get("poissons_ratio", 0) < 0 or mat.get("poissons_ratio", 0) > 0.5:
            warnings.append(
                f"Poisson's ratio {mat.get('poissons_ratio')} is outside typical range [0, 0.5]"
            )

        if mat.get("elastic_modulus_mpa", 0) <= 0:
            errors.append("Elastic modulus must be positive")

    # Validate spatial selectors if present
    selector_ids = set()
    if "spatial_selectors" in data:
        for i, sel in enumerate(data["spatial_selectors"]):
            if "id" not in sel:
                errors.append(f"Spatial selector at index {i} missing 'id'")
            else:
                selector_ids.add(sel["id"])

            if "type" not in sel:
                errors.append(f"Spatial selector '{sel.get('id', i)}' missing 'type'")

            if "query" not in sel:
                errors.append(f"Spatial selector '{sel.get('id', i)}' missing 'query'")

    # Validate boundary conditions
    if "boundary_conditions" in data:
        for i, bc in enumerate(data["boundary_conditions"]):
            if "name" not in bc:
                warnings.append(f"Boundary condition at index {i} missing 'name'")

            # Check for region specification or cylindrical geometry
            bc_type = bc.get("type", "fixed_displacement")
            has_region_id = "region_id" in bc
            has_inline_region = "region" in bc
            is_cylindrical = (
                bc_type == "cylindrical" and "center" in bc and "radius" in bc
            )

            if not has_region_id and not has_inline_region and not is_cylindrical:
                errors.append(
                    f"Boundary condition '{bc.get('name', i)}' missing 'region_id', 'region', or cylindrical geometry (center+radius)"
                )
            elif has_region_id and bc["region_id"] not in selector_ids and selector_ids:
                warnings.append(
                    f"Boundary condition '{bc.get('name', i)}' references unknown region '{bc['region_id']}'"
                )

            if "type" not in bc:
                errors.append(
                    f"Boundary condition '{bc.get('name', i)}' missing 'type'"
                )

    # Validate loads
    if "loads" in data:
        for i, load in enumerate(data["loads"]):
            if "name" not in load:
                warnings.append(f"Load at index {i} missing 'name'")

            load_type = load.get("type", "")

            # Pressure loads can have inline geometry (center, radius) instead of region_id
            if load_type == "pressure":
                has_center = "center" in load
                has_radius = "radius" in load
                has_pressure = "pressure_mpa" in load

                if not (has_center and has_radius):
                    errors.append(
                        f"Pressure load '{load.get('name', i)}' missing 'center' and/or 'radius'"
                    )
                if not has_pressure:
                    errors.append(
                        f"Pressure load '{load.get('name', i)}' missing 'pressure_mpa'"
                    )
            else:
                # Check for region specification (region_id, region, or selector)
                has_region_id = "region_id" in load
                has_inline_region = "region" in load
                has_selector = "selector" in load

                if not has_region_id and not has_inline_region and not has_selector:
                    errors.append(
                        f"Load '{load.get('name', i)}' missing 'region_id', 'region', or 'selector'"
                    )
                elif (
                    has_region_id
                    and load["region_id"] not in selector_ids
                    and selector_ids
                ):
                    warnings.append(
                        f"Load '{load.get('name', i)}' references unknown region '{load['region_id']}'"
                    )

                if "type" not in load:
                    errors.append(f"Load '{load.get('name', i)}' missing 'type'")

                # Check that force is specified
                has_force = "vector_newtons" in load or "magnitude_newtons" in load
                if not has_force:
                    errors.append(
                        f"Load '{load.get('name', i)}' missing force specification "
                        "(need 'vector_newtons' or 'magnitude_newtons')"
                    )

    # Raise if there are critical errors
    if errors:
        error_msg = "Load case validation failed:\n" + "\n".join(
            f"  - {e}" for e in errors
        )
        raise LoadCaseValidationError(error_msg)

    # Log warnings
    for warning in warnings:
        logger.warning(f"Load case validation: {warning}")

    return warnings


@dataclass
class SpatialSelector:
    """Represents a spatial region selector from the load case."""

    id: str
    type: str  # 'box_2d', 'box_3d', 'sphere', 'cylinder', etc.
    query: Dict[
        str, float
    ]  # For boxes: x_min/max, y_min/max, z_min/max. For cylinders: center, radius, normal_axis

    def to_location_string(self) -> str | tuple | dict:
        """
        Convert spatial selector to RapidCadPy location string, coordinate tuple, or box dict.

        For simple box selectors with a thin dimension, we map to standard locations like
        'x_min', 'x_max', 'top', 'bottom', etc.

        For point selectors with single coordinates (x, y, z), returns a tuple.

        For volume/face selectors (boxes with all ranges), returns the query dict directly
        for range-based node selection.
        """
        # Check if this is a point selector (single coordinates x, y, z)
        if all(k in self.query for k in ["x", "y", "z"]) and not any(
            k in self.query
            for k in ["x_min", "x_max", "y_min", "y_max", "z_min", "z_max"]
        ):
            # Point location - return as tuple for point loads
            return (self.query["x"], self.query["y"], self.query["z"])

        # Extract bounds
        x_min = self.query.get("x_min", float("-inf"))
        x_max = self.query.get("x_max", float("inf"))
        y_min = self.query.get("y_min", float("-inf"))
        y_max = self.query.get("y_max", float("inf"))
        z_min = self.query.get("z_min", float("-inf"))
        z_max = self.query.get("z_max", float("inf"))

        x_range = x_max - x_min
        y_range = y_max - y_min
        z_range = z_max - z_min

        if self.type == "box_2d":
            # Check if this is selecting a thin slice (a boundary)
            # If x range is very small, it's selecting x_min or x_max face
            if x_range < 1.0:
                if x_min < 1.0:
                    return "x_min"
                else:
                    return "x_max"

            # If y range is very small, it's selecting y_min or y_max face
            if y_range < 1.0:
                if y_min < 1.0:
                    return "y_min"
                else:
                    return "y_max"

        elif self.type == "box_3d":
            # Determine which face based on thin dimensions
            if x_range < 1.0:
                return "x_min" if x_min < 1.0 else "x_max"
            if y_range < 1.0:
                return "y_min" if y_min < 1.0 else "y_max"
            if z_range < 1.0:
                return "z_min" if z_min < 1.0 else "z_max"

        # For thicker selection boxes, return the query dict directly
        # This allows range-based node selection in boundary_conditions.py
        return self.query


@dataclass
class BoundaryCondition:
    """Represents a boundary condition from the load case."""

    name: str
    region_id: Optional[str] = None  # Reference to a spatial selector
    region: Optional[Dict[str, Any]] = None  # Inline region definition
    type: str = (
        "fixed_displacement"  # 'fixed_displacement', 'prescribed_displacement', 'cylindrical', etc.
    )
    dof_lock: Optional[Dict[str, bool]] = None
    displacement: Optional[Dict[str, float]] = None
    # Cylindrical constraint specific fields
    center: Optional[List[float]] = (
        None  # Center point for cylindrical constraint [x, y, z]
    )
    radius: Optional[float] = None  # Radius for cylindrical constraint
    normal_axis: Optional[str] = (
        "z"  # Axis along which cylinder extends ('x', 'y', 'z')
    )

    def to_constraint(
        self, selectors: Dict[str, SpatialSelector], tolerance: float = 1.0
    ):
        """Convert to RapidCadPy constraint (FixedConstraint or CylindricalConstraint)."""

        # Handle cylindrical constraints separately (they have inline geometry)
        if self.type == "cylindrical":
            if not self.center or self.radius is None:
                raise ValueError(
                    f"Cylindrical constraint '{self.name}' missing required fields: center, radius"
                )

            return CylindricalConstraint(
                center=tuple(self.center),
                radius=self.radius,
                normal_axis=self.normal_axis or "z",
                tolerance=tolerance,
            )

        # For other constraint types, get location from selectors
        # First try to get selector from region_id
        if self.region_id:
            location = selectors.get(self.region_id).query
        # Otherwise, create a temporary selector from inline region
        elif self.region:
            temp_selector = SpatialSelector(
                id=f"inline_{self.name}",
                type=self.region.get("type", "box_2d"),
                query=self.region.get("query", {}),
            )
            location = temp_selector.to_location_string()
        else:
            # Ultimate fallback
            location = "x_min"
            logger.warning(
                f"Boundary condition '{self.name}' has no region, using 'x_min'"
            )

        if self.type == "fixed_displacement":
            return FixedConstraint(location=location, tolerance=tolerance)
        else:
            # For now, treat all as fixed constraints
            # Can be extended for prescribed displacements
            logger.warning(
                f"Unsupported boundary condition type '{self.type}', treating as fixed"
            )
            return FixedConstraint(location=location, tolerance=tolerance)


def direction_string_to_vector(direction: str) -> Dict[str, float]:
    """
    Convert a direction string like 'x', 'y', 'z', '-x', '-y', '-z' to a unit vector.

    Args:
        direction: Direction string (e.g., 'z', '-y', 'x', 'inward', 'outward')

    Returns:
        Unit vector as dict with x, y, z components
    """
    direction = direction.lower().strip()

    # Standard axis directions
    direction_map = {
        "x": {"x": 1.0, "y": 0.0, "z": 0.0},
        "-x": {"x": -1.0, "y": 0.0, "z": 0.0},
        "+x": {"x": 1.0, "y": 0.0, "z": 0.0},
        "y": {"x": 0.0, "y": 1.0, "z": 0.0},
        "-y": {"x": 0.0, "y": -1.0, "z": 0.0},
        "+y": {"x": 0.0, "y": 1.0, "z": 0.0},
        "z": {"x": 0.0, "y": 0.0, "z": 1.0},
        "-z": {"x": 0.0, "y": 0.0, "z": -1.0},
        "+z": {"x": 0.0, "y": 0.0, "z": 1.0},
        # Common aliases
        "up": {"x": 0.0, "y": 0.0, "z": 1.0},
        "down": {"x": 0.0, "y": 0.0, "z": -1.0},
        "left": {"x": -1.0, "y": 0.0, "z": 0.0},
        "right": {"x": 1.0, "y": 0.0, "z": 0.0},
        "forward": {"x": 0.0, "y": 1.0, "z": 0.0},
        "backward": {"x": 0.0, "y": -1.0, "z": 0.0},
        # Pressure load directions (handled by PressureLoad class)
        "inward": {"x": 0.0, "y": 0.0, "z": 0.0},  # Placeholder
        "outward": {"x": 0.0, "y": 0.0, "z": 0.0},  # Placeholder
    }

    if direction in direction_map:
        return direction_map[direction]

    # Default to -z (downward) if unknown
    logger.warning(f"Unknown direction '{direction}', defaulting to -z (downward)")
    return {"x": 0.0, "y": 0.0, "z": -1.0}


@dataclass
class LoadDefinition:
    """Represents a load from the load case."""

    name: str
    region_id: Optional[str] = None  # Reference to a spatial selector
    region: Optional[Dict[str, Any]] = None  # Inline region definition (legacy)
    selector: Optional[Dict[str, Any]] = None  # Inline selector definition (preferred)
    type: str = "point_force"  # 'point_force', 'distributed_force', 'pressure', etc.
    vector_newtons: Optional[Dict[str, float]] = None
    magnitude_newtons: Optional[float] = None
    direction: Optional[str] = (
        None  # String like 'x', 'y', 'z', '-z', 'inward', 'outward'
    )
    direction_vector: Optional[Dict[str, float]] = None  # Named vector {x, y, z}
    distribution: Optional[str] = None
    gradient_axis: Optional[str] = "z"  # For gradient loads (x, y, z)
    gradient_profile: Optional[str] = (
        "triangular_invert"  # triangular, triangular_invert
    )
    # Pressure load specific fields
    pressure_mpa: Optional[float] = None  # Pressure magnitude in MPa
    center: Optional[List[float]] = (
        None  # Center point for cylindrical pressure [x, y, z]
    )
    radius: Optional[float] = None  # Radius for cylindrical pressure
    normal_axis: Optional[str] = (
        "z"  # Axis perpendicular to circular face ('x', 'y', 'z')
    )

    def __post_init__(self):
        """Convert direction string to vector if provided."""
        if self.direction and not self.direction_vector:
            self.direction_vector = direction_string_to_vector(self.direction)

    def get_direction_vector(self) -> Dict[str, float]:
        """
        Get the direction as a unit vector.

        Returns:
            Dict with x, y, z components representing the direction
        """
        if self.direction_vector:
            return self.direction_vector
        elif self.direction:
            return direction_string_to_vector(self.direction)
        elif self.vector_newtons:
            # Derive direction from force vector
            fx = self.vector_newtons.get("x", 0.0)
            fy = self.vector_newtons.get("y", 0.0)
            fz = self.vector_newtons.get("z", 0.0)
            magnitude = (fx**2 + fy**2 + fz**2) ** 0.5
            if magnitude > 0:
                return {
                    "x": fx / magnitude,
                    "y": fy / magnitude,
                    "z": fz / magnitude,
                }
        # Default to -z (downward)
        return {"x": 0.0, "y": 0.0, "z": -1.0}

    def to_load(
        self, selectors: Dict[str, SpatialSelector], tolerance: float = 1.0
    ) -> PointLoad | DistributedLoad | PressureLoad:
        """Convert to RapidCadPy Load."""

        # Handle pressure loads separately (they have inline geometry)
        if self.type == "pressure":
            if not self.center or self.radius is None or self.pressure_mpa is None:
                raise ValueError(
                    f"Pressure load '{self.name}' missing required fields: center, radius, pressure_mpa"
                )

            return PressureLoad(
                center=tuple(self.center),
                radius=self.radius,
                pressure=self.pressure_mpa,
                direction=self.direction or "inward",
                normal_axis=self.normal_axis or "z",
                tolerance=tolerance,
            )

        # For other load types, get location from selectors
        location = None

        # First try to get selector from region_id
        if self.region_id:
            selector = selectors.get(self.region_id)
            location = selector.query
        # Try inline selector (preferred new format)
        elif self.selector:
            query = self.selector.get("query", {})

            # For point loads with single coordinates (x, y, z), use them directly as a tuple
            if self.type == "point_force" and all(k in query for k in ["x", "y", "z"]):
                x = query["x"]
                y = query["y"]
                z = query["z"]
                location = (x, y, z)
                logger.info(f"Point load '{self.name}' at coordinates {location}")
            else:
                # For other cases (ranges, distributed loads), convert to location string
                temp_selector = SpatialSelector(
                    id=f"inline_{self.name}",
                    type=self.selector.get("type", "box_2d"),
                    query=query,
                )
                location = temp_selector.to_location_string()
        # Try legacy inline region format
        elif self.region:
            query = selectors.get(self.region).get("query", {})

            # For point loads with single coordinates (x, y, z), use them directly as a tuple
            if self.type == "point_force" and all(k in query for k in ["x", "y", "z"]):
                x = query["x"]
                y = query["y"]
                z = query["z"]
                location = (x, y, z)
                logger.info(f"Point load '{self.name}' at coordinates {location}")
            else:
                # For other cases (ranges, distributed loads), convert to location string
                temp_selector = SpatialSelector(
                    id=f"inline_{self.name}",
                    type=self.region.get("type", "box_2d"),
                    query=query,
                )
                location = temp_selector.to_location_string()
        else:
            # Ultimate fallback - use load name or default
            location = self.name
            logger.warning(
                f"Load '{self.name}' has no region, using load name as location"
            )

        # Determine force and direction
        if self.type == "distributed_force":
            # Use stored direction or default
            dir_str = self.direction or "z"
            return DistributedLoad(
                location=location,
                force=self.magnitude_newtons,
                direction=dir_str,
                tolerance=tolerance,
            )
        elif self.type == "point_force":
            dir_str = self.direction or "z"
            # If location is a dict (box query), use the centroid as the point
            if isinstance(location, dict):
                cx = (location.get("x_min", 0) + location.get("x_max", 0)) / 2
                cy = (location.get("y_min", 0) + location.get("y_max", 0)) / 2
                cz = (location.get("z_min", 0) + location.get("z_max", 0)) / 2
                point = (cx, cy, cz)
            else:
                point = location
            return PointLoad(
                point=point,
                force=self.magnitude_newtons,
                direction=dir_str,
                tolerance=tolerance,
            )
        elif self.type == "concentrated_force":
            # Direct mapping to ConcentratedLoad
            return ConcentratedLoad(
                location=location,
                force=self.vector_newtons or self.magnitude_newtons,
                direction=self.direction,
                tolerance=tolerance,
            )
        elif (
            self.type == "linear_distributed_force" or self.type == "hydrostatic_force"
        ):
            dir_str = self.direction or "y"
            return LinearDistributedLoad(
                location=location,
                force=self.magnitude_newtons,
                direction=dir_str,
                axis=self.gradient_axis or "z",
                profile=self.gradient_profile or "triangular_invert",
                tolerance=tolerance,
            )
        else:
            raise ValueError(f"Load '{self.name}' has no force specification")


@dataclass
class MaterialDefinition:
    """Represents material properties from the load case."""

    type: str  # 'isotropic', 'orthotropic', etc.
    elastic_modulus_mpa: float
    poissons_ratio: float
    density_g_cm3: float
    yield_strength_mpa: Optional[float] = None

    def to_material_enum(self) -> Material:
        """
        Map material properties to closest RapidCadPy Material enum.

        This finds the best matching material in the Material library
        by comparing Young's Modulus (E).
        """
        from rapidcadpy.fea.materials import MaterialProperties

        E = self.elastic_modulus_mpa

        # Collect all valid materials from the Material class
        available_materials = []
        for attr_name in dir(Material):
            # Skip private/special attributes
            if attr_name.startswith("_"):
                continue

            attr = getattr(Material, attr_name)
            if isinstance(attr, MaterialProperties):
                available_materials.append(attr)

        if not available_materials:
            logger.warning("No materials found in Material class, defaulting to STEEL")
            return Material.STEEL

        # Find best match by minimizing difference in Young's Modulus
        best_match = None
        best_diff = float("inf")

        for mat in available_materials:
            diff = abs(mat.E - E)
            if diff < best_diff:
                best_diff = diff
                best_match = mat

        if best_match:
            logger.info(
                f"Matched input E={E} MPa to material {best_match.name} (E={best_match.E} MPa)"
            )
            return best_match

        return Material.STEEL


@dataclass
class DesignDomain:
    """
    Represents a design domain geometry that can be non-cubic.

    Supports:
    - Simple primitives: box, cylinder, sphere
    - Composite shapes: L-shape, T-shape, plus
    - CSG operations: union, subtract, intersect
    """

    units: str = "mm"
    shape_type: str = "box"  # box, cylinder, sphere, l_shape, t_shape, csg
    bounds: Optional[Dict[str, float]] = None  # For box: x_min, x_max, etc.
    params: Optional[Dict[str, Any]] = None  # Shape-specific parameters
    operations: Optional[List[Dict[str, Any]]] = None  # CSG operations

    def describe(self) -> str:
        """
        Generate a natural language description of the design domain for the LLM.

        This description helps the planner understand the valid spatial region where
        geometry can be created.
        """
        desc = ["Design Domain Configuration:"]
        desc.append(f"- Shape Type: {self.shape_type}")
        desc.append(f"- Units: {self.units}")

        if self.shape_type == "box" and self.bounds:
            desc.append(f"- Bounding Box: {self.bounds}")
            desc.append("  (Rectangular design space)")

        elif self.shape_type in ["l_shape", "t_shape", "plus"]:
            desc.append(f"- Parameters: {self.params}")
            desc.append(f"  (Composite {self.shape_type} design space)")

        elif self.shape_type == "cylinder":
            desc.append(f"- Cylinder Parameters: {self.params}")

        elif self.shape_type == "sphere":
            desc.append(f"- Sphere Parameters: {self.params}")

        elif self.shape_type == "csg" and self.operations:
            desc.append(
                f"- Constructive Solid Geometry (CSG) with {len(self.operations)} operations:"
            )
            for i, op in enumerate(self.operations):
                op_type = op.get("type", "unknown")
                operation = op.get("operation", "add")
                params = op.get("params", {})
                desc.append(f"  {i+1}. {operation.upper()} {op_type} with {params}")

        # Always calculate and add the bounding box as a reference
        try:
            bb = self.get_bounding_box()
            desc.append(f"- Overall Bounding Box: {bb}")
        except Exception as e:
            logger.warning(f"Could not calculate bounding box for description: {e}")

        return "\n".join(desc)

    def build_geometry(self) -> "cq.Workplane":
        """
        Build the design domain geometry using CadQuery.

        Returns:
            CadQuery Workplane with the design domain solid
        """
        import cadquery as cq

        if self.shape_type == "box":
            return self._build_box(cq)
        elif self.shape_type == "cylinder":
            return self._build_cylinder(cq)
        elif self.shape_type == "sphere":
            return self._build_sphere(cq)
        elif self.shape_type == "l_shape":
            return self._build_l_shape(cq)
        elif self.shape_type == "t_shape":
            return self._build_t_shape(cq)
        elif self.shape_type == "plus":
            return self._build_plus(cq)
        elif self.shape_type == "csg":
            return self._build_csg(cq)
        else:
            logger.warning(
                f"Unknown shape type '{self.shape_type}', falling back to box"
            )
            return self._build_box(cq)

    def _build_box(self, cq) -> "cq.Workplane":
        """Build a box from bounds."""
        if not self.bounds:
            raise ValueError("Box shape requires 'bounds' with x_min, x_max, etc.")

        x_min = self.bounds.get("x_min", 0)
        x_max = self.bounds.get("x_max", 100)
        y_min = self.bounds.get("y_min", 0)
        y_max = self.bounds.get("y_max", 100)
        z_min = self.bounds.get("z_min", 0)
        z_max = self.bounds.get("z_max", 100)

        dx = x_max - x_min
        dy = y_max - y_min
        dz = z_max - z_min

        return (
            cq.Workplane("XY")
            .box(dx, dy, dz, centered=False)
            .translate((x_min, y_min, z_min))
        )

    def _build_cylinder(self, cq) -> "cq.Workplane":
        """
        Build a cylinder from params.

        Params:
            radius: Cylinder radius
            height: Cylinder height
            center: (x, y, z) center of base
            axis: 'x', 'y', or 'z' (default 'z')
        """
        p = self.params or {}
        radius = p.get("radius", 50)
        height = p.get("height", 100)
        center = p.get("center", [0, 0, 0])
        axis = p.get("axis", "z").lower()

        # Build cylinder along Z, then rotate if needed
        cyl = cq.Workplane("XY").cylinder(height, radius, centered=(True, True, False))

        if axis == "x":
            cyl = cyl.rotate((0, 0, 0), (0, 1, 0), 90)
        elif axis == "y":
            cyl = cyl.rotate((0, 0, 0), (1, 0, 0), -90)
        # axis == 'z' is default, no rotation needed

        return cyl.translate(tuple(center))

    def _build_sphere(self, cq) -> "cq.Workplane":
        """
        Build a sphere from params.

        Params:
            radius: Sphere radius
            center: (x, y, z) center
        """
        p = self.params or {}
        radius = p.get("radius", 50)
        center = p.get("center", [0, 0, 0])

        return cq.Workplane("XY").sphere(radius).translate(tuple(center))

    def _build_l_shape(self, cq) -> "cq.Workplane":
        """
        Build an L-shaped design domain.

        Params:
            width: Overall width (x dimension)
            height: Overall height (z dimension)
            depth: Depth (y dimension)
            leg_width: Width of the vertical leg
            base_height: Height of the horizontal base
        """
        p = self.params or {}
        width = p.get("width", 200)
        height = p.get("height", 200)
        depth = p.get("depth", 100)
        leg_width = p.get("leg_width", width * 0.4)
        base_height = p.get("base_height", height * 0.3)

        # Horizontal base
        base = cq.Workplane("XY").box(width, depth, base_height, centered=False)

        # Vertical leg on the left
        leg = cq.Workplane("XY").box(leg_width, depth, height, centered=False)

        return base.union(leg)

    def _build_t_shape(self, cq) -> "cq.Workplane":
        """
        Build a T-shaped design domain.

        Params:
            width: Overall width (x dimension)
            height: Overall height (z dimension)
            depth: Depth (y dimension)
            stem_width: Width of the vertical stem
            cap_height: Height of the horizontal cap
        """
        p = self.params or {}
        width = p.get("width", 200)
        height = p.get("height", 200)
        depth = p.get("depth", 100)
        stem_width = p.get("stem_width", width * 0.4)
        cap_height = p.get("cap_height", height * 0.3)

        stem_height = height - cap_height
        stem_x = (width - stem_width) / 2

        # Vertical stem
        stem = (
            cq.Workplane("XY")
            .box(stem_width, depth, stem_height, centered=False)
            .translate((stem_x, 0, 0))
        )

        # Horizontal cap at top
        cap = (
            cq.Workplane("XY")
            .box(width, depth, cap_height, centered=False)
            .translate((0, 0, stem_height))
        )

        return stem.union(cap)

    def _build_plus(self, cq) -> "cq.Workplane":
        """
        Build a plus/cross-shaped design domain.

        Params:
            width: Overall width (x dimension)
            height: Overall height (z dimension)
            depth: Depth (y dimension)
            arm_width: Width of each arm
        """
        p = self.params or {}
        width = p.get("width", 200)
        height = p.get("height", 200)
        depth = p.get("depth", 100)
        arm_width = p.get("arm_width", width * 0.4)

        # Horizontal bar
        h_bar = (
            cq.Workplane("XY")
            .box(width, depth, arm_width, centered=False)
            .translate((0, 0, (height - arm_width) / 2))
        )

        # Vertical bar
        v_bar = (
            cq.Workplane("XY")
            .box(arm_width, depth, height, centered=False)
            .translate(((width - arm_width) / 2, 0, 0))
        )

        return h_bar.union(v_bar)

    def _build_csg(self, cq) -> "cq.Workplane":
        """
        Build geometry using CSG operations.

        Operations is a list of dicts with:
            - type: 'box', 'cylinder', 'sphere'
            - operation: 'add' (union), 'subtract', 'intersect'
            - params: shape-specific parameters
        """
        if not self.operations:
            raise ValueError("CSG shape requires 'operations' list")

        result = None

        for i, op in enumerate(self.operations):
            op_type = op.get("type", "box")
            operation = op.get("operation", "add")
            params = op.get("params", {})

            # Build the primitive
            if op_type == "box":
                x = params.get("x", 0)
                y = params.get("y", 0)
                z = params.get("z", 0)
                dx = params.get("dx", 100)
                dy = params.get("dy", 100)
                dz = params.get("dz", 100)
                shape = (
                    cq.Workplane("XY")
                    .box(dx, dy, dz, centered=False)
                    .translate((x, y, z))
                )
            elif op_type == "cylinder":
                cx = params.get("cx", 0)
                cy = params.get("cy", 0)
                cz = params.get("cz", 0)
                radius = params.get("radius", 50)
                height = params.get("height", 100)
                axis = params.get("axis", "z")
                shape = cq.Workplane("XY").cylinder(
                    height, radius, centered=(True, True, False)
                )
                if axis == "x":
                    shape = shape.rotate((0, 0, 0), (0, 1, 0), 90)
                elif axis == "y":
                    shape = shape.rotate((0, 0, 0), (1, 0, 0), -90)
                shape = shape.translate((cx, cy, cz))
            elif op_type == "sphere":
                cx = params.get("cx", 0)
                cy = params.get("cy", 0)
                cz = params.get("cz", 0)
                radius = params.get("radius", 50)
                shape = cq.Workplane("XY").sphere(radius).translate((cx, cy, cz))
            else:
                logger.warning(f"Unknown CSG primitive type '{op_type}', skipping")
                continue

            # Apply operation
            if result is None or operation == "add":
                if result is None:
                    result = shape
                else:
                    result = result.union(shape)
            elif operation == "subtract":
                result = result.cut(shape)
            elif operation == "intersect":
                result = result.intersect(shape)

        if result is None:
            raise ValueError("No valid CSG operations produced geometry")

        return result

    def export_step(self, filepath: str) -> str:
        """Export the design domain to a STEP file."""
        import cadquery as cq

        geometry = self.build_geometry()
        cq.exporters.export(geometry, filepath)
        logger.info(f"Exported design domain to {filepath}")
        return filepath

    def get_bounding_box(self) -> Dict[str, float]:
        """
        Get the axis-aligned bounding box of the design domain.

        Returns:
            Dict with x_min, x_max, y_min, y_max, z_min, z_max
        """
        if self.shape_type == "box" and self.bounds:
            return self.bounds

        # For other shapes, compute from geometry
        geometry = self.build_geometry()
        bb = geometry.val().BoundingBox()

        return {
            "x_min": bb.xmin,
            "x_max": bb.xmax,
            "y_min": bb.ymin,
            "y_max": bb.ymax,
            "z_min": bb.zmin,
            "z_max": bb.zmax,
        }


@dataclass
class LoadCase:
    """Complete load case definition parsed from JSON."""

    # Metadata
    meta: Optional[Dict[str, Any]] = None
    problem_id: str = ""
    description: str = ""
    analysis_type: str = "3d"
    thickness_mm: Optional[float] = None

    # Design domain (legacy simple bounds)
    units: str = "mm"
    bounds: Optional[Dict[str, float]] = None

    # Analysis parameters
    tolerance: float = 1.0  # Node selection tolerance percentage

    # Design domain (new flexible geometry)
    domain: Optional[DesignDomain] = None

    # Components
    selectors: Dict[str, SpatialSelector] = field(default_factory=dict)
    boundary_conditions: List[BoundaryCondition] = field(default_factory=list)
    loads: List[LoadDefinition] = field(default_factory=list)
    material: Optional[MaterialDefinition] = None

    # Pre-existing mesh data (for imported .inp files)
    mesh_nodes: Optional[np.ndarray] = None
    mesh_elements: Optional[np.ndarray] = None
    mesh_element_type: Optional[str] = None

    def describe_domain(self) -> str:
        """
        Get a description of the design domain constraints.

        Returns:
            String description of the design space to help planning.
        """
        if self.domain:
            return self.domain.describe()
        elif self.bounds:
            return f"Legacy Box Bounds: {self.bounds}"
        else:
            return "Unknown Design Domain (No bounds or domain definition found)"

    def get_constraints(
        self,
    ) -> List[
        Union[FixedConstraint, CylindricalConstraint, DistributedLoad, PressureLoad]
    ]:
        """Get all constraints (boundary conditions + loads) for RapidCadPy FEA."""
        constraints = []

        # Add boundary conditions as FixedConstraints
        for bc in self.boundary_conditions:
            constraints.append(
                bc.to_constraint(self.selectors, tolerance=self.tolerance)
            )

        # Add loads (including PressureLoads)
        for load in self.loads:
            constraints.append(load.to_load(self.selectors, tolerance=self.tolerance))

        return constraints

    def get_material(self) -> Material:
        """Get the RapidCadPy Material enum."""
        if self.material:
            return self.material.to_material_enum()
        return Material.STEEL  # Default

    def get_density(self) -> float:
        """Get material density in g/cm³."""
        if self.material:
            return self.material.density_g_cm3
        return 7.85  # Default steel density

    def get_fea_analyzer(self, mesher: str = "netgen"):
        from rapidcadpy.fea.kernels.base import FEAAnalyzer

        # Get constraints from load case
        constraints = self.get_constraints()

        # Separate boundary conditions and loads
        fixed_constraints = [
            c
            for c in constraints
            if isinstance(c, (FixedConstraint, CylindricalConstraint))
        ]
        load_constraints = [
            c
            for c in constraints
            if isinstance(c, (DistributedLoad, PointLoad, PressureLoad))
        ]

        logger.info(
            f"Plotting {len(fixed_constraints)} fixed constraints, {len(load_constraints)} loads"
        )

        # Create FEA analyzer to get mesh and apply constraints
        material = self.get_material()

        # Create design space geometry and export to STEP
        shape_path = None

        # Prefer new domain object over legacy bounds
        if self.meta and "source_step" in self.meta:
            shape_path = self.meta["source_step"]
            logger.info(f"Using source STEP file from meta: {shape_path}")
        elif self.domain:
            try:
                import tempfile
                import os

                fd, shape_path = tempfile.mkstemp(suffix="_domain.step", prefix="fea_")
                os.close(fd)

                self.domain.export_step(shape_path)
                logger.info(
                    f"Exported design domain ({self.domain.shape_type}) to {shape_path}"
                )

            except Exception as e:
                logger.warning(f"Failed to generate design domain: {e}")
                shape_path = None

        elif self.bounds:
            # Legacy: create simple box from bounds
            try:
                import tempfile
                import os
                from rapidcadpy import OpenCascadeOcpApp

                app = OpenCascadeOcpApp()

                # Extract bounds with defaults
                x_min = self.bounds.get("x_min")
                x_max = self.bounds.get("x_max")
                y_min = self.bounds.get("y_min")
                y_max = self.bounds.get("y_max")
                z_min = self.bounds.get("z_min")
                z_max = self.bounds.get("z_max")

                # Dimensions
                dx = x_max - x_min
                dy = y_max - y_min
                dz = z_max - z_min

                logger.info(f"Generating design domain box: {dx}x{dy}x{dz}")

                # Create design domain box
                box = (
                    app.work_plane("XY")
                    .box(dx, dy, dz, centered=False)
                    .translate(x_min, y_min, z_min)
                )

                # Export to temporary STEP file
                fd, shape_path = tempfile.mkstemp(suffix="_domain.step", prefix="fea_")
                os.close(fd)

                app.to_step(shape_path)
                logger.info(f"Exported design domain to {shape_path}")

            except Exception as e:
                logger.warning(f"Failed to generate design domain box: {e}")
                shape_path = None

        fea = FEAAnalyzer(
            shape=shape_path,
            kernel="torch-fem",
            mesher=mesher,
            loads=load_constraints,
            constraints=fixed_constraints,
            material=material,
            mesh_size=0.1,  # Default mesh size
        )
        return fea


def parse_load_case(source: Union[str, Path, Dict[str, Any]]) -> LoadCase:
    """
    Parse a load case from a JSON file path, JSON string, or dictionary.
    Also supports Abaqus .inp files.

    Args:
        source: Either a file path to a JSON/INP file, a JSON string, or a dict

    Returns:
        LoadCase object with all parsed components
    """
    # Check for Abaqus INP file
    if isinstance(source, (str, Path)):
        s_path = str(source)
        if s_path.lower().endswith(".inp"):
            from tools.abaqus_parser import parse_abaqus_inp

            return parse_abaqus_inp(s_path)

    # Load the JSON data
    if isinstance(source, dict):
        data = source
    elif isinstance(source, Path) or (
        isinstance(source, str) and (source.endswith(".json") or "/" in source)
    ):
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"Load case file not found: {path}")

        with open(path, "r") as f:
            # Remove comments (JSON doesn't support them, but the example has them)
            content = f.read()
            # Simple comment removal for // style comments
            lines = content.split("\n")
            clean_lines = []
            for line in lines:
                # Remove inline comments
                if "//" in line:
                    line = line[: line.index("//")]
                clean_lines.append(line)
            content = "\n".join(clean_lines)
            data = json.loads(content)
    else:
        # Assume it's a JSON string
        data = json.loads(source)

    # Validate the load case data
    validate_load_case_data(data)

    load_case = LoadCase()

    # Parse metadata
    if "meta" in data:
        meta = data["meta"]
        load_case.meta = meta  # Store entire meta dict
        load_case.problem_id = meta.get("problem_id", "")
        load_case.description = meta.get("description", "")
        load_case.analysis_type = meta.get("analysis_type", "3d")
        load_case.thickness_mm = meta.get("thickness_mm")

    # Parse design domain
    if "design_domain" in data:
        domain_data = data["design_domain"]
        load_case.units = domain_data.get("units", "mm")

        # Check for new flexible domain format vs legacy bounds-only format
        shape_type = domain_data.get("shape_type", "box")

        if (
            shape_type != "box"
            or "params" in domain_data
            or "operations" in domain_data
        ):
            # New flexible domain format
            load_case.domain = DesignDomain(
                units=domain_data.get("units", "mm"),
                shape_type=shape_type,
                bounds=domain_data.get("bounds"),
                params=domain_data.get("params"),
                operations=domain_data.get("operations"),
            )
            # Also populate bounds from bounding box for compatibility
            try:
                load_case.bounds = load_case.domain.get_bounding_box()
            except Exception as e:
                logger.warning(f"Could not compute bounding box: {e}")
                load_case.bounds = domain_data.get("bounds")
        else:
            # Legacy bounds-only format
            load_case.bounds = domain_data.get("bounds")

    # Parse spatial selectors
    if "spatial_selectors" in data:
        for sel_data in data["spatial_selectors"]:
            selector = SpatialSelector(
                id=sel_data["id"], type=sel_data["type"], query=sel_data["query"]
            )
            load_case.selectors[selector.id] = selector

    # Parse boundary conditions
    if "boundary_conditions" in data:
        for bc_data in data["boundary_conditions"]:
            bc = BoundaryCondition(
                name=bc_data["name"],
                region_id=bc_data.get("region_id"),  # Optional
                region=bc_data.get("region"),  # Optional inline region
                type=bc_data["type"],
                dof_lock=bc_data.get("dof_lock"),
                displacement=bc_data.get("displacement"),
                # Cylindrical constraint specific fields
                center=bc_data.get("center"),
                radius=bc_data.get("radius"),
                normal_axis=bc_data.get("normal_axis", "z"),
            )
            load_case.boundary_conditions.append(bc)

    # Parse loads
    if "loads" in data:
        for load_data in data["loads"]:
            load = LoadDefinition(
                name=load_data["name"],
                region_id=load_data.get("region_id"),  # Optional
                region=load_data.get("region"),  # Optional inline region (legacy)
                selector=load_data.get(
                    "selector"
                ),  # Optional inline selector (preferred)
                type=load_data["type"],
                vector_newtons=load_data.get("vector_newtons"),
                magnitude_newtons=load_data.get("magnitude_newtons"),
                direction=load_data.get("direction"),
                direction_vector=load_data.get("direction_vector"),
                distribution=load_data.get("distribution"),
                # Pressure load specific fields
                pressure_mpa=load_data.get("pressure_mpa"),
                center=load_data.get("center"),
                radius=load_data.get("radius"),
                normal_axis=load_data.get("normal_axis", "z"),
            )
            load_case.loads.append(load)

    # Parse material
    if "material" in data:
        mat_data = data["material"]
        load_case.material = MaterialDefinition(
            type=mat_data.get("type", "isotropic"),
            elastic_modulus_mpa=mat_data["elastic_modulus_mpa"],
            poissons_ratio=mat_data["poissons_ratio"],
            density_g_cm3=mat_data["density_g_cm3"],
            yield_strength_mpa=mat_data.get("yield_strength_mpa"),
        )

    logger.info(
        f"Parsed load case '{load_case.problem_id}': "
        f"{len(load_case.boundary_conditions)} BCs, {len(load_case.loads)} loads"
    )

    return load_case


def load_case_from_file(filepath: Union[str, Path]) -> LoadCase:
    """Convenience function to load a load case from a file path."""
    return parse_load_case(filepath)


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="Load and visualize FEA load cases from JSON files"
    )
    parser.add_argument("json_file", type=str, help="Path to load case JSON file")
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Visualize the load case interactively (mesh, constraints, loads)",
    )
    parser.add_argument(
        "--show",
        type=str,
        metavar="FILENAME",
        help="Render and save visualization to file (e.g., loadcase.png)",
    )
    parser.add_argument(
        "--export-step", type=str, help="Export design domain to STEP file"
    )
    parser.add_argument(
        "--mesher",
        type=str,
        default="netgen",
        choices=["netgen", "gmsh"],
        help="Mesher to use for visualization (default: netgen)",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1.0,
        help="Node selection tolerance percentage (default: 1.0)",
    )

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    try:
        # Parse load case
        print(f"\n{'='*60}")
        print(f"Loading load case from: {args.json_file}")
        print(f"{'='*60}\n")

        load_case = parse_load_case(args.json_file)

        # Apply tolerance from CLI argument
        load_case.tolerance = args.tolerance
        print(f"Using tolerance: {load_case.tolerance}")

        # Display summary
        print(f"Problem ID: {load_case.problem_id}")
        print(f"Description: {load_case.description}")
        print(f"Analysis Type: {load_case.analysis_type}")
        print(f"Units: {load_case.units}")

        if load_case.thickness_mm:
            print(f"Thickness: {load_case.thickness_mm} mm")

        print(f"\n{'-'*60}")
        print("Design Domain:")
        print(f"{'-'*60}")
        print(load_case.describe_domain())

        # Display spatial selectors
        print(f"\n{'-'*60}")
        print(f"Spatial Selectors: {len(load_case.selectors)}")
        print(f"{'-'*60}")
        for sel_id, selector in load_case.selectors.items():
            print(f"  {sel_id}:")
            print(f"    Type: {selector.type}")
            print(f"    Query: {selector.query}")

        # Display boundary conditions
        print(f"\n{'-'*60}")
        print(f"Boundary Conditions: {len(load_case.boundary_conditions)}")
        print(f"{'-'*60}")
        for bc in load_case.boundary_conditions:
            print(f"  {bc.name}:")
            print(f"    Type: {bc.type}")
            print(f"    Region ID: {bc.region_id or 'inline'}")
            if bc.dof_lock:
                locked_dofs = [dof for dof, locked in bc.dof_lock.items() if locked]
                print(f"    Locked DOFs: {', '.join(locked_dofs)}")

        # Display loads
        print(f"\n{'-'*60}")
        print(f"Loads: {len(load_case.loads)}")
        print(f"{'-'*60}")
        for load in load_case.loads:
            print(f"  {load.name}:")
            print(f"    Type: {load.type}")
            print(f"    Region ID: {load.region_id or 'inline'}")
            if load.magnitude_newtons:
                print(f"    Magnitude: {load.magnitude_newtons} N")
            if load.direction:
                print(f"    Direction: {load.direction}")
            if load.vector_newtons:
                print(f"    Vector: {load.vector_newtons} N")

        # Display material
        print(f"\n{'-'*60}")
        print("Material Properties:")
        print(f"{'-'*60}")
        if load_case.material:
            mat = load_case.material
            print(f"  Type: {mat.type}")
            print(f"  Elastic Modulus: {mat.elastic_modulus_mpa} MPa")
            print(f"  Poisson's Ratio: {mat.poissons_ratio}")
            print(f"  Density: {mat.density_g_cm3} g/cm³")
            if mat.yield_strength_mpa:
                print(f"  Yield Strength: {mat.yield_strength_mpa} MPa")
        else:
            print("  No material specified")

        # Export design domain if requested
        if args.export_step:
            print(f"\n{'-'*60}")
            print("Exporting Design Domain:")
            print(f"{'-'*60}")
            if load_case.domain:
                output_path = load_case.domain.export_step(args.export_step)
                print(f"✓ Exported to: {output_path}")
            else:
                print("✗ No design domain available for export")

        # Visualize or render if requested
        if args.visualize or args.show:
            print(f"\n{'-'*60}")
            if args.show:
                print(f"Rendering Visualization to: {args.show}")
            else:
                print("Generating Interactive Visualization:")
            print(f"{'-'*60}")
            print(f"Using mesher: {args.mesher}")
            print("This may take a moment...")

            try:
                fea = load_case.get_fea_analyzer(mesher=args.mesher)

                # Get constraints for display
                constraints = load_case.get_constraints()
                fixed_constraints = [
                    c
                    for c in constraints
                    if isinstance(c, (FixedConstraint, CylindricalConstraint))
                ]
                load_constraints = [
                    c
                    for c in constraints
                    if isinstance(c, (DistributedLoad, PointLoad, PressureLoad))
                ]

                print(f"✓ Mesh generated")
                print(f"✓ Boundary conditions: {len(fixed_constraints)}")
                print(f"✓ Loads: {len(load_constraints)}")

                # Use show() method with appropriate parameters
                if args.show:
                    print(f"\n✓ Rendering to file: {args.show}")
                    fea.show(filename=args.show, interactive=False, camera_position="x")
                else:
                    print("\n✓ Launching interactive visualization...")
                    fea.show(interactive=True)

            except Exception as e:
                print(f"✗ Visualization failed: {e}")
                import traceback

                traceback.print_exc()

        print(f"\n{'='*60}")
        print("✓ Load case parsed successfully!")
        print(f"{'='*60}\n")

    except FileNotFoundError as e:
        print(f"\n✗ Error: {e}", file=sys.stderr)
        sys.exit(1)
    except LoadCaseValidationError as e:
        print(f"\n✗ Validation Error:\n{e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Error: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        sys.exit(1)
