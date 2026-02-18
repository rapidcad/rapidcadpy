from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from venv import logger

import numpy as np

from .boundary_conditions import (
    BoundaryCondition,
    DistributedLoad,
    FixedConstraint,
    Load,
    PointLoad,
)
from .materials import Material, MaterialProperties


@dataclass
class LoadCase:
    """Per-run FEA data bundle used by `FEAAnalyzer`."""

    # Core run inputs
    material: MaterialProperties = field(default_factory=lambda: Material.STEEL)
    loads: List[Load] = field(default_factory=list)
    constraints: List[BoundaryCondition] = field(default_factory=list)

    # Optional metadata (kept for compatibility with existing parsers)
    meta: Optional[Dict[str, Any]] = None
    problem_id: str = ""
    description: str = ""
    analysis_type: str = "3d"
    thickness_mm: Optional[float] = None

    # Optional design domain info
    units: str = "mm"
    bounds: Optional[Dict[str, float]] = None
    tolerance: float = 1.0
    domain: Optional[Any] = None

    # Optional pre-existing mesh data
    mesh_nodes: Optional[np.ndarray] = None
    mesh_elements: Optional[np.ndarray] = None
    mesh_element_type: Optional[str] = None

    @property
    def boundary_conditions(self) -> List[BoundaryCondition]:
        """Backward-compatible alias for `constraints`."""
        return self.constraints

    @boundary_conditions.setter
    def boundary_conditions(self, value: List[BoundaryCondition]) -> None:
        self.constraints = value

    def add_load(self, load: Load) -> "LoadCase":
        """Add a load (fluent interface)."""
        self.loads.append(load)
        return self

    def add_constraint(self, constraint: BoundaryCondition) -> "LoadCase":
        """Add a boundary condition/constraint (fluent interface)."""
        self.constraints.append(constraint)
        return self

    def get_fea_analyzer(self, mesher: str = "gmsh-subprocess"):
        from config import Config
        from .kernels.base import FEAAnalyzer

        # Create design space geometry and export to STEP
        shape_path = None

        # Prefer new domain object over legacy bounds
        if self.domain:
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
                import cadquery as cq
                import tempfile
                import os

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

                # Create box using CadQuery
                box = (
                    cq.Workplane("XY")
                    .box(dx, dy, dz, centered=False)
                    .translate((x_min, y_min, z_min))
                )

                # Export to temporary STEP file
                fd, shape_path = tempfile.mkstemp(suffix="_domain.step", prefix="fea_")
                os.close(fd)

                cq.exporters.export(box, shape_path)
                logger.info(f"Exported design domain to {shape_path}")

            except Exception as e:
                logger.warning(f"Failed to generate design domain box: {e}")
                shape_path = None

        # Adaptive mesh size: keep global default as upper bound, but refine for tiny domains.
        # This prevents GMSH from missing slender members when bounds are sub-mm scale.
        mesh_size = Config.MESH_SIZE
        if self.bounds:
            try:
                dx = abs(self.bounds.get("x_max", 0) - self.bounds.get("x_min", 0))
                dy = abs(self.bounds.get("y_max", 0) - self.bounds.get("y_min", 0))
                dz = abs(self.bounds.get("z_max", 0) - self.bounds.get("z_min", 0))
                positive_dims = [d for d in [dx, dy, dz] if d > 0]
                if positive_dims:
                    min_dim = min(positive_dims)
                    adaptive = max(min_dim / 50.0, 1e-3)
                    mesh_size = min(mesh_size, adaptive)
                    logger.info(
                        f"Adaptive mesh size: {mesh_size:.6g} (default {Config.MESH_SIZE}, min_dim {min_dim:.6g})"
                    )
            except Exception as e:
                logger.warning(f"Could not compute adaptive mesh size: {e}")

        fea = FEAAnalyzer(
            shape=shape_path,
            kernel="torch-fem",
            mesher=mesher,
            load_case=self,
            mesh_size=mesh_size,
        )
        return fea

    def load_case_to_requirement(self) -> str:
        """
        Convert a parsed LoadCase object into a natural language design requirement.

        This bridges the structured JSON specification with the LLM-based design agent.
        """
        import math

        def _format_location(location) -> str:
            if location is None:
                return "unspecified location"

            if isinstance(location, str):
                return location

            if isinstance(location, dict):
                keys = ["x_min", "x_max", "y_min", "y_max", "z_min", "z_max"]
                if any(k in location for k in keys):
                    return (
                        "box "
                        f"X[{location.get('x_min', '?')}, {location.get('x_max', '?')}], "
                        f"Y[{location.get('y_min', '?')}, {location.get('y_max', '?')}], "
                        f"Z[{location.get('z_min', '?')}, {location.get('z_max', '?')}]"
                    )
                if all(k in location for k in ("x", "y", "z")):
                    return f"point ({location['x']:.3g}, {location['y']:.3g}, {location['z']:.3g})"
                return str(location)

            if isinstance(location, (tuple, list)) and len(location) >= 3:
                return (
                    f"point ({location[0]:.3g}, {location[1]:.3g}, {location[2]:.3g})"
                )

            return str(location)

        def _resolve_region_location(lc, obj) -> str:
            # 1) Native object location/point
            if hasattr(obj, "location"):
                return _format_location(getattr(obj, "location", None))
            if hasattr(obj, "point"):
                return _format_location(getattr(obj, "point", None))

            # 2) Parser compatibility metadata: region_id + selectors
            region_id = getattr(obj, "region_id", None)
            selectors = getattr(lc, "selectors", None) or {}
            if not selectors and getattr(lc, "meta", None):
                selectors = lc.meta.get("selectors", {})

            if region_id and region_id in selectors:
                selector = selectors[region_id]
                query = getattr(selector, "query", None)
                if query is not None:
                    return _format_location(query)

            if region_id:
                return str(region_id)

            return "unspecified location"

        def _force_magnitude(force) -> float:
            if isinstance(force, (int, float)):
                return abs(float(force))
            if isinstance(force, (tuple, list)) and len(force) >= 3:
                fx, fy, fz = float(force[0]), float(force[1]), float(force[2])
                return math.sqrt(fx * fx + fy * fy + fz * fz)
            return 0.0

        def _vector_direction_text(vec) -> str:
            fx, fy, fz = vec
            if abs(fy) >= abs(fx) and abs(fy) >= abs(fz):
                return "downward" if fy < 0 else "upward"
            if abs(fx) >= abs(fz):
                return "in X direction" if fx >= 0 else "in -X direction"
            return "in Z direction" if fz >= 0 else "in -Z direction"

        # Extract bounds from load case
        bounds = self.bounds or {}
        x_min = bounds.get("x_min", 0)
        x_max = bounds.get("x_max", 100)
        y_min = bounds.get("y_min", 0)
        y_max = bounds.get("y_max", 20)
        z_min = bounds.get("z_min", 0)
        z_max = bounds.get("z_max", 10)

        length = x_max - x_min
        height = y_max - y_min
        thickness = z_max - z_min

        # Extract material properties
        material = getattr(self, "material", None)
        if material:
            if hasattr(material, "elastic_modulus_mpa"):
                # Legacy material definition from JSON parser
                E_mpa = material.elastic_modulus_mpa
                _poisson = material.poissons_ratio
                _density = material.density_g_cm3
                yield_strength = material.yield_strength_mpa or 250  # Default for steel
            else:
                # New FEA LoadCase MaterialProperties signature
                E_mpa = getattr(material, "E", 210000)
                _poisson = getattr(material, "nu", 0.3)
                _density = getattr(material, "density", 7.85)
                yield_strength = getattr(material, "yield_strength", None) or 250

            # Determine material name from elastic modulus
            if E_mpa > 180000:
                material_name = "steel"
            elif E_mpa > 100000:
                material_name = "titanium"
            else:
                material_name = "aluminum"
        else:
            material_name = "steel"
            E_mpa = 210000
            yield_strength = 250
            _density = 7.85

        # Extract boundary conditions
        bc_descriptions = []
        for bc in self.boundary_conditions:
            location = _resolve_region_location(self, bc)
            bc_type = bc.__class__.__name__

            if isinstance(bc, FixedConstraint):
                dofs = getattr(bc, "dofs", (True, True, True))
                constrained_axes = [
                    ax for ax, locked in zip(["X", "Y", "Z"], dofs) if locked
                ]
                dof_text = ", ".join(constrained_axes) if constrained_axes else "none"
                bc_descriptions.append(
                    f"{bc_type} at {location} (DOFs locked: {dof_text})"
                )
            else:
                bc_descriptions.append(f"{bc_type} at {location}")

        # Extract loads
        load_descriptions = []
        total_force = 0
        for load in self.loads:
            location = _resolve_region_location(self, load)
            load_type = load.__class__.__name__

            if isinstance(load, PointLoad):
                force = getattr(load, "force", 0.0)
                force_mag = _force_magnitude(force)

                if isinstance(force, (tuple, list)) and len(force) >= 3:
                    direction = _vector_direction_text((force[0], force[1], force[2]))
                    load_descriptions.append(
                        f"{load_type} {force_mag:.0f} N {direction} at {location}"
                    )
                else:
                    direction = getattr(load, "direction", None)
                    direction_str = f" in {direction}" if direction else ""
                    load_descriptions.append(
                        f"{load_type} {force_mag:.0f} N{direction_str} at {location}"
                    )
                total_force += force_mag

            elif isinstance(load, DistributedLoad):
                force = getattr(load, "force", 0.0)
                force_mag = _force_magnitude(force)
                direction = getattr(load, "direction", None)
                direction_str = f" ({direction})" if direction else ""
                load_descriptions.append(
                    f"{load_type} total {force_mag:.0f} N{direction_str} over {location}"
                )
                total_force += force_mag

            else:
                # Generic fallback for other Load subclasses and parser-compat objects
                vector_newtons = getattr(load, "vector_newtons", None)
                magnitude_newtons = getattr(load, "magnitude_newtons", None)

                if vector_newtons:
                    fx = vector_newtons.get("x", 0.0)
                    fy = vector_newtons.get("y", 0.0)
                    fz = vector_newtons.get("z", 0.0)
                    force_mag = math.sqrt(fx * fx + fy * fy + fz * fz)
                    direction = _vector_direction_text((fx, fy, fz))
                    load_descriptions.append(
                        f"{load_type} {force_mag:.0f} N {direction} at {location}"
                    )
                    total_force += force_mag
                elif magnitude_newtons is not None:
                    direction = getattr(load, "direction", None)
                    direction_str = f" in {direction}" if direction else ""
                    force_mag = abs(float(magnitude_newtons))
                    load_descriptions.append(
                        f"{load_type} {force_mag:.0f} N{direction_str} at {location}"
                    )
                    total_force += force_mag
                else:
                    load_descriptions.append(f"{load_type} at {location}")

        if not bc_descriptions:
            bc_descriptions.append("No explicit constraints provided")

        if not load_descriptions:
            load_descriptions.append("No explicit external loads provided")

        # Build the requirement string
        requirement = f"""
        Design a structural component based on the following load case specification:
        
        GEOMETRY:
        - Length (X): {length} {self.units}
        - Beam (Y): {height} {self.units}
        - Height (Z): {thickness} {self.units}
        - Design domain: X=[{x_min}, {x_max}], Y=[{y_min}, {y_max}], Z=[{z_min}, {z_max}]

        MATERIAL:
        - Nominal material: {material_name}
        - Elastic modulus: {E_mpa:.0f} MPa
        
        BOUNDARY CONDITIONS:
        {chr(10).join(f'    - {bc}' for bc in bc_descriptions)}
        
        APPLIED LOADS:
        {chr(10).join(f'    - {ld}' for ld in load_descriptions)}
        - Total applied force: {total_force:.0f} N
        
        STRUCTURAL REQUIREMENTS:
        - Maximum allowable stress: {yield_strength / 2.5:.0f} MPa (Safety Factor = 2.5)
        - Design should efficiently distribute the load path
        - Minimize material usage while maintaining structural integrity
        
        Generate a CAD model that satisfies these requirements. Consider using:
        - Truss structures with triangular elements for efficient load distribution
        - Lattice or honeycomb patterns for lightweight designs
        - Material placement where stress is highest (top/bottom for bending)
        - Smooth transitions to avoid stress concentrations
        """

        return requirement
