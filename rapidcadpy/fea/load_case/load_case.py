from dataclasses import dataclass, field
from pathlib import Path
import logging
import re
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from ..spatial_selector import SpatialSelector

from ..design_domain import DesignDomain

from ..boundary_conditions import (
    AccelerationLoad,
    BoundaryCondition,
    DistributedLoad,
    FixedConstraint,
    Load,
    PressureLoad,
    PointLoad,
)
from ..materials import Material, MaterialProperties
from .freecad_inp_load_case import LoadCaseFromFreeCadInp
import traceback

logger = logging.getLogger(__name__)


@dataclass
class LoadCase(LoadCaseFromFreeCadInp):
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

    # Maximum possible loaded / constrained node counts, computed once during
    # dataset creation by meshing a full design-domain box with this load case.
    # Used to normalise get_bc_node_coverage() raw counts into [0, 1] ratios.
    max_n_loaded_nodes: Optional[int] = None
    max_n_constraint_nodes: Optional[int] = None

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

    def calc_mesh_size(self, num_nodes: int) -> float:
        """
        Recommend a mesh size based on design-space dimensions and a target node count.

        The estimate assumes roughly uniform 3D discretization:

            h ≈ (V / N)^(1/3)

        where:
            - V is the design-space bounding-box volume
            - N is target number of nodes

        For very thin domains, the recommendation is clamped to avoid over-coarsening
        across the smallest dimension.

        Args:
            num_nodes: Target number of mesh nodes (> 0)

        Returns:
            Recommended mesh size in model units (typically mm)

        Raises:
            ValueError: If `num_nodes` is not positive or dimensions are invalid.
        """
        if num_nodes <= 0:
            raise ValueError(f"num_nodes must be > 0, got {num_nodes}")

        # Resolve bounds from domain first, then legacy bounds.
        bounds = None
        if self.domain is not None and hasattr(self.domain, "get_bounding_box"):
            try:
                bounds = self.domain.get_bounding_box()
            except Exception:
                bounds = None
        if bounds is None:
            bounds = self.bounds

        if not bounds:
            raise ValueError(
                "Cannot compute mesh size: no design-space bounds available (domain/bounds missing)."
            )

        dx = float(bounds.get("x_max", 0.0) - bounds.get("x_min", 0.0))
        dy = float(bounds.get("y_max", 0.0) - bounds.get("y_min", 0.0))
        dz = float(bounds.get("z_max", 0.0) - bounds.get("z_min", 0.0))

        positive_dims = [d for d in (dx, dy, dz) if d > 0.0]
        if not positive_dims:
            raise ValueError(
                "Cannot compute mesh size: design-space dimensions are non-positive."
            )

        h_min = 1e-3

        # Unified mesh-size estimate that is robust for all geometries
        # (cubes, beams, thin plates, shells) without branching.
        #
        # Two competing estimates; take the larger:
        #
        #   h_3d = (V / N)^(1/3)   — 3D volumetric estimate
        #   h_2d = sqrt(A_max / N) — 2D in-plane estimate for the largest face
        #
        # For chunky geometries the 3D term dominates.
        # For thin-plate / shell geometries, GMSH enforces ≥1 element across
        # the thin dimension anyway, so in-plane nodes ≈ A_max / h².
        # The 2D term then dominates and matches the actual node count.
        # The max() picks the correct regime automatically — no branching needed.
        sorted_dims = sorted([dx, dy, dz], reverse=True)

        if len(positive_dims) == 3:
            volume = dx * dy * dz
            h_3d = (volume / float(num_nodes)) ** (1.0 / 3.0)
            a_max = sorted_dims[0] * sorted_dims[1]
            h_2d = (a_max / float(num_nodes)) ** 0.5
            h = max(h_min, max(h_3d, h_2d))
        else:
            # Degenerate 2D/1D domain: use largest positive dimension pair / single dim.
            if len(positive_dims) >= 2:
                a_max = sorted_dims[0] * sorted_dims[1]
                h = max(h_min, (a_max / float(num_nodes)) ** 0.5)
            else:
                h = max(h_min, sorted_dims[0] / float(num_nodes))

        return float(h)

    def __str__(self) -> str:
        """Human-readable summary of this load case."""
        material_name = getattr(self.material, "name", str(self.material))
        has_domain = self.domain is not None
        has_bounds = self.bounds is not None
        return (
            f"LoadCase(problem_id={self.problem_id or 'N/A'}, "
            f"material={material_name}, "
            f"loads={len(self.loads)}, "
            f"constraints={len(self.constraints)}, "
            f"domain={'yes' if has_domain else 'no'}, "
            f"bounds={'yes' if has_bounds else 'no'})"
        )

    def __repr__(self) -> str:
        """Developer-friendly representation for debugging."""
        material_name = getattr(self.material, "name", str(self.material))
        return (
            "LoadCase("
            f"problem_id={self.problem_id!r}, "
            f"description={self.description!r}, "
            f"analysis_type={self.analysis_type!r}, "
            f"material={material_name!r}, "
            f"loads={len(self.loads)}, "
            f"constraints={len(self.constraints)}, "
            f"units={self.units!r}, "
            f"bounds={self.bounds!r}, "
            f"domain={'set' if self.domain is not None else None}, "
            f"mesh_nodes={'set' if self.mesh_nodes is not None else None}, "
            f"mesh_elements={'set' if self.mesh_elements is not None else None}"
            ")"
        )

    def get_fea_analyzer(
        self,
        mesher: "Union[MesherBase, str]" = "gmsh-subprocess",
        mesh_size: float = 1,
        device: str = "auto",
        kernel: str = "torch-fem",
        log_exports: bool = True,
    ) -> "FEAAnalyzer":
        from ..fea_analyzer import FEAAnalyzer

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
                if os.environ.get("RCADPY_VERBOSE", False):
                    logger.info(
                        f"Exported design domain ({self.domain.shape_type}) to {shape_path}"
                    )

            except Exception as e:
                logger.warning(f"Failed to generate design domain: {e} /n {traceback.format_exc()}")
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

                if os.environ.get("RCADPY_VERBOSE", False):
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
                if os.environ.get("RCADPY_VERBOSE", False):
                    logger.info(f"Exported design domain to {shape_path}")

            except Exception as e:
                logger.warning(f"Failed to generate design domain box: {e} /n {traceback.format_exc()}")
                shape_path = None

        # Last resort: if no STEP geometry is available but the LoadCase
        # already carries a pre-loaded mesh (e.g. imported from .inp), wrap
        # it as a meshio.Mesh so that get_visualization_data() can use it
        # directly without re-meshing.
        if shape_path is None and self.mesh_nodes is not None and self.mesh_elements is not None:
            try:
                import meshio

                _RAPIDCAD_TO_MESHIO: Dict[str, str] = {
                    "tet4": "tetra",
                    "tet10": "tetra10",
                    "hex8": "hexahedron",
                    "hex20": "hexahedron20",
                    "wed6": "wedge",
                    "wed15": "wedge15",
                    "tri3": "triangle",
                    "tri6": "triangle6",
                    "quad4": "quad",
                    "quad8": "quad8",
                }
                cell_type = _RAPIDCAD_TO_MESHIO.get(
                    self.mesh_element_type or "tet4", "tetra"
                )
                shape_path = meshio.Mesh(
                    points=np.asarray(self.mesh_nodes, dtype=np.float64),
                    cells=[(cell_type, np.asarray(self.mesh_elements, dtype=np.int64))],
                )
                logger.info(
                    f"Using pre-loaded mesh as shape ({self.mesh_nodes.shape[0]} nodes, "
                    f"{self.mesh_elements.shape[0]} {cell_type} elements)"
                )
            except Exception as e:
                logger.warning(f"Failed to build meshio.Mesh from LoadCase mesh data: {e}")
                shape_path = None

        fea = FEAAnalyzer(
            shape=shape_path,
            kernel=kernel,
            mesher=mesher,
            load_case=self,
            mesh_size=mesh_size,
            device=device,
        )
        return fea

    def inspect_boundary_condition_nodes(
        self,
        mesher: "Union[MesherBase, str]" = "gmsh-subprocess",
        mesh_size: float = 1,
        device: str = "auto",
        log_exports: bool = False,
    ) -> Dict[str, Any]:
        """Return loaded/constrained node sets and their overlap."""
        fea = self.get_fea_analyzer(
            mesher=mesher,
            mesh_size=mesh_size,
            device=device,
            log_exports=log_exports,
        )
        return fea.inspect_bc_node_sets()

    def load_case_to_requirement(self) -> str:
        """
        Convert a parsed LoadCase object into a natural language design requirement.

        This bridges the structured JSON specification with the LLM-based design agent.
        """
        import math

        def _fmt_num(value: Any, max_decimals: int = 4) -> str:
            """Format numeric values compactly for prompt readability.

            - Removes floating-point noise (e.g. 100.0000002 -> 100)
            - Trims trailing zeros
            - Keeps non-numeric sentinels (e.g. "?") as-is
            """
            if value is None:
                return "?"

            if isinstance(value, (int, np.integer)):
                return str(int(value))

            if isinstance(value, (float, np.floating)):
                v = float(value)
                if math.isfinite(v):
                    v = round(v, max_decimals)
                    # Avoid negative zero in string output
                    if abs(v) < 10 ** (-max_decimals):
                        v = 0.0
                    s = f"{v:.{max_decimals}f}".rstrip("0").rstrip(".")
                    return s if s else "0"
                return str(v)

            # Numeric strings and other values
            try:
                v = float(value)
                if math.isfinite(v):
                    v = round(v, max_decimals)
                    if abs(v) < 10 ** (-max_decimals):
                        v = 0.0
                    s = f"{v:.{max_decimals}f}".rstrip("0").rstrip(".")
                    return s if s else "0"
                return str(value)
            except (TypeError, ValueError):
                return str(value)

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
                        f"X[{_fmt_num(location.get('x_min', '?'))}, {_fmt_num(location.get('x_max', '?'))}], "
                        f"Y[{_fmt_num(location.get('y_min', '?'))}, {_fmt_num(location.get('y_max', '?'))}], "
                        f"Z[{_fmt_num(location.get('z_min', '?'))}, {_fmt_num(location.get('z_max', '?'))}]"
                    )
                if all(k in location for k in ("x", "y", "z")):
                    return (
                        f"point ({_fmt_num(location['x'])}, "
                        f"{_fmt_num(location['y'])}, {_fmt_num(location['z'])})"
                    )
                return str(location)

            if isinstance(location, (tuple, list)) and len(location) >= 3:
                return f"point ({_fmt_num(location[0])}, {_fmt_num(location[1])}, {_fmt_num(location[2])})"

            return str(location)

        def _resolve_region_location(lc, obj) -> str:
            # 1) Native object location/point
            if hasattr(obj, "location"):
                return _format_location(getattr(obj, "location", None))
            if hasattr(obj, "point"):
                return _format_location(getattr(obj, "point", None))
            if hasattr(obj, "center"):
                return _format_location(getattr(obj, "center", None))

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

            elif isinstance(load, PressureLoad):
                pressure = float(getattr(load, "pressure", 0.0))
                radius = float(getattr(load, "radius", 0.0))
                center = _format_location(getattr(load, "center", None))
                normal_axis = getattr(load, "normal_axis", "?")
                direction = getattr(load, "direction", "?")
                equivalent_force = abs(pressure) * math.pi * max(radius, 0.0) ** 2
                load_descriptions.append(
                    f"{load_type} {pressure:.3g} MPa ({direction}) on circular area "
                    f"centered at {center}, radius {radius:.3g} mm, normal axis {normal_axis}"
                )
                total_force += equivalent_force

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
        - Length (X): {_fmt_num(length)} {self.units}
        - Beam (Y): {_fmt_num(height)} {self.units}
        - Height (Z): {_fmt_num(thickness)} {self.units}
        - Design domain: X=[{_fmt_num(x_min)}, {_fmt_num(x_max)}], Y=[{_fmt_num(y_min)}, {_fmt_num(y_max)}], Z=[{_fmt_num(z_min)}, {_fmt_num(z_max)}]

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
        
        CADQUERY API REFERENCE (use exactly as shown):
        
        # box(xLen, yLen, zLen) — always 3 positional args; centered=True/False optional
        cq.Workplane("XY").box(10, 20, 5)                  # box centered at origin
        cq.Workplane("XY").box(10, 20, 5, centered=False)  # box with corner at origin
        
        # translate((x, y, z)) — move the result solid
        cq.Workplane("XY").box(L, W, t).translate((L/2, W/2, t/2))
        
        # center(x, y) — 2D only, moves the 2D workplane origin (NOT for 3D positioning)
        cq.Workplane("XY").center(cx, cy).box(10, 20, 5)   # only x and y args
        
        # union / cut / intersect — boolean ops between Workplane objects
        result = solid_a.union(solid_b)
        result = solid_a.cut(solid_b)
        
        # export
        cq.exporters.export(result, "output.step")
        
        COMMON MISTAKES TO AVOID:
        - WRONG: .box(xLen, yLen)              # missing height arg
        - WRONG: .center(x, y, z)              # center() is 2D only
        - WRONG: .translate([x, y, z])         # must be a tuple, not a list
        - RIGHT: .box(xLen, yLen, zLen)
        - RIGHT: .center(x, y)
        - RIGHT: .translate((x, y, z))
        """

        return requirement

    def to_inp(
        self,
        filepath: Union[str, Path],
        mesh_size: Optional[float] = None,
        mesher: str = "gmsh-subprocess",
        element_type: str = "tet4",
        geometry_path: Optional[Union[str, Path]] = None,
        analysis_title: Optional[str] = None,
        verbose: bool = False,
    ) -> Path:
        """
        Export this load case to an Abaqus/CalculiX-compatible ``.inp`` file.

        If mesh data is already present on the load case (``mesh_nodes`` and
        ``mesh_elements``), it is reused directly. Otherwise a mesh is generated
        from ``geometry_path`` (if provided), then ``domain``, then ``bounds``.

        Args:
            filepath: Destination ``.inp`` file path.
            mesh_size: Meshing size when a mesh must be generated.
            mesher: Mesher backend name (e.g. ``gmsh-subprocess``, ``netgen``).
            element_type: Target element type for generated meshes.
            geometry_path: Optional STEP/BREP geometry input for meshing.
            analysis_title: Optional header title.
            verbose: Enable mesher verbosity.

        Returns:
            Path to the written ``.inp`` file.
        """

        def _sanitize_set_name(name: str, default: str) -> str:
            raw = (name or default).upper()
            safe = re.sub(r"[^A-Z0-9_]", "_", raw)
            return safe[:80] if safe else default

        def _bounds_from_nodes(nodes_arr: np.ndarray) -> Dict[str, float]:
            return {
                "x_min": float(np.min(nodes_arr[:, 0])),
                "x_max": float(np.max(nodes_arr[:, 0])),
                "y_min": float(np.min(nodes_arr[:, 1])),
                "y_max": float(np.max(nodes_arr[:, 1])),
                "z_min": float(np.min(nodes_arr[:, 2])),
                "z_max": float(np.max(nodes_arr[:, 2])),
            }

        def _select_nodes(
            nodes_arr: np.ndarray,
            location: Any,
            tolerance_scale: float = 1.0,
        ) -> np.ndarray:
            spans = np.ptp(nodes_arr, axis=0)
            model_scale = max(float(np.max(spans)), 1.0)
            base_tol = max(model_scale * 1e-6, 1e-8)
            tol = base_tol * max(float(tolerance_scale), 1.0)
            x = nodes_arr[:, 0]
            y = nodes_arr[:, 1]
            z = nodes_arr[:, 2]

            if isinstance(location, dict):
                if all(
                    k in location
                    for k in ("x_min", "x_max", "y_min", "y_max", "z_min", "z_max")
                ):
                    mask = (
                        (x >= float(location["x_min"]) - tol)
                        & (x <= float(location["x_max"]) + tol)
                        & (y >= float(location["y_min"]) - tol)
                        & (y <= float(location["y_max"]) + tol)
                        & (z >= float(location["z_min"]) - tol)
                        & (z <= float(location["z_max"]) + tol)
                    )
                    idx = np.where(mask)[0]
                    if idx.size > 0:
                        return idx
                if all(k in location for k in ("x", "y", "z")):
                    px, py, pz = (
                        float(location["x"]),
                        float(location["y"]),
                        float(location["z"]),
                    )
                    dist = np.sqrt((x - px) ** 2 + (y - py) ** 2 + (z - pz) ** 2)
                    nearest = int(np.argmin(dist))
                    return np.array([nearest], dtype=np.int64)

            if isinstance(location, (tuple, list)) and len(location) >= 3:
                px, py, pz = float(location[0]), float(location[1]), float(location[2])
                dist = np.sqrt((x - px) ** 2 + (y - py) ** 2 + (z - pz) ** 2)
                nearest = int(np.argmin(dist))
                return np.array([nearest], dtype=np.int64)

            if isinstance(location, str):
                loc = location.lower().strip()
                bounds = self.bounds or _bounds_from_nodes(nodes_arr)
                if loc in ["end_1", "x_min"]:
                    return np.where(np.abs(x - bounds["x_min"]) <= tol)[0]
                if loc in ["end_2", "x_max"]:
                    return np.where(np.abs(x - bounds["x_max"]) <= tol)[0]
                if loc in ["y_min"]:
                    return np.where(np.abs(y - bounds["y_min"]) <= tol)[0]
                if loc in ["y_max"]:
                    return np.where(np.abs(y - bounds["y_max"]) <= tol)[0]
                if loc in ["bottom", "z_min"]:
                    return np.where(np.abs(z - bounds["z_min"]) <= tol)[0]
                if loc in ["top", "z_max"]:
                    return np.where(np.abs(z - bounds["z_max"]) <= tol)[0]

            return np.array([], dtype=np.int64)

        def _resolve_mesh() -> Tuple[np.ndarray, np.ndarray, str]:
            if self.mesh_nodes is not None and self.mesh_elements is not None:
                nodes_arr = np.asarray(self.mesh_nodes, dtype=np.float64)
                elems_arr = np.asarray(self.mesh_elements, dtype=np.int64)
                elem_name = self.mesh_element_type or element_type
                return nodes_arr, elems_arr, elem_name

            step_path: Optional[Path] = None
            cleanup_path: Optional[Path] = None
            if geometry_path is not None:
                step_path = Path(geometry_path)
            elif self.domain is not None:
                import tempfile
                import os

                fd, tmp_path = tempfile.mkstemp(suffix="_domain.step", prefix="fea_")
                os.close(fd)
                step_path = Path(tmp_path)
                cleanup_path = step_path
                self.domain.export_step(str(step_path))
            elif self.bounds is not None:
                import cadquery as cq
                import tempfile
                import os

                x_min = float(self.bounds["x_min"])
                x_max = float(self.bounds["x_max"])
                y_min = float(self.bounds["y_min"])
                y_max = float(self.bounds["y_max"])
                z_min = float(self.bounds["z_min"])
                z_max = float(self.bounds["z_max"])
                box = (
                    cq.Workplane("XY")
                    .box(x_max - x_min, y_max - y_min, z_max - z_min, centered=False)
                    .translate((x_min, y_min, z_min))
                )
                fd, tmp_path = tempfile.mkstemp(suffix="_domain.step", prefix="fea_")
                os.close(fd)
                step_path = Path(tmp_path)
                cleanup_path = step_path
                cq.exporters.export(box, str(step_path))

            if step_path is None:
                raise ValueError(
                    "Cannot generate mesh for to_inp(): provide mesh_nodes/mesh_elements, "
                    "geometry_path, domain, or bounds."
                )

            try:
                from ..mesher import (
                    GmshMesher,
                    GmshSubprocessMesher,
                    IsolatedGmshMesher,
                    NetgenMesher,
                    NetgenSubprocessMesher,
                )

                mesher_key = mesher.lower().strip()
                if mesher_key in {"gmsh-subprocess", "gmsh_subprocess"}:
                    mesher_obj = GmshSubprocessMesher()
                elif mesher_key == "gmsh":
                    mesher_obj = GmshMesher()
                elif mesher_key in {"netgen-subprocess", "netgen_subprocess"}:
                    mesher_obj = NetgenSubprocessMesher()
                elif mesher_key == "netgen":
                    mesher_obj = NetgenMesher()
                elif mesher_key in {"gmsh-isolated", "gmsh_isolated"}:
                    mesher_obj = IsolatedGmshMesher()
                else:
                    raise ValueError(f"Unsupported mesher '{mesher}' for to_inp()")

                h = float(mesh_size) if mesh_size is not None else 1.0
                nodes_t, elems_t = mesher_obj.generate_mesh(
                    str(step_path),
                    mesh_size=h,
                    element_type=element_type,
                    dim=3,
                    verbose=verbose,
                )
                nodes_arr = nodes_t.detach().cpu().numpy().astype(np.float64)
                elems_arr = elems_t.detach().cpu().numpy().astype(np.int64)
                return nodes_arr, elems_arr, element_type
            finally:
                if cleanup_path is not None:
                    try:
                        cleanup_path.unlink()
                    except Exception:
                        pass

        nodes, elements, resolved_elem_type = _resolve_mesh()
        self.mesh_nodes = nodes
        self.mesh_elements = elements
        self.mesh_element_type = resolved_elem_type

        if self.bounds is None:
            self.bounds = _bounds_from_nodes(nodes)

        element_type_to_ccx = {
            "tet4": "C3D4",
            "tet10": "C3D10",
            "hex8": "C3D8",
            "hex20": "C3D20",
            "quad4": "CPS4",
            "tri3": "CPS3",
        }
        nodes_per_elem_to_ccx = {4: "C3D4", 8: "C3D8", 10: "C3D10", 20: "C3D20"}
        ccx_elem_type = element_type_to_ccx.get(
            (resolved_elem_type or "").lower(),
            nodes_per_elem_to_ccx.get(int(elements.shape[1]), "C3D4"),
        )

        selectors = getattr(self, "selectors", None) or {}
        if not selectors and self.meta:
            selectors = self.meta.get("selectors", {})

        constraint_sets: List[Tuple[str, np.ndarray, Tuple[bool, bool, bool]]] = []
        for idx, bc in enumerate(self.boundary_conditions, start=1):
            if not isinstance(bc, FixedConstraint):
                continue

            node_idx = _select_nodes(
                nodes,
                getattr(bc, "location", None),
                tolerance_scale=float(getattr(bc, "tolerance", 1.0)),
            )
            if node_idx.size == 0:
                continue

            set_name = _sanitize_set_name(f"constraint_{idx}", f"CONSTRAINT_{idx}")
            dofs = getattr(bc, "dofs", (True, True, True))
            constraint_sets.append((set_name, np.unique(node_idx), dofs))

        load_sets: List[Tuple[str, np.ndarray, Dict[str, float]]] = []
        # (elset_name, dload_keyword, magnitude, extra_tuple_or_None)
        dload_blocks: List[Tuple[str, str, float, Any]] = []
        for idx, load in enumerate(self.loads, start=1):
            if isinstance(load, PointLoad):
                loc = getattr(load, "point", None)
                region_id = getattr(load, "region_id", None)
                if region_id and region_id in selectors:
                    selector_obj = selectors[region_id]
                    loc = getattr(selector_obj, "query", loc)

                node_idx = _select_nodes(
                    nodes,
                    loc,
                    tolerance_scale=float(getattr(load, "tolerance", 1.0)),
                )
                if node_idx.size == 0:
                    continue

                force = getattr(load, "force", 0.0)
                if isinstance(force, (tuple, list)) and len(force) >= 3:
                    fx, fy, fz = float(force[0]), float(force[1]), float(force[2])
                elif isinstance(force, (int, float)):
                    direction = (getattr(load, "direction", None) or "z").lower()
                    mag = float(force)
                    fx, fy, fz = 0.0, 0.0, 0.0
                    if direction in {"x", "+x"}:
                        fx = mag
                    elif direction == "-x":
                        fx = -abs(mag)
                    elif direction in {"y", "+y"}:
                        fy = mag
                    elif direction == "-y":
                        fy = -abs(mag)
                    elif direction in {"z", "+z"}:
                        fz = mag
                    elif direction == "-z":
                        fz = -abs(mag)
                    else:
                        fz = mag
                else:
                    continue

                load_name = _sanitize_set_name(
                    getattr(load, "name", f"LOAD_{idx}"), f"LOAD_{idx}"
                )
                load_sets.append(
                    (
                        load_name,
                        np.unique(node_idx),
                        {"x": fx, "y": fy, "z": fz},
                    )
                )

            elif isinstance(load, DistributedLoad):
                node_idx = _select_nodes(
                    nodes,
                    getattr(load, "location", None),
                    tolerance_scale=float(getattr(load, "tolerance", 1.0)),
                )
                if node_idx.size == 0:
                    continue

                force = getattr(load, "force", 0.0)
                if isinstance(force, (tuple, list)) and len(force) >= 3:
                    fx, fy, fz = float(force[0]), float(force[1]), float(force[2])
                else:
                    mag = float(force)
                    direction = (getattr(load, "direction", None) or "z").lower()
                    fx, fy, fz = 0.0, 0.0, 0.0
                    if direction in {"x", "+x"}:
                        fx = mag
                    elif direction == "-x":
                        fx = -abs(mag)
                    elif direction in {"y", "+y"}:
                        fy = mag
                    elif direction == "-y":
                        fy = -abs(mag)
                    elif direction in {"z", "+z", "normal"}:
                        fz = mag
                    elif direction == "-z":
                        fz = -abs(mag)
                    else:
                        fz = mag

                load_name = _sanitize_set_name(f"DLOAD_{idx}", f"DLOAD_{idx}")
                load_sets.append(
                    (
                        load_name,
                        np.unique(node_idx),
                        {"x": fx, "y": fy, "z": fz},
                    )
                )

            elif isinstance(load, AccelerationLoad):
                # Write as *DLOAD — no node set needed, uses element set directly.
                # If the load references a specific element set use it; otherwise
                # fall back to the ALL_ELEMENTS set that to_inp() always defines.
                raw_elset = getattr(load, "element_set", None) or ""
                elset = (
                    "ALL_ELEMENTS"
                    if raw_elset.upper() in ("", "EALL", "ALL", "ALL_ELEMENTS")
                    else raw_elset
                )
                accel_type = getattr(load, "load_type", AccelerationLoad.GRAVITY)
                magnitude = float(getattr(load, "magnitude", 0.0))

                if accel_type == AccelerationLoad.GRAVITY:
                    direction = getattr(load, "direction", None) or (0.0, 0.0, -1.0)
                    dx, dy, dz = (
                        float(direction[0]),
                        float(direction[1]),
                        float(direction[2]),
                    )
                    dload_blocks.append((elset, "GRAV", magnitude, (dx, dy, dz)))

                elif accel_type == AccelerationLoad.CENTRIFUGAL:
                    axis = getattr(load, "axis", None) or (0.0, 0.0, 1.0)
                    origin = getattr(load, "origin", None) or (0.0, 0.0, 0.0)
                    dload_blocks.append((elset, "CENTRIF", magnitude, (*origin, *axis)))

                elif accel_type == AccelerationLoad.BODY_FORCE:
                    direction = getattr(load, "direction", None) or (0.0, 0.0, -1.0)
                    dx, dy, dz = (
                        float(direction[0]),
                        float(direction[1]),
                        float(direction[2]),
                    )
                    # Map to independent BX / BY / BZ body-load components
                    for kw, component in (("BX", dx), ("BY", dy), ("BZ", dz)):
                        if abs(component) > 1e-12:
                            dload_blocks.append(
                                (elset, kw, magnitude * component, None)
                            )

        mat_name = _sanitize_set_name(
            getattr(self.material, "name", "MATERIAL"), "MATERIAL"
        )
        E = getattr(self.material, "elastic_modulus_mpa", None)
        if E is None:
            E = getattr(self.material, "E", 210000)
        nu = getattr(self.material, "poissons_ratio", None)
        if nu is None:
            nu = getattr(self.material, "nu", 0.3)

        out_path = Path(filepath)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        title = (
            analysis_title
            or self.description
            or f"LoadCase {self.problem_id or ''}".strip()
        )

        with out_path.open("w") as f:
            f.write("*HEADING\n")
            f.write(f"{title}\n\n")

            f.write("*NODE, NSET=ALL_NODES\n")
            for i, (x, y, z) in enumerate(nodes, start=1):
                f.write(f"{i}, {x:.6f}, {y:.6f}, {z:.6f}\n")
            f.write("\n")

            f.write(f"*ELEMENT, TYPE={ccx_elem_type}, ELSET=ALL_ELEMENTS\n")
            for i, elem in enumerate(elements, start=1):
                node_ids = [int(n) + 1 for n in elem.tolist()]
                f.write(f"{i}, " + ", ".join(map(str, node_ids)) + "\n")
            f.write("\n")

            for set_name, node_idx, _ in constraint_sets:
                node_ids = sorted(int(n) + 1 for n in node_idx.tolist())
                if not node_ids:
                    continue
                f.write(f"*NSET, NSET={set_name}\n")
                for i in range(0, len(node_ids), 16):
                    f.write(", ".join(map(str, node_ids[i : i + 16])) + "\n")
                f.write("\n")

            for set_name, node_idx, _ in load_sets:
                node_ids = sorted(int(n) + 1 for n in node_idx.tolist())
                if not node_ids:
                    continue
                f.write(f"*NSET, NSET={set_name}\n")
                for i in range(0, len(node_ids), 16):
                    f.write(", ".join(map(str, node_ids[i : i + 16])) + "\n")
                f.write("\n")

            f.write(f"*MATERIAL, NAME={mat_name}\n")
            f.write("*ELASTIC\n")
            f.write(f"{float(E):.6f}, {float(nu):.6f}\n\n")

            f.write(f"*SOLID SECTION, ELSET=ALL_ELEMENTS, MATERIAL={mat_name}\n\n")

            for set_name, _, dofs in constraint_sets:
                f.write("*BOUNDARY\n")

                # Emit contiguous DOF ranges to preserve compact boundary
                # representation through INP round-trips.
                locked = [i + 1 for i, is_locked in enumerate(dofs) if is_locked]
                if locked:
                    start = locked[0]
                    prev = locked[0]
                    for dof_idx in locked[1:]:
                        if dof_idx == prev + 1:
                            prev = dof_idx
                            continue
                        f.write(f"{set_name}, {start}, {prev}\n")
                        start = dof_idx
                        prev = dof_idx
                    f.write(f"{set_name}, {start}, {prev}\n")
                f.write("\n")

            f.write("*STEP, NLGEOM=NO\n")
            f.write("*STATIC\n")
            f.write("0.1, 1.0\n\n")

            for set_name, node_idx, force_vec in load_sets:
                # Preserve parsed load-component semantics across INP round-trips:
                # from_inp() stores CLOAD magnitudes directly as PointLoad.force
                # (without multiplying by NSET cardinality), so export should write
                # the same component values back to CLOAD.
                fx = float(force_vec.get("x", 0.0))
                fy = float(force_vec.get("y", 0.0))
                fz = float(force_vec.get("z", 0.0))

                components = ((1, fx), (2, fy), (3, fz))
                for dof, value in components:
                    if abs(value) < 1e-12:
                        continue
                    f.write("*CLOAD\n")
                    f.write(f"{set_name}, {dof}, {value:.6f}\n\n")

            for elset, dload_type, magnitude, extra in dload_blocks:
                f.write("*DLOAD\n")
                if dload_type == "GRAV":
                    dx, dy, dz = extra
                    f.write(
                        f"{elset}, GRAV, {magnitude:.6g},"
                        f" {dx:.6g}, {dy:.6g}, {dz:.6g}\n\n"
                    )
                elif dload_type in ("BX", "BY", "BZ"):
                    f.write(f"{elset}, {dload_type}, {magnitude:.6g}\n\n")
                elif dload_type == "CENTRIF":
                    ox, oy, oz, nx, ny, nz = extra
                    f.write(
                        f"{elset}, CENTRIF, {magnitude:.6g},"
                        f" {ox:.6g}, {oy:.6g}, {oz:.6g},"
                        f" {nx:.6g}, {ny:.6g}, {nz:.6g}\n\n"
                    )

            f.write("*NODE FILE\n")
            f.write("U, RF\n")
            f.write("*EL FILE\n")
            f.write("S, E\n\n")
            f.write("*END STEP\n")

        return out_path

    @staticmethod
    def from_cdb(filepath: str) -> "LoadCase":
        """
        Parse an ANSYS Mechanical CDB (``/CDWRITE``) file and return a LoadCase.

        Uses *ansys-mapdl-reader* (``pip install ansys-mapdl-reader``) to read
        mesh topology and named node components (``NBLOCK``, ``EBLOCK``,
        ``CMBLOCK``), then performs a lightweight single-pass scan for the solver
        directives that the library does not expose:

        * ``D``  commands – displacement constraints → ``FixedConstraint``
        * ``F``  commands – concentrated forces      → ``PointLoad``
        * ``MP`` commands – material properties      → ``MaterialProperties``

        Args:
            filepath: Path to the ``.cdb`` file.

        Returns:
            LoadCase with mesh, material, selectors, constraints and loads populated.

        Raises:
            ImportError: If *ansys-mapdl-reader* is not installed.
            FileNotFoundError: If the file does not exist.
        """
        try:
            from ansys.mapdl.reader import Archive
        except ImportError as exc:
            raise ImportError(
                "ansys-mapdl-reader is required for from_cdb(). "
                "Install it with: pip install ansys-mapdl-reader"
            ) from exc
        import math

        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"ANSYS CDB file not found: {filepath}")

        # ------------------------------------------------------------------
        # 1. Parse mesh + node components via ansys-mapdl-reader
        # ------------------------------------------------------------------
        archive = Archive(str(path), parse_vtk=False, verbose=False)
        nodes_arr = np.asarray(archive.nodes, dtype=np.float64)
        nnum = archive.nnum  # 1-based ANSYS node IDs
        node_id_to_idx: Dict[int, int] = {int(nid): i for i, nid in enumerate(nnum)}

        if len(nodes_arr) == 0:
            raise ValueError("No nodes found in CDB file")

        # ------------------------------------------------------------------
        # 2. Build element connectivity array
        #
        # archive.elem: list of 1-D arrays, one per element.
        # Layout: [mat, etype_ref, rc, sect, esys, death, sm, shape, enum,
        #          base, nid1, nid2, ...]  (ANSYS 1-based node IDs at index 10+)
        # Filter out zero padding (midside-node slots unused in linear elements).
        # ------------------------------------------------------------------
        _ncount_to_type: Dict[int, str] = {
            4: "tet4",
            8: "hex8",
            10: "tet10",
            20: "hex20",
            6: "wedge6",
            15: "wedge15",
        }
        _type_dim: Dict[str, int] = {
            "tet4": 3,
            "tet10": 3,
            "hex8": 3,
            "hex20": 3,
            "wedge6": 3,
            "wedge15": 3,
            "quad4": 2,
            "tri3": 2,
        }

        # Group elements by (type_str, n_nodes_per_elem)
        # Nodes are extracted by filtering elem[8:] against node_id_to_idx.
        # The library stores node IDs starting at _elem[8] (coincides with the
        # sequential element number slot), with _elem[9]=0 (base-element slot).
        # Using node_id_to_idx as a valid-ID filter avoids hard-coding field offsets.
        type_buckets: Dict[Tuple[str, int], List[List[int]]] = {}
        for raw_elem in archive.elem:
            nonzero_ids = [int(v) for v in raw_elem[8:] if int(v) in node_id_to_idx]
            n = len(nonzero_ids)
            if n == 0:
                continue
            elem_type_str = _ncount_to_type.get(n, f"unknown{n}")
            key = (elem_type_str, n)
            idxs = [node_id_to_idx[nid] for nid in nonzero_ids]
            type_buckets.setdefault(key, []).append(idxs)

        # Choose dominant bucket: prefer 3-D elements, then most elements
        best_key: Optional[Tuple[str, int]] = None
        best_score = (-1, 0)
        for key, rows in type_buckets.items():
            dim = _type_dim.get(key[0], 1)
            score = (dim, len(rows))
            if score > best_score:
                best_score = score
                best_key = key

        if best_key is not None and type_buckets[best_key]:
            resolved_elem_type = best_key[0]
            elems_arr = np.array(type_buckets[best_key], dtype=np.int64)
        else:
            resolved_elem_type = "tet4"
            elems_arr = np.empty((0, 4), dtype=np.int64)

        # ------------------------------------------------------------------
        # 3. Derive design-domain bounds from node coordinates
        # ------------------------------------------------------------------
        xs_a, ys_a, zs_a = nodes_arr[:, 0], nodes_arr[:, 1], nodes_arr[:, 2]
        bounds: Dict[str, float] = {
            "x_min": float(xs_a.min()),
            "x_max": float(xs_a.max()),
            "y_min": float(ys_a.min()),
            "y_max": float(ys_a.max()),
            "z_min": float(zs_a.min()),
            "z_max": float(zs_a.max()),
        }
        positive_dims = [
            d
            for d in (
                bounds["x_max"] - bounds["x_min"],
                bounds["y_max"] - bounds["y_min"],
                bounds["z_max"] - bounds["z_min"],
            )
            if d > 0
        ]
        min_dim = min(positive_dims) if positive_dims else 1.0
        selector_tol = max(min_dim * 0.01, 1e-4)
        min_radius = max(min_dim * 0.02, 5e-5)

        # ------------------------------------------------------------------
        # 4. Build LoadCase skeleton
        # ------------------------------------------------------------------
        problem_id = path.stem.upper()
        load_case = LoadCase(
            problem_id=problem_id, description=f"Imported from {path.name}"
        )
        load_case.selectors = {}
        load_case.bounds = bounds
        load_case.domain = DesignDomain(shape_type="box", bounds=bounds, units="mm")
        load_case.mesh_nodes = nodes_arr.astype(np.float32)
        load_case.mesh_elements = elems_arr.astype(np.int32)
        load_case.mesh_element_type = resolved_elem_type

        # ------------------------------------------------------------------
        # 5. Build spatial selectors from node_components (CMBLOCK)
        #    archive.node_components: {name: array_of_ANSYS_1based_nids}
        # ------------------------------------------------------------------
        point_sets: Dict[str, np.ndarray] = {}
        selector_map: Dict[str, str] = {}

        for cm_name, ansys_nids in archive.node_components.items():
            name = cm_name.strip()
            idx_arr = np.array(
                [
                    node_id_to_idx[int(n)]
                    for n in ansys_nids
                    if int(n) in node_id_to_idx
                ],
                dtype=np.int64,
            )
            if idx_arr.size == 0:
                continue
            point_sets[name] = idx_arr
            coords = nodes_arr[idx_arr]
            s_xs, s_ys, s_zs = coords[:, 0], coords[:, 1], coords[:, 2]
            cx = float(np.mean(s_xs))
            cy = float(np.mean(s_ys))
            cz = float(np.mean(s_zs))
            rx = max((float(s_xs.max()) - float(s_xs.min())) / 2.0, min_radius)
            ry = max((float(s_ys.max()) - float(s_ys.min())) / 2.0, min_radius)
            rz = max((float(s_zs.max()) - float(s_zs.min())) / 2.0, min_radius)
            query = {
                "x_min": max(float(s_xs.min()) - selector_tol, bounds["x_min"]),
                "x_max": min(float(s_xs.max()) + selector_tol, bounds["x_max"]),
                "y_min": max(float(s_ys.min()) - selector_tol, bounds["y_min"]),
                "y_max": min(float(s_ys.max()) + selector_tol, bounds["y_max"]),
                "z_min": max(float(s_zs.min()) - selector_tol, bounds["z_min"]),
                "z_max": min(float(s_zs.max()) + selector_tol, bounds["z_max"]),
                "x": cx,
                "y": cy,
                "z": cz,
                "rx": rx,
                "ry": ry,
                "rz": rz,
            }
            sel_id = f"SELECTOR_{name}"
            load_case.selectors[sel_id] = SpatialSelector(
                id=sel_id, type="box_3d", query=query
            )
            selector_map[name] = sel_id

        load_case.meta = {
            "node_sets_count": len(point_sets),
            "node_sets": sorted(point_sets.keys()),
            "selectors": load_case.selectors,
            "source_format": "cdb",
        }

        def _find_selector(target: str) -> Optional[str]:
            """Case-insensitive component name → selector_id lookup."""
            if target in selector_map:
                return selector_map[target]
            tl = target.lower()
            for k, v in selector_map.items():
                if k.lower() == tl:
                    return v
            return None

        # ------------------------------------------------------------------
        # 6. Single-pass scan for D / F / MP solver directives
        #    (ansys-mapdl-reader does not expose these)
        # ------------------------------------------------------------------
        d_lines: List[str] = []
        f_lines: List[str] = []
        mp_lines: List[str] = []

        with open(path, "r", errors="replace") as fh:
            for raw in fh:
                stripped = raw.strip()
                if not stripped or stripped.startswith("!"):
                    continue
                upper = stripped.upper()
                if upper.startswith("D,") or upper.startswith("D ,"):
                    d_lines.append(stripped)
                elif upper.startswith("F,") or upper.startswith("F ,"):
                    f_lines.append(stripped)
                elif upper.startswith("MP,"):
                    mp_lines.append(stripped)

        # ------------------------------------------------------------------
        # 7. Parse MP commands → MaterialProperties
        #    MP,EX,matid,value  /  MP,NUXY,matid,value  /  MP,DENS,matid,value
        # ------------------------------------------------------------------
        mat_E: float = 210000.0
        mat_nu: float = 0.3
        mat_density: float = 7.85e-9

        for mp in mp_lines:
            parts = [p.strip() for p in mp.split(",")]
            if len(parts) < 4:
                continue
            label = parts[1].upper()
            try:
                val = float(parts[3])
            except (ValueError, IndexError):
                continue
            if label == "EX":
                mat_E = val
            elif label in ("NUXY", "PRXY"):
                mat_nu = val
            elif label == "DENS":
                mat_density = val

        if mat_E > 180000:
            _mat_name = "Steel"
        elif mat_E > 100000:
            _mat_name = "Titanium"
        else:
            _mat_name = "Aluminum"
        load_case.material = MaterialProperties(
            name=_mat_name, E=mat_E, nu=mat_nu, density=mat_density
        )

        # ------------------------------------------------------------------
        # 8. Parse D commands → FixedConstraints
        #    D, NodeOrComp, DOFname, Value[, ...]
        #    DOF names: UX / UY / UZ / ALL
        # ------------------------------------------------------------------
        _dof_name_to_idx: Dict[str, int] = {"UX": 0, "UY": 1, "UZ": 2}
        d_accumulator: Dict[str, List[bool]] = {}

        for d_line in d_lines:
            parts = [p.strip() for p in d_line.split(",")]
            if len(parts) < 3:
                continue
            target = parts[1].strip()
            dof_name = parts[2].strip().upper()
            dof_names = ["UX", "UY", "UZ"] if dof_name == "ALL" else [dof_name]
            if target not in d_accumulator:
                d_accumulator[target] = [False, False, False]
            for dn in dof_names:
                idx = _dof_name_to_idx.get(dn)
                if idx is not None:
                    d_accumulator[target][idx] = True

        for target, dofs in d_accumulator.items():
            dofs_tuple = tuple(dofs)
            sel_id = _find_selector(target)
            if sel_id:
                sel = load_case.selectors[sel_id]
                bc = FixedConstraint(
                    location=sel.query,
                    dofs=dofs_tuple,
                    tolerance=selector_tol,
                )
                bc.name = f"BC_{target}"
                bc.region_id = sel_id
            elif target.isdigit():
                nidx = node_id_to_idx.get(int(target))
                if nidx is None:
                    continue
                n_x = float(nodes_arr[nidx, 0])
                n_y = float(nodes_arr[nidx, 1])
                n_z = float(nodes_arr[nidx, 2])
                bc = FixedConstraint(
                    location=(n_x, n_y, n_z),
                    dofs=dofs_tuple,
                    tolerance=selector_tol,
                )
                bc.name = f"BC_NODE_{target}"
                bc.region_id = None
            else:
                continue
            load_case.constraints.append(bc)

        # ------------------------------------------------------------------
        # 9. Parse F commands → PointLoads
        #    F, NodeOrComp, DOFname, Value
        #    DOF names: FX / FY / FZ
        # ------------------------------------------------------------------
        _force_dof_to_axis: Dict[str, str] = {"FX": "x", "FY": "y", "FZ": "z"}
        f_accumulator: Dict[str, Dict[str, float]] = {}

        for f_line in f_lines:
            parts = [p.strip() for p in f_line.split(",")]
            if len(parts) < 4:
                continue
            target = parts[1].strip()
            dof_name = parts[2].strip().upper()
            try:
                val = float(parts[3])
            except ValueError:
                continue
            axis = _force_dof_to_axis.get(dof_name)
            if axis is None:
                continue
            if target not in f_accumulator:
                f_accumulator[target] = {"x": 0.0, "y": 0.0, "z": 0.0}
            f_accumulator[target][axis] += val

        for load_idx, (target, vec) in enumerate(f_accumulator.items(), start=1):
            fx = vec.get("x", 0.0)
            fy = vec.get("y", 0.0)
            fz = vec.get("z", 0.0)
            mag = math.sqrt(fx**2 + fy**2 + fz**2)

            # Determine dominant-axis direction string
            if mag > 0:
                abs_vals = [abs(fx), abs(fy), abs(fz)]
                dom = int(np.argmax(abs_vals))
                axis_str = (
                    ("+x", "+y", "+z")[dom]
                    if [fx, fy, fz][dom] > 0
                    else ("-x", "-y", "-z")[dom]
                )
            else:
                axis_str = None

            sel_id = _find_selector(target)
            if sel_id:
                sel = load_case.selectors[sel_id]
                load = PointLoad(
                    point=sel.query,
                    force=(fx, fy, fz),
                    direction=None,
                    tolerance=selector_tol,
                    search_radius=None,
                )
                load.name = f"LOAD_{target}_{load_idx}"
                load.region_id = sel_id
            elif target.isdigit():
                nidx = node_id_to_idx.get(int(target))
                if nidx is None:
                    continue
                n_x = float(nodes_arr[nidx, 0])
                n_y = float(nodes_arr[nidx, 1])
                n_z = float(nodes_arr[nidx, 2])
                pt_id = f"FPOINT_{target}"
                load_case.selectors.setdefault(
                    pt_id,
                    SpatialSelector(
                        id=pt_id, type="point", query={"x": n_x, "y": n_y, "z": n_z}
                    ),
                )
                load = PointLoad(
                    point=(n_x, n_y, n_z),
                    force=(fx, fy, fz),
                    direction=None,
                    tolerance=1,
                    search_radius=None,
                )
                load.name = f"LOAD_NODE_{target}_{load_idx}"
                load.region_id = pt_id
            else:
                continue
            load.vector_newtons = vec
            load.direction = axis_str
            load.magnitude_newtons = mag
            load_case.loads.append(load)

        logger.info(
            f"Parsed ANSYS CDB (pyansys): {len(nodes_arr)} nodes, "
            f"{len(elems_arr)} {resolved_elem_type} elements, "
            f"{len(point_sets)} CMBLOCKs, "
            f"{len(load_case.boundary_conditions)} BCs, "
            f"{len(load_case.loads)} loads"
        )
        return load_case

    @staticmethod
    def from_dat(filepath: str) -> "LoadCase":
        """
        Parse a Nastran BDF/DAT file and return a LoadCase.

        Uses *pyNastran* (``pip install pyNastran``) to read all cards, then maps:

        * ``GRID``            → mesh nodes
        * ``CTETRA`` / ``CHEXA`` / ``CPENTA`` / ``CQUAD4`` / ``CTRIA3`` → elements
        * ``MAT1``            → ``MaterialProperties``
        * ``SPC`` / ``SPC1`` → ``FixedConstraint`` (DOF components 1-3)
        * ``FORCE``           → ``PointLoad`` (basic coordinate system assumed)
        * ``SET1`` / implicit SPC-node groups → spatial selectors

        Args:
            filepath: Path to the ``.dat``, ``.bdf``, ``.nas``, or ``.fem`` file.

        Returns:
            LoadCase with mesh, material, selectors, constraints and loads populated.

        Raises:
            ImportError: If *pyNastran* is not installed.
            FileNotFoundError: If the file does not exist.
        """
        try:
            from pyNastran.bdf.bdf import BDF
        except ImportError as exc:
            raise ImportError(
                "pyNastran is required for from_dat(). "
                "Install with: pip install pyNastran"
            ) from exc

        import math

        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Nastran file not found: {filepath}")

        bdf = BDF(debug=False)
        try:
            bdf.read_bdf(str(path), punch=False)
        except Exception:
            # Retry in punch mode for bulk-data-only files (no Executive/Case
            # Control decks) – common for *.dat exports from meshing tools.
            bdf = BDF(debug=False)
            bdf.read_bdf(str(path), punch=True)

        # ------------------------------------------------------------------
        # 1. Nodes → nodes_arr, node_id_to_idx (0-based)
        # ------------------------------------------------------------------
        sorted_nids = sorted(bdf.nodes.keys())
        nodes_arr = np.array(
            [bdf.nodes[nid].xyz for nid in sorted_nids], dtype=np.float64
        )
        # Ensure 3-D coordinate array
        if nodes_arr.ndim == 2 and nodes_arr.shape[1] == 2:
            nodes_arr = np.hstack(
                [nodes_arr, np.zeros((len(nodes_arr), 1), dtype=np.float64)]
            )

        node_id_to_idx: Dict[int, int] = {nid: i for i, nid in enumerate(sorted_nids)}

        if len(nodes_arr) == 0:
            raise ValueError("No GRID nodes found in Nastran file")

        # ------------------------------------------------------------------
        # 2. Elements → elems_arr, resolved_elem_type
        #
        # Pick the dominant 3-D element type (prefer higher-order if mixed).
        # ------------------------------------------------------------------
        _nastran_type_map: Dict[str, Tuple[str, int]] = {
            # card_type: (internal_name, dim_rank)
            "CTETRA": ("tet4", 3),  # 4-node; overridden to tet10 if nids==10
            "CHEXA": ("hex8", 3),  # 8-node; overridden to hex20 if nids==20
            "CPENTA": ("wedge", 3),
            "CPYRA": ("pyra5", 3),
            "CQUAD4": ("quad4", 2),
            "CQUADR": ("quad4", 2),
            "CQUAD8": ("quad8", 2),
            "CTRIA3": ("tri3", 2),
            "CTRIAR": ("tri3", 2),
            "CTRIA6": ("tri6", 2),
        }

        # Group all element-node-ID lists by (internal_type, n_nodes_per_elem)
        type_buckets: Dict[Tuple[str, int], List[List[int]]] = {}
        for eid, elem in bdf.elements.items():
            card_type = elem.type.upper()
            nids_raw = elem.node_ids
            n = len(nids_raw)

            if card_type == "CTETRA":
                itype = "tet4" if n == 4 else "tet10"
            elif card_type == "CHEXA":
                itype = "hex8" if n == 8 else "hex20"
            elif card_type in _nastran_type_map:
                itype, _ = _nastran_type_map[card_type]
            else:
                continue

            key = (itype, n)
            if key not in type_buckets:
                type_buckets[key] = []

            row = [node_id_to_idx.get(nid, -1) for nid in nids_raw]
            if all(idx >= 0 for idx in row):
                type_buckets[key].append(row)

        # Choose dominant bucket: prefer 3-D, then most elements
        _type_dim: Dict[str, int] = {
            "tet4": 3,
            "tet10": 3,
            "hex8": 3,
            "hex20": 3,
            "wedge": 3,
            "pyra5": 3,
            "quad4": 2,
            "quad8": 2,
            "tri3": 2,
            "tri6": 2,
        }
        best_key: Optional[Tuple[str, int]] = None
        best_score = (-1, 0)
        for key, rows in type_buckets.items():
            dim = _type_dim.get(key[0], 1)
            score = (dim, len(rows))
            if score > best_score:
                best_score = score
                best_key = key

        if best_key is not None and type_buckets[best_key]:
            resolved_elem_type = best_key[0]
            elems_arr = np.array(type_buckets[best_key], dtype=np.int64)
        else:
            resolved_elem_type = "tet4"
            elems_arr = np.empty((0, 4), dtype=np.int64)

        # ------------------------------------------------------------------
        # 3. Material → MaterialProperties (first MAT1 wins)
        # ------------------------------------------------------------------
        mat_E, mat_nu, mat_density = 210000.0, 0.3, 7.85e-9
        _mat_name = "Steel"
        for mid, mat in bdf.materials.items():
            if mat.type == "MAT1":
                mat_E = float(mat.e or 210000.0)
                mat_nu = float(mat.nu if mat.nu is not None else 0.3)
                mat_density = float(mat.rho if mat.rho is not None else 7.85e-9)
                if mat_E > 180000:
                    _mat_name = "Steel"
                elif mat_E > 100000:
                    _mat_name = "Titanium"
                else:
                    _mat_name = "Aluminum"
                break  # use first material found

        # ------------------------------------------------------------------
        # 4. Derive design-domain bounds
        # ------------------------------------------------------------------
        xs_a, ys_a, zs_a = nodes_arr[:, 0], nodes_arr[:, 1], nodes_arr[:, 2]
        bounds: Dict[str, float] = {
            "x_min": float(xs_a.min()),
            "x_max": float(xs_a.max()),
            "y_min": float(ys_a.min()),
            "y_max": float(ys_a.max()),
            "z_min": float(zs_a.min()),
            "z_max": float(zs_a.max()),
        }
        positive_dims = [
            d
            for d in (
                bounds["x_max"] - bounds["x_min"],
                bounds["y_max"] - bounds["y_min"],
                bounds["z_max"] - bounds["z_min"],
            )
            if d > 0
        ]
        min_dim = min(positive_dims) if positive_dims else 1.0
        selector_tol = max(min_dim * 0.01, 1e-4)
        min_radius = max(min_dim * 0.02, 5e-5)

        # ------------------------------------------------------------------
        # 5. Build LoadCase skeleton
        # ------------------------------------------------------------------
        problem_id = path.stem.upper()
        load_case = LoadCase(
            problem_id=problem_id, description=f"Imported from {path.name}"
        )
        load_case.selectors = {}
        load_case.bounds = bounds
        load_case.domain = DesignDomain(shape_type="box", bounds=bounds, units="mm")
        load_case.mesh_nodes = nodes_arr.astype(np.float32)
        load_case.mesh_elements = elems_arr.astype(np.int32)
        load_case.mesh_element_type = resolved_elem_type
        load_case.material = MaterialProperties(
            name=_mat_name, E=mat_E, nu=mat_nu, density=mat_density
        )

        # ------------------------------------------------------------------
        # 6. Build spatial selectors from SPC node groups
        #
        # pyNastran's bdf.spcs is {conid: [SPC | SPC1, ...]}.  We collect the
        # unique node set for every constraint-set ID so each gets a selector.
        # ------------------------------------------------------------------
        selector_map: Dict[str, str] = {}  # conid_str → selector_id
        spc_node_sets: Dict[str, np.ndarray] = {}  # conid_str → 0-based idx array

        def _register_selector(conid_str: str, idx_arr: np.ndarray) -> Optional[str]:
            """Build a box-query selector for a node group and register it."""
            if idx_arr.size == 0:
                return None
            coords = nodes_arr[idx_arr]
            s_xs = coords[:, 0]
            s_ys = coords[:, 1]
            s_zs = coords[:, 2]
            cx = float(s_xs.mean())
            cy = float(s_ys.mean())
            cz = float(s_zs.mean())
            rx = max((float(s_xs.max()) - float(s_xs.min())) / 2.0, min_radius)
            ry = max((float(s_ys.max()) - float(s_ys.min())) / 2.0, min_radius)
            rz = max((float(s_zs.max()) - float(s_zs.min())) / 2.0, min_radius)
            query = {
                "x_min": max(float(s_xs.min()) - selector_tol, bounds["x_min"]),
                "x_max": min(float(s_xs.max()) + selector_tol, bounds["x_max"]),
                "y_min": max(float(s_ys.min()) - selector_tol, bounds["y_min"]),
                "y_max": min(float(s_ys.max()) + selector_tol, bounds["y_max"]),
                "z_min": max(float(s_zs.min()) - selector_tol, bounds["z_min"]),
                "z_max": min(float(s_zs.max()) + selector_tol, bounds["z_max"]),
                "x": cx,
                "y": cy,
                "z": cz,
                "rx": rx,
                "ry": ry,
                "rz": rz,
            }
            sel_id = f"SELECTOR_SPC_{conid_str}"
            load_case.selectors[sel_id] = SpatialSelector(
                id=sel_id, type="box_3d", query=query
            )
            selector_map[conid_str] = sel_id
            return sel_id

        # Accumulate node indices per SPC constraint-set ID
        spc_conid_nodes: Dict[str, List[int]] = {}
        spc_conid_dofs: Dict[str, List[bool]] = {}

        for conid, spc_cards in bdf.spcs.items():
            conid_str = str(conid)
            if conid_str not in spc_conid_nodes:
                spc_conid_nodes[conid_str] = []
                spc_conid_dofs[conid_str] = [False, False, False]
            for card in spc_cards:
                # Both SPC and SPC1 expose .nodes (list of node IDs)
                nids = card.nodes if hasattr(card, "nodes") else []
                # .components is a string of digit chars for SPC1, or list for SPC
                if hasattr(card, "components"):
                    comp = card.components
                    # SPC1: single string applied to all nodes
                    if isinstance(comp, (str, int)):
                        comp_str = str(comp)
                        for i, dof_char in enumerate(("1", "2", "3")):
                            if dof_char in comp_str:
                                spc_conid_dofs[conid_str][i] = True
                    # SPC: list of per-node component strings
                    elif isinstance(comp, (list, tuple)):
                        for c in comp:
                            for i, dof_char in enumerate(("1", "2", "3")):
                                if dof_char in str(c):
                                    spc_conid_dofs[conid_str][i] = True
                for nid in nids:
                    idx = node_id_to_idx.get(nid)
                    if idx is not None:
                        spc_conid_nodes[conid_str].append(idx)

        for conid_str, nidxs in spc_conid_nodes.items():
            if not nidxs:
                continue
            idx_arr = np.unique(np.array(nidxs, dtype=np.int64))
            spc_node_sets[conid_str] = idx_arr
            sel_id = _register_selector(conid_str, idx_arr)
            if sel_id is None:
                continue
            dofs = tuple(spc_conid_dofs[conid_str])
            if not any(dofs):
                dofs = (True, True, True)  # full fix if no specific DOF info
            load_case.boundary_conditions.append(
                FixedConstraint(
                    location=load_case.selectors[sel_id].query,
                    dofs=dofs,
                    tolerance=1,
                )
            )

        # ------------------------------------------------------------------
        # 7. Parse FORCE cards → PointLoads
        #
        # bdf.loads is {sid: [FORCE, MOMENT, PLOAD4, ...]}
        # FORCE: .node (nid), .mag (scalar), .xyz (unit vector in coord .cid)
        # Multiple FORCE cards for the same node are summed.
        # ------------------------------------------------------------------
        # node_id → accumulated force vector
        force_by_node: Dict[int, np.ndarray] = {}

        for sid, load_cards in bdf.loads.items():
            for card in load_cards:
                if card.type != "FORCE":
                    continue
                nid = card.node
                mag = float(card.mag)
                xyz = np.asarray(card.xyz, dtype=np.float64)
                fvec = mag * xyz
                if nid in force_by_node:
                    force_by_node[nid] += fvec
                else:
                    force_by_node[nid] = fvec.copy()

        for load_idx, (nid, fvec) in enumerate(force_by_node.items(), start=1):
            nidx = node_id_to_idx.get(nid)
            if nidx is None:
                continue
            fx, fy, fz = float(fvec[0]), float(fvec[1]), float(fvec[2])
            abs_comps = [(abs(fx), "x", fx), (abs(fy), "y", fy), (abs(fz), "z", fz)]
            dom_abs, dom_ax, dom_mag_v = max(abs_comps, key=lambda t: t[0])
            axis: Optional[str] = (
                (dom_ax if dom_mag_v >= 0 else f"-{dom_ax}") if dom_abs > 0 else None
            )
            n_x = float(nodes_arr[nidx, 0])
            n_y = float(nodes_arr[nidx, 1])
            n_z = float(nodes_arr[nidx, 2])
            pt_id = f"FPOINT_{nid}"
            load_case.selectors.setdefault(
                pt_id,
                SpatialSelector(
                    id=pt_id, type="point", query={"x": n_x, "y": n_y, "z": n_z}
                ),
            )
            vec: Dict[str, float] = {"x": fx, "y": fy, "z": fz}
            load = PointLoad(
                point=(n_x, n_y, n_z),
                force=(fx, fy, fz),
                direction=None,
                tolerance=1,
                search_radius=None,
            )
            load.name = f"LOAD_NODE_{nid}_{load_idx}"
            load.region_id = pt_id
            load.vector_newtons = vec
            load.direction = axis
            load.magnitude_newtons = math.sqrt(fx**2 + fy**2 + fz**2)
            load_case.loads.append(load)

        load_case.meta = {
            "node_sets_count": len(spc_node_sets),
            "node_sets": sorted(spc_node_sets.keys()),
            "selectors": load_case.selectors,
            "source_format": "dat",
            "nastran_spc_ids": sorted(bdf.spcs.keys()),
            "nastran_load_ids": sorted(bdf.loads.keys()),
        }

        logger.info(
            f"Parsed Nastran DAT (pyNastran): {len(nodes_arr)} nodes, "
            f"{len(elems_arr)} {resolved_elem_type} elements, "
            f"{len(bdf.spcs)} SPC sets, "
            f"{len(load_case.boundary_conditions)} BCs, "
            f"{len(load_case.loads)} loads"
        )
        return load_case
