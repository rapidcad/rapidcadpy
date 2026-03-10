from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Tuple, Union
from venv import logger

import numpy as np

from .spatial_selector import SpatialSelector

from .design_domain import DesignDomain

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

        min_dim = min(positive_dims)

        # Use full 3D volume estimate when all dimensions are positive.
        if dx > 0.0 and dy > 0.0 and dz > 0.0:
            volume = dx * dy * dz
            h_est = (volume / float(num_nodes)) ** (1.0 / 3.0)
        else:
            # Degenerate fallback (effectively 2D/1D bounds): use minimum positive dim.
            h_est = min_dim

        # Clamp to practical range relative to geometry scale.
        # - Don't exceed ~1/2 of smallest dimension.
        # - Don't go below a tiny absolute floor.
        h_max = 0.5 * min_dim
        h_min = 1e-3
        h = max(h_min, min(h_est, h_max))

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
        self, mesher: str = "gmsh-subprocess", mesh_size: float = 1
    ) -> "FEAAnalyzer":
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
                from .mesher import (
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

            f.write("*NODE FILE\n")
            f.write("U, RF\n")
            f.write("*EL FILE\n")
            f.write("S, E\n\n")
            f.write("*END STEP\n")

        return out_path

    @staticmethod
    def from_inp(filepath: str) -> "LoadCase":
        """
        Parse an Abaqus .inp file and return a LoadCase.

        Mesh data (nodes, elements, NSETs, ELSETs) is read via *meshio* for
        robustness and broad format support.  Boundary conditions (*BOUNDARY)
        and concentrated loads (*CLOAD) are extracted with a targeted custom
        pass because meshio does not parse FEA solver directives.

        Args:
            filepath: Path to the ``.inp`` file.

        Returns:
            LoadCase with mesh, selectors, constraints and loads populated.
        """
        import meshio

        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Abaqus file not found: {filepath}")

        # ------------------------------------------------------------------
        # 1. Parse mesh via meshio (nodes, elements, NSETs, ELSETs)
        # ------------------------------------------------------------------
        mesh = meshio.read(str(path))

        nodes_arr = np.asarray(mesh.points, dtype=np.float64)
        # Ensure 3-D coordinate array even when meshio returns 2-D geometry
        if nodes_arr.ndim == 2 and nodes_arr.shape[1] == 2:
            nodes_arr = np.hstack(
                [nodes_arr, np.zeros((len(nodes_arr), 1), dtype=np.float64)]
            )

        # point_sets: name -> 0-based index array
        point_sets: Dict[str, np.ndarray] = {
            k: np.asarray(v, dtype=np.int64) for k, v in (mesh.point_sets or {}).items()
        }
        cell_sets: Dict[str, Any] = mesh.cell_sets or {}

        # Choose primary cell block: prefer 3-D (tet/hex) over 2-D/1-D
        _dim_rank = {
            "tetra": 3,
            "tetra10": 3,
            "hexahedron": 3,
            "hexahedron20": 3,
            "wedge": 3,
            "pyramid": 3,
            "quad": 2,
            "triangle": 2,
            "quad8": 2,
            "triangle6": 2,
            "line": 1,
            "vertex": 0,
        }
        best_block = None
        best_rank = -1
        for cb in mesh.cells:
            rank = _dim_rank.get(cb.type, 1)
            if rank > best_rank or (
                rank == best_rank
                and best_block is not None
                and len(cb.data) > len(best_block.data)
            ):
                best_block = cb
                best_rank = rank

        _meshio_to_type = {
            "tetra": "tet4",
            "tetra10": "tet10",
            "hexahedron": "hex8",
            "hexahedron20": "hex20",
            "quad": "quad4",
            "quad8": "quad8",
            "triangle": "tri3",
            "triangle6": "tri6",
        }
        if best_block is not None:
            elems_arr = np.asarray(best_block.data, dtype=np.int64)
            resolved_elem_type = _meshio_to_type.get(best_block.type, best_block.type)
        else:
            elems_arr = np.empty((0, 4), dtype=np.int64)
            resolved_elem_type = "tet4"

        if len(nodes_arr) == 0:
            raise ValueError("No nodes found in .inp file")

        # ------------------------------------------------------------------
        # 2. Single targeted pass: build orig-1-based-node-ID → 0-based-idx
        #    map (needed for CLOAD lines referencing explicit node IDs), and
        #    collect raw *BOUNDARY / *CLOAD data lines.
        # ------------------------------------------------------------------
        p_keyword = re.compile(r"^\*([\w\s]+)(?:,\s*(.*))?")
        p_node_id = re.compile(r"^\s*(\d+),\s*[\d\.\-eE]")

        node_id_to_idx: Dict[int, int] = {}
        raw_bc_lines: List[str] = []
        raw_cload_lines: List[str] = []

        with open(path, "r") as _f:
            raw_lines = _f.readlines()

        section = None
        node_counter = 0

        for raw in raw_lines:
            line = raw.strip()
            if not line or line.startswith("**"):
                continue
            if line.startswith("*"):
                m = p_keyword.match(line)
                if not m:
                    section = None
                    continue
                kw = m.group(1).upper().strip()
                section = kw if kw in {"NODE", "BOUNDARY", "CLOAD"} else None
                continue

            if section == "NODE":
                m = p_node_id.match(line)
                if m:
                    node_id_to_idx[int(m.group(1))] = node_counter
                    node_counter += 1
            elif section == "BOUNDARY":
                raw_bc_lines.append(line)
            elif section == "CLOAD":
                raw_cload_lines.append(line)

        # ------------------------------------------------------------------
        # 3. Derive design-domain bounds from node coordinates
        # ------------------------------------------------------------------
        xs, ys, zs = nodes_arr[:, 0], nodes_arr[:, 1], nodes_arr[:, 2]
        bounds: Dict[str, float] = {
            "x_min": float(xs.min()),
            "x_max": float(xs.max()),
            "y_min": float(ys.min()),
            "y_max": float(ys.max()),
            "z_min": float(zs.min()),
            "z_max": float(zs.max()),
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

        problem_id = path.stem.upper()
        load_case = LoadCase(
            problem_id=problem_id, description=f"Imported from {path.name}"
        )
        load_case.selectors = {}
        load_case.meta = {
            "node_sets_count": len(point_sets),
            "element_sets_count": len(cell_sets),
            "node_sets": sorted(point_sets.keys()),
            "element_sets": sorted(cell_sets.keys()),
            "selectors": load_case.selectors,
        }
        load_case.domain = DesignDomain(shape_type="box", bounds=bounds, units="mm")
        load_case.bounds = bounds
        load_case.mesh_nodes = nodes_arr.astype(np.float32)
        load_case.mesh_elements = elems_arr.astype(np.int32)
        load_case.mesh_element_type = resolved_elem_type

        # ------------------------------------------------------------------
        # 4. Build spatial selectors from NSETs (meshio point_sets)
        # ------------------------------------------------------------------
        selector_map: Dict[str, str] = {}  # nset_name -> selector_id

        for name, idx_arr in point_sets.items():
            if idx_arr.size == 0:
                continue
            coords = nodes_arr[idx_arr]
            s_xs, s_ys, s_zs = coords[:, 0], coords[:, 1], coords[:, 2]
            cx, cy, cz = float(s_xs.mean()), float(s_ys.mean()), float(s_zs.mean())
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

        def _find_selector(target: str) -> Optional[str]:
            """Case-insensitive NSET name → selector_id lookup."""
            if target in selector_map:
                return selector_map[target]
            tl = target.lower()
            for k, v in selector_map.items():
                if k.lower() == tl:
                    return v
            return None

        # ------------------------------------------------------------------
        # 5. Parse *BOUNDARY → FixedConstraints
        # ------------------------------------------------------------------
        for line in raw_bc_lines:
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 3:
                continue
            nset_name = parts[0]
            try:
                first_dof, last_dof = int(parts[1]), int(parts[2])
            except ValueError:
                continue

            dof_lock = tuple(first_dof <= i <= last_dof for i in (1, 2, 3))

            sel_id = _find_selector(nset_name)
            if sel_id:
                bc = FixedConstraint(
                    location=load_case.selectors[sel_id].query,
                    dofs=dof_lock,
                    tolerance=1,
                )
                load_case.boundary_conditions.append(bc)
            elif nset_name.isdigit():
                idx = node_id_to_idx.get(int(nset_name))
                if idx is not None:
                    nx, ny, nz = (
                        float(nodes_arr[idx, 0]),
                        float(nodes_arr[idx, 1]),
                        float(nodes_arr[idx, 2]),
                    )
                    pt_id = f"PT_{nset_name}"
                    load_case.selectors.setdefault(
                        pt_id,
                        SpatialSelector(
                            id=pt_id,
                            type="point",
                            query={"x": nx, "y": ny, "z": nz},
                        ),
                    )
                    load_case.boundary_conditions.append(
                        FixedConstraint(
                            location=(nx, ny, nz), dofs=dof_lock, tolerance=1
                        )
                    )

        # ------------------------------------------------------------------
        # 6. Parse *CLOAD → PointLoads
        # ------------------------------------------------------------------
        for line in raw_cload_lines:
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 3:
                continue
            target = parts[0]
            try:
                dof, mag = int(parts[1]), float(parts[2])
            except ValueError:
                continue

            vec: Dict[str, float] = {"x": 0.0, "y": 0.0, "z": 0.0}
            axis: Optional[str] = None
            if dof == 1:
                vec["x"] = mag
                axis = "x" if mag >= 0 else "-x"
            elif dof == 2:
                vec["y"] = mag
                axis = "y" if mag >= 0 else "-y"
            elif dof == 3:
                vec["z"] = mag
                axis = "z" if mag >= 0 else "-z"

            # Determine if target maps to a NSET
            matched_nset: Optional[str] = None
            for k in point_sets:
                if k == target or k.lower() == target.lower():
                    matched_nset = k
                    break
            singleton_nset = (
                matched_nset is not None and point_sets[matched_nset].size <= 1
            )

            sel_id = _find_selector(target)
            if sel_id:
                query = load_case.selectors[sel_id].query
                cx = query.get(
                    "x", (query.get("x_min", 0.0) + query.get("x_max", 0.0)) / 2
                )
                cy = query.get(
                    "y", (query.get("y_min", 0.0) + query.get("y_max", 0.0)) / 2
                )
                cz = query.get(
                    "z", (query.get("z_min", 0.0) + query.get("z_max", 0.0)) / 2
                )
                rx = query.get(
                    "rx",
                    max(
                        (query.get("x_max", cx) - query.get("x_min", cx)) / 2,
                        min_radius,
                    ),
                )
                ry = query.get(
                    "ry",
                    max(
                        (query.get("y_max", cy) - query.get("y_min", cy)) / 2,
                        min_radius,
                    ),
                )
                rz = query.get(
                    "rz",
                    max(
                        (query.get("z_max", cz) - query.get("z_min", cz)) / 2,
                        min_radius,
                    ),
                )

                load = PointLoad(
                    point=(cx, cy, cz),
                    force=(vec["x"], vec["y"], vec["z"]),
                    direction=None,
                    tolerance=1,
                    search_radius=None if singleton_nset else (rx, ry, rz),
                )
                load.name = f"LOAD_{target}_{dof}"
                load.region_id = sel_id
                load.vector_newtons = vec
                load.direction = axis
                load.magnitude_newtons = abs(mag)
                load_case.loads.append(load)

            elif target.isdigit():
                idx = node_id_to_idx.get(int(target))
                if idx is not None:
                    n_x, n_y, n_z = (
                        float(nodes_arr[idx, 0]),
                        float(nodes_arr[idx, 1]),
                        float(nodes_arr[idx, 2]),
                    )
                    pt_id = f"POINT_{target}"
                    load_case.selectors.setdefault(
                        pt_id,
                        SpatialSelector(
                            id=pt_id,
                            type="box_3d",
                            query={"x": n_x, "y": n_y, "z": n_z},
                        ),
                    )
                    load = PointLoad(
                        point=(n_x, n_y, n_z),
                        force=(vec["x"], vec["y"], vec["z"]),
                        direction=None,
                        tolerance=1,
                        search_radius=None,
                    )
                    load.name = f"LOAD_NODE_{target}_{dof}"
                    load.region_id = pt_id
                    load.vector_newtons = vec
                    load.direction = axis
                    load.magnitude_newtons = abs(mag)
                    load_case.loads.append(load)

        logger.info(
            f"Parsed Abaqus INP (meshio): {len(nodes_arr)} nodes, "
            f"{len(elems_arr)} {resolved_elem_type} elements, "
            f"{len(point_sets)} NSETs, "
            f"{len(load_case.boundary_conditions)} BCs, "
            f"{len(load_case.loads)} loads"
        )
        return load_case
