from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import Any, Dict, List, Optional
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

    @staticmethod
    def from_inp(filepath: str) -> "LoadCase":
        """
        Parse an Abaqus .inp file and convert it to a LoadCase object.

        This function performs a "reverse engineering" of the explicit FEA model:
        1. Calculates the bounding box of all nodes to define the Design Domain.
        2. Converts Node Sets (*NSET) into box-based Spatial Selectors.
        3. Maps *BOUNDARY constraints to fixed constraints.
        4. Maps *CLOAD forces to concentrated loads.

        Args:
            filepath: Path to the .inp file

        Returns:
            LoadCase object
        """
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Abaqus file not found: {filepath}")

        with open(path, "r") as f:
            lines = f.readlines()

        nodes: Dict[int, Tuple[float, float, float]] = {}
        node_sets: Dict[str, List[int]] = {}
        element_sets: Dict[str, List[int]] = {}
        elements: Dict[int, List[int]] = {}  # elem_id -> [node_ids]

        # Regex patterns
        # Matches: 1, 0.0, 0.0, 0.0
        p_node = re.compile(r"^\s*(\d+),\s*([\d\.-]+),\s*([\d\.-]+)(?:,\s*([\d\.-]+))?")
        # Matches: *KEYWORD, PARAM=VALUE
        p_keyword = re.compile(r"^\*([\w\s]+)(?:,\s*(.*))?")

        current_section = None
        current_nset_name = None
        current_elset_name = None
        current_nset_generate = False
        current_elset_generate = False
        current_element_type = None

        problem_id = path.stem.upper()
        load_case = LoadCase(
            problem_id=problem_id, description=f"Imported from {path.name}"
        )
        # Keep selector mapping available for existing region-based consumers.
        load_case.meta = load_case.meta or {}
        load_case.selectors = {}
        load_case.meta["selectors"] = load_case.selectors

        # -------------------------------------------------------------
        # PASS 1: Parse Nodes and Node Sets
        # -------------------------------------------------------------
        for line in lines:
            line = line.strip()
            if not line or line.startswith("**"):
                continue

            if line.startswith("*"):
                match = p_keyword.match(line)
                if not match:
                    continue

                keyword = match.group(1).upper().strip()
                params = match.group(2) or ""

                if keyword == "NODE":
                    current_section = "NODE"
                    current_nset_name = None
                    current_elset_name = None
                    current_nset_generate = False
                    current_elset_generate = False
                elif keyword == "NSET":
                    current_section = "NSET"
                    current_elset_name = None
                    current_nset_generate = "GENERATE" in params.upper()
                    # Extract NSET name: NSET=FIXED_BACK -> FIXED_BACK
                    nset_match = re.search(r"NSET=([\w_\-\.]+)", params, re.IGNORECASE)
                    if nset_match:
                        current_nset_name = nset_match.group(1)
                        if current_nset_name not in node_sets:
                            node_sets[current_nset_name] = []
                    else:
                        current_nset_name = None
                elif keyword == "ELSET":
                    current_section = "ELSET"
                    current_nset_name = None
                    current_elset_generate = "GENERATE" in params.upper()
                    elset_match = re.search(r"ELSET=([\w_\-\.]+)", params, re.IGNORECASE)
                    if elset_match:
                        current_elset_name = elset_match.group(1)
                        if current_elset_name not in element_sets:
                            element_sets[current_elset_name] = []
                    else:
                        current_elset_name = None
                elif keyword == "ELEMENT":
                    current_section = "ELEMENT"
                    current_nset_name = None
                    current_elset_name = None
                    current_nset_generate = False
                    current_elset_generate = False
                    # Extract element type: TYPE=CPS4 or TYPE=C3D4
                    type_match = re.search(r"TYPE=([\w\d]+)", params, re.IGNORECASE)
                    if type_match:
                        current_element_type = type_match.group(1).upper()
                    else:
                        current_element_type = None
                else:
                    current_section = None  # Skip materials, steps for now in this pass
                    current_nset_name = None
                    current_elset_name = None
                    current_nset_generate = False
                    current_elset_generate = False
                continue

            # Data parsing based on section
            if current_section == "NODE":
                m = p_node.match(line)
                if m:
                    nid = int(m.group(1))
                    x = float(m.group(2))
                    y = float(m.group(3))
                    z = float(m.group(4)) if m.group(4) else 0.0
                    nodes[nid] = (x, y, z)

            elif current_section == "NSET" and current_nset_name:
                # Parse comma separated node IDs.
                # Note: Abaqus can split lists across 'generate' statements but valid lists are just comma sep
                if current_nset_generate:
                    # Handle generation: start, end, step
                    parts = [p.strip() for p in line.split(",") if p.strip()]
                    if len(parts) >= 2:
                        start = int(parts[0])
                        end = int(parts[1])
                        step = int(parts[2]) if len(parts) > 2 else 1
                        node_sets[current_nset_name].extend(range(start, end + 1, step))
                else:
                    # Standard list
                    try:
                        ids = [int(x) for x in line.split(",") if x.strip()]
                        node_sets[current_nset_name].extend(ids)
                    except ValueError:
                        pass  # Ignore lines that might not match simple int lists

            elif current_section == "ELSET" and current_elset_name:
                if current_elset_generate:
                    parts = [p.strip() for p in line.split(",") if p.strip()]
                    if len(parts) >= 2:
                        try:
                            start = int(parts[0])
                            end = int(parts[1])
                            step = int(parts[2]) if len(parts) > 2 else 1
                            element_sets[current_elset_name].extend(
                                range(start, end + 1, step)
                            )
                        except ValueError:
                            pass
                else:
                    try:
                        ids = [int(x) for x in line.split(",") if x.strip()]
                        element_sets[current_elset_name].extend(ids)
                    except ValueError:
                        pass

            elif current_section == "ELEMENT":
                # Parse element: elem_id, node1, node2, node3, ...
                parts = [p.strip() for p in line.split(",") if p.strip()]
                if len(parts) >= 2:
                    try:
                        elem_id = int(parts[0])
                        node_ids = [int(parts[i]) for i in range(1, len(parts))]
                        elements[elem_id] = node_ids
                    except ValueError:
                        pass

        if not nodes:
            raise ValueError("No nodes found in .inp file")

        # -------------------------------------------------------------
        # 2. Derive Design Domain from all Nodes
        # -------------------------------------------------------------
        all_coords = list(nodes.values())
        xs, ys, zs = zip(*all_coords)

        # Create valid bounds
        bounds = {
            "x_min": min(xs),
            "x_max": max(xs),
            "y_min": min(ys),
            "y_max": max(ys),
            "z_min": min(zs),
            "z_max": max(zs),
        }

        dx = bounds["x_max"] - bounds["x_min"]
        dy = bounds["y_max"] - bounds["y_min"]
        dz = bounds["z_max"] - bounds["z_min"]
        positive_dims = [d for d in (dx, dy, dz) if d > 0]
        min_dim = min(positive_dims) if positive_dims else 1.0

        # Scale-aware tolerances: avoid hard-coded 0.5mm which is enormous for sub-mm models.
        selector_tol = max(min_dim * 0.01, 1e-4)
        min_radius = max(min_dim * 0.02, 5e-5)

        load_case.domain = DesignDomain(
            shape_type="box", bounds=bounds, units="mm"  # Assumption
        )
        load_case.bounds = bounds  # Populate legacy bounds too
        load_case.meta = load_case.meta or {}
        load_case.meta["node_sets_count"] = len(node_sets)
        load_case.meta["element_sets_count"] = len(element_sets)
        load_case.meta["abaqus_element_type"] = current_element_type
        load_case.meta["node_sets"] = sorted(node_sets.keys())
        load_case.meta["element_sets"] = sorted(element_sets.keys())

        # -------------------------------------------------------------
        # 3. Create Spatial Selectors from Node Sets
        # -------------------------------------------------------------
        selector_map = {}  # nset_name -> selector_id

        for name, nids in node_sets.items():
            if not nids:
                continue

            # Get coords for this set
            set_coords = [nodes[nid] for nid in nids if nid in nodes]
            if not set_coords:
                continue

            s_xs, s_ys, s_zs = zip(*set_coords)

            # Centroid and anisotropic search radii for robust remapping
            cx = float(np.mean(s_xs))
            cy = float(np.mean(s_ys))
            cz = float(np.mean(s_zs))

            # Use half extents as anisotropic radii and enforce a small scale-aware minimum
            rx = max((max(s_xs) - min(s_xs)) / 2.0, min_radius)
            ry = max((max(s_ys) - min(s_ys)) / 2.0, min_radius)
            rz = max((max(s_zs) - min(s_zs)) / 2.0, min_radius)

            # Tolerance expansion is clamped to global design bounds to prevent oversized regions.
            query = {
                "x_min": max(min(s_xs) - selector_tol, bounds["x_min"]),
                "x_max": min(max(s_xs) + selector_tol, bounds["x_max"]),
                "y_min": max(min(s_ys) - selector_tol, bounds["y_min"]),
                "y_max": min(max(s_ys) + selector_tol, bounds["y_max"]),
                "z_min": max(min(s_zs) - selector_tol, bounds["z_min"]),
                "z_max": min(max(s_zs) + selector_tol, bounds["z_max"]),
                # Extra metadata used for anisotropic point-load remapping
                "x": cx,
                "y": cy,
                "z": cz,
                "rx": rx,
                "ry": ry,
                "rz": rz,
            }

            sel_id = f"SELECTOR_{name}"
            selector = SpatialSelector(id=sel_id, type="box_3d", query=query)
            load_case.selectors[sel_id] = selector
            selector_map[name] = sel_id

        # -------------------------------------------------------------
        # 4. PASS 2: Parse BCs and Loads requiring the selectors
        # -------------------------------------------------------------
        current_section = None

        for line in lines:
            line = line.strip()
            if not line or line.startswith("**"):
                continue
            if line.startswith("*"):
                match = p_keyword.match(line)
                if not match:
                    continue
                keyword = match.group(1).upper().strip()

                if keyword in ["BOUNDARY", "CLOAD"]:
                    current_section = keyword
                else:
                    # Reset if it's a new keyword we don't care about here
                    current_section = None
                continue

            if current_section == "BOUNDARY":
                # Format: NodeSet, FirstDOF, LastDOF, Magnitude
                # e.g. FIXED_BACK, 1, 2, 0.0
                parts = [p.strip() for p in line.split(",")]
                if len(parts) < 3:
                    continue

                nset_name = parts[0]
                first_dof = int(parts[1])
                last_dof = int(parts[2])
                # mag = float(parts[3]) if len(parts) > 3 else 0.0 # Magnitude usually 0 for fixed

                # Map DOFs to boolean lock
                # 1=x, 2=y, 3=z.
                # If 1,3 -> Locks 1, 2, 3.
                # If 1,1 -> Locks 1.

                dof_lock = {"x": False, "y": False, "z": False}
                if first_dof <= 1 <= last_dof:
                    dof_lock["x"] = True
                if first_dof <= 2 <= last_dof:
                    dof_lock["y"] = True
                if first_dof <= 3 <= last_dof:
                    dof_lock["z"] = True

                # If strict NSET name match failed, try case-insensitive or search
                target_sel_id = selector_map.get(nset_name)
                if not target_sel_id:
                    # Try finding it
                    for k in selector_map:
                        if k.lower() == nset_name.lower():
                            target_sel_id = selector_map[k]
                            break

                if target_sel_id:
                    selector = load_case.selectors.get(target_sel_id)
                    location = selector.query if selector else target_sel_id
                    bc = FixedConstraint(
                        location=location,
                        dofs=(dof_lock["x"], dof_lock["y"], dof_lock["z"]),
                        tolerance=1,
                    )
                    load_case.boundary_conditions.append(bc)
                else:
                    # It might be a single node ID?
                    if nset_name.isdigit():
                        nid = int(nset_name)
                        if nid in nodes:
                            # Create point selector for this node
                            pt_id = f"PT_{nid}"
                            nx, ny, nz = nodes[nid]
                            load_case.selectors[pt_id] = SpatialSelector(
                                id=pt_id, type="point", query={"x": nx, "y": ny, "z": nz}
                            )
                            bc = FixedConstraint(
                                location=(nx, ny, nz),
                                dofs=(dof_lock["x"], dof_lock["y"], dof_lock["z"]),
                                tolerance=1,
                            )
                            load_case.boundary_conditions.append(bc)

            elif current_section == "CLOAD":
                # Format: NodeID/NodeSet, DOF, Magnitude
                # e.g. 19, 2, -500.
                parts = [p.strip() for p in line.split(",")]
                if len(parts) < 3:
                    continue

                target = parts[0]  # Could be int ID or string NSET
                dof = int(parts[1])
                mag = float(parts[2])

                # Vector mapping
                vec = {"x": 0.0, "y": 0.0, "z": 0.0}
                axis = None
                if dof == 1:
                    vec["x"] = mag
                    axis = "x" if mag > 0 else "-x"
                elif dof == 2:
                    vec["y"] = mag
                    axis = "y" if mag > 0 else "-y"
                elif dof == 3:
                    vec["z"] = mag
                    axis = "z" if mag > 0 else "-z"

                # Check if target is a NSET
                target_sel_id = selector_map.get(target)
                if not target_sel_id:
                    for k in selector_map:
                        if k.lower() == target.lower():
                            target_sel_id = selector_map[k]
                            break

                if target_sel_id:
                    selector = load_case.selectors.get(target_sel_id)
                    query = selector.query if selector else {}

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
                        search_radius=(rx, ry, rz),
                    )
                    # Compatibility attrs for downstream utilities expecting parser-style load fields
                    load.name = f"LOAD_{target}_{dof}"
                    load.region_id = target_sel_id
                    load.vector_newtons = vec
                    load.direction = axis
                    load.magnitude_newtons = abs(mag)
                    load_case.loads.append(load)

                # Check if target is a NODE ID
                elif target.isdigit():
                    nid = int(target)
                    if nid in nodes:
                        n_x, n_y, n_z = nodes[nid]
                        pt_id = f"POINT_{nid}"
                        # Add selector only if not exists
                        if pt_id not in load_case.selectors:
                            load_case.selectors[pt_id] = SpatialSelector(
                                id=pt_id,
                                type="box_3d",
                                query={"x": n_x, "y": n_y, "z": n_z},  # Treats as point
                            )

                        load = PointLoad(
                            point=(n_x, n_y, n_z),
                            force=(vec["x"], vec["y"], vec["z"]),
                            direction=None,
                            tolerance=1,
                            search_radius=(min_radius, min_radius, min_radius),
                        )
                        load.name = f"LOAD_NODE_{nid}_{dof}"
                        load.region_id = pt_id
                        load.vector_newtons = vec
                        load.direction = axis
                        load.magnitude_newtons = abs(mag)
                        load_case.loads.append(load)

        # -------------------------------------------------------------
        # 5. Convert mesh data to numpy arrays for LoadCase
        # -------------------------------------------------------------
        if nodes and elements:
            # Create node array (sorted by node ID)
            sorted_node_ids = sorted(nodes.keys())
            node_id_map = {nid: idx for idx, nid in enumerate(sorted_node_ids)}

            node_coords = np.array(
                [nodes[nid] for nid in sorted_node_ids], dtype=np.float32
            )

            # Create element array (remap node IDs to indices)
            elem_list = []
            for eid in sorted(elements.keys()):
                node_ids = elements[eid]
                # Remap to 0-based indices
                indices = [node_id_map[nid] for nid in node_ids if nid in node_id_map]
                if len(indices) == len(node_ids):  # Only add if all nodes found
                    elem_list.append(indices)

            if elem_list:
                # Determine element type from first element
                nodes_per_elem = len(elem_list[0])
                elem_array = np.array(elem_list, dtype=np.int32)

                # Map Abaqus element types to standard names
                elem_type_map = {
                    "C3D4": "tet4",
                    "C3D10": "tet10",
                    "CPS4": "quad4",
                    "CPS3": "tri3",
                    "CPE4": "quad4",
                    "CPE3": "tri3",
                }

                mesh_elem_type = elem_type_map.get(
                    current_element_type or "", f"generic_{nodes_per_elem}node"
                )

                load_case.mesh_nodes = node_coords
                load_case.mesh_elements = elem_array
                load_case.mesh_element_type = mesh_elem_type

                logger.info(
                    f"Extracted mesh: {len(node_coords)} nodes, {len(elem_array)} {mesh_elem_type} elements"
                )

        logger.info(
            f"Parsed Abaqus INP: {len(nodes)} nodes, {len(node_sets)} node sets, {len(element_sets)} element sets, {len(load_case.boundary_conditions)} BCs, {len(load_case.loads)} Loads"
        )
        return load_case
