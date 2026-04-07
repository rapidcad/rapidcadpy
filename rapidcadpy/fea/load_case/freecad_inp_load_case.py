from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import meshio
import numpy as np

from ..boundary_conditions import FixedConstraint, PointLoad
from ..design_domain import DesignDomain
from ..spatial_selector import SpatialSelector

if TYPE_CHECKING:
    from .load_case import LoadCase

logger = logging.getLogger(__name__)

class LoadCaseFromFreeCadInp:

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
        # Local import to avoid circular dependency (load_case imports this module)
        from .load_case import LoadCase  # noqa: PLC0415

        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Abaqus file not found: {filepath}")

        # ------------------------------------------------------------------
        # 1. Parse mesh via meshio (nodes, elements, NSETs, ELSETs)
        # ------------------------------------------------------------------
        try:
            mesh = meshio.read(str(path))
        except (Exception, SystemExit) as _meshio_err:
            logger.debug(
                "meshio failed to parse %s (%s); falling back to native "
                "Abaqus parser.",
                path.name,
                _meshio_err,
            )
            from .abaqus_inp_load_case import AbaqusInpLoadCase  # noqa: PLC0415
            return AbaqusInpLoadCase.from_inp(filepath)

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
        #
        # FreeCAD may group nodes from multiple disjoint faces into a single
        # NSET (e.g. two perpendicular fixed faces → one ConstraintFixed set).
        # A single bounding-box selector would then span the entire model and
        # incorrectly match all nodes.  We therefore detect "primary" faces —
        # model faces that have NSET nodes not shared with any other face —
        # and create one tight selector per primary face when there are multiple.
        # ------------------------------------------------------------------
        selector_map: Dict[str, List[str]] = {}  # nset_name -> [selector_id, ...]

        # Six axis-aligned planes of the overall model bounding box
        _face_specs = [
            ("xmin", 0, bounds["x_min"]),
            ("xmax", 0, bounds["x_max"]),
            ("ymin", 1, bounds["y_min"]),
            ("ymax", 1, bounds["y_max"]),
            ("zmin", 2, bounds["z_min"]),
            ("zmax", 2, bounds["z_max"]),
        ]

        def _build_selector(sel_id: str, face_coords: np.ndarray) -> SpatialSelector:
            f_xs = face_coords[:, 0]
            f_ys = face_coords[:, 1]
            f_zs = face_coords[:, 2]
            query = {
                "x_min": max(float(f_xs.min()) - selector_tol, bounds["x_min"]),
                "x_max": min(float(f_xs.max()) + selector_tol, bounds["x_max"]),
                "y_min": max(float(f_ys.min()) - selector_tol, bounds["y_min"]),
                "y_max": min(float(f_ys.max()) + selector_tol, bounds["y_max"]),
                "z_min": max(float(f_zs.min()) - selector_tol, bounds["z_min"]),
                "z_max": min(float(f_zs.max()) + selector_tol, bounds["z_max"]),
                "x": float(f_xs.mean()),
                "y": float(f_ys.mean()),
                "z": float(f_zs.mean()),
                "rx": max((float(f_xs.max()) - float(f_xs.min())) / 2.0, min_radius),
                "ry": max((float(f_ys.max()) - float(f_ys.min())) / 2.0, min_radius),
                "rz": max((float(f_zs.max()) - float(f_zs.min())) / 2.0, min_radius),
            }
            return SpatialSelector(id=sel_id, type="box_3d", query=query)

        for name, idx_arr in point_sets.items():
            if idx_arr.size == 0:
                continue
            coords = nodes_arr[idx_arr]

            # Compute per-face membership masks for this NSET's nodes
            face_masks: Dict[str, np.ndarray] = {
                fname: np.abs(coords[:, axis] - pval) <= selector_tol
                for fname, axis, pval in _face_specs
            }

            # A face is "primary" when it has nodes not shared with any other face
            primary_faces = []
            for fname, mask in face_masks.items():
                others = np.zeros(len(idx_arr), dtype=bool)
                for fn2, m2 in face_masks.items():
                    if fn2 != fname:
                        others |= m2
                if (mask & ~others).any():
                    primary_faces.append(fname)

            sel_ids: List[str] = []
            if len(primary_faces) > 1:
                # NSET spans multiple disjoint faces → one tight selector per face
                for fname in primary_faces:
                    face_idx = idx_arr[face_masks[fname]]
                    sel_id = f"SELECTOR_{name}_{fname}"
                    sel = _build_selector(sel_id, nodes_arr[face_idx])
                    load_case.selectors[sel_id] = sel
                    sel_ids.append(sel_id)
            else:
                # Single face (or no clear split) → original bounding-box selector
                sel_id = f"SELECTOR_{name}"
                sel = _build_selector(sel_id, coords)
                load_case.selectors[sel_id] = sel
                sel_ids.append(sel_id)

            selector_map[name] = sel_ids

        def _find_selectors(target: str) -> List[str]:
            """Case-insensitive NSET name → list of selector_ids."""
            if target in selector_map:
                return selector_map[target]
            tl = target.lower()
            for k, v in selector_map.items():
                if k.lower() == tl:
                    return v
            return []

        def _find_selector(target: str) -> Optional[str]:
            """Return the first selector_id for *target*, or None."""
            ids = _find_selectors(target)
            return ids[0] if ids else None

        # ------------------------------------------------------------------
        # 5. Parse *BOUNDARY → FixedConstraints
        #
        # FreeCAD / CalculiX commonly writes one line per DOF:
        #   ConstraintFixed, 1
        #   ConstraintFixed, 2
        #   ConstraintFixed, 3
        # We aggregate all DOFs per target into a single FixedConstraint so
        # that the result matches the physical intent.
        # ------------------------------------------------------------------
        # Collect locked DOFs per NSET / node-ID target before creating BCs.
        _bc_dofs_by_target: Dict[str, set] = {}
        for line in raw_bc_lines:
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 2:
                continue
            nset_name = parts[0]
            try:
                if len(parts) == 2:
                    # Single-DOF form: "NSET, dof"
                    first_dof = last_dof = int(parts[1])
                else:
                    # Range form: "NSET, first_dof, last_dof[, magnitude]"
                    first_dof, last_dof = int(parts[1]), int(parts[2])
            except ValueError:
                continue
            _bc_dofs_by_target.setdefault(nset_name, set()).update(
                range(first_dof, last_dof + 1)
            )

        for nset_name, locked_dofs in _bc_dofs_by_target.items():
            dof_lock = tuple(i in locked_dofs for i in (1, 2, 3))
            sel_ids = _find_selectors(nset_name)
            if sel_ids:
                for sel_id in sel_ids:
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
        #
        # Two styles exist in FreeCAD-generated files:
        #   a) NSET-based: "SetName, dof, magnitude"  → one PointLoad per line
        #   b) Node-ID-based: "nodeID, dof, magnitude" → FreeCAD distributes a
        #      total force across many surface nodes.  We aggregate all entries
        #      that share the same DOF into a single resultant PointLoad placed
        #      at the centroid of the contributing nodes.
        # ------------------------------------------------------------------
        # Accumulator for node-based CLOAD: dof → (total_mag, [node_indices])
        _node_cload: Dict[int, List] = {}  # dof -> [total_mag, [node_idx,...]]

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
                # NSET-based load → one PointLoad per line (existing behaviour)
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
                # Node-ID-based load → accumulate for later aggregation
                idx = node_id_to_idx.get(int(target))
                if idx is not None:
                    if dof not in _node_cload:
                        _node_cload[dof] = [0.0, []]
                    _node_cload[dof][0] += mag
                    _node_cload[dof][1].append(idx)

        # Emit one aggregated PointLoad per DOF for node-based CLOADs
        for dof in sorted(_node_cload.keys()):
            total_mag, node_indices = _node_cload[dof]
            coords = nodes_arr[node_indices]
            cx, cy, cz = (
                float(coords[:, 0].mean()),
                float(coords[:, 1].mean()),
                float(coords[:, 2].mean()),
            )
            rx = max(float(coords[:, 0].max() - coords[:, 0].min()) / 2, min_radius)
            ry = max(float(coords[:, 1].max() - coords[:, 1].min()) / 2, min_radius)
            rz = max(float(coords[:, 2].max() - coords[:, 2].min()) / 2, min_radius)

            vec_agg: Dict[str, float] = {"x": 0.0, "y": 0.0, "z": 0.0}
            axis_agg: Optional[str] = None
            if dof == 1:
                vec_agg["x"] = total_mag
                axis_agg = "x" if total_mag >= 0 else "-x"
            elif dof == 2:
                vec_agg["y"] = total_mag
                axis_agg = "y" if total_mag >= 0 else "-y"
            elif dof == 3:
                vec_agg["z"] = total_mag
                axis_agg = "z" if total_mag >= 0 else "-z"

            agg_pt_id = f"LOAD_REGION_DOF{dof}"
            load_case.selectors.setdefault(
                agg_pt_id,
                SpatialSelector(
                    id=agg_pt_id,
                    type="box_3d",
                    query={
                        "x": cx, "y": cy, "z": cz,
                        "x_min": float(coords[:, 0].min()),
                        "x_max": float(coords[:, 0].max()),
                        "y_min": float(coords[:, 1].min()),
                        "y_max": float(coords[:, 1].max()),
                        "z_min": float(coords[:, 2].min()),
                        "z_max": float(coords[:, 2].max()),
                        "rx": rx, "ry": ry, "rz": rz,
                    },
                ),
            )
            load = PointLoad(
                point=(cx, cy, cz),
                force=(vec_agg["x"], vec_agg["y"], vec_agg["z"]),
                direction=None,
                tolerance=1,
                search_radius=(rx, ry, rz),
            )
            load.name = f"LOAD_NODES_DOF{dof}"
            load.region_id = agg_pt_id
            load.vector_newtons = vec_agg
            load.direction = axis_agg
            load.magnitude_newtons = abs(total_mag)
            load_case.loads.append(load)

        logger.info(
            f"Parsed Abaqus INP (meshio): {len(nodes_arr)} nodes, "
            f"{len(elems_arr)} {resolved_elem_type} elements, "
            f"{len(point_sets)} NSETs, "
            f"{len(load_case.boundary_conditions)} BCs, "
            f"{len(load_case.loads)} loads"
        )
        return load_case
