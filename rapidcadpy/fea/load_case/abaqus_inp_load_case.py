"""
Pure-Python Abaqus-native .inp parser.

Used as a fallback when meshio cannot handle a file (e.g. files that contain
substructure element types such as Z9994, external *NODE INPUT= references,
or other Abaqus-only constructs).

Supported Abaqus keywords
--------------------------
*HEADING           – skipped (informational)
*NODE              – inline coordinates parsed; ``INPUT=file`` form skipped
*NCOPY             – skipped
*ELEMENT           – C3D20R / C3D8R / C3D4 / C3D10 inline elements parsed;
                     elements that carry a ``FILE=`` parameter (substructure
                     types such as Z9994) are skipped entirely
*SUBSTRUCTURE PROPERTY – skipped
*NSET              – GENERATE (start,stop,step) and direct/UNSORTED lists
*ELSET             – GENERATE and direct lists
*BOUNDARY          – NSET,dof  or  NSET,first,last[,mag]  and node-ID forms
*CLOAD             – node_id,dof,mag  and  NSET,dof,mag
*ELCOPY            – skipped
*TRANSFORM         – skipped
*EQUATION          – skipped
*SURFACE           – skipped
*CONTACT *         – skipped
*FRICTION          – skipped
*MATERIAL / *ELASTIC – skipped
*SOLID SECTION     – skipped
*PRE-TENSION …     – skipped
*STEP / *END STEP  – skipped
*STATIC etc.       – skipped
*MPC               – skipped
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Set, Tuple

import numpy as np

from ..boundary_conditions import FixedConstraint, PointLoad
from ..design_domain import DesignDomain
from ..spatial_selector import SpatialSelector

if TYPE_CHECKING:
    from .load_case import LoadCase

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Abaqus element type → RapidCAD element type string, and nodes per element
_ELEM_TYPE_MAP: Dict[str, Tuple[str, int]] = {
    "C3D20R": ("hex20", 20),
    "C3D20":  ("hex20", 20),
    "C3D8R":  ("hex8",  8),
    "C3D8":   ("hex8",  8),
    "C3D10":  ("tet10", 10),
    "C3D4":   ("tet4",  4),
    "C3D6":   ("wed6",  6),
    "C3D15":  ("wed15", 15),
}

# Keywords whose data lines are consumed but whose content is ignored
_SKIP_KEYWORDS: Set[str] = {
    "HEADING",
    "NCOPY",
    "SUBSTRUCTURE PROPERTY",
    "ELCOPY",
    "TRANSFORM",
    "EQUATION",
    "SURFACE",
    "CONTACT PAIR",
    "SURFACE INTERACTION",
    "FRICTION",
    "SURFACE BEHAVIOR",
    "PRE-TENSION SECTION",
    "RESTART",
    "PREPRINT",
    "STEP",
    "STATIC",
    "NODE PRINT",
    "NODE FILE",
    "CONTACT PRINT",
    "CONTACT FILE",
    "PRINT",
    "END STEP",
    "MPC",
    "MATERIAL",
    "ELASTIC",
    "SOLID SECTION",
}


def _parse_keyword_line(line: str) -> Tuple[str, Dict[str, str]]:
    """
    Split an Abaqus keyword line such as::

        *ELEMENT, TYPE=C3D20R, ELSET=PID1

    into ``("ELEMENT", {"TYPE": "C3D20R", "ELSET": "PID1"})``.

    The keyword name (before the first comma or end of string) is returned
    upper-cased and stripped.  Parameter values are kept as-is.
    """
    # Strip the leading '*'
    body = line.lstrip("*").strip()
    parts = [p.strip() for p in body.split(",")]
    keyword = parts[0].upper()
    params: Dict[str, str] = {}
    for p in parts[1:]:
        if "=" in p:
            k, v = p.split("=", 1)
            params[k.strip().upper()] = v.strip()
        else:
            # Flag-only parameter (e.g. GENERATE, UNSORTED)
            params[p.strip().upper()] = ""
    return keyword, params


def _expand_generate(start: int, stop: int, step: int) -> List[int]:
    """Return the list produced by Abaqus GENERATE: start,stop,step."""
    if step == 0:
        return []
    if step > 0:
        return list(range(start, stop + 1, step))
    return list(range(start, stop - 1, step))  # descending – rare


# ---------------------------------------------------------------------------
# Main parser class
# ---------------------------------------------------------------------------

class AbaqusInpLoadCase:
    """
    Parse an Abaqus-format ``.inp`` file into a :class:`~.load_case.LoadCase`
    without relying on *meshio*.

    This class mirrors the public interface of
    :class:`~.freecad_inp_load_case.LoadCaseFromFreeCadInp` so that either
    can be used as the back-end of :py:meth:`LoadCase.from_inp`.
    """

    @staticmethod
    def from_inp(filepath: str) -> "LoadCase":
        """
        Parse *filepath* and return a populated :class:`~.load_case.LoadCase`.

        Parameters
        ----------
        filepath:
            Absolute or relative path to the ``.inp`` file.

        Returns
        -------
        LoadCase
        """
        # Deferred import to break the circular dependency
        from .load_case import LoadCase  # noqa: PLC0415

        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Abaqus .inp file not found: {filepath}")

        with open(path, "r", errors="replace") as fh:
            raw_lines = fh.readlines()

        # ------------------------------------------------------------------
        # Single-pass line parser
        # ------------------------------------------------------------------
        # section tracking
        section: Optional[str] = None   # e.g. "NODE", "ELEMENT", "NSET", …
        section_params: Dict[str, str] = {}

        # Accumulated data
        nodes: Dict[int, Tuple[float, float, float]] = {}   # id1-based → (x,y,z)
        elements: Dict[str, List[Tuple[int, ...]]] = {}     # elem_type → list of connectivity tuples (1-based node IDs)
        nsets: Dict[str, List[int]] = {}    # name → list of 1-based node IDs
        elsets: Dict[str, List[int]] = {}   # name → list of 1-based element IDs

        raw_bc_lines: List[str] = []
        raw_cload_lines: List[str] = []

        # For multi-line element connectivity we accumulate partial token lists
        _pending_elem_tokens: List[str] = []
        _current_elem_type: Optional[str] = None
        _current_elem_nodes_per: int = 0    # expected nodes-per-element
        _skip_current_element_block = False  # True when FILE= substructure

        def _flush_pending_elem():
            """Commit a completed element connectivity record."""
            if _pending_elem_tokens and _current_elem_type:
                nums = [int(t) for t in _pending_elem_tokens]
                if len(nums) >= 1 + _current_elem_nodes_per:
                    connectivity = tuple(nums[1 : 1 + _current_elem_nodes_per])
                    elements.setdefault(_current_elem_type, []).append(connectivity)
            _pending_elem_tokens.clear()

        for raw in raw_lines:
            line = raw.rstrip("\n").rstrip("\r")
            stripped = line.strip()

            # Skip blank lines and full-line comments
            if not stripped or stripped.startswith("**"):
                continue

            # ----------------------------------------------------------
            # Keyword line
            # ----------------------------------------------------------
            if stripped.startswith("*") and not stripped.startswith("**"):
                # Flush any pending multi-line element before changing section
                if section == "ELEMENT" and not _skip_current_element_block:
                    if _pending_elem_tokens:
                        _flush_pending_elem()

                keyword, params = _parse_keyword_line(stripped)

                # Determine new section
                if keyword in _SKIP_KEYWORDS:
                    section = "SKIP"
                    section_params = {}

                elif keyword == "NODE":
                    # Skip when nodes come from an external file
                    if "INPUT" in params:
                        section = "SKIP"
                    else:
                        section = "NODE"
                    section_params = params

                elif keyword == "ELEMENT":
                    raw_etype = params.get("TYPE", "").upper()
                    # Skip substructure / external element blocks
                    if "FILE" in params or raw_etype not in _ELEM_TYPE_MAP:
                        section = "SKIP"
                        _skip_current_element_block = True
                    else:
                        section = "ELEMENT"
                        _current_elem_type, _current_elem_nodes_per = _ELEM_TYPE_MAP[raw_etype]
                        _skip_current_element_block = False
                        _pending_elem_tokens.clear()
                    section_params = params

                elif keyword == "NSET":
                    set_name = params.get("NSET", params.get("NAME", "")).strip()
                    section = "NSET"
                    section_params = {"NAME": set_name, **params}
                    if set_name and set_name not in nsets:
                        nsets[set_name] = []

                elif keyword == "ELSET":
                    set_name = params.get("ELSET", params.get("NAME", "")).strip()
                    section = "ELSET"
                    section_params = {"NAME": set_name, **params}
                    if set_name and set_name not in elsets:
                        elsets[set_name] = []

                elif keyword == "BOUNDARY":
                    section = "BOUNDARY"
                    section_params = params

                elif keyword == "CLOAD":
                    section = "CLOAD"
                    section_params = params

                else:
                    # Unknown keyword — skip its data lines
                    section = "SKIP"
                    section_params = {}

                continue

            # ----------------------------------------------------------
            # Data line — dispatch by current section
            # ----------------------------------------------------------
            if section == "SKIP":
                continue

            elif section == "NODE":
                # Format: node_id, x, y, z
                try:
                    parts = [p.strip() for p in stripped.split(",")]
                    node_id = int(parts[0])
                    x = float(parts[1]) if len(parts) > 1 else 0.0
                    y = float(parts[2]) if len(parts) > 2 else 0.0
                    z = float(parts[3]) if len(parts) > 3 else 0.0
                    nodes[node_id] = (x, y, z)
                except (ValueError, IndexError):
                    pass

            elif section == "ELEMENT":
                # Elements may span multiple lines.  Token stream:
                # elem_id, n1, n2, …  (may be split across continuation lines)
                tokens = [t.strip() for t in stripped.split(",") if t.strip()]
                # A line that starts with a fresh integer AND we already have
                # a full accumulated record → flush first.
                if _pending_elem_tokens:
                    # Check if this line starts a new element (first token is
                    # numeric and we already have enough tokens buffered)
                    try:
                        _ = int(tokens[0])
                        total_needed = 1 + _current_elem_nodes_per
                        if len(_pending_elem_tokens) >= total_needed:
                            _flush_pending_elem()
                    except (ValueError, IndexError):
                        pass
                _pending_elem_tokens.extend(tokens)
                # Auto-flush when we have exactly enough tokens
                total_needed = 1 + _current_elem_nodes_per
                if len(_pending_elem_tokens) >= total_needed:
                    _flush_pending_elem()

            elif section == "NSET":
                set_name = section_params.get("NAME", "")
                if not set_name:
                    continue
                if "GENERATE" in section_params:
                    parts = [p.strip() for p in stripped.split(",")]
                    try:
                        start, stop = int(parts[0]), int(parts[1])
                        step = int(parts[2]) if len(parts) > 2 and parts[2] else 1
                        nsets[set_name].extend(_expand_generate(start, stop, step))
                    except (ValueError, IndexError):
                        pass
                else:
                    # Direct list (may have trailing comma)
                    for tok in stripped.split(","):
                        tok = tok.strip()
                        if tok:
                            try:
                                nsets[set_name].append(int(tok))
                            except ValueError:
                                pass

            elif section == "ELSET":
                set_name = section_params.get("NAME", "")
                if not set_name:
                    continue
                if "GENERATE" in section_params:
                    parts = [p.strip() for p in stripped.split(",")]
                    try:
                        start, stop = int(parts[0]), int(parts[1])
                        step = int(parts[2]) if len(parts) > 2 and parts[2] else 1
                        elsets[set_name].extend(_expand_generate(start, stop, step))
                    except (ValueError, IndexError):
                        pass
                else:
                    for tok in stripped.split(","):
                        tok = tok.strip()
                        if tok:
                            try:
                                elsets[set_name].append(int(tok))
                            except ValueError:
                                pass

            elif section == "BOUNDARY":
                raw_bc_lines.append(stripped)

            elif section == "CLOAD":
                raw_cload_lines.append(stripped)

        # Flush any remaining pending element
        if section == "ELEMENT" and not _skip_current_element_block and _pending_elem_tokens:
            _flush_pending_elem()

        # ------------------------------------------------------------------
        # Build numpy arrays
        # ------------------------------------------------------------------
        if not nodes:
            raise ValueError(
                f"No inline nodes found in {filepath}. "
                "The file may rely entirely on external INPUT= node files."
            )

        # Sort nodes by ID and build 0-based index map
        sorted_ids = sorted(nodes.keys())
        node_id_to_idx: Dict[int, int] = {nid: i for i, nid in enumerate(sorted_ids)}

        nodes_arr = np.array(
            [(nodes[nid][0], nodes[nid][1], nodes[nid][2]) for nid in sorted_ids],
            dtype=np.float64,
        )

        # Choose element block: prefer 3D, then by count
        _dim_rank = {
            "hex20": 3, "hex8": 3,
            "tet10": 3, "tet4":  3,
            "wed6":  3, "wed15": 3,
            "quad4": 2, "quad8": 2,
            "tri3":  2, "tri6":  2,
        }
        best_etype: Optional[str] = None
        best_rank = -1
        best_count = 0
        for etype, conns in elements.items():
            rank = _dim_rank.get(etype, 1)
            cnt = len(conns)
            if rank > best_rank or (rank == best_rank and cnt > best_count):
                best_etype = etype
                best_rank = rank
                best_count = cnt

        if best_etype is not None:
            raw_conns = elements[best_etype]
            # Map 1-based Abaqus node IDs → 0-based indices
            # Nodes not present in the inline node table are mapped to 0
            # (silently) so the array shape remains consistent.
            elems_arr = np.array(
                [
                    [node_id_to_idx.get(nid, 0) for nid in conn]
                    for conn in raw_conns
                ],
                dtype=np.int64,
            )
            resolved_elem_type = best_etype
        else:
            elems_arr = np.empty((0, 4), dtype=np.int64)
            resolved_elem_type = "tet4"

        # ------------------------------------------------------------------
        # Bounds
        # ------------------------------------------------------------------
        xs, ys, zs = nodes_arr[:, 0], nodes_arr[:, 1], nodes_arr[:, 2]
        bounds = {
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

        # ------------------------------------------------------------------
        # Build LoadCase skeleton
        # ------------------------------------------------------------------
        problem_id = path.stem.upper()
        load_case = LoadCase(
            problem_id=problem_id, description=f"Imported from {path.name}"
        )
        load_case.selectors: Dict[str, SpatialSelector] = {}
        load_case.meta = {
            "node_sets_count": len(nsets),
            "element_sets_count": len(elsets),
            "node_sets": sorted(nsets.keys()),
            "element_sets": sorted(elsets.keys()),
            "selectors": load_case.selectors,
        }
        load_case.domain = DesignDomain(shape_type="box", bounds=bounds, units="mm")
        load_case.bounds = bounds
        load_case.mesh_nodes = nodes_arr.astype(np.float32)
        load_case.mesh_elements = elems_arr.astype(np.int32)
        load_case.mesh_element_type = resolved_elem_type

        # ------------------------------------------------------------------
        # Build spatial selectors from NSETs
        #
        # Only NSETs whose members are present in the inline node table can
        # be used.  For Abaqus files that reference external node files the
        # node_id_to_idx map will be small (or even empty), so many NSETs
        # will yield empty coordinate arrays and will be skipped gracefully.
        # ------------------------------------------------------------------
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

        selector_map: Dict[str, List[str]] = {}  # nset_name → [sel_id, …]

        # Build a 0-based index array for each NSET (only known nodes)
        point_sets: Dict[str, np.ndarray] = {}
        for name, id_list in nsets.items():
            indices = [node_id_to_idx[nid] for nid in id_list if nid in node_id_to_idx]
            if indices:
                point_sets[name] = np.array(indices, dtype=np.int64)

        for name, idx_arr in point_sets.items():
            if idx_arr.size == 0:
                continue
            coords = nodes_arr[idx_arr]

            face_masks: Dict[str, np.ndarray] = {
                fname: np.abs(coords[:, axis] - pval) <= selector_tol
                for fname, axis, pval in _face_specs
            }

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
                for fname in primary_faces:
                    face_idx = idx_arr[face_masks[fname]]
                    sel_id = f"SELECTOR_{name}_{fname}"
                    sel = _build_selector(sel_id, nodes_arr[face_idx])
                    load_case.selectors[sel_id] = sel
                    sel_ids.append(sel_id)
            else:
                sel_id = f"SELECTOR_{name}"
                sel = _build_selector(sel_id, coords)
                load_case.selectors[sel_id] = sel
                sel_ids.append(sel_id)

            selector_map[name] = sel_ids

        def _find_selectors(target: str) -> List[str]:
            if target in selector_map:
                return selector_map[target]
            tl = target.lower()
            for k, v in selector_map.items():
                if k.lower() == tl:
                    return v
            return []

        def _find_selector(target: str) -> Optional[str]:
            ids = _find_selectors(target)
            return ids[0] if ids else None

        # ------------------------------------------------------------------
        # Parse *BOUNDARY → FixedConstraints
        # ------------------------------------------------------------------
        _bc_dofs_by_target: Dict[str, set] = {}
        for line in raw_bc_lines:
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 2:
                continue
            nset_name = parts[0]
            try:
                if len(parts) == 2:
                    first_dof = last_dof = int(parts[1])
                else:
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
        # Parse *CLOAD → PointLoads
        # ------------------------------------------------------------------
        _node_cload: Dict[int, List] = {}  # dof → [total_mag, [node_idx,…]]

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

            # Check if target is a known NSET
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
                cx = query.get("x", (query.get("x_min", 0.0) + query.get("x_max", 0.0)) / 2)
                cy = query.get("y", (query.get("y_min", 0.0) + query.get("y_max", 0.0)) / 2)
                cz = query.get("z", (query.get("z_min", 0.0) + query.get("z_max", 0.0)) / 2)
                rx = query.get("rx", max((query.get("x_max", cx) - query.get("x_min", cx)) / 2, min_radius))
                ry = query.get("ry", max((query.get("y_max", cy) - query.get("y_min", cy)) / 2, min_radius))
                rz = query.get("rz", max((query.get("z_max", cz) - query.get("z_min", cz)) / 2, min_radius))

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
                # Node-ID-based load – accumulate
                idx = node_id_to_idx.get(int(target))
                if idx is not None:
                    if dof not in _node_cload:
                        _node_cload[dof] = [0.0, []]
                    _node_cload[dof][0] += mag
                    _node_cload[dof][1].append(idx)

        # Aggregate node-based CLOADs into one PointLoad per DOF
        for dof in sorted(_node_cload.keys()):
            total_mag, node_indices = _node_cload[dof]
            coords = nodes_arr[node_indices]
            cx = float(coords[:, 0].mean())
            cy = float(coords[:, 1].mean())
            cz = float(coords[:, 2].mean())
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
            "Parsed Abaqus INP (native): %d nodes, %d %s elements, "
            "%d NSETs (%d with selector), %d BCs, %d loads",
            len(nodes_arr),
            len(elems_arr),
            resolved_elem_type,
            len(nsets),
            len(point_sets),
            len(load_case.boundary_conditions),
            len(load_case.loads),
        )

        return load_case
