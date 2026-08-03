"""
Publication-quality load case visualizer
=========================================
Produces a 4-panel figure suitable for academic papers (IJNME, CMAME, etc.):

    ┌──────────────────────────────┬──────────┬──────────┐
    │                              │  TOP     │  FRONT   │
    │     Main 3-D isometric       │  XY      │  XZ      │
    │                              ├──────────┼──────────┤
    │                              │  SIDE    │  LEGEND  │
    │                              │  YZ      │  + meta  │
    └──────────────────────────────┴──────────┴──────────┘

Features
--------
- Design domain: transparent wireframe box with dimension annotations
- Fixed constraints: cross-hatched volumes (engineering convention for fixed support)
- Loads: bold 3-D quiver arrows, magnitude labels, shaded load region
- Two colour-blind-safe, journal-neutral accent colours (red / blue)
- Clean axis spines, 300 DPI PNG + optional PDF (vector) output

Usage
-----
    python scripts/visualize_load_case.py <path/to/load_case.json> [output.png]
    python scripts/visualize_load_case.py vendor/structbench/data/json/a_frame_lateral.json
    python scripts/visualize_load_case.py vendor/structbench/data/json/l_bracket.json --pdf

Dependencies: numpy, matplotlib  (no PyVista / VTK required)
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np

# ── Typography ────────────────────────────────────────────────────────────────
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
        "mathtext.fontset": "cm",
        "axes.labelsize": 8,
        "axes.titlesize": 9,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "figure.dpi": 150,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "lines.linewidth": 1.0,
    }
)

# ── Palette (colour-blind-safe + journal-neutral) ─────────────────────────────
C_DOMAIN = "#2C3E50"  # dark slate  – domain edges
C_DOMAIN_F = "#ECF0F1"  # very light grey – domain fill
C_CONSTRAINT = "#C0392B"  # deep red    – fixed support
C_LOAD = "#2471A3"  # medium blue – force arrows / load region
C_LOAD_PATCH = "#AED6F1"  # pale blue   – load region fill
C_ARROW = "#1A5276"  # dark blue   – arrow shaft
C_MOMENT = "#B9770E"  # amber       – moment/torque glyph
C_DIM = "#5D6D7E"  # slate       – dimension lines

ALPHA_DOMAIN = 0.08
ALPHA_BC = 0.22
ALPHA_LOAD = 0.18

# ── Helpers ───────────────────────────────────────────────────────────────────


def _box_vertices(b: Dict[str, float]) -> np.ndarray:
    """Return 8 corners of an axis-aligned box from a bounds dict."""
    xs = [b["x_min"], b["x_max"]]
    ys = [b["y_min"], b["y_max"]]
    zs = [b["z_min"], b["z_max"]]
    corners = np.array([[x, y, z] for x in xs for y in ys for z in zs], dtype=float)
    return corners


_BOX_EDGES = [
    (0, 1),
    (0, 2),
    (0, 4),
    (1, 3),
    (1, 5),
    (2, 3),
    (2, 6),
    (3, 7),
    (4, 5),
    (4, 6),
    (5, 7),
    (6, 7),
]

_BOX_FACE_INDICES = [
    [0, 2, 3, 1],  # z_min face  (bottom)
    [4, 6, 7, 5],  # z_max face  (top)
    [0, 4, 5, 1],  # y_min face  (front)
    [2, 6, 7, 3],  # y_max face  (back)
    [0, 4, 6, 2],  # x_min face  (left)
    [1, 5, 7, 3],  # x_max face  (right)
]


def _draw_box_3d(
    ax: Axes3D,
    b: Dict[str, float],
    edge_color: str,
    face_color: str,
    alpha_face: float,
    lw: float = 1.2,
    linestyle: str = "-",
    hatch: Optional[str] = None,
    zorder: int = 2,
) -> None:
    """Draw a solid-edge wireframe box with optional translucent filled faces."""
    verts = _box_vertices(b)

    # Filled faces
    faces = [[verts[i] for i in face] for face in _BOX_FACE_INDICES]
    poly = Poly3DCollection(
        faces,
        alpha=alpha_face,
        facecolor=face_color,
        edgecolor="none",
        zorder=zorder,
    )
    if hatch:
        poly.set_hatch(hatch)
    ax.add_collection3d(poly)

    # Edges drawn on top
    for i, j in _BOX_EDGES:
        ax.plot(
            [verts[i, 0], verts[j, 0]],
            [verts[i, 1], verts[j, 1]],
            [verts[i, 2], verts[j, 2]],
            color=edge_color,
            lw=lw,
            ls=linestyle,
            zorder=zorder + 1,
        )


def _draw_dim_bar(
    ax: Axes3D,
    p1: np.ndarray,
    p2: np.ndarray,
    label: str,
    offset: np.ndarray,
    fontsize: float = 7.0,
) -> None:
    """Draw a single dimension bar between two 3D points with a centred label."""
    q1 = p1 + offset
    q2 = p2 + offset
    mid = (q1 + q2) / 2
    off_norm = np.linalg.norm(offset)
    if off_norm > 1e-10:
        mid = mid + (offset / off_norm) * (off_norm * 0.18)

    # Leader lines from domain edge to dim bar
    for pt, q in zip([p1, p2], [q1, q2]):
        ax.plot(
            [pt[0], q[0]],
            [pt[1], q[1]],
            [pt[2], q[2]],
            color=C_DIM,
            lw=0.5,
            ls=":",
            zorder=5,
        )
    # Dim bar
    ax.plot(
        [q1[0], q2[0]],
        [q1[1], q2[1]],
        [q1[2], q2[2]],
        color=C_DIM,
        lw=0.8,
        zorder=5,
    )
    ax.text(
        mid[0],
        mid[1],
        mid[2],
        label,
        fontsize=fontsize,
        ha="center",
        va="bottom",
        color=C_DIM,
        zorder=6,
    )


def _fmt_mm_dim(length_mm: float) -> str:
    """Format geometric dimension labels as integer millimetres."""
    return f"{int(round(length_mm))} mm"


def _dir_to_vec(direction: str) -> np.ndarray:
    """Convert a direction string like '-z', 'x', 'y' into a unit vector."""
    mapping = {
        "x": (1, 0, 0),
        "+x": (1, 0, 0),
        "-x": (-1, 0, 0),
        "y": (0, 1, 0),
        "+y": (0, 1, 0),
        "-y": (0, -1, 0),
        "z": (0, 0, 1),
        "+z": (0, 0, 1),
        "-z": (0, 0, -1),
    }
    return np.array(mapping.get(direction.lower().strip(), (0, 0, -1)), dtype=float)


def _box_centroid(b: Dict[str, float]) -> np.ndarray:
    return np.array(
        [
            (b["x_min"] + b["x_max"]) / 2,
            (b["y_min"] + b["y_max"]) / 2,
            (b["z_min"] + b["z_max"]) / 2,
        ],
        dtype=float,
    )


def _center_anchor_box(center: List[float], radius: float) -> Dict[str, float]:
    cx, cy, cz = float(center[0]), float(center[1]), float(center[2])
    r = max(float(radius), 1e-6)
    return {
        "x_min": cx - r,
        "x_max": cx + r,
        "y_min": cy - r,
        "y_max": cy + r,
        "z_min": cz - r,
        "z_max": cz + r,
    }


def _resolve_load_selector_box(
    load: Dict[str, Any],
    selectors: Dict[str, Dict[str, float]],
    span: float,
) -> Optional[Dict[str, float]]:
    """Resolve a load anchor region from region_id, inline selector, or center/radius metadata."""
    sel_id = load.get("region_id", "")
    sel = selectors.get(sel_id)
    if sel is not None:
        return sel

    inline = load.get("selector", {})
    if inline:
        q = inline.get("query", {})
        if "x" in q and "y" in q and "z" in q:
            eps = span * 0.01
            return {
                "x_min": q["x"] - eps,
                "x_max": q["x"] + eps,
                "y_min": q.get("y", 0) - eps,
                "y_max": q.get("y", 0) + eps,
                "z_min": q.get("z", 0) - eps,
                "z_max": q.get("z", 0) + eps,
            }

    center = load.get("center", None)
    if isinstance(center, list) and len(center) == 3:
        radius = float(load.get("radius", span * 0.04))
        return _center_anchor_box(center, max(radius, span * 0.02))

    return None


def _expand_scene_bounds_with_point(scene: Dict[str, float], point: np.ndarray) -> None:
    scene["x_min"] = min(scene["x_min"], float(point[0]))
    scene["x_max"] = max(scene["x_max"], float(point[0]))
    scene["y_min"] = min(scene["y_min"], float(point[1]))
    scene["y_max"] = max(scene["y_max"], float(point[1]))
    scene["z_min"] = min(scene["z_min"], float(point[2]))
    scene["z_max"] = max(scene["z_max"], float(point[2]))


def _compute_scene_bounds(
    domain: Dict[str, float],
    selectors: Dict[str, Dict[str, float]],
    loads: List[Dict[str, Any]],
    arrow_len: float,
    label_offset: float,
) -> Dict[str, float]:
    """Extend the visible bounds to include rendered load glyphs and labels."""
    scene = dict(domain)
    span = max(
        domain["x_max"] - domain["x_min"],
        domain["y_max"] - domain["y_min"],
        domain["z_max"] - domain["z_min"],
    )

    for load in loads:
        if _is_pressure_load(load) and "center" in load and "radius" in load:
            center = np.array(load["center"], dtype=float)
            radius = float(load["radius"])
            normal_axis = str(load.get("normal_axis", "z")).lower().strip()
            direction = str(load.get("direction", "-z")).lower().strip()

            for dx in (-radius, radius):
                for dy in (-radius, radius):
                    for dz in (-radius, radius):
                        _expand_scene_bounds_with_point(
                            scene, center + np.array([dx, dy, dz], dtype=float)
                        )

            if direction in {"outward", "inward"}:
                label_axis = _dir_to_vec(normal_axis)
                _expand_scene_bounds_with_point(
                    scene, center + label_axis * (arrow_len * 0.35)
                )
            else:
                vec = _dir_to_vec(direction)
                _expand_scene_bounds_with_point(
                    scene, center + vec * (arrow_len * label_offset)
                )
            continue

        sel = _resolve_load_selector_box(load, selectors, span)
        if sel is not None:
            for pt in _box_vertices(sel):
                _expand_scene_bounds_with_point(scene, pt)

        if _is_moment_load(load):
            center = load.get("center", None)
            if isinstance(center, list) and len(center) == 3:
                c = np.array(center, dtype=float)
            elif sel is not None:
                c = _box_centroid(sel)
            else:
                continue
            vec = _moment_axis_vec(load)
            nrm = np.linalg.norm(vec)
            if nrm < 1e-10:
                continue
            vec = vec / nrm
            half = arrow_len * 0.52
            _expand_scene_bounds_with_point(scene, c - vec * half)
            _expand_scene_bounds_with_point(scene, c + vec * half)
            _expand_scene_bounds_with_point(scene, c + vec * (half * label_offset))
            continue

        if sel is not None:
            centroid = _box_centroid(sel)
            vec = _dir_to_vec(str(load.get("direction", "z")))
            _expand_scene_bounds_with_point(scene, centroid)
            _expand_scene_bounds_with_point(
                scene, centroid + vec * (arrow_len * label_offset)
            )

    return scene


def _domain_from_design_domain(
    design_domain: Dict[str, Any],
    json_path: Optional[Path] = None,
) -> Tuple[Dict[str, float], str, Dict[str, Any]]:
    """
    Parse design_domain into:
      1) a cartesian bounds dict (always returned for axis setup),
            2) canonical shape type ("box" | "cylinder" | "step" | "l_shape" | "t_shape"),
      3) normalized shape params.
    """
    bounds = dict(design_domain.get("bounds", {}))
    shape_type = str(design_domain.get("shape_type", "box")).lower().strip()

    if shape_type in {"", "box"}:
        if all(
            k in bounds for k in ("x_min", "x_max", "y_min", "y_max", "z_min", "z_max")
        ):
            return bounds, "box", {}
        raise ValueError(
            "design_domain must provide complete bounds for box domains "
            "(x_min/x_max/y_min/y_max/z_min/z_max)."
        )

    if shape_type == "step":
        params = dict(design_domain.get("params", {}))
        raw_step_path = (
            params.get("step_path") or params.get("step_file") or params.get("path")
        )
        if not raw_step_path:
            raise ValueError(
                "STEP design_domain requires params.step_path (or step_file/path)."
            )

        step_path = Path(raw_step_path)
        if not step_path.is_absolute():
            if json_path is not None:
                step_path = (json_path.parent / step_path).resolve()
            else:
                step_path = step_path.resolve()
        if not step_path.exists():
            raise ValueError(f"STEP file not found: {step_path}")

        has_bounds = all(
            k in bounds for k in ("x_min", "x_max", "y_min", "y_max", "z_min", "z_max")
        )
        if not has_bounds:
            try:
                import cadquery as cq  # type: ignore

                bb = cq.importers.importStep(str(step_path)).val().BoundingBox()
                bounds = {
                    "x_min": float(bb.xmin),
                    "x_max": float(bb.xmax),
                    "y_min": float(bb.ymin),
                    "y_max": float(bb.ymax),
                    "z_min": float(bb.zmin),
                    "z_max": float(bb.zmax),
                }
            except Exception as exc:
                raise ValueError(
                    "STEP design_domain must provide bounds when cadquery is unavailable "
                    f"or STEP import fails ({exc})."
                )

        return bounds, "step", {"step_path": str(step_path)}

    if shape_type == "l_shape":
        params = dict(design_domain.get("params", {}))
        width = float(params.get("width", 0.0))
        height = float(params.get("height", 0.0))
        depth = float(params.get("depth", 0.0))
        leg_width = float(params.get("leg_width", 0.0))
        base_height = float(params.get("base_height", 0.0))

        if not (
            width > 0.0
            and height > 0.0
            and depth > 0.0
            and 0.0 < leg_width <= width
            and 0.0 < base_height <= height
        ):
            raise ValueError(
                "Invalid l_shape design_domain.params "
                "(width/height/depth/leg_width/base_height)."
            )

        bounds = {
            "x_min": 0.0,
            "x_max": width,
            "y_min": 0.0,
            "y_max": depth,
            "z_min": 0.0,
            "z_max": height,
        }
        return (
            bounds,
            "l_shape",
            {
                "width": width,
                "height": height,
                "depth": depth,
                "leg_width": leg_width,
                "base_height": base_height,
            },
        )

    if shape_type == "t_shape":
        params = dict(design_domain.get("params", {}))
        width = float(params.get("width", 0.0))
        height = float(params.get("height", 0.0))
        depth = float(params.get("depth", 0.0))
        stem_width = float(params.get("stem_width", 0.0))
        cap_height = float(params.get("cap_height", 0.0))

        if not (
            width > 0.0
            and height > 0.0
            and depth > 0.0
            and 0.0 < stem_width <= width
            and 0.0 < cap_height < height
        ):
            raise ValueError(
                "Invalid t_shape design_domain.params "
                "(width/height/depth/stem_width/cap_height)."
            )

        bounds = {
            "x_min": 0.0,
            "x_max": width,
            "y_min": 0.0,
            "y_max": depth,
            "z_min": 0.0,
            "z_max": height,
        }
        return (
            bounds,
            "t_shape",
            {
                "width": width,
                "height": height,
                "depth": depth,
                "stem_width": stem_width,
                "cap_height": cap_height,
            },
        )

    if shape_type != "cylinder":
        raise ValueError(f"Unsupported design_domain shape_type: {shape_type}")

    params = design_domain.get("params", {})
    radius = float(params.get("radius", 0.0))
    height = float(params.get("height", 0.0))
    center = params.get("center", [0.0, 0.0, 0.0])
    axis = str(params.get("axis", "z")).lower().strip()

    if (
        len(center) != 3
        or radius <= 0.0
        or height <= 0.0
        or axis not in {"x", "y", "z"}
    ):
        raise ValueError(
            "Invalid cylinder design_domain.params (radius/height/center/axis)."
        )

    cx, cy, cz = float(center[0]), float(center[1]), float(center[2])
    if axis == "z":
        bounds = {
            "x_min": cx - radius,
            "x_max": cx + radius,
            "y_min": cy - radius,
            "y_max": cy + radius,
            "z_min": cz,
            "z_max": cz + height,
        }
    elif axis == "x":
        bounds = {
            "x_min": cx,
            "x_max": cx + height,
            "y_min": cy - radius,
            "y_max": cy + radius,
            "z_min": cz - radius,
            "z_max": cz + radius,
        }
    else:  # axis == "y"
        bounds = {
            "x_min": cx - radius,
            "x_max": cx + radius,
            "y_min": cy,
            "y_max": cy + height,
            "z_min": cz - radius,
            "z_max": cz + radius,
        }

    norm_params = {
        "radius": radius,
        "height": height,
        "center": [cx, cy, cz],
        "axis": axis,
    }
    return bounds, "cylinder", norm_params


def _draw_cylinder_3d(
    ax: Axes3D,
    params: Dict[str, Any],
    edge_color: str,
    face_color: str,
    alpha_face: float,
    lw: float = 1.2,
    linestyle: str = "--",
    zorder: int = 2,
) -> None:
    """Draw a cylinder domain using a side surface and two circular end caps."""
    radius = float(params["radius"])
    height = float(params["height"])
    cx, cy, cz = params["center"]
    axis = params["axis"]

    theta = np.linspace(0.0, 2.0 * np.pi, 48)
    t = np.linspace(0.0, height, 2)
    tt, hh = np.meshgrid(theta, t)

    if axis == "z":
        xx = cx + radius * np.cos(tt)
        yy = cy + radius * np.sin(tt)
        zz = cz + hh
        ring0 = np.column_stack(
            (
                cx + radius * np.cos(theta),
                cy + radius * np.sin(theta),
                np.full_like(theta, cz),
            )
        )
        ring1 = np.column_stack(
            (
                cx + radius * np.cos(theta),
                cy + radius * np.sin(theta),
                np.full_like(theta, cz + height),
            )
        )
    elif axis == "x":
        xx = cx + hh
        yy = cy + radius * np.cos(tt)
        zz = cz + radius * np.sin(tt)
        ring0 = np.column_stack(
            (
                np.full_like(theta, cx),
                cy + radius * np.cos(theta),
                cz + radius * np.sin(theta),
            )
        )
        ring1 = np.column_stack(
            (
                np.full_like(theta, cx + height),
                cy + radius * np.cos(theta),
                cz + radius * np.sin(theta),
            )
        )
    else:  # axis == "y"
        xx = cx + radius * np.cos(tt)
        yy = cy + hh
        zz = cz + radius * np.sin(tt)
        ring0 = np.column_stack(
            (
                cx + radius * np.cos(theta),
                np.full_like(theta, cy),
                cz + radius * np.sin(theta),
            )
        )
        ring1 = np.column_stack(
            (
                cx + radius * np.cos(theta),
                np.full_like(theta, cy + height),
                cz + radius * np.sin(theta),
            )
        )

    ax.plot_surface(
        xx,
        yy,
        zz,
        color=face_color,
        alpha=alpha_face,
        linewidth=0,
        antialiased=True,
        shade=False,
    )
    caps = Poly3DCollection(
        [ring0.tolist(), ring1.tolist()],
        facecolor=face_color,
        alpha=alpha_face,
        edgecolor="none",
        zorder=zorder,
    )
    ax.add_collection3d(caps)
    ax.plot(
        ring0[:, 0],
        ring0[:, 1],
        ring0[:, 2],
        color=edge_color,
        lw=lw,
        ls=linestyle,
        zorder=zorder + 1,
    )
    ax.plot(
        ring1[:, 0],
        ring1[:, 1],
        ring1[:, 2],
        color=edge_color,
        lw=lw,
        ls=linestyle,
        zorder=zorder + 1,
    )

    guide_angles = (0.0, np.pi / 2.0, np.pi, 3.0 * np.pi / 2.0)
    for ang in guide_angles:
        if axis == "z":
            p0 = np.array([cx + radius * np.cos(ang), cy + radius * np.sin(ang), cz])
            p1 = np.array(
                [cx + radius * np.cos(ang), cy + radius * np.sin(ang), cz + height]
            )
        elif axis == "x":
            p0 = np.array([cx, cy + radius * np.cos(ang), cz + radius * np.sin(ang)])
            p1 = np.array(
                [cx + height, cy + radius * np.cos(ang), cz + radius * np.sin(ang)]
            )
        else:  # axis == "y"
            p0 = np.array([cx + radius * np.cos(ang), cy, cz + radius * np.sin(ang)])
            p1 = np.array(
                [cx + radius * np.cos(ang), cy + height, cz + radius * np.sin(ang)]
            )

        ax.plot(
            [p0[0], p1[0]],
            [p0[1], p1[1]],
            [p0[2], p1[2]],
            color=edge_color,
            lw=lw,
            ls=linestyle,
            zorder=zorder + 1,
        )


def _draw_domain_2d(
    ax: plt.Axes,
    domain: Dict[str, float],
    shape_type: str,
    shape_params: Dict[str, Any],
    axis_h: str,
    axis_v: str,
    edge_color: str,
    face_color: str,
    alpha: float,
    lw: float = 1.0,
    linestyle: str = "-",
) -> None:
    """Draw projected design domain shape (box/cylinder/step/l_shape/t_shape) on a 2-D axes."""
    if shape_type == "l_shape":
        width = float(shape_params["width"])
        height = float(shape_params["height"])
        depth = float(shape_params["depth"])
        leg_width = float(shape_params["leg_width"])
        base_height = float(shape_params["base_height"])

        # L profile in X-Z, extruded along Y(depth).
        vertical_leg = {
            "x_min": 0.0,
            "x_max": leg_width,
            "y_min": 0.0,
            "y_max": depth,
            "z_min": 0.0,
            "z_max": height,
        }
        base_leg = {
            "x_min": 0.0,
            "x_max": width,
            "y_min": 0.0,
            "y_max": depth,
            "z_min": 0.0,
            "z_max": base_height,
        }
        _draw_box_2d(
            ax,
            vertical_leg,
            axis_h,
            axis_v,
            edge_color,
            face_color,
            alpha,
            lw=lw,
            linestyle=linestyle,
        )
        _draw_box_2d(
            ax,
            base_leg,
            axis_h,
            axis_v,
            edge_color,
            face_color,
            alpha,
            lw=lw,
            linestyle=linestyle,
        )
        return

    if shape_type == "t_shape":
        width = float(shape_params["width"])
        height = float(shape_params["height"])
        depth = float(shape_params["depth"])
        stem_width = float(shape_params["stem_width"])
        cap_height = float(shape_params["cap_height"])

        stem_x_min = (width - stem_width) / 2.0
        stem_x_max = (width + stem_width) / 2.0
        cap = {
            "x_min": 0.0,
            "x_max": width,
            "y_min": 0.0,
            "y_max": depth,
            "z_min": height - cap_height,
            "z_max": height,
        }
        stem = {
            "x_min": stem_x_min,
            "x_max": stem_x_max,
            "y_min": 0.0,
            "y_max": depth,
            "z_min": 0.0,
            "z_max": height - cap_height,
        }
        _draw_box_2d(
            ax,
            cap,
            axis_h,
            axis_v,
            edge_color,
            face_color,
            alpha,
            lw=lw,
            linestyle=linestyle,
        )
        _draw_box_2d(
            ax,
            stem,
            axis_h,
            axis_v,
            edge_color,
            face_color,
            alpha,
            lw=lw,
            linestyle=linestyle,
        )
        return

    if shape_type != "cylinder":
        _draw_box_2d(
            ax,
            domain,
            axis_h,
            axis_v,
            edge_color,
            face_color,
            alpha,
            lw=lw,
            linestyle=linestyle,
        )
        return

    axis = shape_params.get("axis", "z")
    radius = float(shape_params.get("radius", 0.0))
    height = float(shape_params.get("height", 0.0))
    center = shape_params.get("center", [0.0, 0.0, 0.0])
    idx = {"x": 0, "y": 1, "z": 2}
    idx_h = idx[axis_h]
    idx_v = idx[axis_v]
    c = np.array(center, dtype=float)

    if axis not in {axis_h, axis_v}:
        circ = mpatches.Circle(
            (c[idx_h], c[idx_v]),
            radius,
            edgecolor=edge_color,
            facecolor=face_color,
            linewidth=lw,
            alpha=alpha,
            linestyle=linestyle,
            zorder=3,
        )
        ax.add_patch(circ)
        return

    other = [a for a in ["x", "y", "z"] if a not in {axis}][0]
    if other == axis_h:
        lo_h = c[idx_h] - radius
        hi_h = c[idx_h] + radius
        lo_v = c[idx_v]
        hi_v = c[idx_v] + height
    elif other == axis_v:
        lo_h = c[idx_h]
        hi_h = c[idx_h] + height
        lo_v = c[idx_v] - radius
        hi_v = c[idx_v] + radius
    else:
        if axis == axis_h:
            lo_h = c[idx_h]
            hi_h = c[idx_h] + height
            lo_v = c[idx_v] - radius
            hi_v = c[idx_v] + radius
        else:
            lo_h = c[idx_h] - radius
            hi_h = c[idx_h] + radius
            lo_v = c[idx_v]
            hi_v = c[idx_v] + height

    rect = mpatches.FancyBboxPatch(
        (lo_h, lo_v),
        max(hi_h - lo_h, 1e-6),
        max(hi_v - lo_v, 1e-6),
        boxstyle="square,pad=0",
        linewidth=lw,
        edgecolor=edge_color,
        facecolor=face_color,
        alpha=alpha,
        linestyle=linestyle,
        zorder=3,
    )
    ax.add_patch(rect)


def _is_pressure_load(load: Dict[str, Any]) -> bool:
    ltype = str(load.get("type", "")).lower()
    return "pressure" in ltype or any(
        k in load for k in ("pressure_pa", "pressure_kpa", "pressure_mpa")
    )


def _is_moment_load(load: Dict[str, Any]) -> bool:
    ltype = str(load.get("type", "")).lower()
    return (
        "moment" in ltype
        or "torque" in ltype
        or any(
            k in load for k in ("magnitude_nm", "magnitude_nmm", "magnitude_newton_mm")
        )
    )


def _moment_nmm(load: Dict[str, Any]) -> float:
    if "magnitude_nmm" in load:
        return float(load.get("magnitude_nmm", 0.0))
    if "magnitude_newton_mm" in load:
        return float(load.get("magnitude_newton_mm", 0.0))
    # StructBench currently uses magnitude_nm but values are in N*mm.
    if "magnitude_nm" in load:
        return float(load.get("magnitude_nm", 0.0))
    return float(load.get("magnitude_newtons", 0.0))


def _moment_axis_vec(load: Dict[str, Any]) -> np.ndarray:
    axis = str(load.get("axis", "z")).lower().strip()
    axis_vec = _dir_to_vec(axis)
    rot_dir = str(load.get("direction", "ccw")).lower().strip()
    # Right-hand-rule convention: ccw is positive along +axis, cw along -axis.
    if rot_dir in {"cw", "clockwise", "-", "negative"}:
        axis_vec = -axis_vec
    return axis_vec


def _fmt_moment(nmm: float) -> str:
    nm = nmm / 1000.0
    a = abs(nm)
    if a >= 1e6:
        return f"{nm/1e6:.3g} MN·m"
    if a >= 1e3:
        return f"{nm/1e3:.3g} kN·m"
    return f"{nm:.4g} N·m"


def _stagger_index(n: int) -> int:
    """Sequence 0, +1, -1, +2, -2, ... for deterministic label spreading."""
    if n <= 0:
        return 0
    mag = (n + 1) // 2
    return mag if n % 2 == 1 else -mag


def _spread_label_position_3d(
    base_pos: np.ndarray,
    direction: np.ndarray,
    used_positions: List[np.ndarray],
    span: float,
    label_offset: float,
) -> np.ndarray:
    """Spread nearby 3-D labels sideways and slightly outward based on label_offset."""
    d = np.array(direction, dtype=float)
    nrm = np.linalg.norm(d)
    if nrm < 1e-10:
        d = np.array([0.0, 0.0, 1.0])
    else:
        d = d / nrm

    ref = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(d, ref)) > 0.92:
        ref = np.array([0.0, 1.0, 0.0])
    side = np.cross(d, ref)
    side_nrm = np.linalg.norm(side)
    if side_nrm < 1e-10:
        side = np.array([1.0, 0.0, 0.0])
    else:
        side = side / side_nrm

    cluster_threshold = span * (0.10 + 0.03 * max(label_offset - 1.0, 0.0))
    collisions = sum(
        np.linalg.norm(base_pos - p) < cluster_threshold for p in used_positions
    )
    k = _stagger_index(collisions)
    side_step = span * 0.06 * max(label_offset, 1.0)
    outward_step = span * 0.025 * abs(k) * max(label_offset, 1.0)
    return base_pos + side * (k * side_step) + d * outward_step


def _spread_label_position_2d(
    base_pos: np.ndarray,
    direction: np.ndarray,
    used_positions: List[np.ndarray],
    span: float,
    label_offset: float,
) -> np.ndarray:
    """Spread nearby 2-D labels sideways and slightly outward based on label_offset."""
    d = np.array(direction, dtype=float)
    nrm = np.linalg.norm(d)
    if nrm < 1e-10:
        d = np.array([0.0, 1.0])
    else:
        d = d / nrm

    side = np.array([-d[1], d[0]])
    cluster_threshold = span * (0.09 + 0.03 * max(label_offset - 1.0, 0.0))
    collisions = sum(
        np.linalg.norm(base_pos - p) < cluster_threshold for p in used_positions
    )
    k = _stagger_index(collisions)
    side_step = span * 0.055 * max(label_offset, 1.0)
    outward_step = span * 0.02 * abs(k) * max(label_offset, 1.0)
    return base_pos + side * (k * side_step) + d * outward_step


def _draw_moment_load_3d(
    ax: Axes3D,
    load: Dict[str, Any],
    centroid: np.ndarray,
    arrow_len: float,
    label_offset: float,
    label_font_size: float,
    used_label_positions: List[np.ndarray],
    span: float,
) -> bool:
    if not _is_moment_load(load):
        return False

    vec = _moment_axis_vec(load)
    nrm = np.linalg.norm(vec)
    if nrm < 1e-10:
        return False
    vec = vec / nrm

    half = arrow_len * 0.52
    p0 = centroid - vec * half
    p1 = centroid + vec * half

    # Axis line with two heads to indicate pure couple/moment.
    ax.plot(
        [p0[0], p1[0]],
        [p0[1], p1[1]],
        [p0[2], p1[2]],
        color=C_MOMENT,
        lw=2.0,
        zorder=7,
    )
    ax.quiver(
        centroid[0],
        centroid[1],
        centroid[2],
        vec[0],
        vec[1],
        vec[2],
        length=half,
        color=C_MOMENT,
        arrow_length_ratio=0.28,
        linewidth=2.0,
        zorder=8,
    )
    ax.quiver(
        centroid[0],
        centroid[1],
        centroid[2],
        -vec[0],
        -vec[1],
        -vec[2],
        length=half,
        color=C_MOMENT,
        arrow_length_ratio=0.28,
        linewidth=2.0,
        zorder=8,
    )

    label_pos = centroid + vec * (half * label_offset)
    label_pos = _spread_label_position_3d(
        label_pos, vec, used_label_positions, span, label_offset
    )
    used_label_positions.append(label_pos)
    ax.text(
        label_pos[0],
        label_pos[1],
        label_pos[2],
        _fmt_moment(_moment_nmm(load)),
        fontsize=label_font_size,
        color=C_MOMENT,
        ha="center",
        va="center",
        fontweight="bold",
        zorder=9,
    )
    return True


def _draw_moment_load_2d(
    ax: plt.Axes,
    load: Dict[str, Any],
    center2: np.ndarray,
    axis_h: str,
    axis_v: str,
    arrow_len: float,
    label_font_size: float,
    label_offset: float,
    used_label_positions: List[np.ndarray],
    span: float,
) -> bool:
    if not _is_moment_load(load):
        return False

    vec3 = _moment_axis_vec(load)
    idx = {"x": 0, "y": 1, "z": 2}
    vec2 = np.array([vec3[idx[axis_h]], vec3[idx[axis_v]]], dtype=float)
    nrm = np.linalg.norm(vec2)

    if nrm < 1e-10:
        # Axis is normal to this view: indicate by a centered ± marker and label.
        axis = str(load.get("axis", "z")).lower().strip()
        sign = "⊙" if np.sign(_moment_axis_vec(load)[idx.get(axis, 2)]) >= 0 else "⊗"
        ax.text(
            center2[0],
            center2[1],
            sign,
            fontsize=label_font_size + 3,
            color=C_MOMENT,
            ha="center",
            va="center",
            fontweight="bold",
            zorder=8,
        )
        label_pos = _spread_label_position_2d(
            center2 + np.array([0.0, arrow_len * 0.45]),
            np.array([0.0, 1.0]),
            used_label_positions,
            span,
            label_offset,
        )
        used_label_positions.append(label_pos)
        ax.text(
            label_pos[0],
            label_pos[1],
            _fmt_moment(_moment_nmm(load)),
            fontsize=label_font_size,
            color=C_MOMENT,
            ha="center",
            va="center",
            fontweight="bold",
            zorder=8,
        )
        return True

    u = vec2 / nrm
    half = arrow_len * 0.52
    p0 = center2 - u * half
    p1 = center2 + u * half

    ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color=C_MOMENT, lw=2.0, zorder=7)
    ax.annotate(
        "",
        xy=(p1[0], p1[1]),
        xytext=(center2[0], center2[1]),
        arrowprops=dict(arrowstyle="-|>", color=C_MOMENT, lw=1.8, mutation_scale=11),
        zorder=8,
    )
    ax.annotate(
        "",
        xy=(p0[0], p0[1]),
        xytext=(center2[0], center2[1]),
        arrowprops=dict(arrowstyle="-|>", color=C_MOMENT, lw=1.8, mutation_scale=11),
        zorder=8,
    )
    label_pos = _spread_label_position_2d(
        p1 + u * (arrow_len * 0.15),
        u,
        used_label_positions,
        span,
        label_offset,
    )
    used_label_positions.append(label_pos)
    ax.text(
        label_pos[0],
        label_pos[1],
        _fmt_moment(_moment_nmm(load)),
        fontsize=label_font_size,
        color=C_MOMENT,
        ha="center",
        va="center",
        fontweight="bold",
        zorder=9,
    )
    return True


def _pressure_pa(load: Dict[str, Any]) -> float:
    if "pressure_pa" in load:
        return float(load.get("pressure_pa", 0.0))
    if "pressure_kpa" in load:
        return float(load.get("pressure_kpa", 0.0)) * 1e3
    if "pressure_mpa" in load:
        return float(load.get("pressure_mpa", 0.0)) * 1e6
    return float(load.get("magnitude_newtons", 0.0))


def _draw_pressure_load_3d(
    ax: Axes3D,
    load: Dict[str, Any],
    arrow_len: float,
    label_offset: float,
    label_font_size: float,
    used_label_positions: List[np.ndarray],
    span: float,
) -> bool:
    """Draw pressure load with explicit center/radius/normal_axis metadata."""
    center = load.get("center", None)
    radius = load.get("radius", None)
    if not isinstance(center, list) or len(center) != 3 or radius is None:
        return False

    c = np.array(center, dtype=float)
    r = float(radius)
    if r <= 0.0:
        return False

    direction = str(load.get("direction", "-z")).lower().strip()
    normal_axis = str(load.get("normal_axis", "z")).lower().strip()
    if normal_axis not in {"x", "y", "z"}:
        normal_axis = "z"

    theta = np.linspace(0.0, 2.0 * np.pi, 64)
    if normal_axis == "z":
        ring = np.column_stack(
            (
                c[0] + r * np.cos(theta),
                c[1] + r * np.sin(theta),
                np.full_like(theta, c[2]),
            )
        )
        normal = np.array([0.0, 0.0, 1.0])
    elif normal_axis == "x":
        ring = np.column_stack(
            (
                np.full_like(theta, c[0]),
                c[1] + r * np.cos(theta),
                c[2] + r * np.sin(theta),
            )
        )
        normal = np.array([1.0, 0.0, 0.0])
    else:
        ring = np.column_stack(
            (
                c[0] + r * np.cos(theta),
                np.full_like(theta, c[1]),
                c[2] + r * np.sin(theta),
            )
        )
        normal = np.array([0.0, 1.0, 0.0])

    patch = Poly3DCollection(
        [ring.tolist()],
        facecolor=C_LOAD_PATCH,
        alpha=ALPHA_LOAD,
        edgecolor=C_LOAD,
        linewidths=1.2,
        zorder=5,
    )
    ax.add_collection3d(patch)

    # For outward/inward pressure on an end face, show radial arrows in the face plane.
    if direction in {"outward", "inward"}:
        n_arrows = 8
        angles = np.linspace(0.0, 2.0 * np.pi, n_arrows, endpoint=False)
        for a in angles:
            if normal_axis == "z":
                base = c + np.array([0.65 * r * np.cos(a), 0.65 * r * np.sin(a), 0.0])
                vec = np.array([np.cos(a), np.sin(a), 0.0])
            elif normal_axis == "x":
                base = c + np.array([0.0, 0.65 * r * np.cos(a), 0.65 * r * np.sin(a)])
                vec = np.array([0.0, np.cos(a), np.sin(a)])
            else:
                base = c + np.array([0.65 * r * np.cos(a), 0.0, 0.65 * r * np.sin(a)])
                vec = np.array([np.cos(a), 0.0, np.sin(a)])

            if direction == "inward":
                vec = -vec
            ax.quiver(
                base[0],
                base[1],
                base[2],
                vec[0],
                vec[1],
                vec[2],
                length=arrow_len * 0.45,
                color=C_ARROW,
                arrow_length_ratio=0.25,
                linewidth=1.6,
                zorder=7,
            )

        label_pos = c + normal * (arrow_len * 0.35)
        label_dir = normal
    else:
        vec = _dir_to_vec(direction)
        ax.quiver(
            c[0],
            c[1],
            c[2],
            vec[0],
            vec[1],
            vec[2],
            length=arrow_len,
            color=C_ARROW,
            arrow_length_ratio=0.22,
            linewidth=2.5,
            zorder=7,
        )
        label_pos = c + vec * (arrow_len * label_offset)
        label_dir = vec

    label_pos = _spread_label_position_3d(
        label_pos, label_dir, used_label_positions, span, label_offset
    )
    used_label_positions.append(label_pos)

    ax.text(
        label_pos[0],
        label_pos[1],
        label_pos[2],
        _fmt_pressure(_pressure_pa(load)),
        fontsize=label_font_size,
        color=C_ARROW,
        ha="center",
        va="center",
        fontweight="bold",
        zorder=8,
    )
    return True


def _draw_pressure_load_2d(
    ax: plt.Axes,
    load: Dict[str, Any],
    axis_h: str,
    axis_v: str,
    arrow_len: float,
    label_font_size: float,
    label_offset: float,
    used_label_positions: List[np.ndarray],
    span: float,
) -> bool:
    """Project and draw a pressure load in 2-D views."""
    center = load.get("center", None)
    radius = load.get("radius", None)
    if not isinstance(center, list) or len(center) != 3 or radius is None:
        return False

    c = np.array(center, dtype=float)
    r = float(radius)
    if r <= 0.0:
        return False

    normal_axis = str(load.get("normal_axis", "z")).lower().strip()
    if normal_axis not in {"x", "y", "z"}:
        normal_axis = "z"
    direction = str(load.get("direction", "-z")).lower().strip()

    idx = {"x": 0, "y": 1, "z": 2}
    idx_h = idx[axis_h]
    idx_v = idx[axis_v]
    center2 = np.array([c[idx_h], c[idx_v]])

    if normal_axis not in {axis_h, axis_v}:
        patch = mpatches.Circle(
            center2,
            r,
            edgecolor=C_LOAD,
            facecolor=C_LOAD_PATCH,
            alpha=ALPHA_LOAD * 1.5,
            linewidth=1.0,
            zorder=4,
        )
        ax.add_patch(patch)
    else:
        # Edge-on projection of a circular face appears as a line segment.
        face_axes = [a for a in ["x", "y", "z"] if a != normal_axis]
        shown = face_axes[0] if face_axes[0] in {axis_h, axis_v} else face_axes[1]
        if shown == axis_h:
            lo = center2 + np.array([-r, 0.0])
            hi = center2 + np.array([r, 0.0])
        else:
            lo = center2 + np.array([0.0, -r])
            hi = center2 + np.array([0.0, r])
        ax.plot(
            [lo[0], hi[0]], [lo[1], hi[1]], color=C_LOAD, lw=3.0, alpha=0.8, zorder=5
        )

    def _arrow(start: np.ndarray, vec2: np.ndarray, scale: float = 1.0) -> None:
        nrm = np.linalg.norm(vec2)
        if nrm < 1e-10:
            return
        u = vec2 / nrm
        d = u * (arrow_len * scale)
        ax.annotate(
            "",
            xy=(start[0] + d[0], start[1] + d[1]),
            xytext=(start[0], start[1]),
            arrowprops=dict(arrowstyle="-|>", color=C_ARROW, lw=1.5, mutation_scale=10),
            zorder=6,
        )

    if direction in {"outward", "inward"}:
        face_axes = [a for a in ["x", "y", "z"] if a != normal_axis]
        sign = 1.0 if direction == "outward" else -1.0
        for ax_name in face_axes:
            vec3 = np.array(
                [1.0 if a == ax_name else 0.0 for a in ["x", "y", "z"]], dtype=float
            )
            for sgn in (-1.0, 1.0):
                vec3_use = vec3 * sgn * sign
                vec2 = np.array([vec3_use[idx_h], vec3_use[idx_v]])
                _arrow(center2, vec2, scale=0.60)
        label_dir3 = np.array(
            [1.0 if a == normal_axis else 0.0 for a in ["x", "y", "z"]], dtype=float
        )
        label_dir2 = np.array([label_dir3[idx_h], label_dir3[idx_v]])
    else:
        vec3 = _dir_to_vec(direction)
        vec2 = np.array([vec3[idx_h], vec3[idx_v]])
        _arrow(center2, vec2)
        label_dir2 = vec2

    nrm = np.linalg.norm(label_dir2)
    label_shift = (
        (label_dir2 / nrm) * (arrow_len * label_offset)
        if nrm > 1e-10
        else np.array([0.0, arrow_len * 0.8])
    )
    label_pos = center2 + label_shift
    label_pos = _spread_label_position_2d(
        label_pos, label_dir2, used_label_positions, span, label_offset
    )
    used_label_positions.append(label_pos)
    ax.text(
        label_pos[0],
        label_pos[1],
        _fmt_pressure(_pressure_pa(load)),
        fontsize=label_font_size,
        color=C_ARROW,
        ha="center",
        va="center",
        fontweight="bold",
        zorder=7,
    )
    return True


def _fmt_force(n: float) -> str:
    """Format a force value in SI with k/M prefix."""
    a = abs(n)
    if a >= 1e6:
        return f"{n/1e6:.2g} MN"
    if a >= 1e3:
        return f"{n/1e3:.4g} kN"
    return f"{n:.4g} N"


def _fmt_pressure(n: float) -> str:
    a = abs(n)
    if a >= 1e6:
        return f"{n/1e6:.3g} MPa"
    if a >= 1e3:
        return f"{n/1e3:.3g} kPa"
    return f"{n:.4g} Pa"


def _get_dof_lock_state(bc: Dict[str, Any]) -> Dict[str, bool]:
    """
    Parse a boundary condition and return DOF lock state.
    Default: all DOF locked (x, y, z = True).
    Override with dof_lock dict if present.
    """
    state = {"x": True, "y": True, "z": True}
    dof_lock = bc.get("dof_lock", {})
    if isinstance(dof_lock, dict):
        state.update(dof_lock)
    return state


def _draw_dof_glyph_3d(
    ax: Axes3D,
    centroid: np.ndarray,
    dof_lock: Dict[str, bool],
    span: float,
    fontsize: float = 6.0,
) -> None:
    """
    Draw a small 3-axis DOF indicator at the constraint centroid.
    Only show free DOF: green double-headed arrows (↔).
    No marker shown when all DOF are locked.
    """
    glyph_size = span * 0.08
    for axis, vec_tuple in [("x", (1, 0, 0)), ("y", (0, 1, 0)), ("z", (0, 0, 1))]:
        vec = np.array(vec_tuple)
        is_locked = dof_lock.get(axis, True)

        if not is_locked:
            # Draw a double-headed arrow (free) - green only
            arrow_len = glyph_size * 0.7
            p_tip = centroid + vec * arrow_len
            p_tail = centroid - vec * arrow_len
            # Main arrow shaft
            ax.plot(
                [p_tail[0], p_tip[0]],
                [p_tail[1], p_tip[1]],
                [p_tail[2], p_tip[2]],
                color="#27AE60",
                lw=2.1,
                zorder=10,
            )
            # Add larger markers at ends
            ax.scatter(
                [p_tip[0], p_tail[0]],
                [p_tip[1], p_tail[1]],
                [p_tip[2], p_tail[2]],
                color="#27AE60",
                s=52,
                zorder=11,
                marker="o",
            )


def _draw_dof_glyph_2d(
    ax: plt.Axes,
    centroid: np.ndarray,
    dof_lock: Dict[str, bool],
    axis_h: str,
    axis_v: str,
    span: float,
    fontsize: float = 6.0,
) -> None:
    """
    Draw 2-D DOF indicators showing only the in-plane axes.
    Only show free DOF: green double-headed arrows (↔).
    No marker shown when all DOF are locked.
    Projects the 3-D DOF state onto the 2-D plane (axis_h, axis_v).
    """
    idx_map = {"x": 0, "y": 1, "z": 2}
    idx_h = idx_map[axis_h]
    idx_v = idx_map[axis_v]

    glyph_size = span * 0.08

    # Show DOF for only the in-plane axes (free DOF only)
    for axis in [axis_h, axis_v]:
        is_locked = dof_lock.get(axis, True)
        vec_3d = np.array([1 if a == axis else 0 for a in ["x", "y", "z"]])
        vec_2d = np.array([vec_3d[idx_h], vec_3d[idx_v]])

        if not is_locked:
            # Double-headed arrow - green (free DOF only)
            arrow_len = glyph_size * 0.7
            center = centroid[[idx_h, idx_v]]
            p_tip = center + vec_2d * arrow_len
            p_tail = center - vec_2d * arrow_len
            ax.plot(
                [p_tail[0], p_tip[0]],
                [p_tail[1], p_tip[1]],
                color="#27AE60",
                lw=2.1,
                zorder=10,
            )
            ax.plot(
                [p_tip[0]], [p_tip[1]], "o", color="#27AE60", markersize=6.5, zorder=11
            )
            ax.plot(
                [p_tail[0]],
                [p_tail[1]],
                "o",
                color="#27AE60",
                markersize=6.5,
                zorder=11,
            )


# ── 2-D projection helper ──────────────────────────────────────────────────────


def _draw_box_2d(
    ax: plt.Axes,
    b: Dict[str, float],
    axis_h: str,
    axis_v: str,
    edge_color: str,
    face_color: str,
    alpha: float,
    lw: float = 1.0,
    hatch: Optional[str] = None,
    linestyle: str = "-",
) -> None:
    """Draw a rectangle in 2-D, projecting the given pair of axes."""
    lo_h = b[f"{axis_h}_min"]
    hi_h = b[f"{axis_h}_max"]
    lo_v = b[f"{axis_v}_min"]
    hi_v = b[f"{axis_v}_max"]
    rect = mpatches.FancyBboxPatch(
        (lo_h, lo_v),
        max(hi_h - lo_h, 1e-6),
        max(hi_v - lo_v, 1e-6),
        boxstyle="square,pad=0",
        linewidth=lw,
        edgecolor=edge_color,
        facecolor=face_color,
        alpha=alpha,
        hatch=hatch or "",
        linestyle=linestyle,
        zorder=3,
    )
    ax.add_patch(rect)


def _draw_arrow_2d(
    ax: plt.Axes,
    centroid: np.ndarray,
    vec_2d: np.ndarray,
    arrow_len: float,
    color: str,
    label: str,
    font_size: float = 6.5,
) -> None:
    """Draw a boldly styled 2-D force arrow with a label."""
    norm = np.linalg.norm(vec_2d)
    if norm < 1e-10:
        return
    vec_2d = vec_2d / norm
    dx, dy = vec_2d * arrow_len
    ax.annotate(
        "",
        xy=(centroid[0] + dx, centroid[1] + dy),
        xytext=(centroid[0], centroid[1]),
        arrowprops=dict(
            arrowstyle="-|>",
            color=color,
            lw=1.5,
            mutation_scale=10,
        ),
        zorder=6,
    )
    if label:
        ax.text(
            centroid[0] + dx * 1.12,
            centroid[1] + dy * 1.12,
            label,
            fontsize=font_size,
            color=color,
            ha="center",
            va="center",
            fontweight="bold",
            zorder=7,
        )


# ── Core rendering ─────────────────────────────────────────────────────────────


def render_3d(
    ax: Axes3D,
    domain: Dict[str, float],
    shape_type: str,
    shape_params: Dict[str, Any],
    selectors: Dict[str, Dict[str, float]],
    bcs: List[Dict[str, Any]],
    loads: List[Dict[str, Any]],
    annotate: bool = True,
    equal_axes: bool = False,
    show_grid: bool = True,
    label_offset: float = 1.30,
    label_font_size: float = 7.5,
) -> None:
    """Populate a 3-D axes with the full load-case visualization."""

    span = max(
        domain["x_max"] - domain["x_min"],
        domain["y_max"] - domain["y_min"],
        domain["z_max"] - domain["z_min"],
    )

    # ── Design domain ─────────────────────────────────────────────────────────
    if shape_type == "cylinder":
        _draw_cylinder_3d(
            ax,
            shape_params,
            edge_color=C_DOMAIN,
            face_color=C_DOMAIN_F,
            alpha_face=ALPHA_DOMAIN,
            lw=1.4,
            zorder=1,
        )
    elif shape_type == "l_shape":
        width = float(shape_params["width"])
        height = float(shape_params["height"])
        depth = float(shape_params["depth"])
        leg_width = float(shape_params["leg_width"])
        base_height = float(shape_params["base_height"])

        vertical_leg = {
            "x_min": 0.0,
            "x_max": leg_width,
            "y_min": 0.0,
            "y_max": depth,
            "z_min": 0.0,
            "z_max": height,
        }
        base_leg = {
            "x_min": 0.0,
            "x_max": width,
            "y_min": 0.0,
            "y_max": depth,
            "z_min": 0.0,
            "z_max": base_height,
        }
        _draw_box_3d(
            ax,
            vertical_leg,
            edge_color=C_DOMAIN,
            face_color=C_DOMAIN_F,
            alpha_face=ALPHA_DOMAIN,
            lw=1.4,
            linestyle="--",
            zorder=1,
        )
        _draw_box_3d(
            ax,
            base_leg,
            edge_color=C_DOMAIN,
            face_color=C_DOMAIN_F,
            alpha_face=ALPHA_DOMAIN,
            lw=1.4,
            linestyle="--",
            zorder=1,
        )
    elif shape_type == "t_shape":
        width = float(shape_params["width"])
        height = float(shape_params["height"])
        depth = float(shape_params["depth"])
        stem_width = float(shape_params["stem_width"])
        cap_height = float(shape_params["cap_height"])

        stem_x_min = (width - stem_width) / 2.0
        stem_x_max = (width + stem_width) / 2.0
        cap = {
            "x_min": 0.0,
            "x_max": width,
            "y_min": 0.0,
            "y_max": depth,
            "z_min": height - cap_height,
            "z_max": height,
        }
        stem = {
            "x_min": stem_x_min,
            "x_max": stem_x_max,
            "y_min": 0.0,
            "y_max": depth,
            "z_min": 0.0,
            "z_max": height - cap_height,
        }
        _draw_box_3d(
            ax,
            cap,
            edge_color=C_DOMAIN,
            face_color=C_DOMAIN_F,
            alpha_face=ALPHA_DOMAIN,
            lw=1.4,
            linestyle="--",
            zorder=1,
        )
        _draw_box_3d(
            ax,
            stem,
            edge_color=C_DOMAIN,
            face_color=C_DOMAIN_F,
            alpha_face=ALPHA_DOMAIN,
            lw=1.4,
            linestyle="--",
            zorder=1,
        )
    elif shape_type == "step":
        # STEP geometry is represented by its validated domain bounds in this matplotlib renderer.
        _draw_box_3d(
            ax,
            domain,
            edge_color=C_DOMAIN,
            face_color=C_DOMAIN_F,
            alpha_face=ALPHA_DOMAIN,
            lw=1.4,
            linestyle="--",
            zorder=1,
        )
    else:
        _draw_box_3d(
            ax,
            domain,
            edge_color=C_DOMAIN,
            face_color=C_DOMAIN_F,
            alpha_face=ALPHA_DOMAIN,
            lw=1.4,
            linestyle="--",
            zorder=1,
        )

    # Dimension annotations on three outer edges
    d = domain
    dim_offset_scale = 1.0 + 0.45 * max(label_offset - 1.0, 0.0)
    off_y = np.array([0, span * 0.18 * dim_offset_scale, 0])
    off_x = np.array([span * 0.16 * dim_offset_scale, 0, 0])
    _draw_dim_bar(
        ax,
        np.array([d["x_min"], d["y_min"], d["z_min"]]),
        np.array([d["x_max"], d["y_min"], d["z_min"]]),
        _fmt_mm_dim(d["x_max"] - d["x_min"]),
        -off_y,
        fontsize=label_font_size,
    )
    _draw_dim_bar(
        ax,
        np.array([d["x_max"], d["y_min"], d["z_min"]]),
        np.array([d["x_max"], d["y_max"], d["z_min"]]),
        _fmt_mm_dim(d["y_max"] - d["y_min"]),
        off_x,
        fontsize=label_font_size,
    )
    _draw_dim_bar(
        ax,
        np.array([d["x_max"], d["y_max"], d["z_min"]]),
        np.array([d["x_max"], d["y_max"], d["z_max"]]),
        _fmt_mm_dim(d["z_max"] - d["z_min"]),
        off_x,
        fontsize=label_font_size,
    )

    # ── Fixed constraints ─────────────────────────────────────────────────────
    for bc in bcs:
        sel_id = bc.get("region_id", "")
        if sel_id not in selectors:
            continue
        _draw_box_3d(
            ax,
            selectors[sel_id],
            edge_color=C_CONSTRAINT,
            face_color=C_CONSTRAINT,
            alpha_face=ALPHA_BC,
            lw=1.4,
            hatch="////",
            zorder=3,
        )
        # Draw DOF glyph at constraint centroid
        dof_lock = _get_dof_lock_state(bc)
        centroid = _box_centroid(selectors[sel_id])
        _draw_dof_glyph_3d(ax, centroid, dof_lock, span, fontsize=label_font_size)

    # ── Loads ─────────────────────────────────────────────────────────────────
    arrow_len = span * 0.20
    scene = _compute_scene_bounds(domain, selectors, loads, arrow_len, label_offset)
    used_label_positions_3d: List[np.ndarray] = []

    for load in loads:
        direction = load.get("direction", "z")
        magnitude = load.get("magnitude_newtons", 0.0)
        is_dist = "distributed" in load.get("type", "").lower()

        if _is_pressure_load(load) and "center" in load and "radius" in load:
            if _draw_pressure_load_3d(
                ax,
                load,
                arrow_len,
                label_offset,
                label_font_size,
                used_label_positions_3d,
                span,
            ):
                continue

        sel = _resolve_load_selector_box(load, selectors, span)
        if sel is None:
            continue

        centroid = _box_centroid(sel)

        if _is_moment_load(load):
            # Keep region highlight but render moment as a straight, axis-aligned,
            # double-headed arrow (couple) through the load center.
            moment_center = load.get("center", None)
            if isinstance(moment_center, list) and len(moment_center) == 3:
                centroid = np.array(moment_center, dtype=float)
            if _draw_moment_load_3d(
                ax,
                load,
                centroid,
                arrow_len,
                label_offset,
                label_font_size,
                used_label_positions_3d,
                span,
            ):
                continue

        vec = _dir_to_vec(direction)
        label = _fmt_force(magnitude) if not is_dist else _fmt_pressure(magnitude)

        # Shaded load region
        _draw_box_3d(
            ax,
            sel,
            edge_color=C_LOAD,
            face_color=C_LOAD_PATCH,
            alpha_face=ALPHA_LOAD,
            lw=1.2,
            zorder=3,
        )

        # Force arrow
        ax.quiver(
            centroid[0],
            centroid[1],
            centroid[2],
            vec[0],
            vec[1],
            vec[2],
            length=arrow_len,
            color=C_ARROW,
            arrow_length_ratio=0.22,
            linewidth=2.5,
            zorder=7,
        )
        label_pos = centroid + vec * (arrow_len * label_offset)
        label_pos = _spread_label_position_3d(
            label_pos, vec, used_label_positions_3d, span, label_offset
        )
        used_label_positions_3d.append(label_pos)
        ax.text(
            label_pos[0],
            label_pos[1],
            label_pos[2],
            label,
            fontsize=label_font_size,
            color=C_ARROW,
            ha="center",
            va="center",
            fontweight="bold",
            zorder=8,
        )

    # ── Axes styling ──────────────────────────────────────────────────────────
    if annotate:
        ax.set_xlabel("$x$ [mm]", labelpad=3)
        ax.set_ylabel("$y$ [mm]", labelpad=3)
        ax.set_zlabel("$z$ [mm]", labelpad=3)
    else:
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_zlabel("")
    if show_grid:
        ax.tick_params(labelsize=6, pad=1)
    else:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])

    lx = scene["x_max"] - scene["x_min"]
    ly = scene["y_max"] - scene["y_min"]
    lz = scene["z_max"] - scene["z_min"]
    # Non-annotated views (iso/collage) can use much tighter framing.
    pad = span * (0.02 if not annotate else 0.22)
    if equal_axes:
        ax.set_box_aspect((1.0, 1.0, 1.0))

        # Keep equal scale across x/y/z by fitting all three axes into one cube.
        cx = (scene["x_min"] + scene["x_max"]) / 2
        cy = (scene["y_min"] + scene["y_max"]) / 2
        cz = (scene["z_min"] + scene["z_max"]) / 2
        half_extent = max(lx, ly, lz) / 2 + pad

        ax.set_xlim(cx - half_extent, cx + half_extent)
        ax.set_ylim(cy - half_extent, cy + half_extent)
        ax.set_zlim(cz - half_extent, cz + half_extent)
    else:
        ax.set_box_aspect([lx, ly, lz])
        ax.set_xlim(scene["x_min"] - pad, scene["x_max"] + pad)
        ax.set_ylim(scene["y_min"] - pad, scene["y_max"] + pad)
        ax.set_zlim(scene["z_min"] - pad, scene["z_max"] + pad)

    ax.view_init(elev=22, azim=-55)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    axes_3d = (ax.xaxis, ax.yaxis, ax.zaxis)
    if show_grid:
        ax.xaxis.pane.set_edgecolor("#cccccc")
        ax.yaxis.pane.set_edgecolor("#cccccc")
        ax.zaxis.pane.set_edgecolor("#cccccc")
        for axis in axes_3d:
            axis._axinfo["grid"].update(
                linewidth=0.3,
                linestyle=":",
                color="#dddddd",
            )
    else:
        transparent = (1.0, 1.0, 1.0, 0.0)
        ax.xaxis.pane.set_edgecolor(transparent)
        ax.yaxis.pane.set_edgecolor(transparent)
        ax.zaxis.pane.set_edgecolor(transparent)
        for axis in axes_3d:
            axis._axinfo["grid"].update(linewidth=0.0, color=transparent)
            axis.line.set_color(transparent)

    ax.grid(show_grid, which="major", lw=0.3, color="#dddddd", linestyle=":")


def render_2d(
    ax: plt.Axes,
    domain: Dict[str, float],
    shape_type: str,
    shape_params: Dict[str, Any],
    selectors: Dict[str, Dict[str, float]],
    bcs: List[Dict[str, Any]],
    loads: List[Dict[str, Any]],
    axis_h: str,
    axis_v: str,
    title: str,
    show_grid: bool = True,
    show_title: bool = True,
    label_offset: float = 1.30,
    label_font_size: float = 6.5,
    title_font_size: float = 7.5,
) -> None:
    """Populate a 2-D projection axes."""

    span_h = domain[f"{axis_h}_max"] - domain[f"{axis_h}_min"]
    span_v = domain[f"{axis_v}_max"] - domain[f"{axis_v}_min"]
    arrow_len = max(span_h, span_v) * 0.22
    scene = _compute_scene_bounds(domain, selectors, loads, arrow_len, label_offset)
    used_label_positions_2d: List[np.ndarray] = []

    # Design domain
    _draw_domain_2d(
        ax,
        domain,
        shape_type,
        shape_params,
        axis_h,
        axis_v,
        C_DOMAIN,
        C_DOMAIN_F,
        ALPHA_DOMAIN * 1.5,
        lw=1.1,
        linestyle="--",
    )

    # BCs
    for bc in bcs:
        sel_id = bc.get("region_id", "")
        if sel_id not in selectors:
            continue
        _draw_box_2d(
            ax,
            selectors[sel_id],
            axis_h,
            axis_v,
            C_CONSTRAINT,
            C_CONSTRAINT,
            ALPHA_BC,
            lw=1.2,
            hatch="////",
        )
        # Draw DOF glyph at constraint centroid (2D projection)
        dof_lock = _get_dof_lock_state(bc)
        centroid = _box_centroid(selectors[sel_id])
        _draw_dof_glyph_2d(ax, centroid, dof_lock, axis_h, axis_v, max(span_h, span_v))

    # Loads
    for load in loads:
        direction = load.get("direction", "z")
        magnitude = load.get("magnitude_newtons", 0.0)
        is_dist = "distributed" in load.get("type", "").lower()

        if _is_pressure_load(load) and "center" in load and "radius" in load:
            if _draw_pressure_load_2d(
                ax,
                load,
                axis_h,
                axis_v,
                arrow_len,
                label_font_size,
                label_offset,
                used_label_positions_2d,
                max(span_h, span_v),
            ):
                continue

        sel = _resolve_load_selector_box(load, selectors, max(span_h, span_v))
        if sel is None:
            continue

        _draw_box_2d(
            ax,
            sel,
            axis_h,
            axis_v,
            C_LOAD,
            C_LOAD_PATCH,
            ALPHA_LOAD * 1.5,
            lw=1.0,
        )

        centroid = _box_centroid(sel)
        center2 = np.array(
            [
                centroid[{"x": 0, "y": 1, "z": 2}[axis_h]],
                centroid[{"x": 0, "y": 1, "z": 2}[axis_v]],
            ]
        )

        if _is_moment_load(load):
            moment_center = load.get("center", None)
            if isinstance(moment_center, list) and len(moment_center) == 3:
                c3 = np.array(moment_center, dtype=float)
                center2 = np.array(
                    [
                        c3[{"x": 0, "y": 1, "z": 2}[axis_h]],
                        c3[{"x": 0, "y": 1, "z": 2}[axis_v]],
                    ]
                )
            if _draw_moment_load_2d(
                ax,
                load,
                center2,
                axis_h,
                axis_v,
                arrow_len,
                label_font_size,
                label_offset,
                used_label_positions_2d,
                max(span_h, span_v),
            ):
                continue

        vec3 = _dir_to_vec(direction)
        idx_map = {"x": 0, "y": 1, "z": 2}
        idx_h = idx_map[axis_h]
        idx_v = idx_map[axis_v]
        vec2 = np.array([vec3[idx_h], vec3[idx_v]])
        center2 = np.array([centroid[idx_h], centroid[idx_v]])
        label = _fmt_force(magnitude) if not is_dist else _fmt_pressure(magnitude)
        norm2 = np.linalg.norm(vec2)
        unit2 = vec2 / norm2 if norm2 > 1e-10 else np.array([0.0, 1.0])
        label_pos = center2 + unit2 * (arrow_len * label_offset)
        label_pos = _spread_label_position_2d(
            label_pos, unit2, used_label_positions_2d, max(span_h, span_v), label_offset
        )
        used_label_positions_2d.append(label_pos)
        _draw_arrow_2d(
            ax, center2, vec2, arrow_len, C_ARROW, "", font_size=label_font_size
        )
        ax.text(
            label_pos[0],
            label_pos[1],
            label,
            fontsize=label_font_size,
            color=C_ARROW,
            ha="center",
            va="center",
            fontweight="bold",
            zorder=7,
        )

    # Styling
    pad_h = span_h * 0.30
    pad_v = span_v * 0.30
    ax.set_xlim(scene[f"{axis_h}_min"] - pad_h, scene[f"{axis_h}_max"] + pad_h)
    ax.set_ylim(scene[f"{axis_v}_min"] - pad_v, scene[f"{axis_v}_max"] + pad_v)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(f"${axis_h}$ [mm]", labelpad=2)
    ax.set_ylabel(f"${axis_v}$ [mm]", labelpad=2)
    if show_title:
        ax.set_title(title, fontsize=title_font_size, pad=3)
    for spine in ax.spines.values():
        spine.set_linewidth(0.7)
    ax.tick_params(labelsize=6)
    ax.grid(show_grid, which="major", lw=0.3, color="#dddddd", linestyle=":")


def _bounds_from_load_case(load_case: Any) -> Dict[str, float]:
    domain = getattr(load_case, "domain", None)
    if domain is not None and hasattr(domain, "get_bounding_box"):
        bounds = domain.get_bounding_box()
        if bounds:
            return dict(bounds)

    bounds = getattr(load_case, "bounds", None)
    if bounds:
        return dict(bounds)

    raise ValueError("Load case is missing bounds/domain information for visualization")


def _design_domain_from_load_case(
    load_case: Any, bounds: Dict[str, float]
) -> Dict[str, Any]:
    domain = getattr(load_case, "domain", None)
    if domain is None:
        return {"bounds": bounds, "shape_type": "box"}

    design_domain: Dict[str, Any] = {
        "bounds": dict(getattr(domain, "bounds", None) or bounds),
        "shape_type": str(getattr(domain, "shape_type", "box") or "box"),
    }
    params = getattr(domain, "params", None)
    if params:
        design_domain["params"] = dict(params)
    return design_domain


def _query_from_location(location: Any, bounds: Dict[str, float]) -> Dict[str, float]:
    if isinstance(location, dict):
        return dict(location)

    if isinstance(location, (tuple, list)) and len(location) == 3:
        return {
            "x": float(location[0]),
            "y": float(location[1]),
            "z": float(location[2]),
        }

    if not isinstance(location, str):
        raise ValueError(
            f"Unsupported selector location type: {type(location).__name__}"
        )

    key = location.lower().strip()
    aliases = {
        "end_1": "x_min",
        "end_2": "x_max",
        "bottom": "z_min",
        "top": "z_max",
        "face_left": "x_min",
        "face_right": "x_max",
        "face_front": "y_max",
    }
    key = aliases.get(key, key)
    if key not in {"x_min", "x_max", "y_min", "y_max", "z_min", "z_max"}:
        raise ValueError(f"Unsupported named selector location: {location!r}")

    axis = key[0]
    side = key[-3:]
    value = bounds[f"{axis}_{side}"]
    return {f"{axis}_min": value, f"{axis}_max": value}


def _query_from_point(point: Any, search_radius: Optional[Any]) -> Dict[str, float]:
    if isinstance(point, dict):
        query = dict(point)
        if (
            search_radius is None
            and all(k in query for k in ("rx", "ry", "rz"))
            and all(k in query for k in ("x", "y", "z"))
        ):
            return {
                "x_min": float(query["x"]) - float(query["rx"]),
                "x_max": float(query["x"]) + float(query["rx"]),
                "y_min": float(query["y"]) - float(query["ry"]),
                "y_max": float(query["y"]) + float(query["ry"]),
                "z_min": float(query["z"]) - float(query["rz"]),
                "z_max": float(query["z"]) + float(query["rz"]),
            }
        return query

    if isinstance(point, (tuple, list)) and len(point) == 3:
        x, y, z = (float(point[0]), float(point[1]), float(point[2]))
        if search_radius is None:
            return {"x": x, "y": y, "z": z}
        rx, ry, rz = (
            float(search_radius[0]),
            float(search_radius[1]),
            float(search_radius[2]),
        )
        return {
            "x_min": x - rx,
            "x_max": x + rx,
            "y_min": y - ry,
            "y_max": y + ry,
            "z_min": z - rz,
            "z_max": z + rz,
        }

    raise ValueError(f"Unsupported point selector type: {type(point).__name__}")


def _register_selector(
    selectors: Dict[str, Dict[str, Any]],
    base_id: str,
    query: Dict[str, float],
) -> str:
    stem = "_".join(str(base_id or "selector").split())
    selector_id = stem
    suffix = 1
    while selector_id in selectors:
        selector_id = f"{stem}_{suffix}"
        suffix += 1
    selectors[selector_id] = {
        "id": selector_id,
        "type": (
            "point"
            if all(k in query for k in ("x", "y", "z"))
            and not any(k.endswith("_min") for k in query)
            else "box_3d"
        ),
        "query": query,
    }
    return selector_id


def _axis_aligned_force(force: Any, direction: Optional[str]) -> Tuple[float, str]:
    if isinstance(force, (int, float)):
        if not direction:
            raise ValueError(
                "Scalar loads require an explicit direction for visualization"
            )
        return abs(float(force)), str(direction).lower().strip()

    vec = np.asarray(force, dtype=float)
    if vec.shape != (3,):
        raise ValueError(f"Unsupported force vector shape: {vec.shape}")
    nonzero = np.flatnonzero(np.abs(vec) > 1e-10)
    if len(nonzero) != 1:
        raise ValueError(
            "Only axis-aligned vector loads are supported by conditions visualization"
        )

    axis = "xyz"[int(nonzero[0])]
    component = float(vec[int(nonzero[0])])
    return abs(component), axis if component >= 0.0 else f"-{axis}"


def _coerce_visualization_data(data: Any) -> Dict[str, Any]:
    if isinstance(data, dict):
        return data

    load_case = getattr(data, "load_case", data)
    if load_case is None:
        raise ValueError("No load case is available for conditions visualization")

    bounds = _bounds_from_load_case(load_case)
    selectors: Dict[str, Dict[str, Any]] = {}

    existing_selectors = getattr(load_case, "selectors", None) or {}
    if isinstance(existing_selectors, dict):
        for key, selector in existing_selectors.items():
            query = getattr(selector, "query", None)
            if isinstance(query, dict):
                selector_id = str(getattr(selector, "id", None) or key)
                selectors[selector_id] = {
                    "id": selector_id,
                    "type": str(getattr(selector, "type", "box_3d")),
                    "query": dict(query),
                }

    boundary_conditions: List[Dict[str, Any]] = []
    for index, constraint in enumerate(getattr(load_case, "constraints", [])):
        region_id = getattr(constraint, "region_id", None)
        if region_id is None:
            location = getattr(constraint, "location", None)
            if location is None:
                raise ValueError(
                    f"Unsupported constraint type for conditions visualization: {type(constraint).__name__}"
                )
            region_id = _register_selector(
                selectors,
                f"constraint_{index}",
                _query_from_location(location, bounds),
            )
        elif region_id not in selectors:
            location = getattr(constraint, "location", None)
            if location is None:
                raise ValueError(
                    f"Constraint {type(constraint).__name__} references selector {region_id!r} but no selector data is available"
                )
            region_id = _register_selector(
                selectors, str(region_id), _query_from_location(location, bounds)
            )

        dofs = tuple(bool(v) for v in getattr(constraint, "dofs", (True, True, True)))
        boundary_conditions.append(
            {
                "name": getattr(constraint, "name", type(constraint).__name__),
                "type": "fixed_displacement",
                "region_id": region_id,
                "dof_lock": {"x": dofs[0], "y": dofs[1], "z": dofs[2]},
            }
        )

    loads: List[Dict[str, Any]] = []
    for index, load in enumerate(getattr(load_case, "loads", [])):
        load_type = type(load).__name__
        load_name = getattr(load, "name", f"load_{index}")

        if load_type == "AccelerationLoad":
            acc_kind = str(getattr(load, "load_type", "")).lower().strip()
            if acc_kind in {"gravity", "body_force"}:
                # Body accelerations act over the full domain; represent them as
                # a distributed force proxy so publication rendering remains stable.
                direction_vec = getattr(load, "direction", None)
                if direction_vec is None:
                    raise ValueError(
                        "AccelerationLoad for conditions visualization requires a direction vector"
                    )
                magnitude, direction = _axis_aligned_force(
                    [
                        float(getattr(load, "magnitude", 0.0)) * float(v)
                        for v in direction_vec
                    ],
                    None,
                )
                region_id = _register_selector(selectors, f"load_{index}", dict(bounds))
                loads.append(
                    {
                        "name": load_name,
                        "type": "distributed_force",
                        "region_id": region_id,
                        "direction": direction,
                        "magnitude_newtons": magnitude,
                    }
                )
                continue

            if acc_kind == "centrifugal":
                raise ValueError(
                    "Unsupported AccelerationLoad subtype for conditions visualization: centrifugal"
                )

            raise ValueError(
                f"Unsupported AccelerationLoad subtype for conditions visualization: {acc_kind or 'unknown'}"
            )

        if load_type == "MomentLoad":
            loads.append(
                {
                    "name": load_name,
                    "type": "moment",
                    "center": list(getattr(load, "center")),
                    "axis": str(getattr(load, "axis", "z")).lower(),
                    "radius": float(getattr(load, "radius")),
                    "magnitude_nm": float(getattr(load, "magnitude_nm")),
                    "direction": str(getattr(load, "direction", "cw")).lower(),
                }
            )
            continue

        if load_type == "PressureLoad":
            loads.append(
                {
                    "name": load_name,
                    "type": "pressure",
                    "center": list(getattr(load, "center")),
                    "radius": float(getattr(load, "radius")),
                    "pressure_mpa": float(getattr(load, "pressure")),
                    "direction": str(getattr(load, "direction", "inward")).lower(),
                    "normal_axis": str(getattr(load, "normal_axis", "z")).lower(),
                }
            )
            continue

        if load_type in {
            "PointLoad",
            "ConcentratedLoad",
            "DistributedLoad",
            "LinearDistributedLoad",
        }:
            region_id = getattr(load, "region_id", None)
            if region_id is None:
                if hasattr(load, "point"):
                    query = _query_from_point(
                        getattr(load, "point"), getattr(load, "search_radius", None)
                    )
                elif hasattr(load, "location"):
                    query = _query_from_location(getattr(load, "location"), bounds)
                else:
                    raise ValueError(
                        f"Unsupported load type for conditions visualization: {load_type}"
                    )
                region_id = _register_selector(selectors, f"load_{index}", query)
            elif region_id not in selectors:
                if hasattr(load, "point"):
                    query = _query_from_point(
                        getattr(load, "point"), getattr(load, "search_radius", None)
                    )
                elif hasattr(load, "location"):
                    query = _query_from_location(getattr(load, "location"), bounds)
                else:
                    raise ValueError(
                        f"Load {load_type} references selector {region_id!r} but no selector data is available"
                    )
                region_id = _register_selector(selectors, str(region_id), query)

            magnitude, direction = _axis_aligned_force(
                getattr(load, "force", None),
                getattr(load, "direction", None),
            )
            loads.append(
                {
                    "name": load_name,
                    "type": "force",
                    "region_id": region_id,
                    "direction": direction,
                    "magnitude_newtons": magnitude,
                }
            )
            continue

        raise ValueError(
            f"Unsupported load type for conditions visualization: {load_type}"
        )

    meta = dict(getattr(load_case, "meta", None) or {})
    meta.setdefault(
        "problem_id", getattr(load_case, "problem_id", "Load Case") or "Load Case"
    )
    meta.setdefault("description", getattr(load_case, "description", "") or "")
    meta.setdefault("analysis_type", getattr(load_case, "analysis_type", "3d") or "3d")

    material = getattr(load_case, "material", None)
    material_dict: Dict[str, Any] = {}
    if material is not None:
        material_dict = {
            "type": getattr(material, "name", "material"),
            "elastic_modulus_mpa": getattr(material, "elastic_modulus", None),
            "poissons_ratio": getattr(material, "poissons_ratio", None),
            "density_g_cm3": getattr(material, "density", None),
        }

    return {
        "meta": meta,
        "design_domain": _design_domain_from_load_case(load_case, bounds),
        "spatial_selectors": list(selectors.values()),
        "boundary_conditions": boundary_conditions,
        "loads": loads,
        "material": material_dict,
    }


def render_legend(ax: plt.Axes, meta: Dict[str, Any], mat: Dict[str, Any]) -> None:
    """Render a meta-data / legend panel (no axes, pure text + patches)."""
    ax.axis("off")

    pid = meta.get("problem_id", "Load Case")
    descr = meta.get("description", "")
    atype = meta.get("analysis_type", "3d").upper()

    # Title block
    ax.text(
        0.0,
        1.00,
        pid.replace("_", " "),
        fontsize=8.5,
        fontweight="bold",
        va="top",
        transform=ax.transAxes,
    )
    ax.text(
        0.0,
        0.89,
        descr,
        fontsize=6.5,
        va="top",
        transform=ax.transAxes,
        color="#444444",
    )

    # Material block
    ax.text(
        0.0,
        0.74,
        "Material properties",
        fontsize=7.5,
        fontweight="semibold",
        va="top",
        transform=ax.transAxes,
    )

    E = mat.get("elastic_modulus_mpa", None)
    nu = mat.get("poissons_ratio", "—")
    rho = mat.get("density_g_cm3", "—")
    mtype = mat.get("type", "—").capitalize()

    E_str = f"{E/1000:.0f} GPa" if isinstance(E, (int, float)) else "—"
    lines = [
        f"Type:       {mtype}",
        f"$E$    = {E_str}",
        f"$\\nu$    = {nu}",
        f"$\\rho$   = {rho} g/cm³",
        f"Analysis: {atype}",
    ]
    for k, line in enumerate(lines):
        ax.text(
            0.03,
            0.68 - k * 0.088,
            line,
            fontsize=7,
            va="top",
            transform=ax.transAxes,
            color="#222222",
        )

    # Legend swatches
    y0 = 0.22
    ax.text(
        0.0,
        y0,
        "Legend",
        fontsize=7.5,
        fontweight="semibold",
        va="top",
        transform=ax.transAxes,
    )

    items = [
        (C_DOMAIN, None, C_DOMAIN, "Design domain (dashed)"),
        (C_CONSTRAINT, "////", C_CONSTRAINT, "Fixed constraint (hatched)"),
        (C_LOAD_PATCH, None, C_LOAD, "Load application region"),
        (C_ARROW, None, C_ARROW, "Applied force (arrow)"),
    ]
    for k, (fcolor, hatch, ecolor, label) in enumerate(items):
        y_swatch = y0 - 0.12 - k * 0.10
        rect = mpatches.FancyBboxPatch(
            (0.0, y_swatch),
            0.10,
            0.065,
            boxstyle="square,pad=0",
            linewidth=0.8,
            edgecolor=ecolor,
            facecolor=fcolor,
            alpha=0.65,
            hatch=hatch or "",
            transform=ax.transAxes,
            clip_on=False,
        )
        ax.add_patch(rect)
        ax.text(
            0.15,
            y_swatch + 0.032,
            label,
            fontsize=6.5,
            va="center",
            transform=ax.transAxes,
            color="#222222",
        )

    # DOF annotation
    ax.text(
        0.0,
        -0.08,
        "DOF: ↔ free (green shown)",
        fontsize=5.5,
        va="top",
        transform=ax.transAxes,
        color="#666666",
        style="italic",
    )


# ── Main figure assembly ──────────────────────────────────────────────────────


def visualize_load_case(
    data: Any,
    json_path: Optional[Path] = None,
    output_png: Optional[str] = None,
    output_pdf: Optional[str] = None,
    iso_only: bool = False,
    equal_axes: bool = False,
    show_grid: bool = True,
    show_title: bool = True,
    label_offset: float = 1.30,
    font_size: float = 7.5,
    title_font_size: Optional[float] = None,
) -> plt.Figure:
    """
    Build and return the publication-quality figure.

    Args:
        data:       Parsed JSON dict of the load case.
        output_png: If given, saves a 300-DPI PNG to this path.
        output_pdf: If given, saves a vector PDF to this path.
        iso_only:   If True, output only the isometric 3-D panel (useful
                    for assembling a multi-case collage).
        equal_axes: If True, render the 3-D panel with matched x/y/z scales.
        show_grid:  If True, show axis grids on rendered panels.
        show_title: If True, show figure and panel titles.
        label_offset: Multiplier for placing load labels farther from glyphs and
                  for increasing collision-avoidance spreading.
        font_size: Font size for dimension (mm) and force magnitude labels.
        title_font_size: Font size for titles (panel + figure).

    Returns:
        matplotlib.figure.Figure
    """
    data = _coerce_visualization_data(data)

    meta = data.get("meta", {})
    design_domain = data.get("design_domain", {})
    domain, shape_type, shape_params = _domain_from_design_domain(
        design_domain, json_path=json_path
    )
    raw_sel = data.get("spatial_selectors", [])
    bcs = data.get("boundary_conditions", [])
    loads = data.get("loads", [])
    mat = data.get("material", {})

    # Build id → bounds mapping for selectors
    selectors: Dict[str, Dict[str, float]] = {}
    for s in raw_sel:
        q = dict(s.get("query", {}))
        # Normalise point selectors (x,y,z) to tiny boxes
        if "x" in q and "x_min" not in q:
            eps = 0.5
            q = {
                "x_min": q["x"] - eps,
                "x_max": q["x"] + eps,
                "y_min": q.get("y", 0) - eps,
                "y_max": q.get("y", 0) + eps,
                "z_min": q.get("z", 0) - eps,
                "z_max": q.get("z", 0) + eps,
            }
        # Fill any missing axis bounds from domain (e.g. face/slice selectors)
        for axis in ("x", "y", "z"):
            if f"{axis}_min" not in q:
                q[f"{axis}_min"] = domain.get(f"{axis}_min", 0.0)
            if f"{axis}_max" not in q:
                q[f"{axis}_max"] = domain.get(f"{axis}_max", 0.0)
        selectors[s["id"]] = q

    pid = meta.get("problem_id", "Load Case").replace("_", " ")
    panel_title_size = title_font_size if title_font_size is not None else 7.5
    iso_title_size = title_font_size if title_font_size is not None else 9.0
    fig_title_size = title_font_size if title_font_size is not None else 11.0

    if iso_only:
        # ── Single-panel isometric figure ─────────────────────────────────────
        fig = plt.figure(figsize=(5.0, 5.0))
        ax_3d = fig.add_subplot(111, projection="3d")
        render_3d(
            ax_3d,
            domain,
            shape_type,
            shape_params,
            selectors,
            bcs,
            loads,
            annotate=False,
            equal_axes=equal_axes,
            show_grid=show_grid,
            label_offset=label_offset,
            label_font_size=font_size,
        )
        fig.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0)
    else:
        # ── Full 5-panel figure ────────────────────────────────────────────────
        fig = plt.figure(figsize=(11.0, 6.0))
        gs = GridSpec(
            2,
            3,
            figure=fig,
            width_ratios=[2.5, 1.0, 1.0],
            height_ratios=[1.0, 1.0],
            hspace=0.40,
            wspace=0.34,
            left=0.04,
            right=0.97,
            top=0.91,
            bottom=0.07,
        )

        ax_3d = fig.add_subplot(gs[:, 0], projection="3d")
        ax_top = fig.add_subplot(gs[0, 1])
        ax_front = fig.add_subplot(gs[0, 2])
        ax_side = fig.add_subplot(gs[1, 1])
        ax_leg = fig.add_subplot(gs[1, 2])

        # ── 3-D view ──────────────────────────────────────────────────────────
        render_3d(
            ax_3d,
            domain,
            shape_type,
            shape_params,
            selectors,
            bcs,
            loads,
            equal_axes=equal_axes,
            show_grid=show_grid,
            label_offset=label_offset,
            label_font_size=font_size,
        )
        if show_title:
            ax_3d.set_title("Isometric View", fontsize=iso_title_size, pad=5)
            ax_3d.text2D(
                0.0,
                1.01,
                "(a)",
                transform=ax_3d.transAxes,
                fontsize=8,
                fontweight="bold",
                va="bottom",
            )

        # ── 2-D projections ───────────────────────────────────────────────────
        render_2d(
            ax_top,
            domain,
            shape_type,
            shape_params,
            selectors,
            bcs,
            loads,
            "x",
            "y",
            "Top view  ($xy$-plane)",
            show_grid=show_grid,
            show_title=show_title,
            label_offset=label_offset,
            label_font_size=font_size,
            title_font_size=panel_title_size,
        )
        render_2d(
            ax_front,
            domain,
            shape_type,
            shape_params,
            selectors,
            bcs,
            loads,
            "x",
            "z",
            "Front view  ($xz$-plane)",
            show_grid=show_grid,
            show_title=show_title,
            label_offset=label_offset,
            label_font_size=font_size,
            title_font_size=panel_title_size,
        )
        render_2d(
            ax_side,
            domain,
            shape_type,
            shape_params,
            selectors,
            bcs,
            loads,
            "y",
            "z",
            "Side view  ($yz$-plane)",
            show_grid=show_grid,
            show_title=show_title,
            label_offset=label_offset,
            label_font_size=font_size,
            title_font_size=panel_title_size,
        )

        if show_title:
            for letter, ax in zip(["(b)", "(c)", "(d)"], [ax_top, ax_front, ax_side]):
                ax.text(
                    -0.20,
                    1.05,
                    letter,
                    transform=ax.transAxes,
                    fontsize=8,
                    fontweight="bold",
                    va="bottom",
                )

        # ── Legend + meta panel ───────────────────────────────────────────────
        render_legend(ax_leg, meta, mat)

        # ── Figure title ─────────────────────────────────────────────────────
        if show_title:
            fig.suptitle(
                f"Load Case Specification — {pid}",
                fontsize=fig_title_size,
                fontweight="bold",
                y=0.97,
            )

        # Caption-style footnote
        fig.text(
            0.97,
            0.005,
            "Dimensions in mm  ·  Red hatching = fixed constraint  ·  Blue arrow = applied force",
            ha="right",
            va="bottom",
            fontsize=5.5,
            color="#777777",
        )

    # ── Save ──────────────────────────────────────────────────────────────────
    if output_png:
        Path(output_png).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_png, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"✓ Saved PNG  → {output_png}")
    if output_pdf:
        Path(output_pdf).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_pdf, format="pdf", bbox_inches="tight", facecolor="white")
        print(f"✓ Saved PDF  → {output_pdf}")

    return fig


# ── Collage assembly ──────────────────────────────────────────────────────────


def _load_bounds(json_path: Path) -> Optional[Dict[str, Any]]:
    """Return parsed JSON if the domain is renderable (bounds or supported shape)."""
    try:
        data = json.loads(json_path.read_text())
        _domain_from_design_domain(data.get("design_domain", {}), json_path=json_path)
        return data
    except Exception:
        pass
    return None


def _to_sentence_case(text: str) -> str:
    """Convert text to sentence case."""
    s = " ".join(text.split()).strip()
    if not s:
        return s
    s = s.lower()
    return s[0].upper() + s[1:]


def make_collage(
    json_paths: List[Path],
    output_png: str,
    cols: int = 8,
    equal_axes: bool = False,
    show_grid: bool = True,
    show_title: bool = True,
    label_offset: float = 1.30,
    font_size: float = 7.5,
    title_font_size: Optional[float] = None,
    cell_size: float = 3.0,
    zoom: float = 4.8,
) -> plt.Figure:
    """
    Render each load-case JSON as an isometric panel and arrange them in a grid.

    Args:
        json_paths:  Sorted list of JSON paths to include.
        output_png:  Output file path.
        cols:        Number of columns in the grid (default 8).
        cell_size:   Width and height of each cell in inches (default 3.0).
        show_title:  If True, place the problem_id as a caption below each cell.
        All other kwargs forwarded to render_3d().
    """
    n = len(json_paths)
    rows = (n + cols - 1) // cols

    fig = plt.figure(figsize=(cols * cell_size, rows * cell_size))

    for idx, jp in enumerate(json_paths):
        data = json.loads(jp.read_text())
        meta = data.get("meta", {})
        design_domain = data.get("design_domain", {})
        domain, shape_type, shape_params = _domain_from_design_domain(
            design_domain, json_path=jp
        )
        raw_sel = data.get("spatial_selectors", [])
        bcs = data.get("boundary_conditions", [])
        loads = data.get("loads", [])

        selectors: Dict[str, Dict[str, float]] = {}
        for s in raw_sel:
            q = dict(s.get("query", {}))
            if "x" in q and "x_min" not in q:
                eps = 0.5
                q = {
                    "x_min": q["x"] - eps,
                    "x_max": q["x"] + eps,
                    "y_min": q.get("y", 0) - eps,
                    "y_max": q.get("y", 0) + eps,
                    "z_min": q.get("z", 0) - eps,
                    "z_max": q.get("z", 0) + eps,
                }
            # Fill any missing axis bounds from domain (e.g. face/slice selectors)
            for axis in ("x", "y", "z"):
                if f"{axis}_min" not in q:
                    q[f"{axis}_min"] = domain.get(f"{axis}_min", 0.0)
                if f"{axis}_max" not in q:
                    q[f"{axis}_max"] = domain.get(f"{axis}_max", 0.0)
            selectors[s["id"]] = q

        ax = fig.add_subplot(rows, cols, idx + 1, projection="3d")
        render_3d(
            ax,
            domain,
            shape_type,
            shape_params,
            selectors,
            bcs,
            loads,
            annotate=False,
            equal_axes=equal_axes,
            show_grid=show_grid,
            label_offset=label_offset,
            label_font_size=font_size,
        )
        ax.dist = zoom

        if show_title:
            pid = meta.get("problem_id", jp.stem).replace("_", " ")
            pid = _to_sentence_case(pid)
            # Hard-wrap to avoid adjacent long titles overlapping
            chars_per_line = max(10, int(cell_size * 7))
            wrapped = textwrap.fill(pid, width=chars_per_line)
            title_fs = (
                title_font_size
                if title_font_size is not None
                else max(font_size * 0.75, 5)
            )
            ax.set_title(wrapped, fontsize=title_fs, pad=2, linespacing=1.2)

    fig.subplots_adjust(
        left=0.005,
        right=0.995,
        top=0.975,
        bottom=0.005,
        hspace=0.12,
        wspace=0.01,
    )

    Path(output_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=300, facecolor="white")
    print(f"✓ Saved collage ({n} panels, {cols}×{rows}) → {output_png}")

    return fig


# ── CLI ───────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Publication-quality FEA load case visualizer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "json_file",
        help="Path to a load case JSON file, or a directory for collage mode",
    )
    parser.add_argument(
        "output",
        nargs="?",
        default=None,
        help=(
            "Output file path.  Extension determines format: "
            ".png → 300 DPI raster, .pdf → vector.  "
            "Omit to auto-derive from problem_id into results/"
        ),
    )
    parser.add_argument(
        "--pdf",
        action="store_true",
        help="Also save a companion PDF (vector) alongside the PNG",
    )
    parser.add_argument(
        "--iso",
        action="store_true",
        help="Output only the isometric 3-D panel (square, 5×5 in) — ideal for collages",
    )
    parser.add_argument(
        "--equal-axes",
        action="store_true",
        help="Render the 3-D panel with equal x/y/z scales",
    )
    parser.add_argument(
        "--no-grid",
        action="store_true",
        help="Hide grid lines on the rendered panels",
    )
    parser.add_argument(
        "--no-title",
        action="store_true",
        help="Hide figure and panel titles for collage output",
    )
    parser.add_argument(
        "--label-offset",
        type=float,
        default=1.30,
        help="3-D force label distance multiplier from load centroid (default: 1.30)",
    )
    parser.add_argument(
        "--font-size",
        type=float,
        default=7.5,
        help="Font size for mm and force-magnitude labels (default: 7.5)",
    )
    parser.add_argument(
        "--title-font-size",
        type=float,
        default=None,
        help="Font size for panel/figure titles (default: auto)",
    )
    parser.add_argument(
        "--cols",
        type=int,
        default=8,
        help="Number of columns in collage grid (default: 8)",
    )
    parser.add_argument(
        "--cell-size",
        type=float,
        default=3.0,
        help="Width/height of each collage cell in inches (default: 3.0)",
    )
    parser.add_argument(
        "--zoom",
        type=float,
        default=4.8,
        help="Collage camera distance (smaller = closer, default: 4.8)",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Randomize collage panel order when input is a directory",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed for --shuffle (for reproducible order)",
    )
    args = parser.parse_args()

    if args.label_offset <= 1.0:
        print("Error: --label-offset must be > 1.0", file=sys.stderr)
        sys.exit(2)
    if args.font_size <= 0.0:
        print("Error: --font-size must be > 0", file=sys.stderr)
        sys.exit(2)
    if args.title_font_size is not None and args.title_font_size <= 0.0:
        print("Error: --title-font-size must be > 0", file=sys.stderr)
        sys.exit(2)
    if args.zoom <= 0.0:
        print("Error: --zoom must be > 0", file=sys.stderr)
        sys.exit(2)

    json_path = Path(args.json_file)
    if not json_path.exists():
        print(f"Error: path not found: {json_path}", file=sys.stderr)
        sys.exit(1)

    # ── Collage mode: input is a directory ─────────────────────────────────
    if json_path.is_dir():
        candidates = sorted(json_path.glob("*.json"))
        valid = [p for p in candidates if _load_bounds(p) is not None]
        skipped = len(candidates) - len(valid)
        if not valid:
            print(
                f"Error: no renderable load-case JSON files found in {json_path}",
                file=sys.stderr,
            )
            sys.exit(1)
        if skipped:
            print(
                f"  ⚠ Skipped {skipped} file(s) with unsupported or invalid design_domain."
            )
        if args.shuffle:
            random.Random(args.seed).shuffle(valid)
            if args.seed is None:
                print("  • Collage order randomized.")
            else:
                print(f"  • Collage order randomized with seed={args.seed}.")

        out_png = (
            args.output if args.output else f"results/collage_{json_path.name}.png"
        )
        if not out_png.endswith(".png"):
            out_png += ".png"

        make_collage(
            valid,
            output_png=out_png,
            cols=args.cols,
            equal_axes=args.equal_axes,
            show_grid=not args.no_grid,
            show_title=not args.no_title,
            label_offset=args.label_offset,
            font_size=args.font_size,
            title_font_size=args.title_font_size,
            cell_size=args.cell_size,
            zoom=args.zoom,
        )
        return

    # ── Single-file mode ────────────────────────────────────────────────────
    with open(json_path) as f:
        data = json.load(f)

    try:
        _domain_from_design_domain(data.get("design_domain", {}), json_path=json_path)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    problem_id = data.get("meta", {}).get("problem_id", json_path.stem)
    slug = problem_id.lower().replace(" ", "_")
    iso_suffix = "_iso" if args.iso else ""

    if args.output:
        out_path = Path(args.output)
        out_png = str(out_path) if out_path.suffix.lower() == ".png" else None
        out_pdf = str(out_path) if out_path.suffix.lower() == ".pdf" else None
        if not out_png and not out_pdf:
            out_png = str(out_path.with_suffix(".png"))
    else:
        out_png = f"results/loadcase_{slug}{iso_suffix}.png"
        out_pdf = None

    if args.pdf and out_png:
        out_pdf = out_png.replace(".png", ".pdf")

    visualize_load_case(
        data,
        json_path=json_path,
        output_png=out_png,
        output_pdf=out_pdf,
        iso_only=args.iso,
        equal_axes=args.equal_axes,
        show_grid=not args.no_grid,
        show_title=not args.no_title,
        label_offset=args.label_offset,
        font_size=args.font_size,
        title_font_size=args.title_font_size,
    )


if __name__ == "__main__":
    main()
