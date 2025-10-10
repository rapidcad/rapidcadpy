# pip install OCP  (pythonocc-core modern fork)
# Generates FRONT/TOP/RIGHT orthographic SVGs via OCCT HLR.
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple

from OCP.BRep import BRep_Tool
from OCP.BRepAdaptor import BRepAdaptor_Curve
from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
from OCP.GCPnts import GCPnts_QuasiUniformAbscissa
from OCP.gp import gp_Ax2, gp_Ax3, gp_Dir, gp_Pnt, gp_Vec
from OCP.HLRAlgo import HLRAlgo_Projector
from OCP.HLRBRep import HLRBRep_Algo, HLRBRep_HLRToShape
from OCP.TopAbs import TopAbs_EDGE
from OCP.TopExp import TopExp_Explorer
from OCP.TopoDS import TopoDS_Compound, TopoDS_Edge, TopoDS_Shape

Point2D = Tuple[float, float]


# ---------------------------
# View definitions
# ---------------------------
@dataclass(frozen=True)
class OrthoView:
    name: str
    ax2: gp_Ax2  # projection plane: origin + Z (view dir) + X (right)
    svg_file: str


VIEWS = [
    OrthoView(
        "FRONT", gp_Ax2(gp_Pnt(0, 0, 0), gp_Dir(0, 1, 0), gp_Dir(1, 0, 0)), "front.svg"
    ),
    OrthoView(
        "TOP", gp_Ax2(gp_Pnt(0, 0, 0), gp_Dir(0, 0, 1), gp_Dir(1, 0, 0)), "top.svg"
    ),
    OrthoView(
        "RIGHT", gp_Ax2(gp_Pnt(0, 0, 0), gp_Dir(1, 0, 0), gp_Dir(0, 1, 0)), "right.svg"
    ),
]


# ---------------------------
# Core: HLR → edge compounds
# ---------------------------
def hlr_compounds(
    shape: TopoDS_Shape, ax2: gp_Ax2
) -> Tuple[TopoDS_Compound, TopoDS_Compound]:
    """Return (visible_edges, hidden_edges) as compounds for a given view ax2."""
    algo = HLRBRep_Algo()
    algo.Add(shape)
    proj = HLRAlgo_Projector(ax2)
    algo.Projector(proj)
    algo.Update()
    conv = HLRBRep_HLRToShape(algo)
    return conv.VCompound(), conv.HCompound()


# ---------------------------
# Discretize edges → 3D polyline
# ---------------------------
def edge_polyline_pts(edge: TopoDS_Edge, segments: int = 24) -> List[gp_Pnt]:
    """Sample an edge to a list of 3D points."""
    # Prefer parametric sampling over length-based for simplicity
    adaptor = BRepAdaptor_Curve(edge)
    first, last = adaptor.FirstParameter(), adaptor.LastParameter()
    sampler = GCPnts_QuasiUniformAbscissa(adaptor, segments)
    pts = []
    if sampler.IsDone():
        for i in range(1, sampler.NbPoints() + 1):
            u = sampler.Parameter(i)
            p = gp_Pnt()
            adaptor.D0(u, p)
            pts.append(p)
    else:
        # Fallback: endpoints
        p1, p2 = gp_Pnt(), gp_Pnt()
        curve, f, l = BRep_Tool.Curve(edge)
        curve.D0(first, p1)
        curve.D0(last, p2)
        pts = [p1, p2]
    # Drop duplicates
    out = [pts[0]]
    for p in pts[1:]:
        if p.Distance(out[-1]) > 1e-9:
            out.append(p)
    return out


# ---------------------------
# Project 3D → 2D in view plane
# ---------------------------
def project_pts_to_2d(ax2: gp_Ax2, pts3d: Iterable[gp_Pnt]) -> List[Point2D]:
    """Orthographically project to the plane defined by ax2 (Z=look direction, X=right)."""
    origin = ax2.Location()
    xdir = gp_Vec(ax2.XDirection())  # plane X axis
    ydir = gp_Vec(gp_Ax3(ax2).YDirection())  # plane Y axis derived from ax2
    out: List[Point2D] = []
    for p in pts3d:
        v = gp_Vec(origin, p)
        x = v.Dot(xdir)  # coordinate along X axis
        y = v.Dot(ydir)  # coordinate along Y axis
        out.append((x, y))
    return out


def compound_to_2d_paths(
    comp: TopoDS_Compound, ax2: gp_Ax2, segs: int = 32
) -> List[List[Point2D]]:
    """Discretize all edges in a compound and project to 2D polylines."""
    paths: List[List[Point2D]] = []
    if comp.IsNull():
        return paths
    ex = TopExp_Explorer(comp, TopAbs_EDGE)
    while ex.More():
        edge = TopoDS_Edge(ex.Current())
        pts3d = edge_polyline_pts(edge, segments=segs)
        path2d = project_pts_to_2d(ax2, pts3d)
        if len(path2d) >= 2:
            paths.append(path2d)
        ex.Next()
    return paths


# ---------------------------
# Fit to sheet + SVG writer
# ---------------------------
@dataclass
class Sheet:
    width_px: int = 1200
    height_px: int = 900
    margin_px: int = 40


def bbox_2d(paths: List[List[Point2D]]) -> Tuple[float, float, float, float]:
    xs, ys = [], []
    for path in paths:
        for x, y in path:
            xs.append(x)
            ys.append(y)
    return (min(xs), min(ys), max(xs), max(ys)) if xs else (0, 0, 1, 1)


def fit_paths(paths: List[List[Point2D]], sheet: Sheet) -> List[List[Point2D]]:
    """Scale & translate paths to fit into the sheet (Y flipped for SVG)."""
    minx, miny, maxx, maxy = bbox_2d(paths)
    w, h = maxx - minx, maxy - miny
    avail_w = sheet.width_px - 2 * sheet.margin_px
    avail_h = sheet.height_px - 2 * sheet.margin_px
    s = 0.95 * min(avail_w / (w if w > 0 else 1), avail_h / (h if h > 0 else 1))
    ox = sheet.margin_px + (avail_w - s * w) * 0.5
    oy = sheet.margin_px + (avail_h - s * h) * 0.5
    # SVG Y grows down; flip Y
    fitted: List[List[Point2D]] = []
    for path in paths:
        fitted.append(
            [
                (ox + s * (x - minx), sheet.height_px - (oy + s * (y - miny)))
                for (x, y) in path
            ]
        )
    return fitted


def svg_polyline(points: List[Point2D]) -> str:
    d = " ".join(f"{x:.3f},{y:.3f}" for x, y in points)
    return f'<polyline fill="none" stroke="black" stroke-width="1.5" points="{d}"/>'


def svg_document(
    visible: List[List[Point2D]], hidden: List[List[Point2D]], sheet: Sheet, title: str
) -> str:
    vis = "\n".join(svg_polyline(p) for p in visible)
    hid = "\n".join(
        f'<polyline fill="none" stroke="#555" stroke-width="0.8" stroke-dasharray="5,4" points="{" ".join(f"{x:.3f},{y:.3f}" for x,y in p)}"/>'
        for p in hidden
    )
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<svg width="{sheet.width_px}" height="{sheet.height_px}" viewBox="0 0 {sheet.width_px} {sheet.height_px}"
     xmlns="http://www.w3.org/2000/svg">
  <title>{title}</title>
  <rect x="0" y="0" width="{sheet.width_px}" height="{sheet.height_px}" fill="white"/>
  {hid}
  {vis}
</svg>
"""


# ---------------------------
# Example: build a shape
# ---------------------------
def build_example_shape() -> TopoDS_Shape:
    # 100 x 60 x 30 box; replace with your parametric pipeline output
    return BRepPrimAPI_MakeBox(100.0, 60.0, 30.0).Shape()


# ---------------------------
# End-to-end: one view to SVG
# ---------------------------
def make_view_svg(
    shape: TopoDS_Shape, view: OrthoView, sheet: Sheet = Sheet(), edge_segs: int = 48
) -> str:
    v_comp, h_comp = hlr_compounds(shape, view.ax2)
    v_paths = compound_to_2d_paths(v_comp, view.ax2, segs=edge_segs)
    h_paths = compound_to_2d_paths(h_comp, view.ax2, segs=edge_segs)
    # Fit both together so styles align
    all_paths = v_paths + h_paths
    fitted = fit_paths(all_paths, sheet)
    v_fitted = fitted[: len(v_paths)]
    h_fitted = fitted[len(v_paths) :]
    return svg_document(v_fitted, h_fitted, sheet, f"Ortho {view.name}")


# ---------------------------
# Run
# ---------------------------
if __name__ == "__main__":
    shape = build_example_shape()
    for view in VIEWS:
        svg = make_view_svg(shape, view)
        with open(view.svg_file, "w", encoding="utf-8") as f:
            f.write(svg)
        print(f"Wrote {view.svg_file}")
