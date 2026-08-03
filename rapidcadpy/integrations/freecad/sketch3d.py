"""FreeCAD implementation of `rapidcadpy.sketch3d.Sketch3D`."""

from __future__ import annotations

from typing import Any

from ...sketch3d import Polyline3D, Sketch3D


class FreeCADSketch3D(Sketch3D):
    """FreeCAD-backed 3D sketch (wire builder)."""

    def _primitive_to_edges(self, primitive: object):
        import FreeCAD
        import Part

        if isinstance(primitive, Polyline3D):
            pts = primitive.points
            for a, b in zip(pts, pts[1:]):
                yield Part.makeLine(FreeCAD.Vector(*a), FreeCAD.Vector(*b))
            return

        raise TypeError(f"Unsupported 3D primitive: {type(primitive).__name__}")

    def wire(self):
        import Part

        if len(self._primitives) == 0:
            raise ValueError("Cannot create wire: no primitives in 3D sketch")

        edges: list[Any] = []
        for primitive in self._primitives:
            edges.extend(self._primitive_to_edges(primitive))

        try:
            sorted_edges = Part.__sortEdges__(edges)
            return Part.Wire(sorted_edges)
        except Exception as exc:
            raise ValueError(f"3D wire construction failed: {exc}") from exc

    def pipe(
        self,
        diameter: float,
        is_frenet: bool = True,
        transition_mode: str = "right",
    ):
        import Part

        from .shape import FreeCADShape

        spine = self.wire()
        radius = diameter / 2.0

        wire_edges = spine.Edges
        if not wire_edges:
            raise ValueError("Pipe: wire has no edges after construction.")

        first_edge = wire_edges[0]
        start_pt = first_edge.Vertexes[0].Point
        tangent = first_edge.tangentAt(first_edge.FirstParameter)
        tangent.normalize()

        profile_edge = Part.makeCircle(radius, start_pt, tangent)
        profile_wire = Part.Wire([profile_edge])

        try:
            solid = spine.makePipeShell([profile_wire], True, is_frenet)
        except Exception as exc:
            raise RuntimeError(f"Pipe creation failed: {exc}") from exc

        return FreeCADShape(solid, self.app)

    def sweep(
        self,
        profile: Any,
        make_solid: bool = True,
        is_frenet: bool = True,
        transition_mode: str = "right",
        auto_align_profile: bool = False,
    ):
        from .shape import FreeCADShape

        spine = self.wire()
        profile_wire = profile._make_wire()

        if profile_wire is None:
            raise ValueError("Sweep failed: profile sketch did not produce a wire")

        try:
            solid = spine.makePipeShell([profile_wire], make_solid, is_frenet)
        except Exception as exc:
            raise RuntimeError(f"Sweep failed: {exc}") from exc

        return FreeCADShape(solid, self.app)
