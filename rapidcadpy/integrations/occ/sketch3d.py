"""OpenCASCADE implementation of `rapidcadpy.sketch3d.Sketch3D`.

This is a 3D *path* sketch used as the spine for sweeps/pipes.
"""

from __future__ import annotations

from typing import Any

from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeEdge, BRepBuilderAPI_MakeWire
from OCC.Core.gp import gp_Pnt

from .sketch3d import Polyline3D, Sketch3D


class OccSketch3D(Sketch3D):
    """OpenCASCADE-backed 3D sketch (wire builder)."""

    def _primitive_to_edges(self, primitive: object):
        if isinstance(primitive, Polyline3D):
            pts = primitive.points
            for a, b in zip(pts, pts[1:]):
                yield BRepBuilderAPI_MakeEdge(
                    gp_Pnt(float(a[0]), float(a[1]), float(a[2])),
                    gp_Pnt(float(b[0]), float(b[1]), float(b[2])),
                ).Edge()
            return
        raise TypeError(f"Unsupported 3D primitive: {type(primitive).__name__}")

    def wire(self):
        if len(self._primitives) == 0:
            raise ValueError("Cannot create wire: no primitives in 3D sketch")

        wire_builder: Any = BRepBuilderAPI_MakeWire()
        for prim in self._primitives:
            for edge in self._primitive_to_edges(prim):
                wire_builder.Add(edge)
        wire_builder.Build()

        if not wire_builder.IsDone():
            raise ValueError("3D wire construction failed")

        return wire_builder.Wire()

    def pipe(
        self,
        diameter: float,
        is_frenet: bool = True,
        transition_mode: str = "right",
    ):
        """Create a pipe along this 3D sketch (wire) spine."""

        from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeFace
        from OCC.Core.BRepOffsetAPI import BRepOffsetAPI_MakePipeShell
        from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Fuse
        from OCC.Core.gp import gp_Ax2, gp_Circ, gp_Dir, gp_Pnt, gp_Vec
        from OCC.Core.BRepAdaptor import BRepAdaptor_CompCurve
        from OCC.Core.BRep import BRep_Tool
        from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_TransitionMode
        from typing import cast

        from .integrations.occ.shape import OccShape

        spine = self.wire()
        radius = diameter / 2.0

        wire_adaptor = BRepAdaptor_CompCurve(spine)
        first_param = wire_adaptor.FirstParameter()
        last_param = wire_adaptor.LastParameter()
        is_closed = BRep_Tool.IsClosed(spine)

        start_point = wire_adaptor.Value(first_param)
        tangent_vec = gp_Vec()
        tmp_p = gp_Pnt()
        wire_adaptor.D1(first_param, tmp_p, tangent_vec)
        tangent_dir = gp_Dir(tangent_vec)

        ref_dir = gp_Dir(0, 0, 1)
        if abs(tangent_dir.Z()) > 0.99:
            ref_dir = gp_Dir(1, 0, 0)

        profile_axis = gp_Ax2(start_point, tangent_dir, ref_dir)
        profile_circle = gp_Circ(profile_axis, radius)
        profile_edge = BRepBuilderAPI_MakeEdge(profile_circle).Edge()
        profile_wire = BRepBuilderAPI_MakeWire(profile_edge).Wire()

        pipe_builder = BRepOffsetAPI_MakePipeShell(spine)
        pipe_builder.SetMode(bool(is_frenet))

        trans_map = {
            "transformed": BRepBuilderAPI_TransitionMode.BRepBuilderAPI_Transformed,
            "round": BRepBuilderAPI_TransitionMode.BRepBuilderAPI_RoundCorner,
            "right": BRepBuilderAPI_TransitionMode.BRepBuilderAPI_RightCorner,
        }
        pipe_builder.SetTransitionMode(
            cast(
                BRepBuilderAPI_TransitionMode,
                trans_map.get(
                    transition_mode,
                    BRepBuilderAPI_TransitionMode.BRepBuilderAPI_RightCorner,
                ),
            )
        )

        pipe_builder.Add(profile_wire, False, True)
        pipe_builder.Build()
        if not pipe_builder.IsDone():
            raise RuntimeError("Failed to create pipe")

        pipe_builder.MakeSolid()
        result_shape = pipe_builder.Shape()

        if not is_closed:
            start_cap_face = BRepBuilderAPI_MakeFace(profile_wire).Face()

            end_point = wire_adaptor.Value(last_param)
            end_tangent_vec = gp_Vec()
            end_tmp = gp_Pnt()
            wire_adaptor.D1(last_param, end_tmp, end_tangent_vec)
            end_tangent_dir = gp_Dir(end_tangent_vec)

            end_ref_dir = gp_Dir(0, 0, 1)
            if abs(end_tangent_dir.Z()) > 0.99:
                end_ref_dir = gp_Dir(1, 0, 0)

            end_profile_axis = gp_Ax2(end_point, end_tangent_dir, end_ref_dir)
            end_profile_circle = gp_Circ(end_profile_axis, radius)
            end_profile_edge = BRepBuilderAPI_MakeEdge(end_profile_circle).Edge()
            end_profile_wire = BRepBuilderAPI_MakeWire(end_profile_edge).Wire()
            end_cap_face = BRepBuilderAPI_MakeFace(end_profile_wire).Face()

            fuse_op = BRepAlgoAPI_Fuse(result_shape, start_cap_face)
            fuse_op.Build()
            if fuse_op.IsDone():
                result_shape = fuse_op.Shape()

            fuse_op2 = BRepAlgoAPI_Fuse(result_shape, end_cap_face)
            fuse_op2.Build()
            if fuse_op2.IsDone():
                result_shape = fuse_op2.Shape()

        return OccShape(obj=result_shape, app=self.app)

    def sweep(
        self,
        profile: "Any",
        make_solid: bool = True,
        is_frenet: bool = True,
        transition_mode: str = "right",
        auto_align_profile: bool = False,
    ):
        """Sweep a closed 2D OCC profile sketch along this 3D spine."""

        from OCC.Core.BRepOffsetAPI import BRepOffsetAPI_MakePipeShell
        from OCC.Core.BRepAdaptor import BRepAdaptor_CompCurve
        from OCC.Core.gp import gp_Ax2, gp_Ax3, gp_Dir, gp_Pnt, gp_Vec
        from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Transform
        from OCC.Core.gp import gp_Trsf
        from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_TransitionMode
        from typing import cast

        from .integrations.occ.shape import OccShape

        spine = self.wire()
        profile_wire = profile._make_wire()

        if auto_align_profile:
            wire_adaptor = BRepAdaptor_CompCurve(spine)
            first_param = wire_adaptor.FirstParameter()

            start_point: gp_Pnt = wire_adaptor.Value(first_param)
            tangent_vec = gp_Vec()
            tmp_p = gp_Pnt()
            wire_adaptor.D1(first_param, tmp_p, tangent_vec)
            tangent_dir = gp_Dir(tangent_vec)

            ref_dir = gp_Dir(0, 0, 1)
            if abs(tangent_dir.Z()) > 0.99:
                ref_dir = gp_Dir(1, 0, 0)

            target_ax2 = gp_Ax2(start_point, tangent_dir, ref_dir)

            p0_3d = profile._workplane._to_3d(0.0, 0.0)
            prof_origin = gp_Pnt(float(p0_3d[0]), float(p0_3d[1]), float(p0_3d[2]))
            prof_normal = gp_Dir(
                float(profile._workplane.normal_vector[0]),
                float(profile._workplane.normal_vector[1]),
                float(profile._workplane.normal_vector[2]),
            )
            prof_xdir = gp_Dir(
                float(profile._workplane._local_x[0]),
                float(profile._workplane._local_x[1]),
                float(profile._workplane._local_x[2]),
            )
            source_ax2 = gp_Ax2(prof_origin, prof_normal, prof_xdir)

            trsf = gp_Trsf()
            trsf.SetDisplacement(gp_Ax3(source_ax2), gp_Ax3(target_ax2))
            profile_wire = BRepBuilderAPI_Transform(profile_wire, trsf, True).Shape()

        sweep_builder = BRepOffsetAPI_MakePipeShell(spine)
        sweep_builder.SetMode(bool(is_frenet))

        trans_map = {
            "transformed": BRepBuilderAPI_TransitionMode.BRepBuilderAPI_Transformed,
            "round": BRepBuilderAPI_TransitionMode.BRepBuilderAPI_RoundCorner,
            "right": BRepBuilderAPI_TransitionMode.BRepBuilderAPI_RightCorner,
        }
        sweep_builder.SetTransitionMode(
            cast(
                BRepBuilderAPI_TransitionMode,
                trans_map.get(
                    transition_mode,
                    BRepBuilderAPI_TransitionMode.BRepBuilderAPI_RightCorner,
                ),
            )
        )

        sweep_builder.Add(profile_wire, False, True)
        sweep_builder.Build()
        if not sweep_builder.IsDone():
            raise RuntimeError("Failed to sweep profile")

        if make_solid:
            sweep_builder.MakeSolid()

        return OccShape(obj=sweep_builder.Shape(), app=self.app)
