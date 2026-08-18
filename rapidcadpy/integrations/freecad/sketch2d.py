"""
FreeCAD Sketch2D – builds Part.Wire / Part.Face from 2D primitives and
supports extrude, pipe, and sweep operations.
"""

from typing import Any, Optional, cast

from ...sketch2d import Sketch2D
from ...primitives import Arc, Circle, Line
from .errors import FreeCADNativeFeatureError
from .shape import FreeCADShape


class FreeCADSketch2D(Sketch2D):
    """
    Concrete Sketch2D backed by FreeCAD's ``Part`` module.

    The sketch is stored as a list of 2-D primitives (``Line``,
    ``Circle``, ``Arc``).  Geometry is only materialised into
    ``Part.Wire`` / ``Part.Face`` when an operation such as
    ``extrude()`` is called.
    """

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _silent_fail_enabled(self) -> bool:
        return bool(getattr(self.app, "silent_geometry_failures", False))

    def _primitive_to_edge(self, primitive):
        """Convert a 2-D primitive into a FreeCAD ``Part.Shape`` edge."""
        import FreeCAD
        import Part

        if isinstance(primitive, Line):
            start_3d = self._workplane._to_3d(primitive.start[0], primitive.start[1])
            end_3d = self._workplane._to_3d(primitive.end[0], primitive.end[1])
            return Part.makeLine(
                FreeCAD.Vector(*start_3d),
                FreeCAD.Vector(*end_3d),
            )

        elif isinstance(primitive, Circle):
            center_3d = self._workplane._to_3d(primitive.center[0], primitive.center[1])
            n = self._workplane.normal_vector
            return Part.makeCircle(
                float(primitive.radius),
                FreeCAD.Vector(*center_3d),
                FreeCAD.Vector(float(n[0]), float(n[1]), float(n[2])),
            )

        elif isinstance(primitive, Arc):
            start_3d = self._workplane._to_3d(primitive.start[0], primitive.start[1])
            mid_3d = self._workplane._to_3d(primitive.mid[0], primitive.mid[1])
            end_3d = self._workplane._to_3d(primitive.end[0], primitive.end[1])

            s = FreeCAD.Vector(*start_3d)
            m = FreeCAD.Vector(*mid_3d)
            e = FreeCAD.Vector(*end_3d)

            # Detect degenerate arc: collinear points or mid == end.
            # Strategy: check whether the area of the triangle formed by the
            # three points is below a small tolerance.  If so, fall back to a
            # straight line from start to end.
            sm = m.sub(s)
            se = e.sub(s)
            cross = sm.cross(se)
            is_degenerate = cross.Length < 1e-10 or m.isEqual(e, 1e-10)

            if is_degenerate:
                # Fall back to a straight line start → end
                return Part.makeLine(s, e)

            return Part.Arc(s, m, e).toShape()

        else:
            raise NotImplementedError(
                f"Primitive type {type(primitive).__name__} is not supported "
                "by the FreeCAD integration."
            )

    def _make_wire(self):
        """Build and return a ``Part.Wire`` from the stored primitives."""
        import Part

        if not self._primitives:
            if self._silent_fail_enabled():
                return None
            raise ValueError("Cannot create wire: no primitives in sketch.")

        try:
            edges = [self._primitive_to_edge(p) for p in self._primitives]
            # __sortEdges__ reorders and snaps endpoints so arcs/lines connect
            sorted_edges = Part.__sortEdges__(edges)
            wire = Part.Wire(sorted_edges)
            if not wire.isClosed():
                # Try to close with a small tolerance
                wire = Part.Wire(Part.__sortEdges__(edges))
            return wire
        except Exception as exc:
            if self._silent_fail_enabled():
                return None
            raise ValueError(f"Wire construction failed: {exc}") from exc

    def _make_face(self):
        """Build and return a planar ``Part.Face`` from the stored primitives."""
        import Part

        if not self._primitives:
            if self._silent_fail_enabled():
                return None
            raise ValueError(
                "Face construction failed: sketch has no elements. "
                "Call rect(), circle(), or line_to() before extruding."
            )

        wire = self._make_wire()
        if wire is None:
            return None

        try:
            # Pass wire as a list – more robust in FreeCAD's Part module
            face = Part.Face([wire])
            if not face.isValid():
                # Fall back to the single-arg form
                face = Part.Face(wire)
            if not face.isValid():
                if self._silent_fail_enabled():
                    return None
                raise ValueError(
                    "Face construction failed: the resulting face is invalid. "
                    "Check that the sketch forms a closed, planar, non-self-intersecting loop."
                )
            return face
        except Exception as exc:
            if self._silent_fail_enabled():
                return None
            raise ValueError(f"Face construction failed: {exc}") from exc

    def _create_editable_sketch(self, doc, name: str):
        """Create a Sketcher sketch for the stored 2-D primitives."""
        import FreeCAD
        import Part
        import Sketcher  # noqa: F401

        sketch_obj = doc.addObject("Sketcher::SketchObject", name)

        # Reconstruct the editable sketch so its world geometry matches `_to_3d`
        # / `.obj` exactly. Two subtleties:
        #  * A FreeCAD Placement must be a proper (right-handed) rotation, but the
        #    workplane's local frame is LEFT-handed for the XZ plane
        #    (lx x ly = -lz). So we build the placement from a right-handed
        #    in-plane Y axis ry = lz x lx ...
        #  * ... and compensate by flipping the sketch's local Y when the
        #    workplane's stated ly points opposite ry (``y_sign``), so points
        #    still land at origin + px*lx + py*ly like `_to_3d`.
        wp = self._workplane
        ox, oy, oz = wp._to_3d(0.0, 0.0)
        lx, ly, lz = wp._local_x, wp._local_y, wp._local_z
        # ry = lz x lx (right-handed with lz as the sketch normal)
        ry_x = lz.y * lx.z - lz.z * lx.y
        ry_y = lz.z * lx.x - lz.x * lx.z
        ry_z = lz.x * lx.y - lz.y * lx.x
        y_sign = 1.0 if (ly.x * ry_x + ly.y * ry_y + ly.z * ry_z) >= 0 else -1.0
        matrix = FreeCAD.Matrix(
            float(lx.x),
            float(ry_x),
            float(lz.x),
            float(ox),
            float(lx.y),
            float(ry_y),
            float(lz.y),
            float(oy),
            float(lx.z),
            float(ry_z),
            float(lz.z),
            float(oz),
            0.0,
            0.0,
            0.0,
            1.0,
        )
        sketch_obj.Placement = FreeCAD.Placement(matrix)

        def _v(px, py):
            return FreeCAD.Vector(float(px), y_sign * float(py), 0.0)

        for primitive in self._primitives:
            if isinstance(primitive, Line):
                geom = Part.LineSegment(
                    _v(primitive.start[0], primitive.start[1]),
                    _v(primitive.end[0], primitive.end[1]),
                )
            elif isinstance(primitive, Circle):
                geom = Part.Circle(
                    _v(primitive.center[0], primitive.center[1]),
                    FreeCAD.Vector(0.0, 0.0, 1.0),
                    float(primitive.radius),
                )
            elif isinstance(primitive, Arc):
                geom = Part.ArcOfCircle(
                    _v(primitive.start[0], primitive.start[1]),
                    _v(primitive.mid[0], primitive.mid[1]),
                    _v(
                        primitive.end[0],
                        primitive.end[1],
                    ),
                )
            else:
                continue
            sketch_obj.addGeometry(geom, False)

        return sketch_obj

    # ------------------------------------------------------------------
    # Sketch2D abstract method implementations
    # ------------------------------------------------------------------

    def extrude(
        self,
        distance: float,
        operation: str = "NewBodyFeatureOperation",
        symmetric: bool = False,
    ) -> Optional[FreeCADShape]:
        """
        Extrude the sketch face along the workplane normal.

        Args:
            distance:  Extrusion depth.  Negative values reverse the direction.
            operation: "NewBodyFeatureOperation" (new shape, default),
                       "JoinBodyFeatureOperation" (fuse with last shape),
                       "Cut" / "CutOperation" (subtract from existing shapes).
            symmetric: Extrude distance/2 in each direction around the sketch plane.

        Returns:
            FreeCADShape wrapping the resulting solid, or the modified existing
            shape for Join/Cut operations.
        """
        import FreeCAD

        face = self._make_face()
        if face is None:
            return None

        n = self._workplane.normal_vector

        if symmetric:
            # Translate the face by –distance/2 along normal, then extrude by +distance
            half = distance / 2.0
            offset_vec = FreeCAD.Vector(
                float(n[0]) * (-half),
                float(n[1]) * (-half),
                float(n[2]) * (-half),
            )
            face.translate(offset_vec)

        extrude_vec = FreeCAD.Vector(
            float(n[0]) * distance,
            float(n[1]) * distance,
            float(n[2]) * distance,
        )

        try:
            solid = face.extrude(extrude_vec)
        except Exception as exc:
            if self._silent_fail_enabled():
                return None
            raise RuntimeError(f"Extrusion failed: {exc}") from exc

        self._last_extrude_distance = float(distance)
        self._last_extrude_symmetric = bool(symmetric)
        try:
            return self._apply_operation(solid, operation)
        finally:
            self._last_extrude_distance = None
            self._last_extrude_symmetric = None

    def pipe(self, diameter: float) -> Optional[FreeCADShape]:
        """
        Sweep a circular cross-section along the sketch's wire path.

        Args:
            diameter: Outer diameter of the pipe.

        Returns:
            FreeCADShape wrapping the resulting solid.
        """
        if self.app is not None and hasattr(self.app, "get_doc"):
            raise FreeCADNativeFeatureError(
                "Native FreeCAD pipe creation is not implemented; refusing to "
                "return an unbacked OCC shape that would later be baked."
            )

        import FreeCAD
        import Part

        if not self._primitives:
            if self._silent_fail_enabled():
                return None
            raise ValueError("Pipe: no primitives in sketch.")

        radius = diameter / 2.0

        # Build edges from primitives, then strip any "return" edge — an edge
        # that is the exact reverse of a previously seen edge.  This handles
        # the common case where close() appended a closing line that sends the
        # spine back along the same axis (e.g. (0,0)→(100,0)→(0,0)), which
        # would otherwise cause the two swept cylinders to overlap and cancel.
        raw_edges = [self._primitive_to_edge(p) for p in self._primitives]

        def _pt_key(pt, decimals=6):
            return (round(pt.x, decimals), round(pt.y, decimals), round(pt.z, decimals))

        seen_forward: set = set()
        spine_edges = []
        for e in raw_edges:
            verts = e.Vertexes
            p1 = _pt_key(verts[0].Point)
            p2 = _pt_key(verts[-1].Point)
            reverse_key = (p2, p1)
            if reverse_key in seen_forward:
                # This edge exactly reverses a previous edge — skip it.
                continue
            seen_forward.add((p1, p2))
            spine_edges.append(e)

        if not spine_edges:
            if self._silent_fail_enabled():
                return None
            raise ValueError("Pipe: all edges cancelled out (degenerate path).")

        try:
            sorted_edges = Part.__sortEdges__(spine_edges)
            spine = Part.Wire(sorted_edges)
        except Exception as exc:
            if self._silent_fail_enabled():
                return None
            raise ValueError(f"Pipe: wire construction failed: {exc}") from exc

        # Determine tangent at wire start to orient the profile circle
        wire_edges = spine.Edges
        if not wire_edges:
            if self._silent_fail_enabled():
                return None
            raise ValueError("Pipe: wire has no edges after construction.")

        first_edge = wire_edges[0]
        start_pt = first_edge.Vertexes[0].Point
        tangent = first_edge.tangentAt(first_edge.FirstParameter)
        tangent.normalize()

        # Build a profile circle perpendicular to the tangent
        profile_edge = Part.makeCircle(radius, start_pt, tangent)
        profile_wire = Part.Wire([profile_edge])

        try:
            # makePipeShell(profiles, make_solid, is_frenet)
            solid = spine.makePipeShell([profile_wire], True, True)
            return FreeCADShape(solid, self.app)
        except Exception as exc:
            if self._silent_fail_enabled():
                return None
            raise RuntimeError(f"Pipe creation failed: {exc}") from exc

    def sweep(
        self,
        profile: "FreeCADSketch2D",
        make_solid: bool = True,
        is_frenet: bool = True,
        transition_mode: str = "right",
    ) -> Optional[FreeCADShape]:
        """
        Sweep a profile sketch along this sketch's wire path.

        Args:
            profile:         Profile cross-section (another FreeCADSketch2D).
            make_solid:      Produce a solid (True) or a shell (False).
            is_frenet:       Use Frenet frame so the profile stays perpendicular to path.
            transition_mode: Ignored (placeholder for API compatibility).

        Returns:
            FreeCADShape wrapping the resulting solid.
        """
        if self.app is not None and hasattr(self.app, "get_doc"):
            raise FreeCADNativeFeatureError(
                "Native Part::Sweep creation is not implemented; refusing to "
                "return an unbacked OCC shape that would later be baked."
            )

        path_wire = self._make_wire()
        profile_wire = profile._make_wire()

        if path_wire is None or profile_wire is None:
            return None

        try:
            solid = path_wire.makePipeShell([profile_wire], make_solid, is_frenet)
            return FreeCADShape(solid, self.app)
        except Exception as exc:
            if self._silent_fail_enabled():
                return None
            raise RuntimeError(f"Sweep failed: {exc}") from exc

    def to_png(
        self,
        file_name: Optional[str] = None,
        width: int = 800,
        height: int = 600,
        margin: float = 0.1,
    ) -> None:
        """Delegate 2-D sketch rendering to the workplane's to_png method."""
        self._workplane.to_png(
            file_name=file_name,
            width=width,
            height=height,
            margin=int(margin * max(width, height)),
        )

    # ------------------------------------------------------------------
    # Internal operation dispatcher
    # ------------------------------------------------------------------

    def _apply_operation(
        self,
        solid,
        operation: str,
    ) -> Optional[FreeCADShape]:
        """
        Apply *operation* to *solid* with respect to existing app shapes.

        Returns a FreeCADShape (new or modified existing).
        """

        def _make_freecad_shape(obj, app):
            """Create a native extrusion, never a baked ``Part::Feature``."""
            doc = None
            feat = None
            if app is not None and hasattr(app, "get_doc"):
                doc = app.get_doc()
                feat_idx = app.get_next_feature_index()
                created_names = []
                try:
                    sketch = self._create_editable_sketch(doc, f"Sketch_{feat_idx}")
                    created_names.append(str(sketch.Name))
                    extrusion = doc.addObject("Part::Extrusion", f"Extrude_{feat_idx}")
                    created_names.append(str(extrusion.Name))
                    extrusion.Base = sketch
                    extrusion.DirMode = "Normal"
                    extrusion.Solid = True
                    extrusion.TaperAngle = 0.0
                    extrusion.TaperAngleRev = 0.0

                    distance = float(getattr(self, "_last_extrude_distance", 0.0))
                    symmetric = bool(getattr(self, "_last_extrude_symmetric", False))
                    if symmetric:
                        extrusion.LengthFwd = abs(distance) / 2.0
                        extrusion.LengthRev = abs(distance) / 2.0
                    elif distance >= 0.0:
                        extrusion.LengthFwd = distance
                        extrusion.LengthRev = 0.0
                    else:
                        extrusion.LengthFwd = 0.0
                        extrusion.LengthRev = abs(distance)

                    doc.recompute()
                    native_shape = getattr(extrusion, "Shape", None)
                    if (
                        native_shape is None
                        or getattr(native_shape, "isNull", lambda: False)()
                    ):
                        raise ValueError(
                            "FreeCAD recomputed an empty Part::Extrusion shape."
                        )
                    feat = extrusion
                except Exception as exc:
                    for name in reversed(created_names):
                        try:
                            doc.removeObject(name)
                        except Exception:
                            pass
                    try:
                        doc.recompute()
                    except Exception:
                        pass
                    raise FreeCADNativeFeatureError(
                        "Could not create Sketcher::SketchObject -> Part::Extrusion; "
                        "refusing to replace the failed native feature with baked "
                        "Part::Feature geometry."
                    ) from exc
            return FreeCADShape(obj, app, doc=doc, current_feature=feat)

        if operation != "NewBodyFeatureOperation":
            raise FreeCADNativeFeatureError(
                f"Native FreeCAD extrusion operation {operation!r} is not "
                "implemented. Create a new extrusion and apply cut() or union() "
                "so RapidCADPy can create a linked Part::Cut or Part::MultiFuse."
            )
        return _make_freecad_shape(solid, self.app)
