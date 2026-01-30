"""
OccSketch2D - OpenCASCADE implementation of Sketch2D.
"""

from typing import Any, Optional

from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakePrism
from OCC.Core.gp import gp_Vec, gp_Pnt, gp_Ax2, gp_Circ, gp_Dir
from OCC.Core.BOPAlgo import BOPAlgo_Tools
from OCC.Core.BRepBuilderAPI import (
    BRepBuilderAPI_MakeEdge,
    BRepBuilderAPI_MakeWire,
    BRepBuilderAPI_WireError,
)
from OCC.Core.TopTools import TopTools_ListOfShape
from OCC.Core.TopoDS import TopoDS_Compound
from .integrations.occ.shape import OccShape
from .primitives import Circle, Line
from .sketch2d import Sketch2D


class OccSketch2D(Sketch2D):
    """
    OpenCASCADE-specific implementation of a 2D sketch.

    This class holds a constructed TopoDS_Face and can extrude it
    into a 3D solid along the workplane's normal direction.

    Additional operations:
        - ``pipe(diameter)``: sweep a circular profile along the sketch path.
        - ``sweep(profile)``: sweep an arbitrary *closed* profile sketch along the sketch path.
    """

    def _primitive_to_edge(self, primitive):
        """Convert a 2D primitive to a 3D OCC edge.

        Args:
            primitive: A 2D primitive (Line, Circle, Arc, etc.)
            q
        Returns:
            TopoDS_Edge: The corresponding 3D edge
        """
        if isinstance(primitive, Line):
            # Convert line endpoints to 3D
            start_3d = self._workplane._to_3d(primitive.start[0], primitive.start[1])
            end_3d = self._workplane._to_3d(primitive.end[0], primitive.end[1])

            start_point_ocp = gp_Pnt(start_3d[0], start_3d[1], start_3d[2])
            end_point_ocp = gp_Pnt(end_3d[0], end_3d[1], end_3d[2])

            return BRepBuilderAPI_MakeEdge(start_point_ocp, end_point_ocp).Edge()

        elif isinstance(primitive, Circle):
            # Convert circle to 3D

            center_3d = self._workplane._to_3d(primitive.center[0], primitive.center[1])
            center = gp_Pnt(center_3d[0], center_3d[1], center_3d[2])

            # Use the workplane's normal vector for the circle orientation
            normal = gp_Dir(
                float(self._workplane.normal_vector[0]),
                float(self._workplane.normal_vector[1]),
                float(self._workplane.normal_vector[2]),
            )

            # Use local X direction as reference for circle orientation
            x_dir = gp_Dir(
                float(self._workplane._local_x[0]),
                float(self._workplane._local_x[1]),
                float(self._workplane._local_x[2]),
            )

            # Create circle geometry
            circle_gp = gp_Circ(gp_Ax2(center, normal, x_dir), primitive.radius)

            # Create edge from circle
            return BRepBuilderAPI_MakeEdge(circle_gp).Edge()

    def _make_wire(self):
        num_primitives = len(self._primitives)

        if num_primitives == 0:
            raise ValueError("Cannot create wire: no primitives in sketch")

        # Convert 2D primitives to 3D edges
        occ_edges_list = TopTools_ListOfShape()
        for primitive in self._primitives:
            edge = self._primitive_to_edge(primitive)
            occ_edges_list.Append(edge)

        wire_builder = BRepBuilderAPI_MakeWire()
        wire_builder.Add(occ_edges_list)
        wire_builder.Build()

        if not wire_builder.IsDone():
            error_code = wire_builder.Error()

            # Map error codes to human-readable messages
            error_messages = {
                BRepBuilderAPI_WireError.BRepBuilderAPI_EmptyWire: "Empty wire - no edges provided",
                BRepBuilderAPI_WireError.BRepBuilderAPI_DisconnectedWire: "Disconnected wire - edges don't connect to form a continuous path",
                BRepBuilderAPI_WireError.BRepBuilderAPI_NonManifoldWire: "Non-manifold wire - more than two edges meet at a vertex",
            }

            error_msg = error_messages.get(
                error_code, f"Unknown error code: {error_code}"
            )

            raise ValueError(
                f"Wire construction failed with {num_primitives} primitive(s): {error_msg}\n"
                f"This usually means:\n"
                f"  1. The sketch edges are not properly connected (gaps between line_to calls)\n"
                f"  2. The sketch needs to be closed with .close()\n"
                f"  3. Multiple disconnected shapes in one sketch\n"
                f"Workplane: {self.__class__.__name__}, Normal: {self.normal_vector if hasattr(self, 'normal_vector') else 'unknown'}"
            )

        return wire_builder.Wire()

    def _make_face(self):
        from OCC.Core.TopExp import TopExp_Explorer
        from OCC.Core.TopAbs import TopAbs_FACE

        # Get diagnostic information about the sketch
        num_edges = len(self._primitives)

        if num_edges == 0:
            raise ValueError(
                "Face construction failed: No sketch elements found. "
                "Create a sketch (e.g., rect(), circle(), line_to()) before extruding."
            )

        try:
            wires = self._make_wire()
        except Exception as e:
            raise ValueError(
                f"Face construction failed: Could not create wire from {num_edges} edge(s). "
                f"The sketch edges may not form a valid closed loop. "
                f"Original error: {str(e)}"
            ) from e

        rv = TopoDS_Compound()
        BOPAlgo_Tools.WiresToFaces(wires, rv)

        # Extract the first face from the compound
        explorer = TopExp_Explorer(rv, TopAbs_FACE)
        if not explorer.More():
            raise ValueError(
                f"Face construction failed: BOPAlgo_Tools.WiresToFaces produced no faces. "
                f"The wire with {num_edges} edge(s) could not be converted to a face. "
                f"This usually means:\n"
                f"  1. The sketch is not closed (check if you need to call .close())\n"
                f"  2. The edges are self-intersecting\n"
                f"  3. The edges don't form a planar loop\n"
                f"Workplane normal: {self.normal_vector if hasattr(self, 'normal_vector') else 'unknown'}"
            )

        return explorer.Current()

    def extrude(
        self, distance: float, operation: str = "NewBodyFeatureOperation"
    ) -> OccShape:
        """
        Extrude the sketch face along the workplane's normal direction.

        Args:
            distance: Distance to extrude (can be negative for opposite direction)
            operation: Operation type (reserved for future use)

        Returns:
            OccShape: The extruded 3D solid
        """
        # Calculate extrude vector based on workplane normal
        up_dir_vec = self._workplane.normal_vector * distance

        # Convert to gp_Vec using indexing to avoid attribute access issues
        extrude_vector = gp_Vec(
            float(up_dir_vec[0]), float(up_dir_vec[1]), float(up_dir_vec[2])
        )

        face = self._make_face()

        # Create the prism (extrusion)
        prism_builder: Any = BRepPrimAPI_MakePrism(face, extrude_vector, True)

        # Return the extruded shape
        return OccShape(obj=prism_builder.Shape(), app=self.app)

    def pipe(
        self,
        diameter: float,
        is_frenet: bool = True,
        transition_mode: str = "right",
    ) -> OccShape:
        """
        Create a pipe (cylindrical extrusion) along the sketch path.

        For open wires (non-closed paths), circular end caps are automatically
        added to create a closed solid volume.

        Args:
            diameter: Diameter of the pipe (outer diameter)

        Returns:
            OccShape: The resulting pipe shape
        """
        from OCC.Core.BRepBuilderAPI import (
            BRepBuilderAPI_MakeEdge,
            BRepBuilderAPI_MakeWire,
            BRepBuilderAPI_MakeFace,
        )
        from OCC.Core.BRepOffsetAPI import BRepOffsetAPI_MakePipeShell
        from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Fuse
        from OCC.Core.gp import gp_Ax2, gp_Circ, gp_Dir, gp_Pnt, gp_Vec
        from OCC.Core.BRepAdaptor import BRepAdaptor_CompCurve
        from OCC.Core.BRep import BRep_Tool
        from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_TransitionMode

        # Get the wire that represents the path (spine)
        spine = self._make_wire()

        # Calculate the radius from diameter
        radius = diameter / 2.0

        # Use BRepAdaptor_CompCurve to work with the wire directly
        wire_adaptor = BRepAdaptor_CompCurve(spine)
        first_param = wire_adaptor.FirstParameter()
        last_param = wire_adaptor.LastParameter()

        # Check if wire is closed
        is_closed = BRep_Tool.IsClosed(spine)

        # Get the starting point
        start_point = wire_adaptor.Value(first_param)

        # Get tangent vector at the start
        tangent_vec = gp_Vec()
        point_temp = gp_Pnt()
        wire_adaptor.D1(first_param, point_temp, tangent_vec)

        # Convert tangent vector to direction
        tangent_dir = gp_Dir(tangent_vec)

        # Create a perpendicular direction for the circle plane
        # Use a reference direction (try Z-axis first, then X-axis if parallel to tangent)
        ref_dir = gp_Dir(0, 0, 1)
        if abs(tangent_dir.Z()) > 0.99:  # Nearly parallel to Z
            ref_dir = gp_Dir(1, 0, 0)

        # Create axis system for the circular profile
        # The normal to the circle plane is the tangent direction
        profile_axis = gp_Ax2(start_point, tangent_dir, ref_dir)

        # Create the circular profile
        profile_circle = gp_Circ(profile_axis, radius)
        profile_edge = BRepBuilderAPI_MakeEdge(profile_circle).Edge()
        profile_wire = BRepBuilderAPI_MakeWire(profile_edge).Wire()

        pipe_builder = BRepOffsetAPI_MakePipeShell(spine)

        # Orientation handling
        pipe_builder.SetMode(bool(is_frenet))

        # Transition handling at C1 discontinuities (e.g., polyline corners)
        trans_map = {
            "transformed": BRepBuilderAPI_TransitionMode.BRepBuilderAPI_Transformed,
            "round": BRepBuilderAPI_TransitionMode.BRepBuilderAPI_RoundCorner,
            "right": BRepBuilderAPI_TransitionMode.BRepBuilderAPI_RightCorner,
        }
        # NOTE: pythonocc-core's stub typing represents these enum values as ints.
        # The OCC runtime expects a BRepBuilderAPI_TransitionMode value.
        from typing import cast

        pipe_builder.SetTransitionMode(
            cast(
                BRepBuilderAPI_TransitionMode,
                trans_map.get(
                    transition_mode,
                    BRepBuilderAPI_TransitionMode.BRepBuilderAPI_RightCorner,
                ),
            )
        )

        # Add profile: translate/rotate flags
        # For pipe, we want OCC to rotate the section along the path.
        pipe_builder.Add(profile_wire, False, True)

        # Build the pipe
        pipe_builder.Build()

        if not pipe_builder.IsDone():
            raise RuntimeError("Failed to create pipe - sweep operation failed")

        # Make it a solid (fills the pipe)
        pipe_builder.MakeSolid()

        result_shape = pipe_builder.Shape()

        # If the wire is open (not closed), add end caps
        if not is_closed:
            # Create end cap at the start
            start_cap_face = BRepBuilderAPI_MakeFace(profile_wire).Face()

            # Get end point and tangent
            end_point = wire_adaptor.Value(last_param)
            end_tangent_vec = gp_Vec()
            end_point_temp = gp_Pnt()
            wire_adaptor.D1(last_param, end_point_temp, end_tangent_vec)
            end_tangent_dir = gp_Dir(end_tangent_vec)

            # Create perpendicular direction for end cap
            end_ref_dir = gp_Dir(0, 0, 1)
            if abs(end_tangent_dir.Z()) > 0.99:
                end_ref_dir = gp_Dir(1, 0, 0)

            # Create end cap profile at the end
            end_profile_axis = gp_Ax2(end_point, end_tangent_dir, end_ref_dir)
            end_profile_circle = gp_Circ(end_profile_axis, radius)
            end_profile_edge = BRepBuilderAPI_MakeEdge(end_profile_circle).Edge()
            end_profile_wire = BRepBuilderAPI_MakeWire(end_profile_edge).Wire()
            end_cap_face = BRepBuilderAPI_MakeFace(end_profile_wire).Face()

            # Union the pipe with the end caps
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
        profile: "OccSketch2D",
        make_solid: bool = True,
        is_frenet: bool = True,
        transition_mode: str = "right",
        auto_align_profile: bool = False,
        with_contact: bool = False,
        with_correction: bool = True,
    ) -> OccShape:
        """Sweep a *profile* sketch along this sketch (the path).

        This is a generalization of :meth:`pipe`:
        - ``self`` provides the *spine/path* (a wire)
        - ``profile`` provides the *section/profile* (a closed wire)

        Notes / expectations:
        - The ``profile`` sketch must be *closed* (able to form a valid wire).
        - The resulting shell can optionally be converted to a solid with ``make_solid``.
        - ``frenet=True`` uses an OCC corrected frame to reduce twisting along the path.

        Args:
            profile: The cross-section sketch to sweep (typically closed).
            make_solid: If True, attempt to turn the swept shell into a solid.
            frenet: If True, use Frenet mode for orientation along the path.
            with_contact: Passed through to OCC `Add()`.
            with_correction: Passed through to OCC `Add()`.

        Returns:
            OccShape: The swept shape.
        """

        from OCC.Core.BRepOffsetAPI import BRepOffsetAPI_MakePipeShell
        from OCC.Core.BRepAdaptor import BRepAdaptor_CompCurve
        from OCC.Core.gp import gp_Ax2, gp_Ax3, gp_Dir, gp_Pnt, gp_Vec
        from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Transform
        from OCC.Core.gp import gp_Trsf
        from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_TransitionMode

        # Path (spine)
        spine = self._make_wire()

        # Profile section
        profile_wire = profile._make_wire()

        # Optional: align the profile's plane so its normal matches the path tangent at the start.
        # This mimics what `pipe()` does when constructing its circular profile.
        if auto_align_profile:
            wire_adaptor = BRepAdaptor_CompCurve(spine)
            first_param = wire_adaptor.FirstParameter()

            # Start point + tangent
            start_point: gp_Pnt = wire_adaptor.Value(first_param)
            tangent_vec = gp_Vec()
            tmp_p = gp_Pnt()
            wire_adaptor.D1(first_param, tmp_p, tangent_vec)
            tangent_dir = gp_Dir(tangent_vec)

            # Choose a stable reference direction to define the rotated frame
            ref_dir = gp_Dir(0, 0, 1)
            if abs(tangent_dir.Z()) > 0.99:
                ref_dir = gp_Dir(1, 0, 0)

            # Build a target frame where the profile plane's normal is tangent to the path
            target_ax2 = gp_Ax2(start_point, tangent_dir, ref_dir)

            # Current profile frame: use its workplane origin and normal
            # (profile sketches are planar, so aligning their plane is usually sufficient)
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
            # SetDisplacement expects gp_Ax3 coordinate systems
            trsf.SetDisplacement(gp_Ax3(source_ax2), gp_Ax3(target_ax2))
            profile_wire = BRepBuilderAPI_Transform(profile_wire, trsf, True).Shape()

        sweep_builder = BRepOffsetAPI_MakePipeShell(spine)

        # Let OCC manage the moving trihedron.
        sweep_builder.SetMode(bool(is_frenet))

        # Transition handling at corners
        trans_map = {
            "transformed": BRepBuilderAPI_TransitionMode.BRepBuilderAPI_Transformed,
            "round": BRepBuilderAPI_TransitionMode.BRepBuilderAPI_RoundCorner,
            "right": BRepBuilderAPI_TransitionMode.BRepBuilderAPI_RightCorner,
        }
        # NOTE: pythonocc-core's stub typing represents these enum values as ints.
        # The OCC runtime expects a BRepBuilderAPI_TransitionMode value.
        from typing import cast

        sweep_builder.SetTransitionMode(
            cast(
                BRepBuilderAPI_TransitionMode,
                trans_map.get(
                    transition_mode,
                    BRepBuilderAPI_TransitionMode.BRepBuilderAPI_RightCorner,
                ),
            )
        )

        # The `Add` signature is Add(profile, translate, rotate)
        # We want OCC to rotate the profile along the path; translate is typically False.
        translate = False
        rotate = True
        sweep_builder.Add(profile_wire, translate, rotate)
        sweep_builder.Build()

        if not sweep_builder.IsDone():
            raise RuntimeError("Failed to sweep profile along path")

        if make_solid:
            # For closed profiles OCC can usually cap/solidify the swept shell
            sweep_builder.MakeSolid()

        return OccShape(obj=sweep_builder.Shape(), app=self.app)

    def to_png(
        self,
        file_name: Optional[str] = None,
        width: int = 800,
        height: int = 600,
        margin: float = 0.1,
    ) -> None:
        """
        Render the sketch to a PNG image.

        Args:
            file_name: Path to save the PNG file. If None, displays in a UI window instead.
            width: Image width in pixels (default: 800)
            height: Image height in pixels (default: 600)
            margin: Margin around the sketch as a fraction of size (default: 0.1)

        Raises:
            ValueError: If the sketch has no edges
            ImportError: If matplotlib is not installed
        """
        # Delegate to the workplane's rendering method
        # We need access to the edges that were used to create this face
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError(
                "matplotlib is required for sketch rendering. Install with: pip install matplotlib"
            )

        from OCC.Core.BRepAdaptor import BRepAdaptor_Curve
        from OCC.Core.GeomAbs import GeomAbs_Circle, GeomAbs_Line

        # Create figure
        fig, ax = plt.subplots(figsize=(width / 100, height / 100), dpi=100)
        ax.set_aspect("equal")

        # Collect all points to calculate bounds
        all_points = []

        # Get the most recent extruded sketch edges
        # edges = self._workplane._extruded_sketches[-1]

        # Process each edge
        for edge in self._primitives:
            occ_edge = self._primitive_to_edge(edge)
            if occ_edge is None:
                continue
            adaptor = BRepAdaptor_Curve(occ_edge)
            curve_type = adaptor.GetType()

            if curve_type == GeomAbs_Line:
                # Sample line
                first, last = adaptor.FirstParameter(), adaptor.LastParameter()
                p1 = adaptor.Value(first)
                p2 = adaptor.Value(last)
                all_points.extend([(p1.X(), p1.Y()), (p2.X(), p2.Y())])
                ax.plot([p1.X(), p2.X()], [p1.Y(), p2.Y()], "k-", linewidth=2)

            elif curve_type == GeomAbs_Circle:
                # Handle circle/arc
                circle = adaptor.Circle()
                center = circle.Location()
                radius = circle.Radius()
                first, last = adaptor.FirstParameter(), adaptor.LastParameter()

                # Check if it's a full circle or arc
                is_full_circle = abs(last - first - 6.283185307179586) < 0.01

                if is_full_circle:
                    # Draw full circle
                    from matplotlib.patches import Circle as MplCircle

                    circ = MplCircle(
                        (center.X(), center.Y()),
                        radius,
                        fill=False,
                        edgecolor="black",
                        linewidth=2,
                    )
                    ax.add_patch(circ)
                    all_points.extend(
                        [
                            (center.X() - radius, center.Y()),
                            (center.X() + radius, center.Y()),
                            (center.X(), center.Y() - radius),
                            (center.X(), center.Y() + radius),
                        ]
                    )
                else:
                    # Draw arc - sample points along the arc
                    num_points = 50
                    arc_x = []
                    arc_y = []
                    for i in range(num_points + 1):
                        t = first + (last - first) * i / num_points
                        pnt = adaptor.Value(t)
                        arc_x.append(pnt.X())
                        arc_y.append(pnt.Y())
                        all_points.append((pnt.X(), pnt.Y()))
                    ax.plot(arc_x, arc_y, "k-", linewidth=2)

            else:
                # Generic curve - sample points
                first, last = adaptor.FirstParameter(), adaptor.LastParameter()
                num_points = 50
                curve_x = []
                curve_y = []
                for i in range(num_points + 1):
                    t = first + (last - first) * i / num_points
                    pnt = adaptor.Value(t)
                    curve_x.append(pnt.X())
                    curve_y.append(pnt.Y())
                    all_points.append((pnt.X(), pnt.Y()))
                ax.plot(curve_x, curve_y, "k-", linewidth=2)

        # Calculate bounds with margin
        if all_points:
            xs = [p[0] for p in all_points]
            ys = [p[1] for p in all_points]
            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)

            # Add margin
            x_range = max(max_x - min_x, 1)
            y_range = max(max_y - min_y, 1)
            margin_x = x_range * margin
            margin_y = y_range * margin

            ax.set_xlim(min_x - margin_x, max_x + margin_x)
            ax.set_ylim(min_y - margin_y, max_y + margin_y)

        # Style
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_title("2D Sketch")

        # Save or show
        plt.tight_layout()
        if file_name:
            plt.savefig(file_name, dpi=100, bbox_inches="tight", facecolor="white")
            plt.close()
        else:
            # Display in interactive window
            plt.show()
