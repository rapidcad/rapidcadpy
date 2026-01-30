"""
OccSketch2D - OpenCASCADE implementation of Sketch2D.
"""

from typing import Any, Optional

from OCP.BRepPrimAPI import BRepPrimAPI_MakePrism
from OCP.gp import gp_Vec
from OCP.BOPAlgo import BOPAlgo_Tools
from OCP.BRepBuilderAPI import (
    BRepBuilderAPI_MakeEdge,
    BRepBuilderAPI_MakeWire,
    BRepBuilderAPI_WireError,
)
from OCP.GC import GC_MakeArcOfCircle
from OCP.gp import gp_Pnt
from OCP.gp import gp_Ax2, gp_Circ, gp_Dir
from OCP.TopTools import TopTools_ListOfShape
from OCP.TopoDS import TopoDS_Compound
from .integrations.ocp.shape import OccShape
from .primitives import Arc, Circle, Line
from .sketch2d import Sketch2D


class OccSketch2D(Sketch2D):
    """
    OpenCASCADE-specific implementation of a 2D sketch.

    This class holds a constructed TopoDS_Face and can extrude it
    into a 3D solid along the workplane's normal direction.
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
        elif isinstance(primitive, Arc):
            # Convert arc points to 3D
            start_3d = self._workplane._to_3d(primitive.start[0], primitive.start[1])
            mid_3d = self._workplane._to_3d(primitive.mid[0], primitive.mid[1])
            end_3d = self._workplane._to_3d(primitive.end[0], primitive.end[1])

            start_point_ocp = gp_Pnt(start_3d[0], start_3d[1], start_3d[2])
            mid_point_ocp = gp_Pnt(mid_3d[0], mid_3d[1], mid_3d[2])
            end_point_ocp = gp_Pnt(end_3d[0], end_3d[1], end_3d[2])

            circle_geom = GC_MakeArcOfCircle(
                start_point_ocp, mid_point_ocp, end_point_ocp
            ).Value()

            return BRepBuilderAPI_MakeEdge(circle_geom).Edge()
        else:
            raise NotImplementedError(
                f"Primitive type {type(primitive)} not supported in OCC sketch."
            )

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
        from OCP.TopExp import TopExp_Explorer
        from OCP.TopAbs import TopAbs_FACE

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
        status = BOPAlgo_Tools.WiresToFaces_s(wires, rv)

        if not status:
            raise ValueError(
                f"Face construction failed: BOPAlgo_Tools.WiresToFaces_s returned False. "
                f"The wire with {num_edges} edge(s) could not be converted to a face. "
                f"This usually means:\n"
                f"  1. The sketch is not closed (check if you need to call .close())\n"
                f"  2. The edges are self-intersecting\n"
                f"  3. The edges don't form a planar loop\n"
                f"Workplane normal: {self.normal_vector if hasattr(self, 'normal_vector') else 'unknown'}"
            )

        # Extract the first face from the compound
        explorer = TopExp_Explorer(rv, TopAbs_FACE)
        if explorer.More():
            return explorer.Current()
        else:
            raise ValueError(
                f"Face construction failed: No face found in the resulting compound. "
                f"Wire was created from {num_edges} edge(s) but did not produce a face. "
                f"This may indicate:\n"
                f"  1. The wire is degenerate (has zero area)\n"
                f"  2. The edges are collinear\n"
                f"  3. The wire is not properly closed"
            )

    def extrude(
        self,
        distance: float,
        operation: str = "NewBodyFeatureOperation",
        symmetric: bool = False,
    ) -> OccShape:
        """
        Extrude the sketch face along the workplane's normal direction.

        Args:
            distance: Distance to extrude (can be negative for opposite direction)
            operation: Operation type - "NewBodyFeatureOperation", "JoinBodyFeatureOperation", "Cut", or "CutOperation"
            symmetric: Whether to extrude symmetrically in both directions (distance/2 each way)
        Returns:
            OccShape: The extruded 3D solid (or modified existing shape for cut operations)
        """
        face = self._make_face()

        if symmetric:
            # For symmetric extrusion, translate the face back by distance/2,
            # then extrude by the full distance
            from OCP.gp import gp_Trsf
            from OCP.BRepBuilderAPI import BRepBuilderAPI_Transform

            # Calculate translation vector (move back by half distance)
            half_distance = distance / 2.0
            translate_vec = self._workplane.normal_vector * (-half_distance)

            # Create translation transformation
            transform = gp_Trsf()
            transform.SetTranslation(
                gp_Vec(
                    float(translate_vec[0]),
                    float(translate_vec[1]),
                    float(translate_vec[2]),
                )
            )

            # Apply transformation to face
            transform_builder = BRepBuilderAPI_Transform(face, transform, True)
            face = transform_builder.Shape()

            # Now extrude by the full distance
            up_dir_vec = self._workplane.normal_vector * distance
        else:
            # Normal extrusion
            up_dir_vec = self._workplane.normal_vector * distance

        # Convert to gp_Vec using indexing to avoid attribute access issues
        extrude_vector = gp_Vec(
            float(up_dir_vec[0]), float(up_dir_vec[1]), float(up_dir_vec[2])
        )

        # Create the prism (extrusion)
        prism_builder: Any = BRepPrimAPI_MakePrism(face, extrude_vector, True)
        extruded_shape = prism_builder.Shape()

        # Handle different operations
        if operation in ["Cut", "CutOperation"]:
            # Cut operation - subtract the extruded shape from existing shapes
            from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut

            # Get all existing shapes from the app
            if self.app and hasattr(self.app, "_shapes") and self.app._shapes:
                # Perform cut on all existing shapes
                modified_shapes = []
                for existing_shape in self.app._shapes:
                    if hasattr(existing_shape, "obj"):
                        try:
                            # Subtract the extruded shape from the existing shape
                            cut_builder = BRepAlgoAPI_Cut(
                                existing_shape.obj, extruded_shape
                            )
                            cut_builder.Build()

                            if cut_builder.IsDone():
                                # Update the existing shape with the cut result
                                existing_shape.obj = cut_builder.Shape()
                                modified_shapes.append(existing_shape)
                        except Exception:
                            # If cut fails, keep the original shape
                            continue

                # Return the last modified shape (or first if available)
                if modified_shapes:
                    return modified_shapes[-1]

            # If no shapes to cut from, return the extruded shape as-is
            return OccShape(obj=extruded_shape, app=self.app)

        elif operation == "JoinBodyFeatureOperation":
            # Join operation - union with existing shapes
            from OCP.BRepAlgoAPI import BRepAlgoAPI_Fuse

            # Get all existing shapes from the app
            if self.app and hasattr(self.app, "_shapes") and self.app._shapes:
                # Get the last shape to join with
                last_shape = self.app._shapes[-1]
                if hasattr(last_shape, "obj"):
                    try:
                        # Union the extruded shape with the existing shape
                        fuse_builder = BRepAlgoAPI_Fuse(last_shape.obj, extruded_shape)
                        fuse_builder.Build()

                        if fuse_builder.IsDone():
                            # Update the existing shape with the union result
                            last_shape.obj = fuse_builder.Shape()
                            return last_shape
                    except Exception:
                        pass

            # If join fails, return as new shape
            return OccShape(obj=extruded_shape, app=self.app)

        else:
            # NewBodyFeatureOperation or default - return as new shape
            return OccShape(obj=extruded_shape, app=self.app)

    def pipe(self, diameter: float) -> OccShape:
        """
        Create a pipe (cylindrical extrusion) along the sketch path.

        For open wires (non-closed paths), circular end caps are automatically
        added to create a closed solid volume.

        Args:
            diameter: Diameter of the pipe (outer diameter)

        Returns:
            OccShape: The resulting pipe shape
        """
        from OCP.BRepBuilderAPI import (
            BRepBuilderAPI_MakeEdge,
            BRepBuilderAPI_MakeWire,
            BRepBuilderAPI_MakeFace,
        )
        from OCP.BRepOffsetAPI import BRepOffsetAPI_MakePipeShell
        from OCP.BRepAlgoAPI import BRepAlgoAPI_Fuse
        from OCP.gp import gp_Ax2, gp_Circ, gp_Dir, gp_Pnt, gp_Vec
        from OCP.BRepAdaptor import BRepAdaptor_CompCurve
        from OCP.BRep import BRep_Tool

        # Get the wire that represents the path (spine)
        spine = self._make_wire()

        # Calculate the radius from diameter
        radius = diameter / 2.0

        # Use BRepAdaptor_CompCurve to work with the wire directly
        wire_adaptor = BRepAdaptor_CompCurve(spine)
        first_param = wire_adaptor.FirstParameter()
        last_param = wire_adaptor.LastParameter()

        # Check if wire is closed
        is_closed = BRep_Tool.IsClosed_s(spine)

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

        # Set Frenet mode to automatically handle profile orientation along the path
        # This prevents twisting and handles corners properly
        pipe_builder.SetMode(True)  # True = Frenet mode (corrected frame)

        # Add the profile to the pipe
        # Parameters: profile_wire, withContact=False, withCorrection=False
        pipe_builder.Add(profile_wire, True, True)

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
        if (
            not hasattr(self._workplane, "_extruded_sketches")
            or not self._workplane._extruded_sketches
        ):
            raise ValueError("No sketch edges available for rendering")

        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError(
                "matplotlib is required for sketch rendering. Install with: pip install matplotlib"
            )

        from OCP.BRepAdaptor import BRepAdaptor_Curve
        from OCP.GeomAbs import GeomAbs_Circle, GeomAbs_Line

        # Create figure
        fig, ax = plt.subplots(figsize=(width / 100, height / 100), dpi=100)
        ax.set_aspect("equal")

        # Collect all points to calculate bounds
        all_points = []

        # Get the most recent extruded sketch edges
        edges = self._workplane._extruded_sketches[-1]

        # Process each edge
        for edge in edges:
            adaptor = BRepAdaptor_Curve(edge)
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

    def sweep(
        self,
        profile: Sketch2D,
        make_solid: bool = True,
        is_frenet: bool = True,
        transition_mode: str = "right",
    ) -> OccShape:
        """
        Sweep a profile along this sketch's path to create a 3D shape.

        This method uses the current sketch as the sweep path (spine) and
        sweeps the provided profile along it to create a solid or shell.

        Args:
            profile: A Sketch2D defining the cross-section profile to sweep.
                    Must be a closed wire (e.g., a rectangle, circle, or closed polygon).
            make_solid: If True (default), creates a solid. If False, creates a shell.
            is_frenet: If True (default), uses Frenet frame for profile orientation.
                      This keeps the profile perpendicular to the path.
            transition_mode: Transition mode at path corners. Options:
                           - "right": Right corner (sharp)
                           - "round": Rounded corner
                           - "transformed": Transformed corner

        Returns:
            OccShape: The resulting swept 3D shape

        Raises:
            ValueError: If the profile sketch is empty or invalid
            RuntimeError: If the sweep operation fails

        Example:
            # Create a path
            path = app.work_plane("XY").line_to(10, 0).line_to(10, 10)

            # Create a circular profile
            profile = app.work_plane("YZ").circle(1.0)

            # Sweep the circle along the path
            shape = path.sweep(profile)
        """
        from OCP.BRepBuilderAPI import (
            BRepBuilderAPI_MakeEdge,
            BRepBuilderAPI_MakeWire,
            BRepBuilderAPI_MakeFace,
        )
        from OCP.BRepOffsetAPI import BRepOffsetAPI_MakePipeShell
        from OCP.BRepBuilderAPI import BRepBuilderAPI_TransitionMode
        from OCP.gp import gp_Ax2, gp_Dir, gp_Pnt, gp_Vec
        from OCP.BRepAdaptor import BRepAdaptor_CompCurve
        from OCP.BRep import BRep_Tool

        # Get the spine (path) from this sketch
        spine = self._make_wire()

        # Get the profile wire from the provided sketch
        if not isinstance(profile, OccSketch2D):
            raise ValueError(
                "Profile must be an OccSketch2D instance. "
                "Create it using app.work_plane(...).circle/rect/etc."
            )

        profile_wire = profile._make_wire()

        # Set up the pipe shell builder with the spine
        pipe_builder = BRepOffsetAPI_MakePipeShell(spine)

        # Set Frenet mode for profile orientation
        if is_frenet:
            pipe_builder.SetMode(True)  # Frenet mode

        # Set transition mode
        if transition_mode.lower() == "round":
            pipe_builder.SetTransitionMode(
                BRepBuilderAPI_TransitionMode.BRepBuilderAPI_RoundCorner
            )
        elif transition_mode.lower() == "transformed":
            pipe_builder.SetTransitionMode(
                BRepBuilderAPI_TransitionMode.BRepBuilderAPI_Transformed
            )
        else:
            pipe_builder.SetTransitionMode(
                BRepBuilderAPI_TransitionMode.BRepBuilderAPI_RightCorner
            )

        # Add the profile to the pipe
        # Parameters: profile, withContact=False, withCorrection=True
        pipe_builder.Add(profile_wire, False, True)

        # Build the sweep
        pipe_builder.Build()

        if not pipe_builder.IsDone():
            raise RuntimeError(
                "Failed to create sweep - the profile could not be swept along the path. "
                "Ensure the profile is properly oriented relative to the path start."
            )

        # Make solid if requested
        if make_solid:
            pipe_builder.MakeSolid()

        result_shape = pipe_builder.Shape()

        return OccShape(obj=result_shape, app=self.app)
