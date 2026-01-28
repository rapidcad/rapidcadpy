import math
from typing import TYPE_CHECKING, Any, Optional

from win32com.client import constants

from rapidcadpy import Workplane
from rapidcadpy.cad_types import VectorLike, Vertex
from rapidcadpy.integrations.inventor.shape import InventorShape

if TYPE_CHECKING:
    pass


class InventorWorkPlane(Workplane):
    def __init__(self, app: Any, sketch):
        super().__init__(app=app)
        self.tg = self.app.transient_geom
        self.app: Any = app  # type: ignore
        self.sketch = sketch
        # Track all sketches for multi-sketch extrusion
        self.sketches = [sketch]
        # no duplicate point objects in the sketch, use dict
        self.point2d_dict = {}
        self.sketchpoint_dict = {}
        # Track the start of the current loop for close() - must be Vertex to match base class
        self._loop_start: Vertex = Vertex(0, 0)

    @classmethod
    def from_origin_normal(
        cls, app: Any, origin: VectorLike, normal: VectorLike
    ) -> "InventorWorkPlane":
        """Create an InventorWorkPlane from origin and normal vector.

        Args:
            app: InventorApp instance
            origin: Origin point of the workplane (3D coordinates)
            normal: Normal vector (up-axis direction)
        Returns:
            New InventorWorkPlane with specified origin and normal
        """
        tg = app.transient_geom

        # Ensure we have 3D coordinates
        origin_3d = origin if len(origin) == 3 else (origin[0], origin[1], 0.0)
        normal_3d = normal if len(normal) == 3 else (normal[0], normal[1], 0.0)

        # create a workplane in inventor using up_dir as the normal
        origin_pt = tg.CreatePoint(origin_3d[0], origin_3d[1], origin_3d[2])
        normal_vec = tg.CreateUnitVector(normal_3d[0], normal_3d[1], normal_3d[2])

        # Choose a reference vector that is not parallel to the normal
        # Check if normal is aligned with X-axis
        if abs(normal_3d[0]) > 0.9:
            # Use Y-axis as reference
            ref_vec = tg.CreateUnitVector(0, 1, 0)
        else:
            # Use X-axis as reference
            ref_vec = tg.CreateUnitVector(1, 0, 0)

        # Create perpendicular vectors for the workplane
        x_vec = normal_vec.CrossProduct(ref_vec)
        y_vec = x_vec.CrossProduct(normal_vec)

        # in inventor the y vector is the up direction (normal)
        work_plane = app.comp_def.WorkPlanes.AddFixed(
            origin_pt, x_vec, normal_vec, False
        )
        work_plane.Visible = False

        # add a sketch on the workplane
        sketch = app.comp_def.Sketches.Add(work_plane)
        return cls(
            app=app,
            sketch=sketch,
        )

    @classmethod
    def xy_plane(
        cls, app: Optional[Any] = None, offset: Optional[float] = None
    ) -> "InventorWorkPlane":
        """Create an InventorWorkPlane in the XY orientation at the given origin.

        Args:
            app: InventorApp instance
            offset: Offset distance along the normal (not yet implemented for Inventor)
        Returns:
            New InventorWorkPlane in the XY orientation at the specified origin
        """
        if app is None:
            raise ValueError("app parameter is required for InventorWorkPlane")
        return cls(
            app=app,
            sketch=app.comp_def.Sketches.Add(app.comp_def.WorkPlanes(3)),
        )

    @classmethod
    def xz_plane(
        cls, app: Optional[Any] = None, offset: Optional[float] = None
    ) -> "InventorWorkPlane":
        """Create an InventorWorkPlane in the XZ orientation at the given origin.

        Args:
            app: InventorApp instance
            offset: Offset distance along the normal (not yet implemented for Inventor)
        """
        if app is None:
            raise ValueError("app parameter is required for InventorWorkPlane")
        return cls(
            app=app,
            sketch=app.comp_def.Sketches.Add(app.comp_def.WorkPlanes(2)),
        )

    @classmethod
    def yz_plane(
        cls, app: Optional[Any] = None, offset: Optional[float] = None
    ) -> "InventorWorkPlane":
        """Create an InventorWorkPlane in the YZ orientation at the given origin.

        Args:
            app: InventorApp instance
            offset: Offset distance along the normal (not yet implemented for Inventor)
        """
        if app is None:
            raise ValueError("app parameter is required for InventorWorkPlane")
        return cls(
            app=app,
            sketch=app.comp_def.Sketches.Add(app.comp_def.WorkPlanes(1)),
        )

    @classmethod
    def create_offset_plane(
        cls, app: Any, name: str = "XY", offset: float = 0.0, *args, **kwargs
    ) -> "InventorWorkPlane":
        """Create a standard named workplane with an offset.

        Args:
            name: Standard plane name ("XY", "XZ", "YZ")
            offset: Offset distance from the standard plane
            app: Optional app instance
        Returns:
            InventorWorkPlane instance at the specified offset
        """
        if name == "XY":
            plane = app.comp_def.WorkPlanes.AddByPlaneAndOffset(
                app.comp_def.WorkPlanes(3), offset
            )
        elif name == "XZ":
            plane = app.comp_def.WorkPlanes.AddByPlaneAndOffset(
                app.comp_def.WorkPlanes(2), offset
            )
        elif name == "YZ":
            plane = app.comp_def.WorkPlanes.AddByPlaneAndOffset(
                app.comp_def.WorkPlanes(1), offset
            )
        else:
            raise ValueError(f"Unsupported plane name '{name}' for offset plane")
        sketch = app.comp_def.Sketches.Add(plane)
        return cls(app=app, sketch=sketch)

    def _get_point2d(self, x, y):
        key = (round(x, 6), round(y, 6))
        if key not in self.point2d_dict:
            self.point2d_dict[key] = self.tg.CreatePoint2d(x, y)
        return self.point2d_dict[key]

    def _get_sketch_point(self, x, y):
        key = (round(x, 6), round(y, 6))
        if key not in self.sketchpoint_dict:
            pt2d = self._get_point2d(x, y)
            self.sketchpoint_dict[key] = self.sketch.SketchPoints.Add(pt2d)
        return self.sketchpoint_dict[key]

    def _create_new_sketch(self):
        """Create a new sketch on the same workplane for subsequent operations."""
        # Get the current workplane
        current_workplane = self.sketch.PlanarEntity

        # Create a new sketch on the same workplane
        self.sketch = self.app.comp_def.Sketches.Add(current_workplane)

        # Append the new sketch to the sketches list for multi-sketch extrusion
        self.sketches.append(self.sketch)

        # Reset the point dictionaries to avoid conflicts
        self.point2d_dict = {}
        self.sketchpoint_dict = {}

        # Reset current position to origin
        self._current_position = Vertex(0, 0)
        # Reset loop start for the new sketch
        self._loop_start = Vertex(0, 0)

    def close(self) -> "InventorWorkPlane":  # type: ignore[override]
        """Start a new Inventor Sketch on the same workplane.

        Useful for segmenting geometry into separate sketches before an extrude.
        """
        # add a line from current position to loop start if they are not the same
        if (
            abs(self._current_position.x - self._loop_start.x) > 1e-9
            or abs(self._current_position.y - self._loop_start.y) > 1e-9
        ):
            self.line_to(self._loop_start.x, self._loop_start.y)
        self._create_new_sketch()
        return self

    def add(self, loops) -> "InventorWorkPlane":
        """ " not needed yet"""
        return self

    def move_to(self, x: float, y: float) -> "InventorWorkPlane":
        """Move the current position to the specified point."""
        self._current_position = Vertex(x, y)
        # Each move_to is treated as the start of a new loop within the sketch
        self._loop_start = Vertex(x, y)
        return self

    def line_to(self, x: float, y: float) -> "InventorWorkPlane":
        """Draw a line from the current position to the specified point."""
        # Get start point from current position
        start_point = self._current_position

        # Create sketch points for start and end
        sp = self._get_sketch_point(start_point.x, start_point.y)
        ep = self._get_sketch_point(x, y)

        # Only create a line if start and end points are different
        if abs(start_point.x - x) > 1e-9 or abs(start_point.y - y) > 1e-9:
            self.sketch.SketchLines.AddByTwoPoints(sp, ep)

        # Update current position
        self._current_position = Vertex(x, y)
        return self

    def circle(self, radius: float) -> "InventorWorkPlane":
        """
        Create a circle at the current position.

        Args:
            radius: Radius of the circle

        Returns:
            Self for method chaining
        """
        center = self._get_sketch_point(
            self._current_position.x, self._current_position.y
        )
        self.sketch.SketchCircles.AddByCenterRadius(center, radius)
        return self

    def rect(self, width: float, height: float, centered: bool = True) -> "Workplane":
        if centered:
            start_x = self._current_position.x - width / 2
            start_y = self._current_position.y - height / 2
        else:
            start_x = self._current_position.x
            start_y = self._current_position.y
        x2 = start_x + width
        y2 = start_y + height
        if centered:
            c = self.tg.CreatePoint2d(
                self._current_position.x, self._current_position.y
            )
            cr = self.tg.CreatePoint2d(
                self._current_position.x + width / 2,
                self._current_position.y + height / 2,
            )
            self.sketch.SketchLines.AddAsTwoPointCenteredRectangle(c, cr)
        else:
            bl = self.tg.CreatePoint2d(start_x, start_y)
            tr = self.tg.CreatePoint2d(x2, y2)
            self.sketch.SketchLines.AddAsTwoPointRectangle(bl, tr)
        return self

    def three_point_arc(self, p1: VectorLike, p2: VectorLike) -> "InventorWorkPlane":
        """
        Create a three-point arc using the current position as the first point.

        Args:
            p1: Middle point of the arc (x, y) or (x, y, z)
            p2: End point of the arc (x, y) or (x, y, z)

        Returns:
            InventorWorkPlane: Self for method chaining
        """
        # Get the current position as the start point
        start_point = self._current_position

        # Create the three points for the arc
        pt1 = self._get_sketch_point(start_point.x, start_point.y)  # Start point
        pt2 = self._get_point2d(p1[0], p1[1])  # Middle point (for arc definition)
        pt3 = self._get_sketch_point(p2[0], p2[1])  # End point

        # Create the arc using Inventor's SketchArcs.AddByThreePoints method
        self.sketch.SketchArcs.AddByThreePoints(pt1, pt2, pt3)

        # Update current position to the end point
        self._current_position = Vertex(p2[0], p2[1])

        return self

    def extrude(
        self,
        distance: float,
        operation: str = "NewBodyFeatureOperation",
        both: bool = False,
        symmetric: bool = False,
    ) -> InventorShape:
        """Extrude all sketches in the workplane.

        Args:
            distance: Extrusion distance (positive or negative)
            operation: Operation type for extrusion
            both: Whether to extrude both directions (deprecated, use symmetric instead)
            symmetric: Whether to extrude symmetrically in both directions

        Returns:
            InventorShape representing the last extruded body (or combined result)
        """
        # Map operation string to Inventor constants
        ext_op = {
            "JoinBodyFeatureOperation": constants.kJoinOperation,
            "Cut": constants.kCutOperation,
            "CutOperation": constants.kCutOperation,
            "Intersect": constants.kIntersectOperation,
            "NewBodyFeatureOperation": constants.kNewBodyOperation,
        }.get(operation, constants.kNewBodyOperation)

        # Determine extrusion direction
        if symmetric or both:
            direction = constants.kSymmetricExtentDirection
        elif distance < 0:
            direction = constants.kNegativeExtentDirection
        else:
            direction = constants.kPositiveExtentDirection

        extruded_shapes = []

        # Helper function to check if a sketch has geometry
        def sketch_has_geometry(sketch):
            """Check if sketch has any geometric entities."""
            return (
                sketch.SketchLines.Count > 0
                or sketch.SketchCircles.Count > 0
                or sketch.SketchArcs.Count > 0
                or sketch.SketchEllipses.Count > 0
                or sketch.SketchSplines.Count > 0
            )

        # Filter out empty sketches
        valid_sketches = [s for s in self.sketches if sketch_has_geometry(s)]

        if not valid_sketches:
            raise RuntimeError("No valid sketches to extrude - all sketches are empty")

        # Extrude each valid sketch
        for sketch_idx, sketch in enumerate(valid_sketches):
            # Always create a single composite profile per sketch.
            # This preserves inner loops (holes) as voids during extrusion.
            try:
                # Ensure a composite profile exists in the collection
                if sketch.Profiles.Count == 0:
                    # Create a profile representing all closed regions in this sketch
                    # Let Inventor detect outer/inner loops and build one composite profile
                    sketch.Profiles.AddForSolid()
            except Exception as e:
                # If creating a new profile fails, skip this sketch
                if sketch.Profiles.Count == 0:
                    print(
                        f"Warning: Skipping sketch {sketch_idx} - could not create a composite profile: {e}"
                    )
                    continue

            # Choose a profile that contains inner loops if available (to preserve holes)
            selected_profile = None
            try:
                for i in range(1, sketch.Profiles.Count + 1):
                    p = sketch.Profiles.Item(i)
                    loops = getattr(p, "ProfileLoops", None)
                    if loops is None:
                        continue
                    inner_count = 0
                    try:
                        for j in range(1, loops.Count + 1):
                            loop = loops.Item(j)
                            if (
                                getattr(loop, "LoopType", None)
                                == constants.kInnerProfileLoop
                            ):
                                inner_count += 1
                    except Exception:
                        pass
                    if inner_count > 0:
                        selected_profile = p
                        break
                # Fallback: pick the profile with the most loops (likely composite)
                if selected_profile is None and sketch.Profiles.Count > 0:
                    best = None
                    best_loops = -1
                    for i in range(1, sketch.Profiles.Count + 1):
                        p = sketch.Profiles.Item(i)
                        loops = getattr(p, "ProfileLoops", None)
                        loop_count = (
                            getattr(loops, "Count", 0) if loops is not None else 0
                        )
                        if loop_count > best_loops:
                            best = p
                            best_loops = loop_count
                    selected_profile = best or sketch.Profiles.Item(1)
            except Exception:
                if sketch.Profiles.Count > 0:
                    selected_profile = sketch.Profiles.Item(1)
                else:
                    print(
                        f"Warning: Skipping sketch {sketch_idx} - no profiles available after creation"
                    )
                    continue

            try:
                # For the first sketch use the requested op, then join subsequent ones
                current_op = (
                    ext_op if len(extruded_shapes) == 0 else constants.kJoinOperation
                )

                ext_def = (
                    self.app.comp_def.Features.ExtrudeFeatures.CreateExtrudeDefinition(
                        selected_profile, current_op
                    )
                )
                ext_def.SetDistanceExtent(abs(distance), direction)

                extrusion_feature = self.app.comp_def.Features.ExtrudeFeatures.Add(
                    ext_def
                )
                extruded_shapes.append(
                    InventorShape(obj=extrusion_feature.SurfaceBody, app=self.app)
                )
            except Exception as e:
                print(f"Warning: Failed to extrude sketch {sketch_idx}: {e}")
                continue

        # After extrusion, optionally start a new empty sketch for subsequent operations
        self._create_new_sketch()
        # Clear previously extruded sketches; keep only the new empty sketch
        self.sketches = [self.sketch]

        # Return the last extruded shape (which should contain all joined bodies)
        if not extruded_shapes:
            raise RuntimeError("No shapes were successfully extruded")
        return extruded_shapes[-1]

    def revolve(
        self,
        angle: float,
        axis: str,
        operation: str = "NewBodyFeatureOperation",
    ) -> InventorShape:
        """
        Revolve the current sketch around a specified axis.
        """
        # Check if sketch has any geometry
        if self.sketch.SketchLines.Count == 0 and self.sketch.SketchArcs.Count == 0:
            raise RuntimeError("Sketch is empty - cannot create revolve feature")

        # Create profile from the sketch
        try:
            profile = self.sketch.Profiles.AddForSolid()
        except Exception as e:
            # If AddForSolid fails, try to get existing profiles
            if self.sketch.Profiles.Count > 0:
                profile = self.sketch.Profiles.Item(1)
            else:
                raise RuntimeError(f"Failed to create profile for revolve: {e}")

        # Map operation string to Inventor constants
        rev_op = {
            "JoinBodyFeatureOperation": constants.kJoinOperation,
            "NewBodyFeatureOperation": constants.kNewBodyOperation,
            "Cut": constants.kCutOperation,
            "CutOperation": constants.kCutOperation,
        }.get(operation, constants.kNewBodyOperation)

        # For axis (0.0, 0.0), use the Z-axis work axis (typically WorkAxes.Item(3))
        # This assumes the sketch is on a plane perpendicular to Z-axis
        try:
            if axis == "Z":
                # Use the Z-axis (typically the 3rd work axis in Inventor)
                work_axis = self.app.comp_def.WorkAxes.Item(3)
            elif axis == "X":
                work_axis = self.app.comp_def.WorkAxes.Item(1)
            elif axis == "Y":
                work_axis = self.app.comp_def.WorkAxes.Item(2)
            else:
                # For other axes, we'd need to create a custom work axis
                # For now, default to Z-axis
                work_axis = self.app.comp_def.WorkAxes.Item(3)
        except:
            # Fallback: try to get any available work axis
            if self.app.comp_def.WorkAxes.Count > 0:
                work_axis = self.app.comp_def.WorkAxes.Item(1)
            else:
                raise RuntimeError("No work axes available for revolve operation")
        angle = angle * 2 * math.pi
        try:
            if abs(angle - 2 * math.pi) < 1e-6 or abs(angle - 360) < 1e-3:
                # Full revolve (360 degrees)
                revolve_feature = self.app.comp_def.Features.RevolveFeatures.AddFull(
                    profile, work_axis, rev_op
                )
            else:
                # Partial revolve
                revolve_feature = self.app.comp_def.Features.RevolveFeatures.AddByAngle(
                    profile, work_axis, angle, rev_op
                )
        except Exception as e:
            # Provide more detailed error information
            error_msg = "Failed to create revolve feature. "
            error_msg += f"Profile count: {self.sketch.Profiles.Count}, "
            error_msg += f"Work axis: {work_axis}, "
            error_msg += f"Operation: {operation} ({rev_op}), "
            error_msg += f"Angle: {angle}. "
            error_msg += f"Original error: {e}"
            raise RuntimeError(error_msg)

        # self._create_new_sketch()
        return InventorShape(obj=revolve_feature.SurfaceBody, app=self.app)
