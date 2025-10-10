import math
from typing import TYPE_CHECKING, Optional, Tuple

from win32com.client import constants

from rapidcadpy import Workplane
from rapidcadpy.cad_types import VectorLike, Vertex

from rapidcadpy.integrations.inventor.shape import InventorShape

if TYPE_CHECKING:
    from rapidcadpy.integrations.inventor.app import InventorApp

class InventorWorkPlane(Workplane):
    def __init__(
        self,
        origin: VectorLike = (0.0, 0.0, 0.0),
        x_dir: VectorLike = (1.0, 0.0, 0.0),
        y_dir: VectorLike = (0.0, 1.0, 0.0),
        z_dir: VectorLike = (0.0, 0.0, 1.0),
        app: Optional["InventorApp"] = None,
    ):
        super().__init__(origin=origin, x_dir=x_dir, y_dir=y_dir, z_dir=z_dir, app=app)
        self.tg = self.app.transient_geom
        self.app: "InventorApp" = app  # type: ignore

        # create a workplane in inventor
        origin_pt = self.tg.CreatePoint(origin[0], origin[1], origin[2])
        x_vec = self.tg.CreateUnitVector(x_dir[0], x_dir[1], x_dir[2])
        y_vec = self.tg.CreateUnitVector(y_dir[0], y_dir[1], y_dir[2])
        work_plane = self.app.comp_def.WorkPlanes.AddFixed(
            origin_pt, x_vec, y_vec, False
        )
        work_plane.Visible = False

        # add a sketch on the workplane
        self.sketch = self.app.comp_def.Sketches.Add(work_plane)

        # no duplicate point objects in the sketch, use dict
        self.point2d_dict = {}
        self.sketchpoint_dict = {}

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

        # Reset the point dictionaries to avoid conflicts
        self.point2d_dict = {}
        self.sketchpoint_dict = {}

        # Reset current position to origin
        self._current_position = Vertex(0, 0)

    def line_to(self, x: float, y: float) -> "InventorWorkPlane":
        start_point = self._current_position
        sp = self._get_sketch_point(start_point.x, start_point.y)
        ep = self._get_sketch_point(x, y)
        self.sketch.SketchLines.AddByTwoPoints(sp, ep)
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

    def three_point_arc(
        self, p1: Tuple[float, float], p2: Tuple[float, float]
    ) -> "InventorWorkPlane":
        """
        Create a three-point arc using the current position as the first point.

        Args:
            p1: Tuple (x, y) for the middle point of the arc
            p2: Tuple (x, y) for the end point of the arc

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
        self, distance: float, operation: str = "NewBodyFeatureOperation"
    ) -> InventorShape:
        profile = self.sketch.Profiles.AddForSolid()
        ext_op = {
            "JoinBodyFeatureOperation": 1,
            "Cut": 2,
            "Intersect": 3,
            "NewBodyFeatureOperation": constants.kNewBodyOperation,
        }.get(operation, 0)
        ext_def = self.app.comp_def.Features.ExtrudeFeatures.CreateExtrudeDefinition(
            profile, ext_op
        )
        ext_def.SetDistanceExtent(distance, constants.kPositiveExtentDirection)
        extrusion_feature = self.app.comp_def.Features.ExtrudeFeatures.Add(ext_def)
        # After extrusion, create a new sketch for future operations
        self._create_new_sketch()

        # Return the shape representing the extruded body
        return InventorShape(obj=extrusion_feature.SurfaceBody, app=self.app)

    def revolve(
        self,
        angle: float,
        axis: Tuple[float, float],
        operation: str = "NewBodyFeatureOperation",
    ) -> InventorShape:
        """
        Revolve the current sketch around a specified axis.
        """
        profile = self.sketch.Profiles.AddForSolid()
        rev_op = {
            "JoinBodyFeatureOperation": constants.kJoinOperation,
            "NewBodyFeatureOperation": constants.kNewBodyOperation,
        }.get(operation, 0)

        # Find or create a WorkAxis at the given axis point
        # For simplicity, use the first work axis (usually the origin Z axis)
        # You may need to create a custom axis for arbitrary axes
        work_axes = self.app.comp_def.WorkAxes(3)

        if abs(angle - 2 * math.pi) < 1e-6 or abs(angle - 360) < 1e-3:
            # Full revolve
            revolve_feature = self.app.comp_def.Features.RevolveFeatures.AddFull(
                profile, work_axes, rev_op
            )
        else:
            # Partial revolve
            revolve_feature = self.app.comp_def.Features.RevolveFeatures.AddByAngle(
                profile, work_axes, angle, rev_op
            )

        self._create_new_sketch()
        return InventorShape(obj=revolve_feature.SurfaceBody, app=self.app)
