from math import degrees, pi
from typing import Any, Optional


from ...app import App
from ...cad_types import Vector, VectorLike, Vertex
from ...workplane import Workplane
from ...primitives import Line, Circle, Arc


class OccWorkplane(Workplane):
    def __init__(self, app: Optional[Any] = None, *args, **kwargs):
        super().__init__(app=app, *args, **kwargs)
        # Set up coordinate system basis vectors based on normal
        if hasattr(self.__class__, "normal_vector"):
            self._setup_coordinate_system()

    @classmethod
    def xy_plane(cls, app, offset=None) -> "OccWorkplane":
        """Set the workplane to the XY plane."""
        wp = cls(app=app)
        wp.normal_vector = Vector(0, 0, 1)
        app.register_workplane(wp)
        wp._setup_coordinate_system()
        return wp

    @classmethod
    def create_offset_plane(
        cls, app: App, name: str = "XY", offset: float = 0
    ) -> Workplane:
        raise NotImplementedError("Offset plane creation not implemented yet.")

    @classmethod
    def from_origin_normal(
        cls, origin: VectorLike, normal: VectorLike, app: Optional[Any] = None
    ) -> "OccWorkplane":
        """Create an OccWorkplane from origin and normal vector.

        Args:
            origin: Origin point of the workplane
            normal: Normal vector (up-axis direction)
            app: Optional app instance

        Returns:
            New OccWorkplane with specified origin and normal
        """
        from .cad_types import Vector

        # Convert to vectors
        origin_vec = Vector(*origin) if not isinstance(origin, Vector) else origin
        normal_vec = Vector(*normal) if not isinstance(normal, Vector) else normal

        # Use default x and y directions for now
        workplane = cls(
            origin=(origin_vec.x, origin_vec.y, origin_vec.z),
            up_dir=(normal_vec.x, normal_vec.y, normal_vec.z),
        )
        if app is not None:
            app.register_workplane(workplane)
        return workplane

    def _clear_pending_shapes(self):
        """Clear pending shapes after extrusion but preserve them in extruded_sketches for visualization."""
        # Preserve the sketch before clearing
        if self._pending_shapes:
            self._extruded_sketches.append(list(self._pending_shapes))
        self._pending_shapes = []
        self._current_position = Vertex(0, 0)
        self._loop_start = None

    def rect(
        self, width: float, height: float, centered: bool = True
    ) -> "OccWorkplane":
        if centered:
            x_offset = width / 2
            y_offset = height / 2
            start_x = self._current_position.x - x_offset
            start_y = self._current_position.y - y_offset
        else:
            start_x = self._current_position.x
            start_y = self._current_position.y

        # Create rectangle as four 2D line primitives
        p1 = (start_x, start_y)
        p2 = (start_x + width, start_y)
        p3 = (start_x + width, start_y + height)
        p4 = (start_x, start_y + height)

        self._pending_shapes.extend(
            [
                Line(p1, p2),
                Line(p2, p3),
                Line(p3, p4),
                Line(p4, p1),
            ]
        )

        return self

    def box(
        self, length: float, width: float, height: float, centered: bool = True
    ) -> "OccShape":
        """
        Create a 3D box shape on the workplane.

        Args:
            length: Length of the box (along workplane X axis)
            width: Width of the box (along workplane Y axis)
            height: Height of the box (along workplane normal/Z axis)
            centered: If True (default), box is centered at current position.
                     If False, box extends from current position in positive directions.

        Returns:
            OccShape: The created box shape

        Example:
            # Create a centered box on XY plane
            box = app.work_plane("XY").box(10, 20, 30)

            # Create a box on XZ plane (respects plane orientation)
            box = app.work_plane("XZ").box(10, 20, 30)
        """
        from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
        from OCP.gp import gp_Pnt, gp_Ax3, gp_Dir, gp_Trsf
        from OCP.BRepBuilderAPI import BRepBuilderAPI_Transform
        from rapidcadpy.integrations.ocp.shape import OccShape

        # Create box in local coordinates (0,0,0 origin, axis-aligned)
        # First create at origin
        if centered:
            local_x = -length / 2
            local_y = -width / 2
            local_z = -height / 2
        else:
            local_x = 0
            local_y = 0
            local_z = 0

        corner = gp_Pnt(local_x, local_y, local_z)
        box_builder = BRepPrimAPI_MakeBox(corner, length, width, height)
        solid = box_builder.Shape()

        # Get workplane coordinate system
        if not hasattr(self, "_local_x"):
            self._setup_coordinate_system()

        # Calculate the position where the box should be placed
        # Start from current 2D position on the workplane
        center_2d = self._current_position
        center_3d = self._to_3d(center_2d.x, center_2d.y)

        # Create transformation to align box with workplane
        # The box needs to be rotated and translated
        
        # Build transformation matrix
        # Origin point (where box center should be)
        origin = gp_Pnt(center_3d[0], center_3d[1], center_3d[2])
        
        # Direction vectors for the coordinate system
        x_dir = gp_Dir(self._local_x.x, self._local_x.y, self._local_x.z)
        z_dir = gp_Dir(self._local_z.x, self._local_z.y, self._local_z.z)
        
        # Create coordinate system aligned with workplane (gp_Ax3 for 3D coordinate system)
        coord_sys = gp_Ax3(origin, z_dir, x_dir)
        
        # Create transformation from global to workplane coordinates
        trsf = gp_Trsf()
        trsf.SetTransformation(coord_sys)
        
        # Apply transformation
        transform = BRepBuilderAPI_Transform(solid, trsf, False)
        transformed_solid = transform.Shape()

        # Return as OccShape
        return OccShape(obj=transformed_solid, app=self.app)

    def revolve(
        self,
        angle: float,
        axis: str = "Z",
        operation: str = "NewBodyFeatureOperation",
    ) -> "OccWorkplane":
        """
        Revolve the current sketch around a specified axis.

        Args:
            angle: Angle in radians
            axis: Axis to revolve around ('X', 'Y', or 'Z')
            operation: Operation type (default 'NewBodyFeatureOperation')

        Returns:
            OccShape: The revolved 3D solid
        """
        from OCP.BRepPrimAPI import BRepPrimAPI_MakeRevol
        from OCP.gp import gp_Ax1, gp_Pnt, gp_Dir
        from .sketch2d import OccSketch2D

        # Check if there are shapes to revolve
        if not self._pending_shapes:
            if getattr(self.app, "silent_geometry_failures", False):
                return None
            raise ValueError("No shapes to revolve - sketch is empty")

        # Create a Sketch2D from pending shapes to use existing _make_wire and _make_face
        sketch2d = OccSketch2D(
            primitives=self._pending_shapes, workplane=self, app=self.app
        )

        # Build the face to revolve
        try:
            face = sketch2d._make_face()
            if face is None:
                return None
        except Exception:
            if getattr(self.app, "silent_geometry_failures", False):
                return None
            raise

        # Map axis string to direction
        axis_map = {
            "X": gp_Dir(1, 0, 0),
            "Y": gp_Dir(0, 1, 0),
            "Z": gp_Dir(0, 0, 1),
        }

        axis_dir = axis_map.get(axis.upper(), gp_Dir(0, 0, 1))
        axis_origin = gp_Pnt(0, 0, 0)

        # Create the revolve axis
        revolve_axis = gp_Ax1(axis_origin, axis_dir)

        # Perform the revolve operation
        # Testing: BRepPrimAPI_MakeRevol might expect radians despite documentation
        try:
            revol_builder = BRepPrimAPI_MakeRevol(face, revolve_axis, angle, True)
        except Exception:
            if getattr(self.app, "silent_geometry_failures", False):
                return None
            raise

        # Get the resulting shape
        solid = revol_builder.Shape()

        # Clear pending shapes
        self._clear_pending_shapes()

        # Return OccShape
        from .shape import OccShape

        # Handle different operations
        if operation in ["Cut", "CutOperation"]:
            # Cut operation - subtract the revolved shape from existing shapes
            from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut

            # Get all existing shapes from the app
            if self.app and hasattr(self.app, "_shapes") and self.app._shapes:
                # Perform cut on all existing shapes
                modified_shapes = []
                for existing_shape in self.app._shapes:
                    if hasattr(existing_shape, "obj"):
                        try:
                            # Subtract the revolved shape from the existing shape
                            cut_builder = BRepAlgoAPI_Cut(existing_shape.obj, solid)
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

            # If no shapes to cut from, return the revolved shape as-is
            return OccShape(obj=solid, app=self.app)

        elif operation == "JoinBodyFeatureOperation":
            # Join operation - union with existing shapes
            from OCP.BRepAlgoAPI import BRepAlgoAPI_Fuse

            # Get all existing shapes from the app
            if self.app and hasattr(self.app, "_shapes") and self.app._shapes:
                # Get the last shape to join with
                last_shape = self.app._shapes[-1]
                if hasattr(last_shape, "obj"):
                    try:
                        # Union the revolved shape with the existing shape
                        fuse_builder = BRepAlgoAPI_Fuse(last_shape.obj, solid)
                        fuse_builder.Build()

                        if fuse_builder.IsDone():
                            # Update the existing shape with the union result
                            last_shape.obj = fuse_builder.Shape()
                            return last_shape
                    except Exception:
                        pass

            # If join fails, return as new shape
            return OccShape(obj=solid, app=self.app)

        else:
            # NewBodyFeatureOperation or default - return as new shape
            return OccShape(obj=solid, app=self.app)

    def sweep(
        self,
        profile: "OccWorkplane",
        make_solid: bool = True,
        is_frenet: bool = True,
        transition_mode: str = "right",
    ) -> "OccShape":
        """
        Sweep a profile along this workplane's path to create a 3D shape.

        This method uses the current workplane's sketch as the sweep path (spine)
        and sweeps the provided profile workplane's sketch along it.

        Args:
            profile: An OccWorkplane containing the cross-section profile to sweep.
                    Must have a closed wire (e.g., created with rect(), circle()).
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
            ValueError: If either sketch is empty or invalid
            RuntimeError: If the sweep operation fails

        Example:
            # Create a path on XY plane
            path = app.work_plane("XY").line_to(10, 0).line_to(10, 10)

            # Create a circular profile on YZ plane
            profile = app.work_plane("YZ").circle(1.0)

            # Sweep the circle along the path
            shape = path.sweep(profile)
        """
        from .sketch2d import OccSketch2D
        from .shape import OccShape

        # Check if there are shapes to use as path
        if not self._pending_shapes:
            raise ValueError("No shapes in path - sketch is empty")

        # Check if profile has shapes
        if not profile._pending_shapes:
            raise ValueError("No shapes in profile - profile sketch is empty")

        # Create Sketch2D instances from pending shapes
        path_sketch = OccSketch2D(
            primitives=self._pending_shapes, workplane=self, app=self.app
        )
        profile_sketch = OccSketch2D(
            primitives=profile._pending_shapes, workplane=profile, app=self.app
        )

        # Use the sweep method from OccSketch2D
        result = path_sketch.sweep(
            profile=profile_sketch,
            make_solid=make_solid,
            is_frenet=is_frenet,
            transition_mode=transition_mode,
        )

        # Clear pending shapes
        self._clear_pending_shapes()

        return result

    def chamfer(
        self,
        x: float,
        radius: float,
        angle: float,
        distance: float,
        tol: float = 1e-3,
    ):
        """Alias for chamfer_edge to match Inventor's chamfer method name."""
        return self.app.chamfer_edge(x, radius, angle, distance, tol)
