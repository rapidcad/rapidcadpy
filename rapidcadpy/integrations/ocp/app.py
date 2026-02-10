import tempfile
import math
from typing import TYPE_CHECKING, Optional, List, Union, Tuple, Dict

from trimesh import tol

from ...app import App

if TYPE_CHECKING:
    from .workplane import OccWorkplane

from ...fea.boundary_conditions import BoundaryCondition, Load
from ...fea.materials import MaterialProperties
from .workplane import OccWorkplane

# Standard ISO metric thread data: designation -> (pitch, major_diameter)
# Values in mm
ISO_METRIC_THREADS: Dict[str, Tuple[float, float]] = {
    "M1x0.25": (0.25, 1.0),
    "M1.2x0.25": (0.25, 1.2),
    "M1.4x0.3": (0.3, 1.4),
    "M1.6x0.35": (0.35, 1.6),
    "M2x0.4": (0.4, 2.0),
    "M2.5x0.45": (0.45, 2.5),
    "M3x0.5": (0.5, 3.0),
    "M4x0.7": (0.7, 4.0),
    "M5x0.8": (0.8, 5.0),
    "M6x1": (1.0, 6.0),
    "M7x1": (1.0, 7.0),
    "M8x1": (1.0, 8.0),
    "M8x1.25": (1.25, 8.0),
    "M10x1.25": (1.25, 10.0),
    "M10x1.5": (1.5, 10.0),
    "M12x1.5": (1.5, 12.0),
    "M12x1.75": (1.75, 12.0),
    "M14x1.5": (1.5, 14.0),
    "M14x2": (2.0, 14.0),
    "M16x1.5": (1.5, 16.0),
    "M16x2": (2.0, 16.0),
    "M18x1.5": (1.5, 18.0),
    "M18x2.5": (2.5, 18.0),
    "M20x1.5": (1.5, 20.0),
    "M20x2.5": (2.5, 20.0),
    "M24x2": (2.0, 24.0),
    "M24x3": (3.0, 24.0),
    "M30x2": (2.0, 30.0),
    "M30x3.5": (3.5, 30.0),
}


class OpenCascadeOcpApp(App):

    @property
    def workplane_class(self):
        from .workplane import OccWorkplane

        return OccWorkplane

    @property
    def sketch_class(self):
        from .sketch2d import OccSketch2D

        return OccSketch2D

    def chamfer_edge(
        self,
        x: float,
        radius: float,
        angle: float,
        distance: float,
        tol: float = 1e-3,
    ) -> int:
        """Apply a distance-angle chamfer to edges that match a given X coordinate and radius.

        Searches all edges of all registered shapes and applies a chamfer to
        those whose curve center X coordinate and radius match within `tol`.

        Args:
            x: X coordinate to match for circular edges
            radius: Radius to match for circular edges
            angle: Chamfer angle in radians
            distance: Chamfer distance from edge
            tol: Tolerance for matching (default: 1e-3)

        Returns:
            int: Number of chamfers successfully created

        Raises:
            ValueError: If no shapes are registered with the app
        """
        from OCP.BRepFilletAPI import BRepFilletAPI_MakeChamfer
        from OCP.TopExp import TopExp_Explorer
        from OCP.TopAbs import TopAbs_EDGE, TopAbs_FACE
        from OCP.BRepAdaptor import BRepAdaptor_Curve
        from OCP.GeomAbs import GeomAbs_Circle
        from OCP.TopoDS import TopoDS
        from OCP.TopExp import TopExp
        from math import tan

        if not self._shapes:
            raise ValueError("No shapes registered with app for chamfer operation")

        created = 0

        # Process each shape
        for shape in self._shapes:
            if not hasattr(shape, "obj"):
                continue

            shape_obj = shape.obj
            matching_edges = []

            # Find all edges that match the criteria
            edge_explorer = TopExp_Explorer(shape_obj, TopAbs_EDGE)
            while edge_explorer.More():
                edge = TopoDS.Edge_s(edge_explorer.Current())

                # Analyze the edge curve
                curve_adaptor = BRepAdaptor_Curve(edge)
                curve_type = curve_adaptor.GetType()

                # Check if it's a circular edge
                if curve_type == GeomAbs_Circle:
                    circle = curve_adaptor.Circle()
                    center = circle.Location()
                    edge_radius = circle.Radius()

                    # Check if center X and radius match
                    if abs(center.X() - x) <= tol and abs(edge_radius - radius) <= tol:
                        matching_edges.append(edge)

                edge_explorer.Next()

            # Apply chamfer to each matching edge
            for edge in matching_edges:
                try:
                    # Find adjacent faces for this edge
                    map_shape = TopExp.MapShapesAndUniqueAncestors_s(
                        shape_obj, TopAbs_EDGE, TopAbs_FACE
                    )

                    # Get the faces adjacent to this edge
                    idx = map_shape.FindIndex(edge)
                    if idx == 0:
                        continue

                    faces_list = map_shape.FindFromIndex(idx)
                    if faces_list.Size() == 0:
                        continue

                    # Get the first adjacent face
                    face = TopoDS.Face_s(faces_list.First())

                    # Create chamfer builder
                    chamfer_builder = BRepFilletAPI_MakeChamfer(shape_obj)

                    # Add chamfer with distance and angle
                    # The angle defines the chamfer slope: tan(angle) = height/distance
                    chamfer_builder.Add(distance, distance * tan(angle), edge, face)

                    # Build the chamfered shape
                    chamfer_builder.Build()

                    if chamfer_builder.IsDone():
                        # Update the shape object
                        shape.obj = chamfer_builder.Shape()
                        created += 1

                except Exception:
                    # Skip this edge if chamfer fails
                    continue

        return created

    def add_thread(
        self,
        x: Optional[float] = None,
        y: Optional[float] = None,
        z: Optional[float] = None,
        radius: Optional[float] = None,
        axis: Optional[str] = None,
        designation: str = "M8x1.25",
        thread_class: str = "6H",
        thread_type: str = "external",
        right_handed: bool = True,
        length: float = 5.0,
        full_length: bool = True,
        offset: float = 0.0,
        modeled: bool = True,
        thread_axis: Optional[str] = None,
        tol: float = 1e-3,
    ) -> "OccShape":
        """Create a modeled thread using helical sweep.

        This creates actual thread geometry by sweeping a triangular profile
        along a helical path. Unlike Inventor's cosmetic threads, this creates
        real 3D geometry suitable for visualization and analysis.

        Args:
            x: X coordinate of the thread axis start point (default: 0)
            y: Y coordinate of the thread axis start point (default: 0)
            z: Z coordinate of the thread axis start point (default: 0)
            radius: Radius of the cylinder to thread (if None, derived from designation)
            axis: Thread axis direction ("X", "Y", or "Z", default: "Z")
            designation: Thread designation (e.g., "M8x1.25", "M10x1.5")
            thread_class: Thread class/fit (e.g., "6H", "6g") - informational only
            thread_type: "internal" or "external"
            right_handed: True for right-hand thread, False for left-hand
            length: Thread length (required)
            full_length: Whether thread spans the full length (if True, uses length param)
            offset: Offset from start point for thread placement
            modeled: True for modeled thread geometry (always True for OCC)
            thread_axis: Thread direction ("X", "Y", "Z", "-X", "-Y", "-Z") - determines which direction the length applies to
            tol: Tolerance for geometric operations

        Returns:
            OccShape: The thread geometry as a shape

        Raises:
            ValueError: If designation is not recognized or length not provided

        Example:
            # External M8x1.25 thread, 20mm long, along Z-axis
            thread = app.add_thread(
                x=0, y=0, z=0,
                designation="M8x1.25",
                thread_type="external",
                length=20.0,
                axis="Z"
            )

            # Internal M10x1.5 thread (for holes)
            thread = app.add_thread(
                designation="M10x1.5",
                thread_type="internal",
                length=15.0
            )

            # Thread with specified direction
            thread = app.add_thread(
                x=37.1925, radius=4.0, axis='X', thread_axis='-X',
                designation="M80x2", thread_type="external", length=1.7
            )
        """
        from .shape import OccShape

        # Parse thread designation to get pitch and major diameter (in mm)
        pitch_mm, major_diameter_mm = self._parse_thread_designation(designation)

        # Convert from mm to cm (model units are in cm)
        pitch = pitch_mm / 10.0
        major_diameter = major_diameter_mm / 10.0

        # Default values
        x = x if x is not None else 0.0
        y = y if y is not None else 0.0
        z = z if z is not None else 0.0

        # If thread_axis is specified, extract the axis from it
        # thread_axis can be "X", "Y", "Z", "-X", "-Y", or "-Z"
        if thread_axis is not None:
            # Extract the axis letter (remove leading '-' if present)
            axis_from_thread = thread_axis.lstrip("-").upper()
            axis = axis_from_thread
        else:
            axis = axis if axis is not None else "Z"

        if length is None:
            raise ValueError("Thread length must be specified")

        # Calculate thread dimensions based on ISO metric thread profile
        # Thread height H = 0.866025 * pitch (for 60° thread angle)
        H = 0.866025 * pitch

        if thread_type.lower() == "external":
            # External thread: thread extends outward from the base cylinder
            # For M55 thread: inner diameter is 55mm, thread extends outward
            # Minor diameter (inner) = designation diameter
            # Major diameter (outer) = minor + 2 * thread_depth
            if radius is not None:
                minor_radius = radius
            else:
                minor_radius = major_diameter / 2.0

            thread_depth = 5.0 / 8.0 * H  # Standard external thread depth
            thread_radius = minor_radius + thread_depth
        else:
            # Internal thread: cuts into a hole
            # For internal threads, the thread_radius is at the hole surface
            # and the profile extends inward
            if radius is not None:
                thread_radius = radius
            else:
                thread_radius = major_diameter / 2.0

            thread_depth = 5.0 / 8.0 * H

        # Determine direction based on thread_axis
        # The thread_axis parameter specifies which direction the thread extends
        # The x, y, z coordinates are the starting edge of the thread
        # For negative directions ("-X", "-Y", "-Z"), use negative length to reverse helix
        effective_length = length
        effective_offset = offset

        if thread_axis is not None:
            is_negative = thread_axis.startswith("-")

            if is_negative:
                # For negative direction: use negative length to signal reverse direction
                # The provided x, y, z is already the starting edge position
                effective_length = -length

        # Create the helical thread geometry
        thread_shape = self._create_thread_helix(
            center=(x, y, z),
            axis=axis.upper(),
            pitch=pitch,
            length=effective_length,
            thread_radius=thread_radius,
            thread_depth=thread_depth,
            right_handed=right_handed,
            is_internal=(thread_type.lower() == "internal"),
            offset=effective_offset,
        )

        # For internal threads, apply as a cut operation to existing shapes
        if thread_type.lower() == "internal":
            from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut

            if not self._shapes:
                raise ValueError(
                    "Internal thread requires an existing shape to cut from. "
                    "Create a cylinder or other solid before adding an internal thread."
                )

            # Cut the thread from the last existing shape
            last_shape = self._shapes[-1]
            if not hasattr(last_shape, "obj"):
                raise ValueError("Cannot cut thread from shape without geometry")

            try:
                cut_builder = BRepAlgoAPI_Cut(last_shape.obj, thread_shape)
                cut_builder.Build()

                if not cut_builder.IsDone():
                    raise RuntimeError("Thread cut operation failed")

                # Update the shape in place
                last_shape.obj = cut_builder.Shape()

                # Don't add to shapes list again - return None to indicate modification in place
                return None

            except Exception as e:
                raise RuntimeError(f"Failed to cut internal thread: {str(e)}")

        # For external threads, create and return a new shape
        result = OccShape(obj=thread_shape, app=self)
        return result

    def _parse_thread_designation(self, designation: str) -> Tuple[float, float]:
        """Parse thread designation to extract pitch and major diameter.

        Args:
            designation: Thread designation (e.g., "M8x1.25")

        Returns:
            Tuple of (pitch, major_diameter) in mm

        Raises:
            ValueError: If designation cannot be parsed
        """
        # Check if it's a known standard thread
        if designation in ISO_METRIC_THREADS:
            return ISO_METRIC_THREADS[designation]

        # Try to parse MxP format (e.g., "M8x1.25")
        designation = designation.strip().upper()
        if designation.startswith("M"):
            try:
                parts = designation[1:].split("X")
                if len(parts) == 2:
                    major_diameter = float(parts[0])
                    pitch = float(parts[1])
                    return (pitch, major_diameter)
                elif len(parts) == 1:
                    # Just diameter, use coarse pitch default
                    major_diameter = float(parts[0])
                    # Default coarse pitches for common sizes
                    coarse_pitches = {
                        1: 0.25,
                        1.2: 0.25,
                        1.4: 0.3,
                        1.6: 0.35,
                        2: 0.4,
                        2.5: 0.45,
                        3: 0.5,
                        4: 0.7,
                        5: 0.8,
                        6: 1.0,
                        7: 1.0,
                        8: 1.25,
                        10: 1.5,
                        12: 1.75,
                        14: 2.0,
                        16: 2.0,
                        18: 2.5,
                        20: 2.5,
                        24: 3.0,
                    }
                    pitch = coarse_pitches.get(major_diameter, major_diameter / 6.0)
                    return (pitch, major_diameter)
            except (ValueError, IndexError):
                pass

        raise ValueError(
            f"Unknown thread designation: {designation}. "
            f"Use format like 'M8x1.25' or one of: {list(ISO_METRIC_THREADS.keys())[:5]}..."
        )

    def _create_thread_helix(
        self,
        center: Tuple[float, float, float],
        axis: str,
        pitch: float,
        length: float,
        thread_radius: float,
        thread_depth: float,
        right_handed: bool,
        is_internal: bool,
        offset: float = 0.0,
    ) -> "TopoDS_Shape":
        """Create thread geometry using helical sweep.

        Args:
            center: Center point of the thread axis (x, y, z)
            axis: Axis direction ("X", "Y", or "Z")
            pitch: Thread pitch (distance per revolution)
            length: Total thread length
            thread_radius: Outer radius for external, inner radius for internal
            thread_depth: Depth of thread profile
            right_handed: True for right-hand thread
            is_internal: True for internal thread
            offset: Offset from center along axis

        Returns:
            TopoDS_Shape: The thread solid
        """
        from OCP.gp import gp_Pnt, gp_Dir, gp_Ax2, gp_Ax1, gp_Vec, gp_Trsf, gp_Pnt2d
        from OCP.Geom import Geom_CylindricalSurface
        from OCP.Geom2d import Geom2d_Line
        from OCP.BRepBuilderAPI import (
            BRepBuilderAPI_MakeEdge,
            BRepBuilderAPI_MakeWire,
            BRepBuilderAPI_MakeFace,
            BRepBuilderAPI_Transform,
        )
        from OCP.BRepOffsetAPI import BRepOffsetAPI_MakePipeShell
        from OCP.TopoDS import TopoDS_Wire
        from OCP.GCE2d import GCE2d_MakeSegment

        # Determine axis direction
        if axis == "X":
            axis_dir = gp_Dir(1, 0, 0)
            perp_dir = gp_Dir(0, 1, 0)
        elif axis == "Y":
            axis_dir = gp_Dir(0, 1, 0)
            perp_dir = gp_Dir(0, 0, 1)
        else:  # Z
            axis_dir = gp_Dir(0, 0, 1)
            perp_dir = gp_Dir(1, 0, 0)

        # Apply offset to center
        cx, cy, cz = center
        if axis == "X":
            cx += offset
        elif axis == "Y":
            cy += offset
        else:
            cz += offset

        center_pnt = gp_Pnt(cx, cy, cz)

        # Create helical spine (path for sweep)
        # Number of turns
        num_turns = length / pitch

        # Create the helix using parametric approach
        helix_wire = self._make_helix_wire(
            center_pnt, axis_dir, perp_dir, thread_radius, pitch, length, right_handed
        )

        # Create thread profile (triangular cross-section)
        # ISO metric thread has 60° angle
        profile_wire = self._make_thread_profile(
            center_pnt,
            axis_dir,
            perp_dir,
            thread_radius,
            thread_depth,
            pitch,
            is_internal,
        )

        # Sweep the profile along the helix
        pipe_builder = BRepOffsetAPI_MakePipeShell(helix_wire)

        # Use auxiliary spine mode for better handling of helical paths
        # SetMode(True) = Frenet mode
        pipe_builder.SetMode(True)

        # Add the profile
        pipe_builder.Add(profile_wire, False, False)

        # Build the sweep
        pipe_builder.Build()

        if not pipe_builder.IsDone():
            raise RuntimeError(
                f"Failed to create thread sweep. "
                f"Try adjusting thread parameters (pitch={pitch}, length={length})"
            )

        # Make it a solid
        pipe_builder.MakeSolid()

        return pipe_builder.Shape()

    def _make_helix_wire(
        self,
        center: "gp_Pnt",
        axis_dir: "gp_Dir",
        perp_dir: "gp_Dir",
        radius: float,
        pitch: float,
        length: float,
        right_handed: bool,
    ) -> "TopoDS_Wire":
        """Create a helical wire for thread sweep path.

        Args:
            center: Center point of helix start
            axis_dir: Direction of helix axis
            perp_dir: Perpendicular direction for starting point
            radius: Helix radius
            pitch: Distance per revolution (always positive)
            length: Total helix length along axis (negative for reverse direction)
            right_handed: True for right-hand helix

        Returns:
            TopoDS_Wire: The helical wire
        """
        from OCP.gp import gp_Pnt, gp_Vec, gp_Ax2
        from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeEdge, BRepBuilderAPI_MakeWire
        from OCP.TColgp import TColgp_Array1OfPnt
        from OCP.GeomAPI import GeomAPI_PointsToBSpline

        # Handle negative length (reverse direction)
        reverse_direction = length < 0
        abs_length = abs(length)

        # Calculate number of points for smooth helix
        # Use more points for longer threads
        num_turns = abs_length / pitch
        points_per_turn = 36  # 10 degree increments
        total_points = max(int(num_turns * points_per_turn) + 1, 10)

        # Create array of points along helix
        points = TColgp_Array1OfPnt(1, total_points)

        # Get axis vector components
        ax = axis_dir.X()
        ay = axis_dir.Y()
        az = axis_dir.Z()

        # Reverse axis direction if length was negative
        if reverse_direction:
            ax = -ax
            ay = -ay
            az = -az

        # Get perpendicular vectors for helix
        px = perp_dir.X()
        py = perp_dir.Y()
        pz = perp_dir.Z()

        # Calculate second perpendicular (cross product of axis and perp)
        qx = ay * pz - az * py
        qy = az * px - ax * pz
        qz = ax * py - ay * px

        # Generate helix points
        for i in range(total_points):
            t = i / (total_points - 1)  # Parameter from 0 to 1

            # Position along axis
            h = t * abs_length

            # Angle for this position
            angle = t * num_turns * 2 * math.pi
            if not right_handed:
                angle = -angle

            # Calculate point on helix
            cos_a = math.cos(angle)
            sin_a = math.sin(angle)

            x = center.X() + h * ax + radius * (cos_a * px + sin_a * qx)
            y = center.Y() + h * ay + radius * (cos_a * py + sin_a * qy)
            z = center.Z() + h * az + radius * (cos_a * pz + sin_a * qz)

            points.SetValue(i + 1, gp_Pnt(x, y, z))

        # Create B-spline curve through points
        spline_builder = GeomAPI_PointsToBSpline(
            points, 3, 8
        )  # degree 3, max segments 8

        if not spline_builder.IsDone():
            raise RuntimeError("Failed to create helix spline")

        helix_curve = spline_builder.Curve()

        # Create edge and wire from curve
        edge = BRepBuilderAPI_MakeEdge(helix_curve).Edge()
        wire_builder = BRepBuilderAPI_MakeWire()
        wire_builder.Add(edge)

        return wire_builder.Wire()

    def _make_thread_profile(
        self,
        center: "gp_Pnt",
        axis_dir: "gp_Dir",
        perp_dir: "gp_Dir",
        thread_radius: float,
        thread_depth: float,
        pitch: float,
        is_internal: bool,
    ) -> "TopoDS_Wire":
        """Create triangular thread profile for sweep.

        Creates an ISO metric thread profile (60° angle) positioned at the
        start of the helix.

        Args:
            center: Center of thread axis at start
            axis_dir: Direction of thread axis
            perp_dir: Perpendicular direction (radial)
            thread_radius: Major radius for external, minor for internal
            thread_depth: Depth of thread groove
            pitch: Thread pitch (for profile height calculation)
            is_internal: True for internal thread profile

        Returns:
            TopoDS_Wire: Triangular profile wire
        """
        from OCP.gp import gp_Pnt, gp_Vec
        from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeEdge, BRepBuilderAPI_MakeWire

        # ISO metric thread profile: 60° angle
        # Profile height H = pitch * sqrt(3) / 2 ≈ 0.866 * pitch
        # But we use the actual thread_depth which accounts for truncations

        # Calculate profile dimensions
        # For 60° thread, the profile is a triangle with:
        # - Height along radius = thread_depth
        # - Width along axis = pitch / 2 (approximately)
        profile_width = pitch * 0.5  # Half pitch for profile width

        # Get axis and perpendicular vectors
        ax_vec = gp_Vec(axis_dir)
        perp_vec = gp_Vec(perp_dir)

        # Calculate profile points
        # The profile is positioned at the start of the helix
        # Profile is a triangle in the plane containing axis and radial direction

        if is_internal:
            # Internal thread: profile points inward (toward center) from hole surface
            # Point 1: At hole surface (thread_radius), center of profile
            p1 = gp_Pnt(
                center.X() + thread_radius * perp_dir.X(),
                center.Y() + thread_radius * perp_dir.Y(),
                center.Z() + thread_radius * perp_dir.Z(),
            )

            # Point 2: Inward toward center by thread_depth, top of profile
            p2_center = gp_Pnt(
                center.X() + (thread_radius - thread_depth) * perp_dir.X(),
                center.Y() + (thread_radius - thread_depth) * perp_dir.Y(),
                center.Z() + (thread_radius - thread_depth) * perp_dir.Z(),
            )
            p2 = gp_Pnt(
                p2_center.X() - profile_width / 2 * axis_dir.X(),
                p2_center.Y() - profile_width / 2 * axis_dir.Y(),
                p2_center.Z() - profile_width / 2 * axis_dir.Z(),
            )

            # Point 3: At major radius (inward), bottom of profile
            p3 = gp_Pnt(
                p2_center.X() + profile_width / 2 * axis_dir.X(),
                p2_center.Y() + profile_width / 2 * axis_dir.Y(),
                p2_center.Z() + profile_width / 2 * axis_dir.Z(),
            )
        else:
            # External thread: profile points outward from major radius
            # Point 1: At major radius, center of profile
            p1 = gp_Pnt(
                center.X() + thread_radius * perp_dir.X(),
                center.Y() + thread_radius * perp_dir.Y(),
                center.Z() + thread_radius * perp_dir.Z(),
            )

            # Point 2: At minor radius (inward), top of profile
            p2_center = gp_Pnt(
                center.X() + (thread_radius - thread_depth) * perp_dir.X(),
                center.Y() + (thread_radius - thread_depth) * perp_dir.Y(),
                center.Z() + (thread_radius - thread_depth) * perp_dir.Z(),
            )
            p2 = gp_Pnt(
                p2_center.X() - profile_width / 2 * axis_dir.X(),
                p2_center.Y() - profile_width / 2 * axis_dir.Y(),
                p2_center.Z() - profile_width / 2 * axis_dir.Z(),
            )

            # Point 3: At minor radius (inward), bottom of profile
            p3 = gp_Pnt(
                p2_center.X() + profile_width / 2 * axis_dir.X(),
                p2_center.Y() + profile_width / 2 * axis_dir.Y(),
                p2_center.Z() + profile_width / 2 * axis_dir.Z(),
            )

        # Create edges of the triangle
        edge1 = BRepBuilderAPI_MakeEdge(p1, p2).Edge()
        edge2 = BRepBuilderAPI_MakeEdge(p2, p3).Edge()
        edge3 = BRepBuilderAPI_MakeEdge(p3, p1).Edge()

        # Create wire from edges
        wire_builder = BRepBuilderAPI_MakeWire()
        wire_builder.Add(edge1)
        wire_builder.Add(edge2)
        wire_builder.Add(edge3)

        if not wire_builder.IsDone():
            raise RuntimeError("Failed to create thread profile wire")

        return wire_builder.Wire()

    def to_step(self, file_name: str) -> None:
        """Export all shapes in the app to a single STEP file.

        All shapes are combined using union operations before export.

        Args:
            file_name: Path to the output STEP file
        """
        if not self._shapes:
            raise ValueError("No shapes to export")

        if len(self._shapes) == 1:
            # Single shape - export directly
            self._shapes[0].to_step(file_name)
        else:
            # Multiple shapes - union them all together
            from OCP.BRepAlgoAPI import BRepAlgoAPI_Fuse
            from OCP.ShapeFix import ShapeFix_Shape
            from .shape import OccShape

            # Start with the first shape
            combined_obj = self._shapes[0].obj

            # Fuse with each subsequent shape
            for shape in self._shapes[1:]:
                if hasattr(shape, "obj"):
                    fuse_builder = BRepAlgoAPI_Fuse(combined_obj, shape.obj)
                    # Use fuzzy tolerance to handle small gaps/overlaps
                    fuse_builder.SetFuzzyValue(1e-5)
                    fuse_builder.Build()

                    if fuse_builder.IsDone():
                        combined_obj = fuse_builder.Shape()
                    else:
                        # If fuse fails, try to continue with what we have
                        pass

            # Apply ShapeFix to heal any geometry issues from fusion
            fixer = ShapeFix_Shape(combined_obj)
            fixer.SetPrecision(1e-6)
            fixer.SetMaxTolerance(1e-3)
            fixer.Perform()
            combined_obj = fixer.Shape()

            # Create a combined shape and export
            combined_shape = OccShape(obj=combined_obj, app=self)
            combined_shape.to_step(file_name)

    def to_stl(self, file_name: str, ascii: bool = False) -> None:
        """Export all shapes in the app to a single STL file.

        When multiple shapes exist, they are written to the same STL file.
        Note: Some STL viewers may show them as separate objects.

        Args:
            file_name: Path to the output STL file
            ascii: Whether to export as ASCII STL (default: False - binary STL)
        """
        if not self._shapes:
            raise ValueError("No shapes to export")

        if len(self._shapes) == 1:
            # Single shape - export directly
            self._shapes[0].to_stl(file_name, ascii=ascii)
        else:
            # Multiple shapes - write all to the same STL file
            from OCP.BRepMesh import BRepMesh_IncrementalMesh
            from OCP.StlAPI import StlAPI_Writer
            from OCP.TopoDS import TopoDS_Compound, TopoDS_Builder

            # Create a compound to hold all shapes
            compound = TopoDS_Compound()
            builder = TopoDS_Builder()
            builder.MakeCompound(compound)

            # Add all shapes to the compound
            for shape in self._shapes:
                if hasattr(shape, "obj"):
                    builder.Add(compound, shape.obj)

            # Mesh the compound
            tolerance = 1e-3
            angular_tolerance = 0.1
            BRepMesh_IncrementalMesh(compound, tolerance, True, angular_tolerance, True)

            # Write to STL
            writer = StlAPI_Writer()
            writer.ASCIIMode = ascii
            writer.Write(compound, file_name)

    def wrapped(self):
        """Union all registered shapes and return the underlying TopoDS_Shape.

        Returns:
            TopoDS_Shape: The fused OCC shape (equivalent to .val().wrapped)
        """
        if not self._shapes:
            raise ValueError("No shapes registered with app")

        # Single shape - return its underlying TopoDS_Shape
        if len(self._shapes) == 1:
            shape = self._shapes[0]
            if not hasattr(shape, "obj"):
                raise ValueError("Shape has no geometry to wrap")
            return shape.obj

        from OCP.BRepAlgoAPI import BRepAlgoAPI_Fuse
        from OCP.ShapeFix import ShapeFix_Shape

        # Start with the first shape
        combined_obj = self._shapes[0].obj

        # Fuse with each subsequent shape
        for shape in self._shapes[1:]:
            if not hasattr(shape, "obj"):
                continue

            fuse_builder = BRepAlgoAPI_Fuse(combined_obj, shape.obj)
            # Use fuzzy tolerance to handle small gaps/overlaps
            fuse_builder.SetFuzzyValue(1e-5)
            fuse_builder.Build()

            if fuse_builder.IsDone():
                combined_obj = fuse_builder.Shape()

        # Heal any geometry issues from fusion
        fixer = ShapeFix_Shape(combined_obj)
        fixer.SetPrecision(1e-6)
        fixer.SetMaxTolerance(1e-3)
        fixer.Perform()
        combined_obj = fixer.Shape()

        # Return the underlying TopoDS_Shape (equivalent to .val().wrapped)
        return combined_obj
