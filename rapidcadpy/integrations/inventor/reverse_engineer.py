import math
from typing import Dict, List, Tuple

import win32com.client as win32
from win32com.client import constants


class InventorReverseEngineer:
    def __init__(self, doc):
        self.doc = doc
        self.generated_code = []

    def analyze_ipt_file(self) -> str:
        """
        Analyze an IPT file and generate Python code to recreate it.
        """

        # Open the document
        self.comp_def = self.doc.ComponentDefinition

        # Initialize code
        self.generated_code = [
            "from rapidcadpy import InventorApp",
            "",
            "# Initialize Inventor application",
            "app = InventorApp()",
            "app.new_document()",
            "",
        ]

        # Analyze all sketches and their features
        self._analyze_sketches_and_features()

        return "\n".join(self.generated_code)

    def _analyze_sketches_and_features(self):
        """Analyze sketches and their associated features in order."""
        # Get all features that use sketches
        extrude_features = self.comp_def.Features.ExtrudeFeatures
        revolve_features = self.comp_def.Features.RevolveFeatures
        chamfer_features = self.comp_def.Features.ChamferFeatures

        # Collect all features with their sketches
        feature_sketch_pairs = []

        # Add extrude features
        for i in range(1, extrude_features.Count + 1):
            feature = extrude_features.Item(i)
            sketch = feature.Profile.Parent
            feature_sketch_pairs.append(
                {"sketch": sketch, "feature": feature, "type": "extrude", "index": i}
            )

        # Add revolve features
        for i in range(1, revolve_features.Count + 1):
            feature = revolve_features.Item(i)
            sketch = feature.Profile.Parent
            feature_sketch_pairs.append(
                {"sketch": sketch, "feature": feature, "type": "revolve", "index": i}
            )

        # Add chamfer features
        for i in range(1, chamfer_features.Count + 1):
            feature = chamfer_features.Item(i)
            feature_sketch_pairs.append(
                {"sketch": None, "feature": feature, "type": "chamfer", "index": i}
            )

        # Sort by creation order (using sketch index as approximation)
        feature_sketch_pairs.sort(key=lambda x: self._get_sketch_index(x["sketch"]))

        # Generate code for each sketch and feature
        for i, pair in enumerate(feature_sketch_pairs):
            wp_var = f"wp{i+1}"
            self._analyze_sketch(pair["sketch"], wp_var, i + 1)
            self._analyze_feature(pair["feature"], pair["type"], wp_var, i + 1)

    def _get_sketch_index(self, target_sketch) -> int:
        """Get the index of a sketch in the sketches collection."""
        for i in range(1, self.comp_def.Sketches.Count + 1):
            if self.comp_def.Sketches.Item(i) == target_sketch:
                return i
        return 999  # Put unknown sketches at the end
    
    def _analyze_chamfer(self, feature, feature_type: str, wp_var: str, feature_num: int):
        """Analyze a chamfer feature and generate code."""
        chamfer_distance = feature.Definition.Distance.Value
        if feature.DefinitionType == constants.kDistanceAndAngle:
            angle = feature.Definition.Angle.Value
            face = feature.Definition.Face
            edges = feature.Definition.Face.Edges
            self.generated_code.extend(
                [
                    f"# Chamfer feature {feature_num}",
                    f"{wp_var}.chamfer(\"X+\", distance={chamfer_distance}, angle={angle})",
                    "",
                ]
            )
        else:
            # Generate code for chamfer
            self.generated_code.extend(
                [
                    f"{wp_var}.chamfer(\"X+\", distance={chamfer_distance})",
                    "",
                ]
        )

    def _analyze_sketch(self, sketch, wp_var: str, sketch_num: int):
        """Analyze a sketch and generate workplane and geometry code."""
        # Get workplane info
        if sketch is None:
            return
        sketch = win32.CastTo(sketch, "PlanarSketch")
        workplane_info = self._get_workplane_info(sketch.PlanarEntity)

        # Create workplane code
        if "plane_name" in workplane_info and "offset" in workplane_info:
            self.generated_code.extend(
                [
                    f"# Sketch {sketch_num}",
                    f"{wp_var} = app.work_plane(\"{workplane_info['plane_name']}\", offset={workplane_info['offset']})",
                    "",
                ]
            )
        elif "plane_name" in workplane_info:
            self.generated_code.extend(
                [
                    f"# Sketch {sketch_num}",
                    f"{wp_var} = app.work_plane(\"{workplane_info['plane_name']}\")",
                    "",
                ]
            )
        else:
            self.generated_code.extend(
                [
                    f"# Sketch {sketch_num}",
                    f"{wp_var} = app.work_plane(",
                    f"    origin={workplane_info['origin']},",
                    f"    normal={workplane_info['normal']},",
                    "     app=app",
                    ")",
                    "",
                ]
            )

        # Get connected paths to avoid unnecessary move_to calls
        paths = self._get_connected_paths(sketch)

        # Generate code for each path
        for i, path in enumerate(paths):
            if len(path) == 1 and path[0]["type"] == "circle":
                # Single circle
                circle = path[0]
                self.generated_code.append(
                    f"{wp_var}.move_to({circle['center'][0]}, {circle['center'][1]}).circle({circle['radius']})"
                )
            else:
                # Connected path of lines and arcs
                self._generate_path_code(path, wp_var)

        self.generated_code.append("")

    def _get_workplane_info(self, planar_entity):
        """Extract workplane origin and orientation."""
        if planar_entity.DefinitionType == constants.kPlaneAndOffsetWorkPlane:
            offset = planar_entity.Definition.Offset.Value
            base_plane_name = planar_entity.Definition.Plane.Name
            if "XY" in base_plane_name:
                return {"plane_name": "XY", "offset": offset}
        elif hasattr(planar_entity, "Name"):
            name = planar_entity.Name
            if "XY" in name:
                return {"plane_name": "XY"}
            elif "XZ" in name:
                return {"plane_name": "XZ"}
            elif "YZ" in name:
                return {"plane_name": "YZ"}

            # For custom workplanes, extract geometry
            plane = planar_entity.Geometry
            origin = plane.RootPoint
            normal = plane.Normal

            return {
                "origin": (round(origin.X, 6), round(origin.Y, 6), round(origin.Z, 6)),
                "normal": (round(normal.X, 6), round(normal.Y, 6), round(normal.Z, 6)),
            }

    def _get_connected_paths(self, sketch) -> List[List[Dict]]:
        """Get sketch elements organized into connected paths."""
        elements = []

        # Collect lines
        for i in range(1, sketch.SketchLines.Count + 1):
            line = sketch.SketchLines.Item(i)
            try:
                # Check if the line has valid start and end points
                if line.StartSketchPoint is None or line.EndSketchPoint is None:
                    continue  # Skip invalid lines

                start_pt = line.StartSketchPoint.Geometry
                end_pt = line.EndSketchPoint.Geometry

                # Additional check for valid geometry
                if start_pt is None or end_pt is None:
                    continue

                elements.append(
                    {
                        "type": "line",
                        "start": (round(start_pt.X, 6), round(start_pt.Y, 6)),
                        "end": (round(end_pt.X, 6), round(end_pt.Y, 6)),
                        "used": False,
                    }
                )
            except Exception as e:
                # Skip lines that can't be processed
                print(f"Warning: Skipping line {i} due to error: {e}")
                continue

        # Collect arcs
        for i in range(1, sketch.SketchArcs.Count + 1):
            arc = sketch.SketchArcs.Item(i)
            try:
                # Check if the arc has valid start, end, and center points
                if (
                    arc.StartSketchPoint is None
                    or arc.EndSketchPoint is None
                    or arc.CenterSketchPoint is None
                ):
                    continue  # Skip invalid arcs

                start_pt = arc.StartSketchPoint.Geometry
                end_pt = arc.EndSketchPoint.Geometry
                center_pt = arc.CenterSketchPoint.Geometry

                # Additional check for valid geometry
                if start_pt is None or end_pt is None or center_pt is None:
                    continue

                # Try to get angles, but calculate them if not available
                try:
                    start_angle = arc.StartAngle
                    end_angle = arc.EndAngle
                except AttributeError:
                    # Calculate angles from points if properties not available
                    start_angle = math.atan2(
                        start_pt.Y - center_pt.Y, start_pt.X - center_pt.X
                    )
                    end_angle = math.atan2(
                        end_pt.Y - center_pt.Y, end_pt.X - center_pt.X
                    )

                # Try to get radius, calculate if not available
                try:
                    radius = round(arc.Radius, 6)
                except AttributeError:
                    # Calculate radius from center to start point
                    dx = start_pt.X - center_pt.X
                    dy = start_pt.Y - center_pt.Y
                    radius = round(math.sqrt(dx * dx + dy * dy), 6)

                elements.append(
                    {
                        "type": "arc",
                        "start": (round(start_pt.X, 6), round(start_pt.Y, 6)),
                        "end": (round(end_pt.X, 6), round(end_pt.Y, 6)),
                        "center": (round(center_pt.X, 6), round(center_pt.Y, 6)),
                        "radius": radius,
                        "start_angle": start_angle,
                        "end_angle": end_angle,
                        "used": False,
                    }
                )
            except Exception as e:
                # Skip arcs that can't be processed
                print(f"Warning: Skipping arc {i} due to error: {e}")
                continue

        # Build connected paths
        paths = []

        for start_element in elements:
            if start_element["used"] or start_element["type"] == "circle":
                continue

            # Start a new path
            current_path = [start_element]
            start_element["used"] = True
            current_end = start_element["end"]

            # Follow the chain
            while True:
                found_next = False

                for next_element in elements:
                    if next_element["used"] or next_element["type"] == "circle":
                        continue

                    # Check if connects
                    if self._points_equal(current_end, next_element["start"]):
                        current_path.append(next_element)
                        next_element["used"] = True
                        current_end = next_element["end"]
                        found_next = True
                        break

                if not found_next:
                    break

            paths.append(current_path)

        # Add circles as separate paths
        for i in range(1, sketch.SketchCircles.Count + 1):
            circle = sketch.SketchCircles.Item(i)
            try:
                # Check if the circle has valid center point
                if circle.CenterSketchPoint is None:
                    continue  # Skip invalid circles

                center_pt = circle.CenterSketchPoint.Geometry

                # Additional check for valid geometry
                if center_pt is None:
                    continue

                # Try to get radius, use default if not available
                try:
                    radius = round(circle.Radius, 6)
                except AttributeError:
                    # Default radius if property not available
                    radius = 1.0
                    print(
                        f"Warning: Could not get radius for circle {i}, using default: {radius}"
                    )

                paths.append(
                    [
                        {
                            "type": "circle",
                            "center": (round(center_pt.X, 6), round(center_pt.Y, 6)),
                            "radius": radius,
                        }
                    ]
                )
            except Exception as e:
                # Skip circles that can't be processed
                print(f"Warning: Skipping circle {i} due to error: {e}")
                continue

        return paths

    def _points_equal(
        self, p1: Tuple[float, float], p2: Tuple[float, float], tolerance: float = 1e-5
    ) -> bool:
        """Check if two points are equal within tolerance."""
        return abs(p1[0] - p2[0]) < tolerance and abs(p1[1] - p2[1]) < tolerance

    def _generate_path_code(self, path: List[Dict], wp_var: str):
        """Generate code for a connected path."""
        if not path:
            return

        first_element = path[0]
        code_line = f"{wp_var}.move_to({first_element['start'][0]}, {first_element['start'][1]})"

        # Chain all elements
        for element in path:
            if element["type"] == "line":
                code_line += f".line_to({element['end'][0]}, {element['end'][1]})"
            elif element["type"] == "arc":
                # Calculate middle point for three_point_arc
                center = element["center"]
                start = element["start"]
                end = element["end"]
                radius = element["radius"]

                # Calculate middle point on arc
                start_angle = math.atan2(start[1] - center[1], start[0] - center[0])
                end_angle = math.atan2(end[1] - center[1], end[0] - center[0])

                # Handle angle wrapping
                if end_angle < start_angle:
                    end_angle += 2 * math.pi

                mid_angle = (start_angle + end_angle) / 2
                mid_x = center[0] + radius * math.cos(mid_angle)
                mid_y = center[1] + radius * math.sin(mid_angle)

                code_line += f".three_point_arc(({round(mid_x, 6)}, {round(mid_y, 6)}), ({end[0]}, {end[1]}))"

        self.generated_code.append(code_line)

    def _analyze_feature(
        self, feature, feature_type: str, wp_var: str, feature_num: int
    ):
        """Analyze a 3D feature and generate code."""
        if feature_type == "extrude":
            self._analyze_extrude_feature(feature, wp_var, feature_num)
        elif feature_type == "revolve":
            self._analyze_revolve_feature(feature, wp_var, feature_num)
        elif feature_type == "chamfer":
            self._analyze_chamfer(feature, feature_type, wp_var, feature_num)
        else:
            ...

    def _analyze_extrude_feature(self, feature, wp_var: str, feature_num: int):
        """Analyze an extrude feature."""
        # Get distance
        extent = feature.Extent
        try:
            distance_extent = win32.CastTo(extent, "DistanceExtent")

            direction = distance_extent.Direction
            if direction == constants.kNegativeExtentDirection:
                distance = -round(distance_extent.Distance.Value, 6)
            else:
                distance = round(distance_extent.Distance.Value, 6)
            if direction == constants.kSymmetricExtentDirection:
                symmetric = "symmetric=True"
            else:
                symmetric = "symmetric=False"

            # Get operation type
            operation_map = {
                constants.kNewBodyOperation: "NewBodyFeatureOperation",
                constants.kJoinOperation: "JoinBodyFeatureOperation",
                constants.kCutOperation: "Cut",
                constants.kIntersectOperation: "Intersect",
            }
            operation = operation_map.get(feature.Operation, "NewBodyFeatureOperation")

            self.generated_code.extend(
                [
                    f"# Extrude feature {feature_num}",
                    f"shape{feature_num} = {wp_var}.extrude({distance}, '{operation}', {symmetric})",
                    "",
                ]
            )
        except Exception as e:
            raise RuntimeError(f"Failed to analyze extrude feature: {e}")

    def _analyze_revolve_feature(self, feature, wp_var: str, feature_num: int):
        """Analyze a revolve feature."""
        extent = feature.Extent
        extent_type = feature.ExtentType
        # Get angle
        if extent_type == constants.kAngleExtent:
            # Cast to AngleExtent to get the angle
            angle_extent = win32.CastTo(extent, "AngleExtent")
            angle = round(angle_extent.Angle.Value, 6)
        elif extent_type == constants.kFullSweepExtent:
            # Full revolution (360 degrees = 2*pi radians)
            angle = 6.283185307179586  # 2*pi
        else:
            raise ValueError("Unsupported revolve extent type.")

        # Get axis
        axis_entity = feature.AxisEntity

        # Determine which basis vector (X, Y, or Z) the axis is aligned with
        # Get the axis direction vector
        if hasattr(axis_entity, "Geometry"):
            axis_geom = axis_entity.Geometry
            # For a line, get direction vector
            if hasattr(axis_geom, "Direction"):
                direction = axis_geom.Direction

                if abs(direction.X) == 1.0:
                    axis = "X"
                elif abs(direction.Y) == 1.0:
                    axis = "Y"
                elif abs(direction.Z) == 1.0:
                    axis = "Z"
            else:
                # Default to Z axis
                axis = "Z"
        else:
            # Check if it's a work axis by name
            if hasattr(axis_entity, "Name"):
                name = axis_entity.Name.upper()
                if "X" in name:
                    axis = "X"
                elif "Y" in name:
                    axis = "Y"
                elif "Z" in name:
                    axis = "Z"
                else:
                    axis = "Z"  # Default
            else:
                axis = "Z"  # Default

        # Get operation type
        operation_map = {
            constants.kNewBodyOperation: "NewBodyFeatureOperation",
            constants.kJoinOperation: "JoinBodyFeatureOperation",
            constants.kCutOperation: "Cut",
            constants.kIntersectOperation: "Intersect",
        }
        operation = operation_map.get(feature.Operation, "NewBodyFeatureOperation")

        self.generated_code.extend(
            [
                f"# Revolve feature {feature_num}",
                f"shape{feature_num} = {wp_var}.revolve({angle}, '{axis}', '{operation}')",
                "",
            ]
        )
