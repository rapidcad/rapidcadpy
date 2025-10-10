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
            "from rapidcadpy.integrations.inventor import InventorApp",
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

    def _analyze_sketch(self, sketch, wp_var: str, sketch_num: int):
        """Analyze a sketch and generate workplane and geometry code."""
        # Get workplane info
        sketch = win32.CastTo(sketch, "PlanarSketch")
        workplane_info = self._get_workplane_info(sketch.PlanarEntity)

        # Create workplane code
        if "plane_name" in workplane_info:
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
                    f"    x_dir={workplane_info['x_dir']},",
                    f"    y_dir={workplane_info['y_dir']},",
                    "    app=app",
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

    def _get_workplane_info(
        self, planar_entity
    ) -> Dict[str, Tuple[float, float, float]]:
        """Extract workplane origin and orientation."""
        try:
            # Check if it's a standard workplane (XY, XZ, YZ)
            if hasattr(planar_entity, "Name"):
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
            # Create reasonable X and Y directions
            if abs(normal.Z) < 0.9:
                x_dir = normal.CrossProduct(
                    self.app.transient_geom.CreateVector(0, 0, 1)
                )
            else:
                x_dir = normal.CrossProduct(
                    self.app.transient_geom.CreateVector(0, 1, 0)
                )
            x_dir.Normalize()
            y_dir = normal.CrossProduct(x_dir)
            y_dir.Normalize()

            return {
                "origin": (round(origin.X, 6), round(origin.Y, 6), round(origin.Z, 6)),
                "x_dir": (round(x_dir.X, 6), round(x_dir.Y, 6), round(x_dir.Z, 6)),
                "y_dir": (round(y_dir.X, 6), round(y_dir.Y, 6), round(y_dir.Z, 6)),
            }
        except:
            # Default to XY plane
            return {"plane_name": "XY"}

    def _get_connected_paths(self, sketch) -> List[List[Dict]]:
        """Get sketch elements organized into connected paths."""
        elements = []

        # Collect lines
        for i in range(1, sketch.SketchLines.Count + 1):
            line = sketch.SketchLines.Item(i)
            start_pt = line.StartSketchPoint.Geometry
            end_pt = line.EndSketchPoint.Geometry
            elements.append(
                {
                    "type": "line",
                    "start": (round(start_pt.X, 6), round(start_pt.Y, 6)),
                    "end": (round(end_pt.X, 6), round(end_pt.Y, 6)),
                    "used": False,
                }
            )

        # Collect arcs
        for i in range(1, sketch.SketchArcs.Count + 1):
            arc = sketch.SketchArcs.Item(i)
            start_pt = arc.StartSketchPoint.Geometry
            end_pt = arc.EndSketchPoint.Geometry
            center_pt = arc.CenterSketchPoint.Geometry
            elements.append(
                {
                    "type": "arc",
                    "start": (round(start_pt.X, 6), round(start_pt.Y, 6)),
                    "end": (round(end_pt.X, 6), round(end_pt.Y, 6)),
                    "center": (round(center_pt.X, 6), round(center_pt.Y, 6)),
                    "radius": round(arc.Radius, 6),
                    "start_angle": arc.StartAngle,
                    "end_angle": arc.EndAngle,
                    "used": False,
                }
            )

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
            center_pt = circle.CenterSketchPoint.Geometry
            paths.append(
                [
                    {
                        "type": "circle",
                        "center": (round(center_pt.X, 6), round(center_pt.Y, 6)),
                        "radius": round(circle.Radius, 6),
                    }
                ]
            )

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

    def _analyze_extrude_feature(self, feature, wp_var: str, feature_num: int):
        """Analyze an extrude feature."""
        # Get distance
        extent = feature.Extent
        if hasattr(extent, "Distance"):
            distance = round(extent.Distance, 6)
        else:
            distance = 10.0  # Default

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
                f"shape{feature_num} = {wp_var}.extrude({distance}, '{operation}')",
                "",
            ]
        )

    def _analyze_revolve_feature(self, feature, wp_var: str, feature_num: int):
        """Analyze a revolve feature."""
        # Get angle
        extent = feature.Extent
        if hasattr(extent, "Angle"):
            angle = round(extent.Angle, 6)
        else:
            angle = math.pi * 2  # Default full revolution

        # Get axis - this is tricky, we'll use a simple approach
        # For now, assume axis is at origin of sketch
        axis = (0.0, 0.0)  # Simplified

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
                f"shape{feature_num} = {wp_var}.revolve({angle}, {axis}, '{operation}')",
                "",
            ]
        )
