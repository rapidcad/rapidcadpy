import math
from typing import Dict, List, Tuple

import win32com.client as win32
from win32com.client import constants


class InventorReverseEngineer:
    def __init__(self, doc):
        self.doc = doc
        self.generated_code = []
        self.decimal_precision = 4

    # Formatting helpers that respect self.decimal_precision
    def _fmt(self, v):
        try:
            f = round(float(v), self.decimal_precision)
            # Normalize tiny values to zero to avoid "-0.0"
            if abs(f) < 10 ** (-self.decimal_precision):
                f = 0.0
            s = str(f)
            if s in ("-0.0", "-0"):
                s = "0"
            return s
        except Exception:
            return str(v)

    def _fmt2(self, pt):
        try:
            x, y = pt
            return f"({self._fmt(x)}, {self._fmt(y)})"
        except Exception:
            return str(pt)

    def _fmt3(self, pt):
        try:
            x, y, z = pt
            return f"({self._fmt(x)}, {self._fmt(y)}, {self._fmt(z)})"
        except Exception:
            return str(pt)

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
        self._analyze_chamfer_feature()
        self._analyze_thread_features()

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
        workplane_info = self._get_workplane_info(sketch.PlanarEntity) or {}
        plane_name = workplane_info.get("plane_name")
        offset = workplane_info.get("offset")
        origin = workplane_info.get("origin")
        normal = workplane_info.get("normal")

        # Create workplane code
        if plane_name and offset is not None:
            self.generated_code.extend(
                [
                    f"# Sketch {sketch_num}",
                    f'{wp_var} = app.work_plane("{plane_name}", offset={self._fmt(offset)})',
                    "",
                ]
            )
        elif plane_name:
            self.generated_code.extend(
                [
                    f"# Sketch {sketch_num}",
                    f'{wp_var} = app.work_plane("{plane_name}")',
                    "",
                ]
            )
        elif origin is not None and normal is not None:
            self.generated_code.extend(
                [
                    f"# Sketch {sketch_num}",
                    f"{wp_var} = app.work_plane(",
                    f"    origin={self._fmt3(origin)},",
                    f"    normal={self._fmt3(normal)},",
                    "    app=app",
                    ")",
                    "",
                ]
            )
        else:
            # Fallback to XY workplane
            self.generated_code.extend(
                [
                    f"# Sketch {sketch_num}",
                    f'{wp_var} = app.work_plane("XY")',
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
                    f"{wp_var}.move_to({self._fmt(circle['center'][0])}, {self._fmt(circle['center'][1])}).circle({self._fmt(circle['radius'])})"
                )
            else:
                # Connected path of lines and arcs
                self._generate_path_code(path, wp_var)

        self.generated_code.append("")

    def _get_workplane_info(self, planar_entity) -> Dict[str, object]:
        """Extract workplane origin and orientation. Always return a dict with keys: plane_name, offset, origin, normal."""
        info: Dict[str, object] = {
            "plane_name": None,
            "offset": None,
            "origin": None,
            "normal": None,
        }
        try:
            # Offset from base planes
            if (
                getattr(planar_entity, "DefinitionType", None)
                == constants.kPlaneAndOffsetWorkPlane
            ):
                try:
                    offset = planar_entity.Definition.Offset.Value
                except Exception:
                    offset = None
                base_plane_name = (
                    getattr(planar_entity.Definition.Plane, "Name", "") or ""
                )
                if "XY" in base_plane_name:
                    info["plane_name"] = "XY"
                elif "XZ" in base_plane_name:
                    info["plane_name"] = "XZ"
                elif "YZ" in base_plane_name:
                    info["plane_name"] = "YZ"
                info["offset"] = offset
                return info

            # Named base planes
            if hasattr(planar_entity, "Name"):
                name = planar_entity.Name or ""
                if "XY" in name:
                    info["plane_name"] = "XY"
                    return info
                if "XZ" in name:
                    info["plane_name"] = "XZ"
                    return info
                if "YZ" in name:
                    info["plane_name"] = "YZ"
                    return info

            # Custom workplanes - derive from geometry
            if hasattr(planar_entity, "Geometry"):
                plane = planar_entity.Geometry
                if hasattr(plane, "RootPoint") and hasattr(plane, "Normal"):
                    origin = plane.RootPoint
                    normal = plane.Normal
                    info["origin"] = (
                        round(origin.X, 6),
                        round(origin.Y, 6),
                        round(origin.Z, 6),
                    )
                    info["normal"] = (
                        round(normal.X, 6),
                        round(normal.Y, 6),
                        round(normal.Z, 6),
                    )
        except Exception:
            pass
        return info

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
        code_line = f"{wp_var}.move_to({self._fmt(first_element['start'][0])}, {self._fmt(first_element['start'][1])})"

        # Chain all elements
        for element in path:
            if element["type"] == "line":
                code_line += f".line_to({self._fmt(element['end'][0])}, {self._fmt(element['end'][1])})"
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

                code_line += f".three_point_arc({self._fmt2((mid_x, mid_y))}, {self._fmt2((end[0], end[1]))})"

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
                    f"shape{feature_num} = {wp_var}.extrude({self._fmt(distance)}, '{operation}', {symmetric})",
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

        # Convert angle from radians to revolutions for numeric stability
        angle = angle / (2 * math.pi)

        # Get axis (default to Z)
        axis = "Z"
        axis_entity = feature.AxisEntity

        # Determine which basis vector (X, Y, or Z) the axis is aligned with
        if hasattr(axis_entity, "Geometry"):
            axis_geom = axis_entity.Geometry
            if hasattr(axis_geom, "Direction"):
                direction = axis_geom.Direction
                try:
                    if abs(direction.X) == 1.0:
                        axis = "X"
                    elif abs(direction.Y) == 1.0:
                        axis = "Y"
                    elif abs(direction.Z) == 1.0:
                        axis = "Z"
                except Exception:
                    pass
        elif hasattr(axis_entity, "Name"):
            name = (axis_entity.Name or "").upper()
            if "X" in name:
                axis = "X"
            elif "Y" in name:
                axis = "Y"
            elif "Z" in name:
                axis = "Z"

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
                f"shape{feature_num} = {wp_var}.revolve({self._fmt(angle)}, '{axis}', '{operation}')",
                "",
            ]
        )

    def _analyze_chamfer_feature(self):
        # Identify chamfered edges
        chamfer_features = self.comp_def.Features.ChamferFeatures
        original_edges_col = self.comp_def.SurfaceBodies.Item(1).Edges
        n_chamfers = chamfer_features.Count
        n_edges_chamfered = original_edges_col.Count

        edges_chamfered = []
        edges_unchamfered = []
        chamfer_parameters = []

        for e in range(1, n_edges_chamfered + 1):  # edges im gefasten modell
            edges_chamfered.append(original_edges_col.Item(e))

        for i in range(1, n_chamfers + 1):  # suppress all chamfer features
            chamfer_features.Item(i).Suppressed = True
            angle = chamfer_features.Item(i).Definition.Angle.Value
            distance = chamfer_features.Item(i).Definition.Distance.Value
            chamfer_parameters.append({"angle": angle, "distance": distance})

        new_edges_col = self.comp_def.SurfaceBodies.Item(1).Edges
        n_edges_unchamfered = new_edges_col.Count

        for e in range(1, n_edges_unchamfered + 1):  # edges im ungefasten modell
            edges_unchamfered.append(new_edges_col.Item(e))

        for i in range(1, n_chamfers + 1):
            chamfer_features.Item(i).Suppressed = False

        chamfered_params = {
            (e.Geometry.Center.X * 10, e.Geometry.Radius * 10)
            for e in edges_chamfered
            if e.CurveType == 5124
        }
        unchamfered_params = {
            (e.Geometry.Center.X * 10, e.Geometry.Radius * 10)
            for e in edges_unchamfered
            if e.CurveType == 5124
        }
        lost_edges = unchamfered_params - chamfered_params
        print(f"Chamfered Edges (X, Radius): {lost_edges}")
        self.generated_code.append("# Chamfered Edges")
        for (center_x, radius), params in zip(lost_edges, chamfer_parameters):
            self.generated_code.append(
                f"app.chamfer_edge(x={self._fmt(center_x/10)}, radius={self._fmt(radius/10)}, angle={self._fmt(params['angle'])}, distance={self._fmt(params['distance'])})"
            )

    def _analyze_thread_feature(self, thread_feature) -> Dict:
        """
        Analyze a thread feature and extract its parameters.

        Args:
            thread_feature: Inventor ThreadFeature object

        Returns:
            Dictionary containing thread information
        """
        thread_info = {
            "feature_type": "Thread",
            "name": thread_feature.Name,
            "suppressed": thread_feature.Suppressed,
        }

        try:
            # Get thread specification
            thread_spec = thread_feature.ThreadInfo
            
            # Get cylindrical face information for x and radius (similar to chamfer approach)
            if thread_feature.ThreadedFace:
                thread_info["face_type"] = "Cylindrical"
                threaded_face_obj = thread_feature.ThreadedFace
                
                # Check if it's a Faces collection or a single Face
                try:
                    # If it's a collection, get the first face
                    if hasattr(threaded_face_obj, 'Count'):
                        if threaded_face_obj.Count > 0:
                            face = threaded_face_obj.Item(1)
                            print(f"Got face from Faces collection (count: {threaded_face_obj.Count})")
                        else:
                            print("Warning: ThreadedFace collection is empty")
                            face = None
                    else:
                        # It's a single face
                        face = threaded_face_obj
                        print("Got single face object")
                except Exception as e:
                    print(f"Warning: Could not access ThreadedFace: {e}")
                    face = None
                
                if face is not None:
                    try:
                        # Get face geometry for position and radius
                        geom = face.Geometry
                        
                        # Method 1: Try to get BasePoint and Radius directly
                        if hasattr(geom, "BasePoint") and hasattr(geom, "Radius"):
                            try:
                                base_pt = geom.BasePoint
                                thread_info["x"] = round(base_pt.X, 6)
                                thread_info["y"] = round(base_pt.Y, 6)
                                thread_info["z"] = round(base_pt.Z, 6)
                                thread_info["radius"] = round(geom.Radius, 6)
                                print(f"Thread face position (BasePoint): ({thread_info['x']}, {thread_info['y']}, {thread_info['z']})")
                                print(f"Thread face radius: {thread_info['radius']}")
                            except Exception as e:
                                print(f"Warning: BasePoint/Radius extraction failed: {e}")
                        
                        # Method 2: Get from circular edges (similar to chamfer approach)
                        if "x" not in thread_info or "radius" not in thread_info:
                            try:
                                edges = face.Edges
                                for e_idx in range(1, edges.Count + 1):
                                    edge = edges.Item(e_idx)
                                    # Look for circular edges (CurveType 5124)
                                    if edge.CurveType == 5124:
                                        edge_geom = edge.Geometry
                                        if hasattr(edge_geom, "Center") and hasattr(edge_geom, "Radius"):
                                            center = edge_geom.Center
                                            thread_info["x"] = round(center.X, 6)
                                            thread_info["y"] = round(center.Y, 6)
                                            thread_info["z"] = round(center.Z, 6)
                                            thread_info["radius"] = round(edge_geom.Radius, 6)
                                            print(f"Thread face position (from edge): ({thread_info['x']}, {thread_info['y']}, {thread_info['z']})")
                                            print(f"Thread face radius (from edge): {thread_info['radius']}")
                                            break
                            except Exception as e:
                                print(f"Warning: Edge extraction failed: {e}")
                        
                        # Method 3: Use face evaluator range box
                        if "x" not in thread_info or "radius" not in thread_info:
                            try:
                                evaluator = face.Evaluator
                                range_box = evaluator.RangeBox
                                min_pt = range_box.MinPoint
                                max_pt = range_box.MaxPoint
                                
                                if "x" not in thread_info:
                                    thread_info["x"] = round((min_pt.X + max_pt.X) / 2, 6)
                                    thread_info["y"] = round((min_pt.Y + max_pt.Y) / 2, 6)
                                    thread_info["z"] = round((min_pt.Z + max_pt.Z) / 2, 6)
                                    print(f"Thread face position (from range): ({thread_info['x']}, {thread_info['y']}, {thread_info['z']})")
                                
                                if "radius" not in thread_info:
                                    # Try to get radius from geometry again
                                    if hasattr(geom, "Radius"):
                                        thread_info["radius"] = round(geom.Radius, 6)
                                        print(f"Thread face radius (from geom): {thread_info['radius']}")
                            except Exception as e:
                                print(f"Warning: Range box extraction failed: {e}")
                        
                        # Get cylinder axis direction for better matching
                        if hasattr(geom, "AxisVector"):
                            try:
                                axis_vec = geom.AxisVector
                                # Determine primary axis direction
                                if abs(axis_vec.X) > 0.9:
                                    thread_info["axis"] = "X"
                                elif abs(axis_vec.Y) > 0.9:
                                    thread_info["axis"] = "Y"
                                elif abs(axis_vec.Z) > 0.9:
                                    thread_info["axis"] = "Z"
                                print(f"Thread face axis: {thread_info.get('axis', 'Unknown')}")
                            except Exception as e:
                                print(f"Warning: Axis extraction failed: {e}")
                        
                    except Exception as e:
                        print(f"Warning: Could not extract face geometry: {e}")
                        import traceback
                        traceback.print_exc()

            # Thread designation - try different attribute names
            designation = None
            for attr_name in ["Designation", "ThreadDesignation", "Size"]:
                try:
                    designation = getattr(thread_spec, attr_name, None)
                    if designation:
                        break
                except Exception:
                    continue
            
            if not designation:
                # Try to build designation from thread size properties
                try:
                    # Some threads have Size, Pitch, etc.
                    if hasattr(thread_spec, "NominalDiameter"):
                        nominal = thread_spec.NominalDiameter
                        if hasattr(thread_spec, "Pitch"):
                            pitch = thread_spec.Pitch
                            designation = f"M{nominal}x{pitch}"
                        else:
                            designation = f"M{nominal}"
                except Exception:
                    pass
            
            thread_info["designation"] = designation if designation else "M8x1.25"

            # Thread class/fit - try different attribute names
            thread_class = None
            for attr_name in ["ThreadClass", "Class", "FitClass"]:
                try:
                    thread_class = getattr(thread_spec, attr_name, None)
                    if thread_class:
                        break
                except Exception:
                    continue
            thread_info["thread_class"] = thread_class if thread_class else "6H"

            # Direction (Right-hand or Left-hand)
            try:
                thread_info["right_handed"] = getattr(thread_spec, "RightHanded", True)
            except Exception:
                thread_info["right_handed"] = True

            # Check if it's a modeled thread or cosmetic thread
            thread_info["modeled"] = thread_feature.ThreadedFace is not None

            # Thread length
            try:
                if hasattr(thread_spec, "ThreadLength"):
                    length = thread_spec.ThreadLength
                    thread_info["length"] = length
                    thread_info["length_cm"] = length  # Already in cm
            except Exception:
                pass

            # Full length or partial
            try:
                thread_info["full_length"] = getattr(thread_spec, "FullLength", False)
            except Exception:
                thread_info["full_length"] = False

            # Thread offset (if not starting at face)
            try:
                offset = getattr(thread_spec, "ThreadOffset", 0)
                thread_info["offset"] = offset if offset else 0
            except Exception:
                thread_info["offset"] = 0

            # Internal vs External thread
            try:
                thread_type = getattr(thread_spec, "ThreadType", None)
                if thread_type is not None:
                    thread_info["internal"] = thread_type == 67587  # kInternalThread
                    thread_info["external"] = thread_type == 67586  # kExternalThread
                else:
                    # Fallback: check thread type string
                    thread_type_str = str(thread_spec).lower()
                    thread_info["internal"] = "internal" in thread_type_str
                    thread_info["external"] = not thread_info["internal"]
            except Exception:
                # Default to external
                thread_info["internal"] = False
                thread_info["external"] = True

        except Exception as e:
            print(f"Error analyzing thread feature: {e}")
            import traceback
            traceback.print_exc()

        # Validate that we have critical parameters for reconstruction
        if "x" not in thread_info or "radius" not in thread_info:
            print(f"Warning: Thread '{thread_info.get('name')}' missing position/radius - reconstruction may fail")
        else:
            print(f"✓ Thread extracted: x={thread_info['x']}, radius={thread_info['radius']}, {thread_info['designation']}")

        return thread_info

    def _generate_thread_code(self, thread_info: Dict) -> List[str]:
        """
        Generate Python code for creating a thread feature.

        Args:
            thread_info: Dictionary with thread parameters

        Returns:
            List of code lines
        """
        code_lines = []

        # Add comment with thread info
        designation = thread_info.get('designation', 'Unknown')
        thread_type = "internal" if thread_info.get("internal") else "external"
        code_lines.append(f"# Thread: {designation} ({thread_type})")

        # Generate thread creation code
        code_lines.append("thread = app.add_thread(")
        
        # Add geometric parameters if available (critical for face matching)
        if "x" in thread_info and "radius" in thread_info:
            code_lines.append(f"    x={self._fmt(thread_info['x'])},  # Cylindrical face center X")
            code_lines.append(f"    radius={self._fmt(thread_info['radius'])},  # Cylindrical face radius")
            
            # Optionally add Y/Z if non-zero (for non-axial cylinders)
            if "y" in thread_info and abs(thread_info['y']) > 1e-6:
                code_lines.append(f"    y={self._fmt(thread_info['y'])},  # Cylindrical face center Y")
            if "z" in thread_info and abs(thread_info['z']) > 1e-6:
                code_lines.append(f"    z={self._fmt(thread_info['z'])},  # Cylindrical face center Z")
            
            # Add axis direction for better matching
            if "axis" in thread_info:
                code_lines.append(f"    axis='{thread_info['axis']}',  # Cylinder axis direction")
        else:
            code_lines.append("    # WARNING: Missing x/radius - may not match correct face!")
        
        code_lines.append(f"    designation='{thread_info.get('designation', 'M8x1.25')}',")
        code_lines.append(f"    thread_class='{thread_info.get('thread_class', '6H')}',")
        code_lines.append(f"    thread_type='{thread_type}',")
        code_lines.append(f"    right_handed={thread_info.get('right_handed', True)},")

        if not thread_info.get("full_length"):
            length = thread_info.get("length_cm", 0)
            if length > 0:
                code_lines.append(f"    length={self._fmt(length)},")
        else:
            code_lines.append(f"    full_length=True,")

        offset = thread_info.get("offset", 0)
        if offset > 0:
            code_lines.append(f"    offset={self._fmt(offset)},")

        code_lines.append(f"    modeled={thread_info.get('modeled', False)}")
        code_lines.append(")")

        return code_lines

    def _analyze_thread_features(self):
        """
        Process all thread features in the part and add code generation.
        """
        try:
            thread_features = self.comp_def.Features.ThreadFeatures

            if thread_features.Count == 0:
                return

            self.generated_code.append("")
            self.generated_code.append("# === Thread Features ===")

            for i in range(1, thread_features.Count + 1):
                thread_feature = thread_features.Item(i)

                # Skip suppressed features if needed
                if thread_feature.Suppressed:
                    continue

                thread_info = self._analyze_thread_feature(thread_feature)
                thread_code = self._generate_thread_code(thread_info)
                self.generated_code.extend(thread_code)
                self.generated_code.append("")

        except Exception as e:
            print(f"Error processing thread features: {e}")
