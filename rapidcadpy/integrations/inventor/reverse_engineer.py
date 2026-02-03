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
        """Analyze all features in their correct creation order."""
        # Get the PartFeatures collection which contains all features in order
        part_features = self.comp_def.Features

        # Build a list of all features in order using the AllFeatures collection
        # This gives us the browser/timeline order
        all_features_ordered = []

        # Iterate through all features in order
        # Note: PartFeatures has collections for each feature type
        # We need to use a different approach to get the timeline order

        extrude_features = part_features.ExtrudeFeatures
        revolve_features = part_features.RevolveFeatures
        thread_features = part_features.ThreadFeatures
        chamfer_features = part_features.ChamferFeatures
        fillet_features = part_features.FilletFeatures

        # Collect all features with metadata
        all_features = []

        # Add extrude features
        for i in range(1, extrude_features.Count + 1):
            feature = extrude_features.Item(i)
            if not feature.Suppressed:
                sketch = feature.Profile.Parent
                all_features.append(
                    {
                        "feature": feature,
                        "type": "extrude",
                        "sketch": sketch,
                        "sketch_index": self._get_sketch_index(sketch),
                        "name": feature.Name,
                    }
                )

        # Add revolve features
        for i in range(1, revolve_features.Count + 1):
            feature = revolve_features.Item(i)
            if not feature.Suppressed:
                sketch = feature.Profile.Parent
                all_features.append(
                    {
                        "feature": feature,
                        "type": "revolve",
                        "sketch": sketch,
                        "sketch_index": self._get_sketch_index(sketch),
                        "name": feature.Name,
                    }
                )

        # Add thread features
        for i in range(1, thread_features.Count + 1):
            feature = thread_features.Item(i)
            if not feature.Suppressed:
                # Threads don't have a sketch, but we can try to find which face they depend on
                # This helps us determine order based on which extrude/revolve created that face
                dependent_sketch_index = self._get_thread_dependent_sketch_index(
                    feature
                )
                all_features.append(
                    {
                        "feature": feature,
                        "type": "thread",
                        "sketch": None,
                        "sketch_index": dependent_sketch_index,
                        "name": feature.Name,
                    }
                )

        # Add chamfer features
        for i in range(1, chamfer_features.Count + 1):
            feature = chamfer_features.Item(i)
            if not feature.Suppressed:
                dependent_sketch_index = self._get_chamfer_dependent_sketch_index(
                    feature
                )
                all_features.append(
                    {
                        "feature": feature,
                        "type": "chamfer",
                        "sketch": None,
                        "sketch_index": dependent_sketch_index,
                        "name": feature.Name,
                    }
                )

        # Add fillet features
        for i in range(1, fillet_features.Count + 1):
            feature = fillet_features.Item(i)
            if not feature.Suppressed:
                dependent_sketch_index = self._get_fillet_dependent_sketch_index(
                    feature
                )
                all_features.append(
                    {
                        "feature": feature,
                        "type": "fillet",
                        "sketch": None,
                        "sketch_index": dependent_sketch_index,
                        "name": feature.Name,
                    }
                )

        # Sort by sketch_index to maintain feature order
        # Features that depend on later sketches come after features that depend on earlier sketches
        all_features.sort(key=lambda x: (x["sketch_index"], x["name"]))

        # Generate code for each feature in order
        sketch_counter = 0
        processed_sketches = {}  # Changed from set() to {}

        for feat_info in all_features:
            feat_type = feat_info["type"]
            feature = feat_info["feature"]

            if feat_type in ["extrude", "revolve"]:
                # These features have sketches
                sketch = feat_info["sketch"]
                sketch_id = id(sketch)

                if sketch_id not in processed_sketches:
                    sketch_counter += 1
                    wp_var = f"wp{sketch_counter}"
                    self._analyze_sketch(sketch, wp_var, sketch_counter)
                    processed_sketches[sketch_id] = (wp_var, sketch_counter)
                else:
                    # Find the wp_var for this already-processed sketch
                    wp_var, _ = processed_sketches[sketch_id]

                self._analyze_feature(feature, feat_type, wp_var, sketch_counter)

            elif feat_type == "thread":
                thread_info = self._analyze_thread_feature(feature)
                thread_code = self._generate_thread_code(thread_info)
                self.generated_code.append("")
                # Comment is added by _generate_thread_code
                self.generated_code.extend(thread_code)

            elif feat_type == "chamfer":
                self._analyze_single_chamfer_feature(feature)

            elif feat_type == "fillet":
                self._analyze_single_fillet_feature(feature)

    def _get_thread_dependent_sketch_index(self, thread_feature) -> float:
        """
        Determine which sketch/feature the thread depends on.
        Returns a sketch index that places this thread after its parent feature.
        """
        try:
            # Try to find the face the thread is applied to
            if thread_feature.ThreadedFace:
                threaded_face = thread_feature.ThreadedFace

                # Get the face (handle collection or single face)
                if hasattr(threaded_face, "Count") and threaded_face.Count > 0:
                    face = threaded_face.Item(1)
                else:
                    face = threaded_face

                # Try to find which feature created this face
                # by looking at the face's CreatedByFeature
                if hasattr(face, "CreatedByFeature"):
                    parent_feature = face.CreatedByFeature

                    # If the parent feature has a profile/sketch, get its index
                    if hasattr(parent_feature, "Profile"):
                        sketch = parent_feature.Profile.Parent
                        # Add 0.5 to place thread after the parent feature
                        return self._get_sketch_index(sketch) + 0.5
        except Exception as e:
            pass

        # Default: put threads near the end
        return 998

    def _get_chamfer_dependent_sketch_index(self, chamfer_feature) -> float:
        """
        Determine which sketch/feature the chamfer depends on.
        """
        try:
            # Strategy 1: Look at the Faces created by the chamfer (Robust)
            if hasattr(chamfer_feature, "Faces"):
                faces = chamfer_feature.Faces
                if faces.Count > 0:
                    # Look at one face of the chamfer
                    c_face = faces.Item(1)
                    
                    # Look at adjacent faces via edges
                    for j in range(1, c_face.Edges.Count + 1):
                        edge = c_face.Edges.Item(j)
                        for k in range(1, edge.Faces.Count + 1):
                            adj_face = edge.Faces.Item(k)
                            
                            # Check if this face belongs to another feature
                            try:
                                if hasattr(adj_face, "CreatedByFeature"):
                                    parent = adj_face.CreatedByFeature
                                    # Ensure it's not the chamfer itself (by name comparison)
                                    if parent.Name != chamfer_feature.Name:
                                        if hasattr(parent, "Profile"):
                                            sketch = parent.Profile.Parent
                                            return self._get_sketch_index(sketch) + 0.6
                            except Exception:
                                pass

            # Strategy 2: Old/Deprecated Edges approach
            if hasattr(chamfer_feature, "Edges"):
                edges = chamfer_feature.Edges
                if edges and edges.Count > 0:
                    edge = edges.Item(1)

                    # Get the faces connected to this edge
                    if hasattr(edge, "Faces") and edge.Faces.Count > 0:
                        face = edge.Faces.Item(1)

                        if hasattr(face, "CreatedByFeature"):
                            parent_feature = face.CreatedByFeature

                            if hasattr(parent_feature, "Profile"):
                                sketch = parent_feature.Profile.Parent
                                return self._get_sketch_index(sketch) + 0.6
        except Exception:
            pass

        return 999

    def _get_fillet_dependent_sketch_index(self, fillet_feature) -> float:
        """
        Determine which sketch/feature the fillet depends on.
        """
        try:
            if hasattr(fillet_feature, "Edges"):
                edges = fillet_feature.Edges
                if edges and edges.Count > 0:
                    edge = edges.Item(1)

                    if hasattr(edge, "Faces") and edge.Faces.Count > 0:
                        face = edge.Faces.Item(1)

                        if hasattr(face, "CreatedByFeature"):
                            parent_feature = face.CreatedByFeature

                            if hasattr(parent_feature, "Profile"):
                                sketch = parent_feature.Profile.Parent
                                return self._get_sketch_index(sketch) + 0.7
        except Exception:
            pass

        return 999

    def _analyze_single_chamfer_feature(self, chamfer_feature):
        """Analyze and generate code for a single chamfer feature."""
        # Detect chamfer by comparing edges before and after suppression
        try:
             # Get original edges (unchamfered if we roll back)
             # But here we are at the point where chamfer exists.
             # So the current state has NEW edges from chamfer, and MISSING edges that were chamfered.
             
             # This logic used to be global in _analyze_chamfer_feature but user wants it per feature
             # Since we are iterating features in order, we can try to suppress THIS feature
             # to find out which edge it modifies.
             
             # 1. Get current edges (with chamfer)
             current_edges = {
                (e.Geometry.Center.X, e.Geometry.Radius, e.Geometry.Center.Y, e.Geometry.Center.Z)
                for e in self.comp_def.SurfaceBodies.Item(1).Edges
                if e.CurveType == 5124 # kCircleCurve
             }
             
             # 2. Suppress this chamfer
             chamfer_feature.Suppressed = True
             
             # 3. Get edges without chamfer (original state)
             original_edges_objs = []
             surface_bodies = self.comp_def.SurfaceBodies
             if surface_bodies.Count > 0:
                 for e in surface_bodies.Item(1).Edges:
                     if e.CurveType == 5124:
                         original_edges_objs.append(e)
             
             original_edges = {
                (e.Geometry.Center.X, e.Geometry.Radius, e.Geometry.Center.Y, e.Geometry.Center.Z)
                for e in original_edges_objs
             }
             
             # 4. Unsuppress chamfer to restore state
             chamfer_feature.Suppressed = False
             
             # 5. Find the edge that exists in original but NOT in current
             # That is the edge that got chamfered away
             chamfered_edges = original_edges - current_edges
             
             if chamfered_edges:
                 # Extract parameters
                 chamfer_def = chamfer_feature.Definition
                 
                 angle = 0.785 # 45 deg
                 distance = 0.1
                 
                 # Check chamfer type
                 # kTwoDistancesChamfer = 51201
                 # kDistanceAndAngleChamfer = 51202
                 
                 try:
                     if hasattr(chamfer_def, "Distance"):
                         distance = chamfer_def.Distance.Value
                     elif hasattr(chamfer_def, "DistanceOne"):
                         distance = chamfer_def.DistanceOne.Value
                     
                     if hasattr(chamfer_def, "Angle"):
                         angle = chamfer_def.Angle.Value
                 except Exception:
                     pass
                 
                 for (cx, cr, cy, cz) in chamfered_edges:
                      # Generate code
                      self.generated_code.append(f"# Chamfer: {chamfer_feature.Name}")
                      self.generated_code.append(
                        f"app.chamfer_edge(x={self._fmt(cx)}, radius={self._fmt(cr)}, angle={self._fmt(angle)}, distance={self._fmt(distance)})"
                      )
                      self.generated_code.append("")
             else:
                 # Fallback to old extraction method if diff method fails
                 chamfer_info = self._extract_chamfer_info(chamfer_feature)
                 if chamfer_info and "x" in chamfer_info:
                    self.generated_code.append("")
                    self.generated_code.append(f"# Chamfer: {chamfer_feature.Name}")
                    chamfer_code = self._generate_chamfer_code(chamfer_info)
                    self.generated_code.extend(chamfer_code)
                 else:
                    self.generated_code.append(f"# Chamfer: Could not extract edge position for {chamfer_feature.Name} using edge diff or direct property access")

        except Exception as e:
            self.generated_code.append(
                f"# Warning: Could not process chamfer {chamfer_feature.Name}: {e}"
            )

    def _analyze_single_fillet_feature(self, fillet_feature):
        """Analyze and generate code for a single fillet feature."""
        # Fillet extraction not yet implemented.
        # Intentionally omitting from output to avoid TODO spam.
        pass

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
                    f"    normal={self._fmt3(normal)}",
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
                mid_pt = arc.Geometry.MidPoint

                # Additional check for valid geometry
                if start_pt is None or end_pt is None or center_pt is None:
                    continue

                # Try to get angles, but calculate them if not available
                try:
                    start_angle = arc.StartAngle
                    end_angle = arc.EndAngle
                except AttributeError:
                    # Calculate angles from points if not available
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
                        "mid": (round(mid_pt.X, 6), round(mid_pt.Y, 6)),
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
            # Limit iterations to avoid infinite loops in complex geometries
            max_chain_length = len(elements) * 2 
            chain_counter = 0
            
            while chain_counter < max_chain_length:
                chain_counter += 1
                found_next = False

                for next_element in elements:
                    if next_element["used"] or next_element["type"] == "circle":
                        continue

                    # Check if connects to current end
                    if self._points_equal(current_end, next_element["start"]):
                        current_path.append(next_element)
                        next_element["used"] = True
                        current_end = next_element["end"]
                        found_next = True
                        break
                    # Check if connects reversely (start of next == current end handled above)
                    # Check if end of next == current end (need to reverse next)
                    elif self._points_equal(current_end, next_element["end"]):
                         # Must reverse this element to maintain chain direction
                         reversed_element = next_element.copy()
                         reversed_element["start"] = next_element["end"]
                         reversed_element["end"] = next_element["start"]
                         if next_element["type"] == "arc":
                             # Swap angles for arc
                             reversed_element["start_angle"] = next_element["end_angle"]
                             reversed_element["end_angle"] = next_element["start_angle"]
                             # For arcs, traversing in reverse changes CW/CCW sense
                             # Inventor arcs are typically CCW. Code gen assumes P1->P2.
                             # Three point arc code gen handles direction relative to midpoint.
                             pass
                             
                         current_path.append(reversed_element)
                         next_element["used"] = True # Mark original as used
                         current_end = reversed_element["end"]
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
        """Generate code for a connected path of lines and arcs."""
        if not path:
            return

        # Start at the first point
        start_pt = path[0]["start"]
        
        # Build the command chain as a single line
        code_str = f"{wp_var}.move_to({self._fmt(start_pt[0])}, {self._fmt(start_pt[1])})"

        for element in path:
            end_pt = element["end"]
            
            if element["type"] == "line":
                code_str += f".line_to({self._fmt(end_pt[0])}, {self._fmt(end_pt[1])})"
            elif element["type"] == "arc":
                # Use the extracted midpoint directly
                # This avoids issues with arc direction (CW vs CCW) and angle calculations
                if "mid" in element:
                     mid_x, mid_y = element["mid"]
                else:
                    # Fallback for older extraction logic
                    center_pt = element["center"]
                    radius = element["radius"]
                    start_angle = element["start_angle"]
                    end_angle = element["end_angle"]
                    
                    # Calculate midpoint on the arc for three_point_arc
                    # Assume CCW traversal for Inventor SketchArcs
                    diff = end_angle - start_angle
                    while diff <= 0:
                        diff += 2 * math.pi
                    while diff > 2 * math.pi:
                        diff -= 2 * math.pi
                    
                    mid_angle = start_angle + (diff / 2.0)
                    mid_x = center_pt[0] + radius * math.cos(mid_angle)
                    mid_y = center_pt[1] + radius * math.sin(mid_angle)
                
                code_str += (
                    f".three_point_arc("
                    f"p1=({self._fmt(mid_x)}, {self._fmt(mid_y)}), "
                    f"p2=({self._fmt(end_pt[0])}, {self._fmt(end_pt[1])})"
                    f")"
                )
                
        self.generated_code.append(code_str)

    def _analyze_feature(
        self, feature, feat_type: str, wp_var: str, sketch_counter: int
    ):
        """Analyze extrude or revolve feature."""
        try:
            name = feature.Name
            
            if feat_type == "extrude":
                # Get Extrude details (distance, operation type)
                extent = feature.Extent
                extent_type = extent.Type
                
                dist = 1.0
                symmetric = False
                
                try:
                    # In newer Inventor API, it might be separate defined object
                    # Simplified for typical extensive distance:
                    
                    is_distance_extent = (
                        extent_type == constants.kDistanceExtent 
                        or extent_type == constants.kDistanceExtentObject
                    )
                    if is_distance_extent:  # Distance
                        # Check ExtentOne or Distance property
                        # Distance object usually has Value property
                        # cast to DistanceExtent object
                        try:
                            extent = win32.CastTo(extent, "DistanceExtent")
                        except Exception:
                            print(f"Warning: Could not cast extent for {name} to DistanceExtent")

                        try:
                             dist = extent.Distance.Value
                        except Exception as e:
                             print(f"DEBUG: Error getting Distance.Value for {name}: {e}")
                             dist = 1.0 # Default fallback
                        
                        # Determine direction to set sign of distance
                        try:
                            direction = extent.Direction
                            
                            if direction == constants.kNegativeExtentDirection: # Negative
                                 dist = -dist
                            elif direction == constants.kSymmetricExtentDirection: # Symmetric
                                 symmetric = True
                        except Exception as e:
                            print(f"DEBUG: Error getting Direction for {name}: {e}")

                except Exception as e:
                    print(f"Warning: Could not get extrude distance for {name}: {e}")
                
                # Check for symmetric
                # Might be property on Feature or Definition
                
                op_type = feature.Operation
                # kJoinOperation = 20737
                # kCutOperation = 20738
                # kIntersectOperation = 20739
                # kNewBodyOperation = 20740
                
                op_str = "NewBodyFeatureOperation"
                if op_type == 20737:
                    op_str = "JoinBodyFeatureOperation"
                elif op_type == 20738:  # Cut
                    op_str = "Cut"
                elif op_type == 20739:
                    op_str = "Intersect"
                
                self.generated_code.append(
                    f"# Extrude: {name}"
                )
                if symmetric:
                    self.generated_code.append(
                         f'{wp_var}.extrude(distance={self._fmt(dist)}, operation="{op_str}", symmetric=True)'
                    )
                else: 
                    self.generated_code.append(
                        f'{wp_var}.extrude(distance={self._fmt(dist)}, operation="{op_str}")'
                    )
                self.generated_code.append("")
                
            elif feat_type == "revolve":
                # Revolve details
                angle_rad = 6.283185  # 360 deg
                try:
                    extent = feature.Extent
                    if extent.Type == constants.kAngleExtent: 
                         angle_rad = extent.Angle.Value
                except Exception:
                    pass
                
                op_type = feature.Operation
                # Operation mapping (RapidCAD Py constants/strings)
                # kJoinOperation = 20737
                # kCutOperation = 20738
                # kIntersectOperation = 20739
                # kNewBodyOperation = 20740
                
                op_str = "NewBodyFeatureOperation"
                if op_type == 20737:
                    op_str = "JoinBodyFeatureOperation"
                elif op_type == constants.kCutOperation:
                    op_str = "Cut" # Matches Workplane.revolve logic
                elif op_type == 20739:
                    op_str = "Intersect"
                
                # Axis determination
                # RapidCAD Py Revolve expects axis="X", "Y", or "Z"
                axis_char = "X" # Default
                
                try:
                    if hasattr(feature, "AxisEntity"):
                        axis_ent = feature.AxisEntity
                        # Check for WorkAxis or Line alignment
                        # WorkAxes often have names like "X Axis", "Y Axis", "Z Axis"
                        axis_name = getattr(axis_ent, "Name", "")
                        
                        if "X" in axis_name and "Axis" in axis_name:
                            axis_char = "X"
                        elif "Y" in axis_name and "Axis" in axis_name:
                            axis_char = "Y"
                        elif "Z" in axis_name and "Axis" in axis_name:
                            axis_char = "Z"
                        else:
                            # Try geometry analysis
                            geom = None
                            if hasattr(axis_ent, "Geometry"):
                                geom = axis_ent.Geometry # Line or LineSegment
                            elif hasattr(axis_ent, "Line"):
                                geom = axis_ent.Line.Geometry
                            
                            if geom:
                                dir_vec = None
                                if hasattr(geom, "Direction"):
                                    dir_vec = geom.Direction
                                elif hasattr(geom, "StartPoint") and hasattr(geom, "EndPoint"):
                                    # Create direction from points
                                    dx = geom.EndPoint.X - geom.StartPoint.X
                                    dy = geom.EndPoint.Y - geom.StartPoint.Y
                                    dz = geom.EndPoint.Z - geom.StartPoint.Z
                                    mag = math.sqrt(dx*dx + dy*dy + dz*dz)
                                    if mag > 1e-9:
                                        dir_vec = [dx/mag, dy/mag, dz/mag]
                                
                                if dir_vec:
                                    # Check alignment (handling win32 objects or lists)
                                    vx = 0.0
                                    vy = 0.0
                                    vz = 0.0
                                    
                                    # Use getattr to avoid static type checker complaints about missing attributes
                                    if hasattr(dir_vec, "X"):
                                        vx = getattr(dir_vec, "X")
                                        vy = getattr(dir_vec, "Y")
                                        # Check if Z exists (3D) or default to 0 (2D)
                                        if hasattr(dir_vec, "Z"):
                                            vz = getattr(dir_vec, "Z")
                                        else:
                                            vz = 0.0
                                    else:
                                        # Assume list/tuple if attributes missing
                                        vx = dir_vec[0]
                                        vy = dir_vec[1]
                                        # Handle 2D tuples/lists
                                        if len(dir_vec) > 2:
                                            vz = dir_vec[2]
                                        else:
                                            vz = 0.0
                                    
                                    if abs(vx) > 0.9: axis_char = "X"
                                    elif abs(vy) > 0.9: axis_char = "Y"
                                    elif abs(vz) > 0.9: axis_char = "Z"
                except Exception as e:
                    print(f"Warning: Could not determine revolve axis: {e}")
                
                # Angle in "Turns" (0.0 - 1.0) because InventorWorkPlane.revolve does angle * 2 * pi
                # But wait, InventorWorkPlane.revolve calls AddByAngle(..., angle * 2 * PI, ...)
                # Inventor AddByAngle expects Radians.
                # So if we pass 1.0 (turn), it passes 2*PI radians (360 deg). Correct.
                # So we need to convert radians to turns.
                angle_turns = angle_rad
                
                self.generated_code.append(
                    f"# Revolve: {name}"
                )
                self.generated_code.append(
                    f"{wp_var}.revolve(angle={self._fmt(angle_turns)}, axis=\"{axis_char}\", operation=\"{op_str}\")"
                )
                self.generated_code.append("")
                
        except Exception as e:
            self.generated_code.append(f"# Error analyzing feature {feature.Name}: {e}")
            self.generated_code.append("")

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

    def _extract_chamfer_info(self, chamfer_feature) -> Dict:
        """
        Extract chamfer parameters from a single chamfer feature.

        Args:
            chamfer_feature: Inventor ChamferFeature object

        Returns:
            Dictionary containing chamfer information
        """
        try:
            chamfer_def = chamfer_feature.Definition

            info = {
                "name": chamfer_feature.Name,
                "angle": (
                    chamfer_def.Angle.Value if hasattr(chamfer_def, "Angle") else 0.785
                ),  # 45 degrees default
                "distance": (
                    chamfer_def.Distance.Value
                    if hasattr(chamfer_def, "Distance")
                    else 0.1
                ),
            }

            # Try to get edge information
            if hasattr(chamfer_feature, "Edges"):
                edges = chamfer_feature.Edges
                if edges and edges.Count > 0:
                    edge = edges.Item(1)
                    if edge.CurveType == 5124:  # Circular edge
                        geom = edge.Geometry
                        if hasattr(geom, "Center"):
                            center = geom.Center
                            info["x"] = round(center.X, 6)
                            info["y"] = round(center.Y, 6)
                            info["z"] = round(center.Z, 6)
                        if hasattr(geom, "Radius"):
                            info["radius"] = round(geom.Radius, 6)

            return info
        except Exception as e:
            print(f"Warning: Could not extract chamfer info: {e}")
            return {}

    def _generate_chamfer_code(self, chamfer_info: Dict) -> List[str]:
        """
        Generate code for a chamfer feature.

        Args:
            chamfer_info: Dictionary containing chamfer parameters

        Returns:
            List of code lines
        """
        code_lines = []

        if chamfer_info and "x" in chamfer_info and "radius" in chamfer_info:
            code_lines.append(
                f"app.chamfer_edge("
                f"x={self._fmt(chamfer_info['x'])}, "
                f"radius={self._fmt(chamfer_info['radius'])}, "
                f"angle={self._fmt(chamfer_info['angle'])}, "
                f"distance={self._fmt(chamfer_info['distance'])})"
            )
        elif chamfer_info:
            code_lines.append(
                f"# Chamfer: Could not extract edge position for {chamfer_info.get('name', 'unknown')}"
            )

        return code_lines

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
                    if hasattr(threaded_face_obj, "Count"):
                        if threaded_face_obj.Count > 0:
                            face = threaded_face_obj.Item(1)
                            print(
                                f"Got face from Faces collection (count: {threaded_face_obj.Count})"
                            )
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
                                print(
                                    f"Thread face position (BasePoint): ({thread_info['x']}, {thread_info['y']}, {thread_info['z']})"
                                )
                                print(f"Thread face radius: {thread_info['radius']}")
                            except Exception as e:
                                print(
                                    f"Warning: BasePoint/Radius extraction failed: {e}"
                                )

                        # Method 2: Get from circular edges (similar to chamfer approach)
                        if "x" not in thread_info or "radius" not in thread_info:
                            try:
                                edges = face.Edges
                                for e_idx in range(1, edges.Count + 1):
                                    edge = edges.Item(e_idx)
                                    # Look for circular edges (CurveType 5124)
                                    if edge.CurveType == 5124:
                                        edge_geom = edge.Geometry
                                        if hasattr(edge_geom, "Center") and hasattr(
                                            edge_geom, "Radius"
                                        ):
                                            center = edge_geom.Center
                                            thread_info["x"] = round(center.X, 6)
                                            thread_info["y"] = round(center.Y, 6)
                                            thread_info["z"] = round(center.Z, 6)
                                            thread_info["radius"] = round(
                                                edge_geom.Radius, 6
                                            )
                                            print(
                                                f"Thread face position (from edge): ({thread_info['x']}, {thread_info['y']}, {thread_info['z']})"
                                            )
                                            print(
                                                f"Thread face radius (from edge): {thread_info['radius']}"
                                            )
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
                                    thread_info["x"] = round(
                                        (min_pt.X + max_pt.X) / 2, 6
                                    )
                                    thread_info["y"] = round(
                                        (min_pt.Y + max_pt.Y) / 2, 6
                                    )
                                    thread_info["z"] = round(
                                        (min_pt.Z + max_pt.Z) / 2, 6
                                    )
                                    print(
                                        f"Thread face position (from range): ({thread_info['x']}, {thread_info['y']}, {thread_info['z']})"
                                    )

                                if "radius" not in thread_info:
                                    # Try to get radius from geometry again
                                    if hasattr(geom, "Radius"):
                                        thread_info["radius"] = round(geom.Radius, 6)
                                        print(
                                            f"Thread face radius (from geom): {thread_info['radius']}"
                                        )
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
                                print(
                                    f"Thread face axis: {thread_info.get('axis', 'Unknown')}"
                                )
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
            print(
                f"Warning: Thread '{thread_info.get('name')}' missing position/radius - reconstruction may fail"
            )
        else:
            print(
                f"✓ Thread extracted: x={thread_info['x']}, radius={thread_info['radius']}, {thread_info['designation']}"
            )

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
        designation = thread_info.get("designation", "Unknown")
        thread_type = "internal" if thread_info.get("internal") else "external"
        code_lines.append(f"# Thread: {designation} ({thread_type})")

        # Generate thread creation code
        code_lines.append("thread = app.add_thread(")

        # Add geometric parameters if available (critical for face matching)
        if "x" in thread_info and "radius" in thread_info:
            code_lines.append(
                f"    x={self._fmt(thread_info['x'])},  # Cylindrical face center X"
            )
            code_lines.append(
                f"    radius={self._fmt(thread_info['radius'])},  # Cylindrical face radius"
            )

            # Optionally add Y/Z if non-zero (for non-axial cylinders)
            if "y" in thread_info and abs(thread_info["y"]) > 1e-6:
                code_lines.append(
                    f"    y={self._fmt(thread_info['y'])},  # Cylindrical face center Y"
                )
            if "z" in thread_info and abs(thread_info["z"]) > 1e-6:
                code_lines.append(
                    f"    z={self._fmt(thread_info['z'])},  # Cylindrical face center Z"
                )

            # Add axis direction for better matching
            if "axis" in thread_info:
                code_lines.append(
                    f"    axis='{thread_info['axis']}',  # Cylinder axis direction"
                )
        else:
            code_lines.append(
                "    # WARNING: Missing x/radius - may not match correct face!"
            )

        code_lines.append(
            f"    designation='{thread_info.get('designation', 'M8x1.25')}',"
        )
        code_lines.append(
            f"    thread_class='{thread_info.get('thread_class', '6H')}',"
        )
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
