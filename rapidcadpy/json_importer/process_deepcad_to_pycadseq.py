import json
import math
from typing import Dict, List, Tuple, Any, Optional
from collections import OrderedDict

import numpy as np


class DeepCadToPyCadSeqGenerator:
    """
    Generates PyCadSeq Python source code from DeepCAD JSON data.
    Similar to the Inventor reverse engineer but for JSON input.
    """
    
    def __init__(self, backend: str = "inventor"):
        self.generated_code = []
        self.backend = backend.lower()

    @staticmethod
    def build_seq_dict(json_obj):
        seq_dict = {seq["index"]: seq for seq in json_obj["sequence"]}
        return OrderedDict(sorted(seq_dict.items(), key=lambda item: item[0]))

    @staticmethod
    def build_entity_dict(json_obj):
        return {entity_id: entity for entity_id, entity in json_obj["entities"].items()}
    
    @staticmethod
    def load_json(json_file_path):
        with open(json_file_path) as fp:
            return json.load(fp)
        
    def generate_code_from_json(self, json_file_path: str) -> str:
        """
        Generate Python code from a DeepCAD JSON file.
        
        Args:
            json_file_path: Path to the DeepCAD JSON file
            
        Returns:
            Generated Python code as string
        """
        # Load and parse JSON data
        json_data = DeepCadToPyCadSeqGenerator.load_json(json_file_path)
        ordered_seq_dict = DeepCadToPyCadSeqGenerator.build_seq_dict(json_data)
        entity_dict = DeepCadToPyCadSeqGenerator.build_entity_dict(json_data)
        
        # Initialize code
        if self.backend == "inventor":
            self.generated_code = [
                "from rapidcadpy import InventorApp",
                "",
                "# Initialize Inventor application",
                "app = InventorApp()",
                "app.new_document()",
                "",
            ]
        elif self.backend == "occ":
            self.generated_code = [
                "from rapidcadpy import OpenCascadeApp",
                "",
                "# Initialize OCC application",
                "app = OpenCascadeApp()",
                "",
            ]
        else:
            raise ValueError(f"Unsupported backend: {self.backend}. Supported backends: 'inventor', 'occ'")
        
        # Process sequences in order
        self._process_sequences(ordered_seq_dict, entity_dict)
        
        return "\n".join(self.generated_code)
    
    def _process_sequences(self, ordered_seq_dict: OrderedDict, entity_dict: Dict[str, Any]):
        """Process the construction sequences and generate code."""
        sketch_counter = 1
        feature_counter = 1
        current_workplane = None
        # Map: sketch_id -> {profile_name: profile_data}
        sketch_profiles = {}
        # Map: extrude_index -> (sketch_id, profile_name)
        extrude_profile_refs = []
        # First pass: collect all profiles and extrude references
        for seq_index, seq in ordered_seq_dict.items():
            entity = entity_dict[seq["entity"]]
            if entity["type"].lower() == "sketch":
                sketch_id = seq["entity"]
                sketch_profiles[sketch_id] = entity["profiles"]
            elif "extrude" in entity["type"].lower():
                for prof in entity.get("profiles", []):
                    extrude_profile_refs.append((seq_index, prof["sketch"], prof["profile"]))
        # Second pass: generate only referenced sketches/profiles
        for seq_index, seq in ordered_seq_dict.items():
            entity = entity_dict[seq["entity"]]
            if entity["type"].lower() == "sketch":
                sketch_id = seq["entity"]
                # Find all profiles from this sketch that are referenced by extrudes
                used_profiles = [p for (_, s_id, p) in extrude_profile_refs if s_id == sketch_id]
                if used_profiles:
                    current_workplane = f"wp{sketch_counter}"
                    self._generate_sketch_code_filtered(entity, current_workplane, sketch_counter, used_profiles)
                    sketch_counter += 1
            elif "extrude" in entity["type"].lower():
                if current_workplane:
                    self._generate_extrude_code(entity, current_workplane, feature_counter)
                    feature_counter += 1

    def _generate_sketch_code_filtered(self, sketch_entity: Dict[str, Any], wp_var: str, sketch_num: int, used_profiles: list):
        """Generate code for only the used profiles in a sketch entity."""
        # Extract workplane information
        transform = sketch_entity["transform"]
        workplane_info = self._get_workplane_info_from_transform(transform)
        
        # Create workplane code
        if "plane_name" in workplane_info:
            self.generated_code.extend([
                f"# Sketch {sketch_num}",
                f"{wp_var} = app.work_plane(\"{workplane_info['plane_name']}\")",
                "",
            ])
        else:
            self.generated_code.extend([
                f"# Sketch {sketch_num}",
                f"{wp_var} = app.work_plane(",
                f"    origin={workplane_info['origin']},",
                f"    x_dir={workplane_info['x_dir']},",
                f"    y_dir={workplane_info['y_dir']},",
                ")",
                "",
            ])
        
        # Only process profiles referenced by extrude
        for profile_name in used_profiles:
            profile_data = sketch_entity["profiles"][profile_name]
            self._generate_profile_code(profile_data, wp_var)
    
    def _get_workplane_info_from_transform(self, transform: Dict[str, Any]) -> Dict[str, Any]:
        """Extract workplane information from transform data."""
        origin = transform["origin"]
        x_axis = transform["x_axis"]
        y_axis = transform["y_axis"]
        z_axis = transform["z_axis"]
        
        # Check if it's a standard plane (XY, XZ, YZ)
        origin_tuple = (round(origin["x"], 6), round(origin["y"], 6), round(origin["z"], 6))
        x_dir = (round(x_axis["x"], 6), round(x_axis["y"], 6), round(x_axis["z"], 6))
        y_dir = (round(y_axis["x"], 6), round(y_axis["y"], 6), round(y_axis["z"], 6))
        z_dir = (round(z_axis["x"], 6), round(z_axis["y"], 6), round(z_axis["z"], 6))
        
        # Check for standard planes at origin
        if origin_tuple == (0.0, 0.0, 0.0):
            if (x_dir == (1.0, 0.0, 0.0) and y_dir == (0.0, 1.0, 0.0) and z_dir == (0.0, 0.0, 1.0)):
                return {"plane_name": "XY"}
            elif (x_dir == (1.0, 0.0, 0.0) and y_dir == (0.0, 0.0, 1.0) and z_dir == (0.0, -1.0, 0.0)):
                return {"plane_name": "XZ"}
            elif (x_dir == (0.0, 1.0, 0.0) and y_dir == (0.0, 0.0, 1.0) and z_dir == (1.0, 0.0, 0.0)):
                return {"plane_name": "YZ"}
        
        # For custom planes, return full specification
        return {
            'origin': origin_tuple,
            'x_dir': x_dir,
            'y_dir': y_dir,
        }
    
    def _generate_profile_code(self, profile_data: Dict[str, Any], wp_var: str):
        """Generate code for a profile (collection of curves)."""
        # Organize curves into connected paths
        paths = self._organize_curves_into_paths(profile_data["loops"])

        # Only process the first outer loop (like CadQuery)
        if paths:
            path = paths[0]
            if len(path) == 1 and path[0]["type"] == "circle":
                # Single circle
                circle = path[0]
                center = circle["center_point"]
                self.generated_code.append(
                    f"{wp_var}.move_to({center['x']}, {center['y']}).circle({circle['radius']})"
                )
            else:
                # Connected path of lines and arcs, ending with .close()
                code_line = self._generate_path_code_return(path, wp_var)
                self.generated_code.append(code_line)
    
    @staticmethod
    def parse_vector(vector_dict):
        return np.array([vector_dict["x"], vector_dict["y"], vector_dict["z"]])

    def _generate_path_code_return(self, curves: List[Dict[str, Any]], wp_var: str) -> str:
        """Return code for a connected path of curves as a string (for chaining .close())."""
        if not curves:
            return ""
        first_curve = curves[0]
        start_point = self._get_curve_start_point(first_curve)
        code_line = f"{wp_var}.move_to({start_point[0]}, {start_point[1]})"
        for curve in curves:
            if "line" in curve["type"].lower():
                end_point = curve["end_point"]
                code_line += f".line_to({end_point['x']}, {end_point['y']})"
            elif "arc" in curve["type"].lower():
                end_point = curve["end_point"]
                center_point = curve["center_point"]
                radius = curve["radius"]
                start_angle = curve["start_angle"]
                end_angle = curve["end_angle"]
                mid_angle = (start_angle + end_angle) / 2

                reference_vector = DeepCadToPyCadSeqGenerator.parse_vector(
                    curve["reference_vector"]
                )

                ref_vec = reference_vector / np.linalg.norm(reference_vector)
                mid_angle = (start_angle + end_angle) / 2
                # Rotation matrix for the midpoint angle
                rot_matrix = np.array(
                    [
                        [np.cos(mid_angle), -np.sin(mid_angle)],
                        [np.sin(mid_angle), np.cos(mid_angle)],
                    ]
                )

                # Rotate the reference vector to get the direction for the midpoint
                rotated_vec = rot_matrix @ ref_vec[:2]

                # Calculate midpoint coordinates
                mid_x = center_point["x"] + rotated_vec[0] * radius
                mid_y = center_point["y"] + rotated_vec[1] * radius

                code_line += f".three_point_arc(({round(mid_x, 6)}, {round(mid_y, 6)}), ({end_point['x']}, {end_point['y']}))"
            elif "circle" in curve["type"].lower():
                center = curve["center_point"]
                radius = curve["radius"]
                self.generated_code.append(f"{wp_var}.move_to({center['x']}, {center['y']}).circle({radius})")
                return ""
        return code_line

    def _organize_curves_into_paths(self, loops: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """Organize curves from loops into connected paths, ordering curves for connectivity and left-most start."""
        def get_point(curve, key):
            pt = curve.get(key)
            return (pt["x"], pt["y"]) if pt else None

        def reorder_curves(curves):
            if not curves or len(curves) == 1:
                return curves
            
            def points_equal(p1, p2, tolerance=1e-9):
                """Compare points with tolerance for floating-point precision."""
                if p1 is None or p2 is None:
                    return False
                return abs(p1[0] - p2[0]) < tolerance and abs(p1[1] - p2[1]) < tolerance
            
            # Build connectivity map
            points = [(get_point(c, "start_point"), get_point(c, "end_point")) for c in curves]
            used = [False] * len(curves)
            
            # Find left-most start point
            valid_indices = [i for i in range(len(curves)) if points[i][0] is not None]
            if not valid_indices:
                return curves
            
            left_idx = min(valid_indices, key=lambda i: (points[i][0][0], points[i][0][1]))
            ordered = [curves[left_idx]]
            used[left_idx] = True
            last_pt = points[left_idx][1]
            
            # Try to connect remaining curves
            for _ in range(len(curves) - 1):  # Only need to add N-1 more curves
                found = False
                
                # Try forward connection
                for j, (start, end) in enumerate(points):
                    if not used[j] and points_equal(start, last_pt):
                        ordered.append(curves[j])
                        used[j] = True
                        last_pt = end
                        found = True
                        break
                
                if not found:
                    # Try reverse connection
                    for j, (start, end) in enumerate(points):
                        if not used[j] and points_equal(end, last_pt):
                            # Reverse the curve
                            curves[j]["start_point"], curves[j]["end_point"] = (
                                curves[j]["end_point"], 
                                curves[j]["start_point"]
                            )
                            # Update points array
                            points[j] = (end, start)
                            ordered.append(curves[j])
                            used[j] = True
                            last_pt = start  # Use the new end point
                            found = True
                            break
                
                if not found:
                    # Could not connect - this might be a disconnected loop
                    # Add remaining curves as a warning or try starting a new chain
                    print(f"Warning: Could not connect all curves. Connected {len(ordered)} of {len(curves)}")
                    break
            
            # Return all curves, even if some couldn't be connected
            # Add any remaining unused curves at the end
            for j, curve in enumerate(curves):
                if not used[j]:
                    ordered.append(curve)
                    print(f"Warning: Curve {j} added at end (disconnected)")
            
            return ordered
        paths = []
        for loop in loops:
            if loop["is_outer"]:
                curves = loop["profile_curves"]
                if curves:
                    curves = reorder_curves(curves)
                    paths.append(curves)
        return paths
    
    def _generate_path_code(self, curves: List[Dict[str, Any]], wp_var: str):
        """(Deprecated) Generate code for a connected path of curves."""
        pass
    
    def _get_curve_start_point(self, curve: Dict[str, Any]) -> Tuple[float, float]:
        """Get the start point of a curve."""
        if "start_point" in curve:
            start = curve["start_point"]
            return (start["x"], start["y"])
        elif "center_point" in curve:  # For circles
            center = curve["center_point"]
            return (center["x"], center["y"])
        else:
            return (0.0, 0.0)
    
    def _generate_extrude_code(self, extrude_entity: Dict[str, Any], wp_var: str, feature_num: int):
        """Generate code for an extrude feature."""
        extent_one = extrude_entity["extent_one"]["distance"]["value"]
        operation = extrude_entity["operation"]
        
        # Handle negative extrusion
        if extent_one < 0:
            extent_one = -extent_one
        
        # Map operation types
        operation_map = {
            "NewBodyFeatureOperation": "NewBodyFeatureOperation",
            "JoinFeatureOperation": "JoinBodyFeatureOperation", 
            "CutFeatureOperation": "Cut",
            "IntersectFeatureOperation": "Intersect",
        }
        mapped_operation = operation_map.get(operation, "NewBodyFeatureOperation")
        
        self.generated_code.extend([
            f"# Extrude feature {feature_num}",
            f"shape{feature_num} = {wp_var}.extrude({extent_one}, '{mapped_operation}')",
            "",
        ])


def generate_pycadseq_code_from_deepcad_json(json_file_path: str, backend: str = "inventor", output_file: Optional[str] = None) -> str:
    """
    Generate PyCadSeq Python code from a DeepCAD JSON file.
    
    Args:
        json_file_path: Path to the DeepCAD JSON file
        backend: Backend to use ('inventor' or 'occ')
        output_file: Optional path to save the generated code
        
    Returns:
        Generated Python code as string
    """
    generator = DeepCadToPyCadSeqGenerator(backend=backend)
    generated_code = generator.generate_code_from_json(json_file_path)
    
    # Save to file if specified
    if output_file:
        with open(output_file, 'w') as f:
            f.write(generated_code)
        print(f"Generated code saved to: {output_file}")
    
    return generated_code


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python process_deepcad_to_pycadseq.py <json_file_path> [backend] [output_file_path]")
        print("Backends: 'inventor' (default), 'occ'")
        sys.exit(1)
    
    json_file = sys.argv[1]
    backend = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] in ["inventor", "occ"] else "inventor"
    output_file = sys.argv[3] if len(sys.argv) > 3 else (sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] not in ["inventor", "occ"] else None)
    
    try:
        code = generate_pycadseq_code_from_deepcad_json(json_file, backend=backend, output_file=output_file)
        print("Generated Python code:")
        print("-" * 50)
        print(code)
    except Exception as e:
        print(f"Error: {e}")
