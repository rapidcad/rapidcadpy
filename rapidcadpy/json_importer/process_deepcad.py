import copy
import json
from collections import OrderedDict
from typing import Dict, List

import numpy as np

from .cad import Cad
from .cad_types import Vector, Vertex
from .extrude import Extrude
from .primitive import Arc, Circle, Line
from .sketch import Sketch
from .sketch_extrude import Extrude
from .wire import Wire
from .workplane import Workplane


class DeepCadJsonParser:
    @staticmethod
    def load_json(json_file_path):
        with open(json_file_path) as fp:
            return json.load(fp)

    @staticmethod
    def _find_orthogonal_vector(normal):
        # Ensure the normal vector is a numpy array
        normal = np.asarray(normal)

        # Check for special cases where normal vector has specific components equal to zero
        if normal[0] != 0 or normal[1] != 0:
            # Choose (1, 0, 0) or (0, 1, 0) as a candidate vector if normal[0] and normal[1] are not both zero
            candidate = np.array([1, 0, 0])
            if normal[0] == 0:
                candidate = np.array([0, 1, 0])
        else:
            candidate = np.array([0, 1, 0])

        return np.cross(normal, candidate)

    @staticmethod
    def parse_vector(vector_dict):
        return Vector(vector_dict["x"], vector_dict["y"], vector_dict["z"])

    @staticmethod
    def parse_curve(curve_dict, curve_name):
        if "circle" in curve_dict["type"].lower():
            center_point = DeepCadJsonParser._parse_point(curve_dict["center_point"])
            radius = curve_dict["radius"]
            return Circle(center_point, radius, name=curve_dict["curve"])
        elif "line" in curve_dict["type"].lower():
            start_point = DeepCadJsonParser._parse_point(curve_dict["start_point"])
            end_point = DeepCadJsonParser._parse_point(curve_dict["end_point"])
            return Line(
                name="line_" + curve_dict["curve"],
                start_point=start_point,
                end_point=end_point,
            )
        elif "arc" in curve_dict["type"].lower():
            radius = curve_dict["radius"]
            start_point = DeepCadJsonParser._parse_point(curve_dict["start_point"])
            end_point = DeepCadJsonParser._parse_point(curve_dict["end_point"])
            center_point = DeepCadJsonParser._parse_point(curve_dict["center_point"])
            start_angle = curve_dict["start_angle"]
            end_angle = curve_dict["end_angle"]
            reference_vector = DeepCadJsonParser.parse_vector(
                curve_dict["reference_vector"]
            )

            ref_vec = reference_vector.normalize()
            mid_angle = (start_angle + end_angle) / 2
            # Rotation matrix for the midpoint angle
            rot_matrix = np.array(
                [
                    [np.cos(mid_angle), -np.sin(mid_angle)],
                    [np.sin(mid_angle), np.cos(mid_angle)],
                ]
            )

            # Rotate the reference vector to get the direction for the midpoint
            rotated_vec = rot_matrix @ ref_vec.get_2d()

            # Calculate midpoint coordinates
            mid_x = center_point[0] + rotated_vec[0] * radius
            mid_y = center_point[1] + rotated_vec[1] * radius

            return Arc(
                start_point=start_point,
                mid_point=Vertex(x=mid_x, y=mid_y),
                end_point=end_point,
                id=curve_dict["curve"],
            )

    @staticmethod
    def parse_loops(profiles: dict) -> dict:
        parsed_inner_loops = []
        parsed_outer_loops = []
        for loop in profiles["loops"]:
            curves = []
            for i, curve in enumerate(loop["profile_curves"]):
                curves.append(DeepCadJsonParser.parse_curve(curve, curve_name=i))
            if loop["is_outer"]:
                parsed_outer_loops.append(Wire(curves))
            elif not loop["is_outer"]:
                parsed_inner_loops.append(Wire(curves))
        if len(parsed_outer_loops) > 1:
            loops_with_area = [
                {"loop": loop, "area": loop.bounding_box_area()}
                for loop in parsed_outer_loops
            ]
            loops_with_area.sort(key=lambda x: x["area"], reverse=True)

            parsed_outer_loop = loops_with_area[0]["loop"]
            parsed_inner_loops = [entry["loop"] for entry in loops_with_area[1:]]
        else:
            parsed_outer_loop = parsed_outer_loops[0]

        return {"outer_wires": parsed_outer_loop, "inner_wires": parsed_inner_loops}

    @staticmethod
    def _parse_point(point_dict):
        return Vertex(point_dict["x"], point_dict["y"])

    @staticmethod
    def parse_face(profile_id, face_dict, curve_dict):
        outer_loop = []
        inner_loop_curve = []
        num_outer_loop = 0
        edge_counter = 0
        arc_counter = 0
        for loop in face_dict["loops"]:
            if loop["is_outer"]:
                num_outer_loop += 1
                for curve in loop["profile_curves"]:
                    if curve["type"] == "Line3D":
                        outer_loop.append(
                            DeepCadJsonParser._parse_line(
                                curve, edge_name=f"edge_{edge_counter}"
                            )
                        )
                        edge_counter += 1
                    elif curve["type"] == "Arc3D":
                        outer_loop.append(
                            DeepCadJsonParser._parse_arc(
                                curve, edge_name=f"arc_{arc_counter}"
                            )
                        )
                        arc_counter += 1
                    elif curve["type"] == "Circle3D":
                        outer_loop.append(DeepCadJsonParser._parse_circle(curve))
            elif not loop["is_outer"]:
                current_inner_loop = set()
                for curve in loop["profile_curves"]:
                    curve_obj = curve_dict[curve["curve"]]
                    current_inner_loop.add(curve_obj)
                inner_loop_curve.append(Wire(edges=list(current_inner_loop)))

        if num_outer_loop > 1:
            raise NotImplementedError("Multiple outer loops not supported")
        if num_outer_loop == 0:
            raise ValueError("No outer loop found")

        return {
            "outer_wire": Wire(outer_loop),
            "inner_wires": inner_loop_curve,
        }

    @staticmethod
    def parse_transform(plane_dict):
        origin = DeepCadJsonParser.parse_vector(plane_dict["origin"])
        normal = DeepCadJsonParser.parse_vector(plane_dict["z_axis"])
        y_axis = DeepCadJsonParser.parse_vector(plane_dict["y_axis"])
        x_axis = DeepCadJsonParser.parse_vector(plane_dict["x_axis"])
        return Workplane(origin, x_dir=x_axis, y_dir=y_axis, z_dir=normal)

    @staticmethod
    def process_sequences(ordered_seq_dict, entity_dict) -> List[Extrude]:
        skex_seq = []
        profile_dict: Dict[str, Wire] = {}

        for seq_index, seq in ordered_seq_dict.items():
            entity = entity_dict[seq["entity"]]
            if entity["type"].lower() == "sketch":
                profile_dict.update(
                    {
                        profile_id: DeepCadJsonParser.parse_loops(loops)
                        for i, (profile_id, loops) in enumerate(
                            entity["profiles"].items()
                        )
                    }
                )
                plane = DeepCadJsonParser.parse_transform(entity["transform"])
            elif "extrude" in entity["type"].lower():
                extent_one = entity["extent_one"]["distance"]["value"]
                taper_angle_one = entity["extent_one"]["taper_angle"]["value"]
                extent_two = (
                    entity["extent_two"]["distance"]["value"]
                    if "extent_two" in entity
                    else 0.0
                )
                taper_angle_two = (
                    entity["extent_two"]["taper_angle"]["value"]
                    if "extent_two" in entity
                    else 0.0
                )
                extent_type = entity["extent_type"]
                operation = entity["operation"]
                extruded_sketches: List[Sketch] = []
                for profile in entity["profiles"]:
                    parsed_profile = profile_dict[profile["profile"]]
                    sketch_entity_obj = Sketch(
                        id=seq["entity"],
                        outer_wire=parsed_profile["outer_wires"],
                        inner_wires=parsed_profile["inner_wires"],
                        name=profile["profile"],
                    )
                    extruded_sketches.append(sketch_entity_obj)

                if extent_one < 0:
                    extent_one = -extent_one
                    direction = -1
                else:
                    direction = 1

                extent_obj = Extrude(
                    extent_one=extent_one,
                    direction=direction,
                    extent_two=extent_two,
                    extent_type=extent_type,
                    operation=operation,
                    taper_angle_one=taper_angle_one,
                    taper_angle_two=taper_angle_two,
                    name=entity["name"],
                )
                skex_seq.append(
                    Extrude(
                        sketch=extruded_sketches,
                        extrude=extent_obj,
                        sketch_plane=copy.deepcopy(plane),
                        name=entity["name"],
                    )
                )

        return skex_seq

    @staticmethod
    def build_seq_dict(json_obj):
        seq_dict = {seq["index"]: seq for seq in json_obj["sequence"]}
        return OrderedDict(sorted(seq_dict.items(), key=lambda item: item[0]))

    @staticmethod
    def build_entity_dict(json_obj):
        return {entity_id: entity for entity_id, entity in json_obj["entities"].items()}

    @staticmethod
    def process(file: str) -> Cad:
        json_data = DeepCadJsonParser.load_json(json_file_path=file)
        ordered_seq_dict = DeepCadJsonParser.build_seq_dict(json_data)
        entity_dict = DeepCadJsonParser.build_entity_dict(json_data)
        skex_seq: List[Extrude] = DeepCadJsonParser.process_sequences(
            ordered_seq_dict, entity_dict
        )
        cad_seq = Cad(construction_sequence=skex_seq)
        # cad_seq.normalize()
        return cad_seq
