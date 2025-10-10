import json
import uuid
from collections import OrderedDict
from typing import Dict, List, Optional

import numpy as np

from rapidcadpy.cad import Cad
from rapidcadpy.cad_types import Vector, Vertex
from rapidcadpy.constraint import (
    EndToStartCoincidenceConstraint,
    HorizontalConstraint,
    ParallelConstraint,
    PerpendicularConstraint,
    StartToEndCoincidenceConstraint,
    TangentConstraint,
    VerticalConstraint,
)
from rapidcadpy.extrude import Extrude
from rapidcadpy.primitive import Arc, Circle, Line
from rapidcadpy.sketch import Sketch
from rapidcadpy.sketch_extrude import Extrude
from rapidcadpy.wire import Wire
from rapidcadpy.workplane import Workplane


class Fusion360GalleryParser:
    def __init__(self):
        self.start_point_of_primitive: dict = {}
        self.end_point_of_primitive: dict = {}

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

    def parse_curve(
        self, curve_dict, point_dict, curve_name: str, curve_id: Optional[str] = None
    ) -> Circle | Line | Arc:
        if "circle" in curve_dict["type"].lower():
            center_point = point_dict[curve_dict["center_point"]]
            radius = curve_dict["radius"]
            return Circle(center_point, radius, name=f"circle_{curve_name}")
        elif "line" in curve_dict["type"].lower():
            start_point = point_dict[curve_dict["start_point"]]
            end_point = point_dict[curve_dict["end_point"]]
            line = Line(start_point, end_point, name="line_" + curve_name, id=curve_id)
            self.start_point_of_primitive[start_point] = line
            self.end_point_of_primitive[end_point] = line
            return line
        elif "arc" in curve_dict["type"].lower():
            start_point = point_dict[curve_dict["start_point"]]
            end_point = point_dict[curve_dict["end_point"]]
            center_point = point_dict[curve_dict["center_point"]]
            radius = curve_dict["radius"]
            start_angle = curve_dict["start_angle"]
            end_angle = curve_dict["end_angle"]
            reference_vector = Fusion360GalleryParser.parse_vector(
                curve_dict["reference_vector"]
            )
            return Arc(
                radius=radius,
                start_point=start_point,
                end_point=end_point,
                center=center_point,
                start_angle=start_angle,
                end_angle=end_angle,
                ref_vec=reference_vector,
                name=f"arc_{curve_name}",
                ccw=None,
                id=curve_id,
            )

    def parse_loop_curve(
        self, curve_dict, curve_name: str, curve_id: Optional[str] = None
    ) -> Circle | Line | Arc:
        if "circle" in curve_dict["type"].lower():
            center_point = Fusion360GalleryParser._parse_point(
                curve_dict["center_point"]
            )
            radius = curve_dict["radius"]
            return Circle(center_point, radius, name=f"circle_{curve_name}")
        elif "line" in curve_dict["type"].lower():
            start_point = Fusion360GalleryParser._parse_point(curve_dict["start_point"])
            end_point = Fusion360GalleryParser._parse_point(curve_dict["end_point"])
            line = Line(start_point, end_point, name="line_" + curve_name, id=curve_id)
            self.start_point_of_primitive[start_point] = line
            self.end_point_of_primitive[end_point] = line
            return line
        elif "arc" in curve_dict["type"].lower():
            start_point = Fusion360GalleryParser._parse_point(curve_dict["start_point"])
            end_point = Fusion360GalleryParser._parse_point(curve_dict["end_point"])
            center_point = Fusion360GalleryParser._parse_point(
                curve_dict["center_point"]
            )
            radius = curve_dict["radius"]
            start_angle = curve_dict["start_angle"]
            end_angle = curve_dict["end_angle"]
            reference_vector = Fusion360GalleryParser.parse_vector(
                curve_dict["reference_vector"]
            )
            return Arc(
                radius=radius,
                start_point=start_point,
                end_point=end_point,
                center=center_point,
                start_angle=start_angle,
                end_angle=end_angle,
                ref_vec=reference_vector,
                name=f"arc_{curve_name}",
                ccw=None,
                id=curve_id,
            )
        else:
            raise NotImplementedError("NURBS, ellipses, are not supported")

    def parse_constraint(self, constraint, curve_dict, point_dict):
        if "coincident" in constraint["type"].lower():
            try:
                point_1 = point_dict[constraint["entity"]]
                point_2 = point_dict[constraint["point"]]
                if (
                    point_1 in self.start_point_of_primitive
                    and point_2 in self.end_point_of_primitive
                ):
                    return StartToEndCoincidenceConstraint(
                        self.start_point_of_primitive[point_1],
                        self.end_point_of_primitive[point_2],
                    )
                elif (
                    point_1 in self.end_point_of_primitive
                    and point_2 in self.start_point_of_primitive
                ):
                    return EndToStartCoincidenceConstraint(
                        self.end_point_of_primitive[point_1],
                        self.start_point_of_primitive[point_2],
                    )
            except KeyError:
                return None
        if (
            "horizontal" in constraint["type"].lower()
            and "point" not in constraint["type"].lower()
        ):
            return HorizontalConstraint(
                curve_dict[constraint["line"]],
            )
        if (
            "vertical" in constraint["type"].lower()
            and "point" not in constraint["type"].lower()
        ):
            return VerticalConstraint(
                curve_dict[constraint["line"]],
            )
        if "perpendicular" in constraint["type"].lower():
            return PerpendicularConstraint(
                curve_dict[constraint["line_one"]],
                curve_dict[constraint["line_two"]],
            )
        if "parallel" in constraint["type"].lower():
            return ParallelConstraint(
                curve_dict[constraint["line_one"]],
                curve_dict[constraint["line_two"]],
            )
        if "tangent" in constraint["type"].lower():
            return TangentConstraint(
                curve_dict[constraint["curve_one"]],
                curve_dict[constraint["curve_one"]],
            )

    @staticmethod
    def _parse_point(point_dict):
        return Vertex(point_dict["x"], point_dict["y"], id=point_dict)

    def parse_face(self, loops, curve_dict):
        outer_loop = []
        inner_loop_curve = []
        num_outer_loop = 0
        for loop in loops:
            if loop["is_outer"]:
                num_outer_loop += 1
                for curve in loop["profile_curves"]:
                    outer_loop.append(curve_dict[curve["curve"]])
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
    def parse_plane(plane_dict, transform_dict):
        origin = Fusion360GalleryParser.parse_vector(transform_dict["origin"])
        x_dir = Fusion360GalleryParser.parse_vector(transform_dict["x_axis"])
        y_dir = Fusion360GalleryParser.parse_vector(transform_dict["y_axis"])
        z_dir = Fusion360GalleryParser.parse_vector(transform_dict["z_axis"])
        return Workplane(origin, x_dir, y_dir, z_dir)

    @staticmethod
    def build_seq_dict(json_obj):
        seq_dict = {seq["index"]: seq for seq in json_obj["sequence"]}
        return OrderedDict(sorted(seq_dict.items(), key=lambda item: item[0]))

    @staticmethod
    def build_entity_dict(json_obj):
        return {entity_id: entity for entity_id, entity in json_obj["entities"].items()}

    def process_sequences(self, ordered_seq_dict, entity_dict) -> List[Extrude]:
        skex_seq = []
        point_dict: Dict[str, Vertex] = {}
        curve_dict: Dict[str, Circle | Line | Arc] = {}
        face_dict: Dict[str, any] = {}
        profile_dict = {}

        for seq_index, seq in ordered_seq_dict.items():
            entity = entity_dict[seq["entity"]]
            if entity["type"].lower() == "sketch":
                point_dict.update(
                    {
                        point_id: Vertex(
                            point["x"],
                            point["y"],
                            id=point_id,
                            name=f"vertex_{i}",
                        )
                        for i, (point_id, point) in enumerate(entity["points"].items())
                    }
                )
                for profile_id, profile in entity["profiles"].items():
                    profile_dict[profile_id] = profile["loops"]
                    for loop in entity["profiles"][profile_id]["loops"]:
                        for curve in loop["profile_curves"]:
                            curve_id = curve["curve"]
                            curve_obj = self.parse_loop_curve(
                                curve, curve_name=curve_id, curve_id=curve_id
                            )
                            curve_dict[curve_id] = curve_obj

                plane = Fusion360GalleryParser.parse_plane(
                    entity["reference_plane"], entity["transform"]
                )
                if "constraints" in entity:
                    constraint_dict = {
                        constraint_id: constraint
                        for constraint_id, constraint in entity["constraints"].items()
                    }
                else:
                    constraint_dict = {}
            elif "extrude" in entity["type"].lower():
                extent_one = entity["extent_one"]["distance"]["value"]
                taper_angle_one = entity["extent_one"]["taper_angle"]["value"]
                extent_two = (
                    entity["extent_two"]["distance"]["value"]
                    if "extent_two" in entity
                    else 0.0
                )
                extent_type = entity["extent_type"]
                extent_operation = entity["operation"]
                taper_angle_two = (
                    entity["extent_two"]["taper_angle"]["value"]
                    if "extent_two" in entity
                    else 0.0
                )
                for profile in entity["profiles"]:
                    parsed_face = self.parse_face(
                        profile_dict[profile["profile"]], curve_dict
                    )
                    constraints = []
                    for constraint in constraint_dict:
                        try:
                            parsed = self.parse_constraint(
                                constraint_dict[constraint], curve_dict, point_dict
                            )
                            if parsed is not None:
                                constraints.append(parsed)
                        except KeyError:
                            continue
                    sketch_entity_obj = Sketch(
                        id=uuid.UUID(seq["entity"]),
                        outer_wire=parsed_face["outer_wire"],
                        inner_wires=parsed_face["inner_wires"],
                        sketch_plane=plane,
                        constraints=constraints,
                    )

                    extent_obj = Extrude(
                        extent_one=extent_one,
                        extent_two=extent_two,
                        extent_type=extent_type,
                        operation=extent_operation,
                        symmetric=(
                            entity["extent_type"] == "SymmetricFeatureExtentType"
                        ),
                        taper_angle_one=taper_angle_one,
                        taper_angle_two=taper_angle_two,
                    )
                    skex_seq.append(
                        Extrude(sketch=[sketch_entity_obj], extrude=extent_obj)
                    )

        return skex_seq

    @staticmethod
    def assemble_cad_sequence(skex_seq, exploded=False):
        from build123d import Compound

        occ_object = skex_seq.assemble_3d()
        if not exploded:
            return Compound(children=occ_object)
        return [Compound(children=[obj]) for obj in occ_object]

    @staticmethod
    def parse(json_file_path) -> Cad:
        json_obj = Fusion360GalleryParser.load_json(json_file_path)
        ordered_seq_dict = Fusion360GalleryParser.build_seq_dict(json_obj)
        entity_dict = Fusion360GalleryParser.build_entity_dict(json_obj)
        parser = Fusion360GalleryParser()
        skex_seq: List[Extrude] = parser.process_sequences(
            ordered_seq_dict, entity_dict
        )
        return Cad(construction_history=skex_seq)

    @staticmethod
    def process_json_data(json_data: dict) -> Cad:
        ordered_seq_dict = Fusion360GalleryParser.build_seq_dict(json_data)
        entity_dict = Fusion360GalleryParser.build_entity_dict(json_data)
        parser = Fusion360GalleryParser()
        skex_seq: List[Extrude] = parser.process_sequences(
            ordered_seq_dict, entity_dict
        )
        return Cad(construction_sequence=skex_seq)
