from typing import Dict, Optional

from .cad_types import Vertex
from .cadseq import Sketch
from .constraint import CoincidenceConstraint
from .onshape_processing.custom_onshape_client import CustomOnshapeClient
from .primitive import Primitive
from .workplane import Workplane as PlaneOld


class SketchParser:
    def __init__(self, client: CustomOnshapeClient, did, wid, eid):
        self.client = client
        self.did = did
        self.wid = wid
        self.eid = eid

    def _parse_parameter_list(self, parameter_list: list):
        param_dict = {}
        for i, param_item in enumerate(parameter_list):
            param_msg = param_item["message"]
            param_id = param_msg["parameterId"]
            if "queries" in param_msg:
                param_value = []
                for i in range(len(param_msg["queries"])):
                    param_value.extend(
                        param_msg["queries"][i]["message"]["geometryIds"]
                    )  # FIXME: could be error-prone
            elif "expression" in param_msg:
                param_value = param_msg["expression"]
            elif "value" in param_msg:
                param_value = param_msg["value"]
            else:
                raise NotImplementedError("param_msg:\n{}".format(param_msg))

            param_dict.update({param_id: param_value})
        return param_dict

    def _resolve_query(self):
        return "Dummy"

    def parse(self, feature_dict) -> Optional[Sketch]:
        feature_parameters = self._parse_parameter_list(
            feature_dict["message"]["parameters"]
        )
        plane: PlaneOld = self.parse_sketch_plane(feature_parameters)
        primitive_entities = self.parse_primitive_entities(
            feature_dict["message"]["entities"]
        )
        constraints = self.parse_constraints(feature_dict["message"]["constraints"])
        # reference constraints to primitive entities
        constraint_obj_list = []
        for constraint in constraints:
            constraint_type = constraint["type"]
            if ("localFirst" in constraint) and ("localSecond" in constraint):
                local_first: Primitive = primitive_entities[
                    constraint["localFirst"].split(".")[0]
                ]
                local_second: Primitive = primitive_entities[
                    constraint["localSecond"].split(".")[0]
                ]
                if constraint["localFirst"].endswith("start"):
                    local_first: Vertex = local_first.start_point
                else:
                    local_first: Vertex = local_first.end_point
                if constraint["localSecond"].endswith("start"):
                    local_second: Vertex = local_second.start_point
                else:
                    local_second: Vertex = local_second.end_point
                options = {
                    "COINCIDENT": CoincidenceConstraint,
                }
                constraint_class = options.get(constraint_type, None)
                if constraint_class is None:
                    continue
                constraint_obj_list.append(constraint_class(local_first, local_second))

        return Sketch(
            constraints=constraint_obj_list,
            sketch_plane=plane,
            primitives=primitive_entities.values(),
        )

    def parse_sketch_plane(self, feature_parameters):
        geo_id = feature_parameters["sketchPlane"][0]
        response = self.client.get_entity_by_id(
            self.did, self.wid, self.eid, [geo_id], "FACE"
        )
        plane_data = self.client.parse_face_msg(
            response.json()["result"]["message"]["value"]
        )[0]
        plane = PlaneOld.from_n_x_axis(
            origin=plane_data["origin"],
            normal=plane_data["normal"],
            x_axis=plane_data["x"],
        )
        return plane

    def parse_constraints(self, constraint_list):
        constraint_obj = []
        for constraint in constraint_list:
            result_dict = {}
            result_dict["type"] = constraint["message"]["constraintType"]
            constraint_parameters = self._parse_parameter_list(
                constraint["message"]["parameters"]
            )
            if "localFirst" in constraint_parameters:
                result_dict["localFirst"] = constraint_parameters["localFirst"]

            if "localSecond" in constraint_parameters:
                result_dict["localSecond"] = constraint_parameters["localSecond"]

            constraint_obj.append(result_dict)
        return constraint_obj

    def parse_primitive_entities(self, primitive_entities_list: list):
        primitive_entities_obj: Dict[str, Primitive] = {}
        for entity in primitive_entities_list:
            entity_id = entity["message"]["entityId"]
            primitive_entities_obj[entity_id] = Primitive.from_dict(entity)
        return primitive_entities_obj
