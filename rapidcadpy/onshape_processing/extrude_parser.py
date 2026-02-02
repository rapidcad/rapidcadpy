from .onshape_processing.custom_onshape_client import CustomOnshapeClient

EXTENT_TYPE_MAP = {
    "BLIND": "OneSideFeatureExtentType",
    "SYMMETRIC": "SymmetricFeatureExtentType",
}
OPERATION_MAP = {
    "NEW": "NewBodyFeatureOperation",
    "ADD": "JoinFeatureOperation",
    "REMOVE": "CutFeatureOperation",
    "INTERSECT": "IntersectFeatureOperation",
}


class ExtrudeParser:
    def __init__(self, client: CustomOnshapeClient, did, wid, eid):
        self.client = client
        self.did = did
        self.wid = wid
        self.eid = eid

    @staticmethod
    def parse_feature_param(feat_param_data):
        param_dict = {}
        for i, param_item in enumerate(feat_param_data):
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

    def parse(self, feature_data):
        param_dict = self.parse_feature_param(feature_data["message"]["parameters"])
        if "hasOffset" in param_dict and param_dict["hasOffset"] is True:
            raise NotImplementedError(
                "extrude with offset not supported: {}".format(param_dict["hasOffset"])
            )

        entities = param_dict["entities"]  # geometryIds for target face

        extent_one = self._expr2meter(param_dict["depth"])
        if param_dict["endBound"] == "SYMMETRIC":
            extent_one = extent_one / 2
        if (
            "oppositeDirection" in param_dict
            and param_dict["oppositeDirection"] is True
        ):
            extent_one = -extent_one
        extent_two = 0.0
        if param_dict["endBound"] not in ["BLIND", "SYMMETRIC"]:
            raise NotImplementedError(
                "endBound type not supported: {}".format(param_dict["endBound"])
            )
        elif (
            "hasSecondDirection" in param_dict
            and param_dict["hasSecondDirection"] is True
        ):
            if param_dict["secondDirectionBound"] != "BLIND":
                raise NotImplementedError(
                    "secondDirectionBound type not supported: {}".format(
                        param_dict["endBound"]
                    )
                )
            extent_type = "TwoSidesFeatureExtentType"
            extent_two = self._expr2meter(param_dict["secondDirectionDepth"])
            if (
                "secondDirectionOppositeDirection" in param_dict
                and str(param_dict["secondDirectionOppositeDirection"]) == "true"
            ):
                extent_two = -extent_two
        else:
            extent_type = EXTENT_TYPE_MAP[param_dict["endBound"]]

        operation = OPERATION_MAP[param_dict["operationType"]]

        save_dict = {
            "name": feature_data["message"]["name"],
            "type": "ExtrudeFeature",
            "entities": entities,
            "operation": operation,
            "start_extent": {"type": "ProfilePlaneStartDefinition"},
            "extent_type": extent_type,
            "extent_one": {
                "distance": {
                    "type": "ModelParameter",
                    "value": extent_one,
                    "name": "none",
                    "role": "AlongDistance",
                },
                "taper_angle": {
                    "type": "ModelParameter",
                    "value": 0.0,
                    "name": "none",
                    "role": "TaperAngle",
                },
                "type": "DistanceExtentDefinition",
            },
            "extent_two": {
                "distance": {
                    "type": "ModelParameter",
                    "value": extent_two,
                    "name": "none",
                    "role": "AgainstDistance",
                },
                "taper_angle": {
                    "type": "ModelParameter",
                    "value": 0.0,
                    "name": "none",
                    "role": "Side2TaperAngle",
                },
                "type": "DistanceExtentDefinition",
            },
        }
        return save_dict

    def _locateSketchProfile(self, geo_ids):
        return [{"profile": k, "sketch": self.profile2sketch[k]} for k in geo_ids]

    def _expr2meter(self, expr):
        return self.client.exprextent2meter(self.did, self.wid, self.eid, expr)
