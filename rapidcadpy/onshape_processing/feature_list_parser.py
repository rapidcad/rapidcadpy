from collections import OrderedDict

from .cadseq import Sketch
from .extrude import Extrude
from .onshape_processing.extrude_parser import ExtrudeParser
from .onshape_processing.sketch_parser import SketchParser

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


class FeatureListParser:
    def __init__(self, client, feature_list, did, wid, eid):
        self.feature_list = feature_list
        self.client = client
        self.did = did
        self.wid = wid
        self.eid = eid

    def parse(self):
        result = {"entities": OrderedDict(), "properties": {}, "sequence": []}
        sketch_parser = SketchParser(
            client=self.client, did=self.did, wid=self.wid, eid=self.eid
        )
        extrude_parser = ExtrudeParser(
            client=self.client, did=self.did, wid=self.wid, eid=self.eid
        )
        for i, feat_item in enumerate(self.feature_list["features"]):
            feat_type = feat_item["message"]["featureType"]
            feature_id = feat_item["message"]["featureId"]

            if feat_type == "newSketch":
                sketch: Sketch = sketch_parser.parse(feat_item)
                result["entities"].update({feature_id: sketch})

            elif feat_type == "extrude":
                self.client.bodydetails(self.did, self.wid, self.eid).json()
                extrude_dict = extrude_parser.parse(feat_item)
                extruded_entity = result["entities"][extrude_dict["entities"][0]]
                Extrude(
                    sketch=extruded_entity,
                    operation=extrude_dict["operation"],
                    extent_type=extrude_dict["extent_type"],
                    extent_one=extrude_dict["extent_one"],
                    extent_two=extrude_dict["extent_two"],
                )
                # result["entities"].update({feature_id: extrude})

            result["sequence"].append({"index": i, "entity": feature_id})

        return result
