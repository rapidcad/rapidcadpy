from rapidcadpy.onshape_processing.custom_onshape_client import CustomOnshapeClient
from rapidcadpy.onshape_processing.feature_list_parser import FeatureListParser

url = "https://cad.onshape.com/documents/12d7e85cddb60c8cdebe7c33/w/4170277dd65d229c60b56fee/e/41f95128c810bf7238d32d43"
client = CustomOnshapeClient(
    creds="/Users/eliasberger/Documents/PhD/brep2cad/rapidcadpy/creds.json"
)
v_list = url.split("/")
document_id, workplace_id, element_id = v_list[-5], v_list[-3], v_list[-1]
features = client.get_features(did=document_id, wid=workplace_id, eid=element_id)

# check if any other operations than sketch and extrude are used
ofs_data = client.get_features(document_id, workplace_id, element_id).json()
for item in ofs_data["features"]:
    if item["message"]["featureType"] not in ["newSketch", "extrude"]:
        exit(1)


parser = FeatureListParser(
    client, features.json(), document_id, workplace_id, element_id
)
parser.parse()
