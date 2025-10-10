import os
import pathlib

import pytest

from rapidcadpy.cad import Cad
from rapidcadpy.json_importer.process_f360 import Fusion360GalleryParser

data_folder = pathlib.Path(__file__).parent.parent.parent / "data" / "f360_json"
K = 5


@pytest.mark.parametrize(
    "file", ["21231_eb9826e5_0000.json", "20241_6bced5ac_0000.json"]
)
def test_cad_sequence_processing(file):
    json_file_path = os.path.join(data_folder, file)

    # Parse and process the CAD file
    try:
        cad: Cad = Fusion360GalleryParser.parse(json_file_path)
        cad_occ = cad.to_occ()

        # from OCC.Display.SimpleGui import init_display
        # display, start_display, add_menu, add_function_to_menu = init_display()
        # display.DisplayShape(cad_occ, color="BLUE", update=False)
        # start_display()

        stl_path = str(json_file_path).replace(".json", ".stl")

        write_stl_file(
            cad_occ,
            filename=stl_path,
        )
    except Exception as e:
        pytest.fail(f"Failed to parse {file}: {e}")
