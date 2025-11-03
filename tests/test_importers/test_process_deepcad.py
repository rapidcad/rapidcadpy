import os
import pathlib
import random

import pytest

from rapidcadpy.json_importer.process_deepcad import DeepCadJsonParser

DATA_FOLDER = pathlib.Path(__file__).parent / "test_files"


random_files = []
files = os.listdir(DATA_FOLDER)
selected_files = random.sample(files, 1)
random_files.extend(selected_files)


@pytest.mark.parametrize("file", random_files)
def test_json_conversion_1(file):
    """
    Test if export then import of the JSON leads to the same CAD sequence
    :return:
    """
    json_file_path = os.path.join(DATA_FOLDER, file)
    cad = DeepCadJsonParser.process(json_file_path)
    print(cad.to_python())
