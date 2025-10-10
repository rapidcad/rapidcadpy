import os
import pathlib
import random
import traceback

import matplotlib.pyplot as plt
import numpy as np
from models.brep_embeddings.brep_to_graph import SolidModelToGraphConverter
from server.execute_code import execute_cad_code
from settings.pylogging import logger
from tqdm import tqdm

from rapidcadpy.cadseq import Cad
from rapidcadpy.json_importer.process_f360 import Fusion360GalleryParser

data_folder = "/Users/eliasberger/Documents/PhD/brep2cad/data/cad_constraint_json"
# get random file from data folder
cad_parser = Fusion360GalleryParser()
files = os.listdir(data_folder)
k = 30
problem_set = random.sample(os.listdir(data_folder), k=k)

demo_reconstruction = False
bin_dir = pathlib.Path(__file__).parent.parent.parent.joinpath("data", "f360_bin")
vec_dir = pathlib.Path(__file__).parent.parent.parent.joinpath("data", "f360_vec")
brep_dir = pathlib.Path(__file__).parent.parent.parent.joinpath("data", "f360_step")
render_results_base_dir = pathlib.Path(__file__).parent.parent.parent.joinpath("render")
fig, axes = plt.subplots(k, k, figsize=(15, 15))
counter = 0
for file in tqdm(problem_set, desc="Processing CAD files", total=k):
    json_file_path = os.path.join(data_folder, file)
    file = file.split(".")[0]
    skex_seq: Cad = Fusion360GalleryParser.parse(json_file_path)
    skex_seq.apply_data_cleaning(visualize_steps=False)
    skex_seq.normalize()

    # create step file
    if not os.path.exists(f"{file}_reconstructed.step"):
        continue
        # from build123d import export_step
        # assembly = Fusion360GalleryParser.assemble_cad_sequence(skex_seq)
        # export_step(assembly, file_path=brep_dir.joinpath(f"{file}_reconstructed.step"))

    # create bin graph
    if not os.path.exists(f"{file}_reconstructed.bin"):
        solid_model_converter = SolidModelToGraphConverter(
            input_folder=brep_dir,
            output_folder=bin_dir,
        )
        solid_model_converter.process_one_file(
            (brep_dir.joinpath(f"{file}_reconstructed.step"), solid_model_converter)
        )

    # create vector representation
    if not os.path.exists(f"{file}_vec.csv"):
        vec: np.ndarray = skex_seq.to_vector()
        np.savetxt(
            os.path.join(vec_dir, f"{file}_vec.csv"), vec, delimiter=",", fmt="%.6g"
        )

    # plot
    if demo_reconstruction:
        source_code = skex_seq.to_python()
        reconstructed_sequence = execute_cad_code(source_code)
        # reconstructed_sequence.plot("Reconstructed")
        # reconstructed_sequence.render_3d(title="Reconstructed")
        reconstructed_compound = reconstructed_sequence.assemble_3d(as_compound=True)

    counter += 1
    try:
        ...
    except Exception as e:
        logger.error(f"Failed to process {file}: {e}")
        logger.error(traceback.format_exc())

logger.info(f"Processed {counter} files.")
