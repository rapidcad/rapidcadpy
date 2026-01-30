import sys
import os
import glob
from statistics import mean

# Try to import OCP modules. If this script is run in an environment where OCP is not installed
# directly but cadquery is, we might fail here. However, based on user context, we proceed with OCP.
try:
    from OCP.STEPControl import STEPControl_Reader
    from OCP.TopExp import TopExp
    from OCP.TopAbs import TopAbs_SOLID, TopAbs_FACE, TopAbs_EDGE, TopAbs_VERTEX
    from OCP.IFSelect import IFSelect_RetDone
    from OCP.TopTools import TopTools_IndexedMapOfShape
except ImportError:
    print("Error: OCP module not found. Please ensure OCP/CadQuery is installed.")
    sys.exit(1)


def get_element_counts(step_file_path):
    """
    Reads a STEP file and counts its topological elements (Solids, Faces, Edges, Vertices).
    Returns a dict with counts, or None if reading fails.
    """
    if not os.path.exists(step_file_path):
        print(f"Error: File '{step_file_path}' not found.")
        return None

    # Initialize the STEP Reader
    reader = STEPControl_Reader()
    status = reader.ReadFile(step_file_path)

    if status != IFSelect_RetDone:
        # Don't print for every error in bulk mode unless verbose
        # print(f"Error: Unable to read file '{step_file_path}'. Status: {status}")
        return None

    # Transfer the contents to a TopoDS_Shape
    reader.TransferRoots()
    shape = reader.OneShape()

    if shape.IsNull():
        return None

    def count_unique_type(shape, shape_type):
        map_of_shapes = TopTools_IndexedMapOfShape()
        TopExp.MapShapes_s(shape, shape_type, map_of_shapes)
        return map_of_shapes.Extent()

    return {
        "solids": count_unique_type(shape, TopAbs_SOLID),
        "faces": count_unique_type(shape, TopAbs_FACE),
        "edges": count_unique_type(shape, TopAbs_EDGE),
        "vertices": count_unique_type(shape, TopAbs_VERTEX),
    }


def count_single_file(step_file_path):
    counts = get_element_counts(step_file_path)
    if counts:
        print(f"File: {step_file_path}")
        print(f"Solids:   {counts['solids']}")
        print(f"Faces:    {counts['faces']}")
        print(f"Edges:    {counts['edges']}")
        print(f"Vertices: {counts['vertices']}")
    else:
        print(f"Failed to process {step_file_path}")


def process_directory(directory_path):
    if not os.path.isdir(directory_path):
        print(f"Error: Directory '{directory_path}' not found.")
        return

    # Find all .step and .stp files
    step_files = sorted(
        glob.glob(os.path.join(directory_path, "*.step"))
        + glob.glob(os.path.join(directory_path, "*.stp"))
    )

    if not step_files:
        print(f"No STEP files found in '{directory_path}'.")
        return

    total_solids = []
    total_faces = []
    total_edges = []
    total_vertices = []

    print(f"Processing {len(step_files)} files in '{directory_path}'...")

    for i, file_path in enumerate(step_files):
        # Print progress every 10 files
        if i % 10 == 0:
            sys.stdout.write(f"\rProcessing {i+1}/{len(step_files)}...")
            sys.stdout.flush()

        counts = get_element_counts(file_path)
        if counts:
            total_solids.append(counts["solids"])
            total_faces.append(counts["faces"])
            total_edges.append(counts["edges"])
            total_vertices.append(counts["vertices"])

    print(f"\rProcessed {len(total_solids)} files successfully.      ")
    print("=" * 40)

    if total_solids:
        print(f"Average Solids:   {mean(total_solids):.2f}")
        print(f"Average Faces:    {mean(total_faces):.2f}")
        print(f"Average Edges:    {mean(total_edges):.2f}")
        print(f"Average Vertices: {mean(total_vertices):.2f}")
        print("-" * 40)
        print(f"Total Solids:     {sum(total_solids)}")
        print(f"Total Faces:      {sum(total_faces)}")
        print(f"Total Edges:      {sum(total_edges)}")
        print(f"Total Vertices:   {sum(total_vertices)}")
        print("-" * 40)
        print(f"Min/Max Faces:    {min(total_faces)} / {max(total_faces)}")
        print(f"Min/Max Edges:    {min(total_edges)} / {max(total_edges)}")
    else:
        print("No valid STEP files were processed.")


if __name__ == "__main__":
    path_arg = None

    if len(sys.argv) > 1:
        path_arg = sys.argv[1]
    else:
        # Default test directory
        test_dir = "/mnt/data/deepcad_step/0000/"
        if os.path.exists(test_dir):
            path_arg = test_dir

    if path_arg:
        if os.path.isdir(path_arg):
            process_directory(path_arg)
        elif os.path.isfile(path_arg):
            count_single_file(path_arg)
        else:
            print(f"Error: Path '{path_arg}' not found.")
    else:
        print("Usage: python count_step_ocp.py <file_or_directory_path>")
