# import json
# import pathlib


# from rapidcadpy.json_importer.process_f360 import Fusion360GalleryParser
# from rapidcadpy.sketch import Sketch
# from scripts.visualization.visualize_constraint_graph import (
#     visualize_dgl_graph,
#     visualize_ptg_graph,
# )


# class TestSketchBBox:
#     def test_bbox(self):
#         pass
#         # # Create mock edges
#         # edge1 = Line((0, 0), (1, 1))
#         # edge2 = Line((1, 0), (2, 2))
#         # edge3 = Line((-1, -1), (0, 0))
#         #
#         # # Create outer_wire with edges
#         # outer_wire = Wire(edges=[edge1, edge2, edge3])
#         #
#         # # Create Sketch
#         # sketch = Sketch(outer_wire=outer_wire)
#         #
#         # # Expected bbox: min point (-1, -1), max point (2, 2)
#         # expected_bbox = np.array([[-1, -1], [2, 2]])
#         #
#         # # Assert bbox computation
#         # np.testing.assert_array_equal(sketch.bbox, expected_bbox)

#     def test_create_constraint_graph(self):
#         base_dir = (
#             pathlib.Path(__file__).parent.parent.parent
#             / "data"
#             / "f360_json"
#             / "20241_6bced5ac_0000.json"
#         )
#         cad = Fusion360GalleryParser.process_json_data(
#             json_data=json.load(open(base_dir))
#         )
#         for skex in cad.construction_history:
#             for sketch in skex.sketch:
#                 dgl_hg = sketch.create_constraint_graph()
#                 visualize_dgl_graph(dgl_hg)
#         pass

#     def test_create_constraint_graph_pgt(self):
#         base_dir = (
#             pathlib.Path(__file__).parent.parent.parent
#             / "data"
#             / "f360_json"
#             / "json"
#             / "21231_eb9826e5_0001.json"
#         )
#         cad = Fusion360GalleryParser.process_json_data(
#             json_data=json.load(open(base_dir))
#         )
#         cad.apply_data_cleaning()
#         for skex in cad.construction_history:
#             for sketch in skex.sketch:
#                 print(sketch.constraints)
#                 print("###")
#                 sketch.plot(title="Original Sketch")
#                 ptg_hg = sketch.to_graph()
#                 visualize_ptg_graph(ptg_hg)
#                 reconstructed_sketch = Sketch.from_graph(ptg_hg)
#                 reconstructed_sketch.plot(title="Reconstructed Sketch")
#         pass
