from rapidcadpy.shape import Shape


class OccShape(Shape):
    def __init__(self, obj) -> None:
        super().__init__(obj)

    def to_stl(self, file_name: str):
        # The constructor used here automatically calls mesh.Perform(). https://dev.opencascade.org/doc/refman/html/class_b_rep_mesh___incremental_mesh.html#a3a383b3afe164161a3aa59a492180ac6
        from OCP.BRepMesh import BRepMesh_IncrementalMesh
        from OCP.StlAPI import StlAPI_Writer

        tolerance = 1e-3
        angular_tolerance = 0.1
        ascii = False
        relative = True
        parallel = True
        BRepMesh_IncrementalMesh(
            self.obj, tolerance, relative, angular_tolerance, parallel
        )
        writer = StlAPI_Writer()
        writer.ASCIIMode = ascii

        return writer.Write(self.obj, file_name)
