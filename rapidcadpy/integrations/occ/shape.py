from rapidcadpy.shape import Shape


class OccShape(Shape):
    def __init__(self, obj, app) -> None:
        self.app = app
        super().__init__(obj, app)

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

    def to_step(self, file_name: str) -> None:
        raise NotImplementedError("STEP export not implemented yet.")

    def cut(self, other: "OccShape") -> "OccShape":
        """
        Perform a boolean cut operation (subtraction) on this shape.

        This operation modifies the current shape in-place by subtracting
        the other shape from it.

        Args:
            other: The shape to subtract from this shape

        Returns:
            OccShape: Self (modified in-place) for method chaining
        """
        from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut

        cut_result = BRepAlgoAPI_Cut(self.obj, other.obj)
        cut_result.Build()
        if not cut_result.IsDone():
            raise RuntimeError("Cut operation failed.")

        # Update the current object with the cut result (in-place modification)
        self.obj = cut_result.Shape()
        return self

    def union(self, other: Shape) -> Shape:
        """
        Perform a boolean union operation (addition) on this shape.

        This operation modifies the current shape in-place by unioning
        it with the other shape.

        Args:
            other: The shape to union with this shape

        Returns:
            OccShape: Self (modified in-place) for method chaining
        """
        from OCP.BRepAlgoAPI import BRepAlgoAPI_Fuse

        fuse_result = BRepAlgoAPI_Fuse(self.obj, other.obj)
        fuse_result.Build()
        if not fuse_result.IsDone():
            raise RuntimeError("Union operation failed.")

        # Update the current object with the union result (in-place modification)
        self.obj = fuse_result.Shape()
        return self
