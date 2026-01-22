import tempfile
from typing import TYPE_CHECKING, Optional, List, Union

from rapidcadpy.app import App

if TYPE_CHECKING:
    from rapidcadpy.integrations.ocp.workplane import OccWorkplane

from rapidcadpy.fea.boundary_conditions import BoundaryCondition, Load
from rapidcadpy.fea.materials import MaterialProperties
from rapidcadpy.integrations.ocp.workplane import OccWorkplane


class OpenCascadeOcpApp(App):
    def __init__(self):

        super().__init__(OccWorkplane)

    @property
    def sketch_class(self):
        from rapidcadpy.integrations.ocp.sketch2d import OccSketch2D

        return OccSketch2D

    def chamfer_edge(
        self,
        x: float,
        radius: float,
        angle: float,
        distance: float,
        tol: float = 1e-3,
    ) -> int:
        """Apply a distance-angle chamfer to edges that match a given X coordinate and radius.

        Searches all edges of all registered shapes and applies a chamfer to
        those whose curve center X coordinate and radius match within `tol`.

        Args:
            x: X coordinate to match for circular edges
            radius: Radius to match for circular edges
            angle: Chamfer angle in radians
            distance: Chamfer distance from edge
            tol: Tolerance for matching (default: 1e-3)

        Returns:
            int: Number of chamfers successfully created

        Raises:
            ValueError: If no shapes are registered with the app
        """
        from OCP.BRepFilletAPI import BRepFilletAPI_MakeChamfer
        from OCP.TopExp import TopExp_Explorer
        from OCP.TopAbs import TopAbs_EDGE, TopAbs_FACE
        from OCP.BRepAdaptor import BRepAdaptor_Curve
        from OCP.GeomAbs import GeomAbs_Circle
        from OCP.TopoDS import TopoDS
        from OCP.TopExp import TopExp
        from math import tan

        if not self._shapes:
            raise ValueError("No shapes registered with app for chamfer operation")

        created = 0

        # Process each shape
        for shape in self._shapes:
            if not hasattr(shape, "obj"):
                continue

            shape_obj = shape.obj
            matching_edges = []

            # Find all edges that match the criteria
            edge_explorer = TopExp_Explorer(shape_obj, TopAbs_EDGE)
            while edge_explorer.More():
                edge = TopoDS.Edge_s(edge_explorer.Current())

                # Analyze the edge curve
                curve_adaptor = BRepAdaptor_Curve(edge)
                curve_type = curve_adaptor.GetType()

                # Check if it's a circular edge
                if curve_type == GeomAbs_Circle:
                    circle = curve_adaptor.Circle()
                    center = circle.Location()
                    edge_radius = circle.Radius()

                    # Check if center X and radius match
                    if (
                        abs(center.X() - x) <= tol
                        and abs(edge_radius - radius) <= tol
                    ):
                        matching_edges.append(edge)

                edge_explorer.Next()

            # Apply chamfer to each matching edge
            for edge in matching_edges:
                try:
                    # Find adjacent faces for this edge
                    map_shape = TopExp.MapShapesAndUniqueAncestors_s(
                        shape_obj, TopAbs_EDGE, TopAbs_FACE
                    )
                    
                    # Get the faces adjacent to this edge
                    idx = map_shape.FindIndex(edge)
                    if idx == 0:
                        continue

                    faces_list = map_shape.FindFromIndex(idx)
                    if faces_list.Size() == 0:
                        continue

                    # Get the first adjacent face
                    face = TopoDS.Face_s(faces_list.First())

                    # Create chamfer builder
                    chamfer_builder = BRepFilletAPI_MakeChamfer(shape_obj)

                    # Add chamfer with distance and angle
                    # The angle defines the chamfer slope: tan(angle) = height/distance
                    chamfer_builder.Add(distance, distance * tan(angle), edge, face)

                    # Build the chamfered shape
                    chamfer_builder.Build()

                    if chamfer_builder.IsDone():
                        # Update the shape object
                        shape.obj = chamfer_builder.Shape()
                        created += 1
                    
                except Exception:
                    # Skip this edge if chamfer fails
                    continue

        return created
