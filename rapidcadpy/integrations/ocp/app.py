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

    def fea(
        self,
        material: Union["MaterialProperties", str, None] = None,
        loads: Optional[List["Load"]] = None,
        constraints: Optional[List["BoundaryCondition"]] = None,
        mesh_size: float = 2.0,
        element_type: str = "tet4",
        verbose: bool = False,
    ):
        return self._shapes[0].analyze(
            material=material,
            loads=loads,
            constraints=constraints,
            mesh_size=mesh_size,
            element_type=element_type,
            verbose=verbose,
        )
