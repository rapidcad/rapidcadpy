from typing import TYPE_CHECKING, Optional, Union, List

from rapidcadpy.app import App

if TYPE_CHECKING:
    from rapidcadpy.integrations.occ.workplane import OccWorkplane


from rapidcadpy.fea.boundary_conditions import BoundaryCondition, Load
from rapidcadpy.fea.materials import MaterialProperties
from rapidcadpy.integrations.ocp.workplane import OccWorkplane


class OpenCascadeApp(App):
    def __init__(self):
        from rapidcadpy.integrations.occ.workplane import OccWorkplane

        super().__init__(OccWorkplane)

    @property
    def sketch_class(self):
        from rapidcadpy.integrations.occ.sketch import OccSketch2D

        return OccSketch2D

    @property
    def sketch_3d(self):
        """Entry point for building 3D path sketches (wires)."""
        from rapidcadpy.integrations.occ.sketch3d import OccSketch3D

        return OccSketch3D(self)
