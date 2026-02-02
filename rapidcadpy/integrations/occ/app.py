from typing import TYPE_CHECKING, Optional, Union, List

from .app import App

if TYPE_CHECKING:
    from .integrations.occ.workplane import OccWorkplane


from .fea.boundary_conditions import BoundaryCondition, Load
from .fea.materials import MaterialProperties
from .integrations.ocp.workplane import OccWorkplane


class OpenCascadeApp(App):
    def __init__(self):
        from .integrations.occ.workplane import OccWorkplane

        super().__init__(OccWorkplane)

    @property
    def sketch_class(self):
        from .integrations.occ.sketch import OccSketch2D

        return OccSketch2D

    @property
    def sketch_3d(self):
        """Entry point for building 3D path sketches (wires)."""
        from .integrations.occ.sketch3d import OccSketch3D

        return OccSketch3D(self)
