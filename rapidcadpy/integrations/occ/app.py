from typing import Optional

from ...app import App


class OpenCascadeApp(App):
    def __init__(self):
        from .workplane import OccWorkplane

        super().__init__(OccWorkplane)

    @property
    def sketch_class(self):
        from .sketch import OccSketch2D

        return OccSketch2D

    @property
    def sketch_3d(self):
        """Entry point for building 3D path sketches (wires)."""
        from .sketch3d import OccSketch3D

        return OccSketch3D(self)
