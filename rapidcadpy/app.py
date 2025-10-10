from typing import TYPE_CHECKING, Type

if TYPE_CHECKING:
    pass

from rapidcadpy.workplane import Workplane


class App:
    def __init__(self, work_plane_class: Type[Workplane] = Workplane):
        self.work_plane_class = work_plane_class

    def work_plane(self, name: str) -> Workplane:
        if name.upper() == "XY":
            return self.work_plane_class.xy_plane(app=self)
        elif name.upper() == "XZ":
            return self.work_plane_class.xz_plane(app=self)
        elif name.upper() == "YZ":
            return self.work_plane_class.yz_plane(app=self)
        else:
            raise ValueError(f"Unknown workplane: {name}")
