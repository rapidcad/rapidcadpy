from typing import TYPE_CHECKING
from rapidcadpy.app import App
if TYPE_CHECKING:
    from rapidcadpy.integrations.occ.workplane import OccWorkplane
from rapidcadpy.integrations.occ.workplane import OccWorkplane


class OpenCascadeApp(App):
    def __init__(self):
        from rapidcadpy.integrations.occ.workplane import OccWorkplane

        super().__init__(OccWorkplane)

    def work_plane(self, name: str = "XY") -> "OccWorkplane":
        return super().work_plane(name)  # type: ignore