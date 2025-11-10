from typing import TYPE_CHECKING

from pycadseq.app import App

if TYPE_CHECKING:
    from pycadseq.integrations.occ.workplane import OccWorkplane

from pycadseq.integrations.occ.workplane import OccWorkplane


class OpenCascadeApp(App):
    def __init__(self):
        from pycadseq.integrations.occ.workplane import OccWorkplane

        super().__init__(OccWorkplane)

    def work_plane(self, name: str = "XY") -> "OccWorkplane":
        return super().work_plane(name)  # type: ignore

    def new_document(self): ...
