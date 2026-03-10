"""
FreeCAD Shape – wraps a Part.Shape and implements the Shape ABC.
"""

import os
import tempfile
from typing import List, Optional, Union

from ...shape import Shape


class FreeCADShape(Shape):
    """
    Concrete Shape backed by a FreeCAD ``Part.Shape`` object stored in
    ``self.obj``.
    """

    def __init__(self, obj, app) -> None:  # type: ignore[override]
        super().__init__(obj, app)

    # ------------------------------------------------------------------
    # Abstract implementations
    # ------------------------------------------------------------------

    def volume(self) -> float:
        """Return the volume of the shape (model units³)."""
        return float(self.obj.Volume)

    def to_stl(self, file_name: str) -> None:
        """Export shape to an ASCII or binary STL file."""
        self.obj.exportStl(file_name)

    def to_step(self, file_name: str) -> None:
        """Export shape to a STEP file."""
        import Part

        Part.export([self.obj], file_name)

    def to_png(
        self,
        file_name: Optional[str] = None,
        view: str = "iso",
        width: int = 800,
        height: int = 600,
        backend: str = "auto",
    ) -> None:
        """
        Render the shape to a PNG via a temporary STL export.

        Supported rendering back-ends (tried in order when backend='auto'):
        pyvista, vedo.
        """
        with tempfile.NamedTemporaryFile(suffix=".stl", delete=False) as tmp:
            tmp_stl = tmp.name

        try:
            self.to_stl(tmp_stl)
            _render_stl(tmp_stl, file_name, view, width, height, backend)
        finally:
            if os.path.exists(tmp_stl):
                os.remove(tmp_stl)

    def cut(self, other: "FreeCADShape") -> "FreeCADShape":
        """Boolean subtraction – modifies this shape in-place."""
        self.obj = self.obj.cut(other.obj)
        return self

    def union(
        self, other: Union["FreeCADShape", List["FreeCADShape"]]
    ) -> "FreeCADShape":
        """Boolean union – modifies this shape in-place."""
        others = [other] if not isinstance(other, list) else other
        for s in others:
            self.obj = self.obj.fuse(s.obj)
        return self

    # ------------------------------------------------------------------
    # Extra operations (not abstract but useful)
    # ------------------------------------------------------------------

    def translate(
        self, x: float = 0.0, y: float = 0.0, z: float = 0.0
    ) -> "FreeCADShape":
        """Translate the shape in-place by (x, y, z)."""
        import FreeCAD

        vec = FreeCAD.Vector(x, y, z)
        self.obj.translate(vec)
        return self


# ---------------------------------------------------------------------------
# Internal rendering helpers (shared with sketch2d)
# ---------------------------------------------------------------------------


def _render_stl(
    stl_file: str,
    output: Optional[str],
    view: str,
    width: int,
    height: int,
    backend: str = "auto",
) -> None:
    """Render *stl_file* to *output* (or display interactively if None)."""
    backends = ["pyvista", "vedo"] if backend == "auto" else [backend]
    last_error: Optional[Exception] = None

    for name in backends:
        try:
            if name == "pyvista":
                _render_pyvista(stl_file, output, view, width, height)
                return
            elif name == "vedo":
                _render_vedo(stl_file, output, view, width, height)
                return
        except ImportError as exc:
            last_error = exc
            continue

    raise ImportError(
        "No rendering backend available. Install pyvista or vedo. "
        f"Last error: {last_error}"
    )


def _render_pyvista(
    stl_file: str, output: Optional[str], view: str, width: int, height: int
) -> None:
    import pyvista as pv

    if output:
        pv.OFF_SCREEN = True

    mesh = pv.read(stl_file)
    plotter = pv.Plotter(off_screen=(output is not None), window_size=[width, height])
    plotter.add_mesh(mesh, color="lightgray", show_edges=False)
    plotter.background_color = "white"

    view = view.lower()
    if view in ("iso", "isometric"):
        plotter.camera_position = "iso"
    elif view in ("front", "y"):
        plotter.camera.position = (0, -10, 0)
        plotter.camera.focal_point = (0, 0, 0)
        plotter.camera.up = (0, 0, 1)
    elif view in ("top", "z"):
        plotter.camera.position = (0, 0, 10)
        plotter.camera.focal_point = (0, 0, 0)
        plotter.camera.up = (0, 1, 0)
    elif view in ("right", "x"):
        plotter.camera.position = (10, 0, 0)
        plotter.camera.focal_point = (0, 0, 0)
        plotter.camera.up = (0, 0, 1)

    if output:
        plotter.show(screenshot=output, auto_close=True)
    else:
        plotter.show()


def _render_vedo(
    stl_file: str, output: Optional[str], view: str, width: int, height: int
) -> None:
    import vedo

    mesh = vedo.load(stl_file)
    view = view.lower()

    cam_kwargs: dict = {}
    if view in ("iso", "isometric"):
        cam_kwargs = {"azimuth": 45, "elevation": 30}
    elif view in ("front", "y"):
        cam_kwargs = {"azimuth": 0, "elevation": 0}
    elif view in ("top", "z"):
        cam_kwargs = {"azimuth": 0, "elevation": 90}
    elif view in ("right", "x"):
        cam_kwargs = {"azimuth": 90, "elevation": 0}

    if output:
        plotter = vedo.Plotter(offscreen=True, size=(width, height))
        plotter.show(mesh, viewup="z")
        if cam_kwargs:
            plotter.camera.Azimuth(cam_kwargs.get("azimuth", 0))
            plotter.camera.Elevation(cam_kwargs.get("elevation", 0))
        plotter.screenshot(output)
        plotter.close()
    else:
        plotter = vedo.Plotter(size=(width, height))
        plotter.show(mesh, viewup="z", camera=cam_kwargs if cam_kwargs else None)
