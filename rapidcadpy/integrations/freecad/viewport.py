"""FreeCAD implementation of native viewport screenshot capture."""

from __future__ import annotations

import tempfile
from pathlib import Path

from ...cad_objects import CadDocument
from ...viewport import (
    ViewportBackend,
    ViewportBackendFactory,
    ViewportName,
    ViewportScreenshot,
    normalize_viewport_name,
    validate_viewport_size,
)


class FreeCADViewportBackendFactory(ViewportBackendFactory):
    """Construct FreeCAD viewport backends without leaking native handles."""

    def create(self, document: CadDocument) -> ViewportBackend:
        if document.backend.strip().lower() != "freecad":
            raise ValueError("FreeCADViewportBackend requires a FreeCAD document.")
        return FreeCADViewportBackend()


class FreeCADViewportBackend(ViewportBackend):
    """Capture the active FreeCAD GUI viewport as a PNG image."""

    _VIEW_METHODS = {
        "isometric": "viewAxonometric",
        "front": "viewFront",
        "rear": "viewRear",
        "left": "viewLeft",
        "right": "viewRight",
        "top": "viewTop",
        "bottom": "viewBottom",
    }

    def capture_screenshot(
        self,
        *,
        document: CadDocument,
        view: ViewportName,
        width: int,
        height: int,
        fit: bool,
    ) -> ViewportScreenshot:
        normalized_view = normalize_viewport_name(view)
        normalized_width, normalized_height = validate_viewport_size(
            width,
            height,
        )
        if document.backend.strip().lower() != "freecad":
            raise ValueError("FreeCADViewportBackend requires a FreeCAD document.")

        import FreeCAD as App
        import FreeCADGui as Gui

        native_document = document.native_handle
        native_name = str(getattr(native_document, "Name", "")).strip()
        if not native_name:
            raise RuntimeError("The FreeCAD document has no native name.")
        if App.ActiveDocument is not native_document:
            App.setActiveDocument(native_name)
        gui_document = Gui.activeDocument()
        if gui_document is None:
            raise RuntimeError("FreeCAD has no active GUI document.")

        native_document.recompute()
        Gui.updateGui()
        active_view = gui_document.activeView()
        view_method = self._VIEW_METHODS.get(normalized_view)
        if view_method is not None:
            getattr(active_view, view_method)()
        if fit:
            active_view.fitAll()
        Gui.updateGui()

        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as handle:
                temporary_path = Path(handle.name)
            active_view.saveImage(
                str(temporary_path),
                normalized_width,
                normalized_height,
                "Current",
            )
            Gui.updateGui()
            content = temporary_path.read_bytes()
            if not content:
                raise RuntimeError("FreeCAD produced an empty viewport screenshot.")
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

        return ViewportScreenshot(
            content=content,
            width=normalized_width,
            height=normalized_height,
            view=normalized_view,
        )


__all__ = ["FreeCADViewportBackend", "FreeCADViewportBackendFactory"]
