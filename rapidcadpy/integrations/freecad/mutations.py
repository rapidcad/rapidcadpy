"""FreeCAD transactions, including document metadata not covered by native undo."""

from __future__ import annotations

from typing import Any


class FreeCADMutationBackend:
    def __init__(self, document: Any) -> None:
        self.document = document
        self._undo_mode = None
        self._meta = None

    def begin(self, name: str) -> None:
        if self.document.HasPendingTransaction:
            raise RuntimeError(
                "Finish the active native transaction before editing through RapidCADPy."
            )
        self._undo_mode = self.document.UndoMode
        self._meta = dict(self.document.Meta)
        if self._undo_mode == 0:
            self.document.UndoMode = 1
        try:
            self.document.openTransaction(f"RapidCADPy {name}")
        except Exception:
            self.document.UndoMode = self._undo_mode
            raise

    def recompute(self) -> None:
        self.document.recompute()

    def validate(self) -> None:
        for obj in self.document.Objects:
            if any(str(status).lower() in {"invalid", "error"} for status in obj.State):
                raise ValueError(f"Native feature {obj.Name} failed: {obj.State}.")
            shape = getattr(obj, "Shape", None)
            if shape is not None and not shape.isNull() and not shape.isValid():
                raise ValueError(f"Native feature {obj.Name} has invalid geometry.")
            kind = getattr(obj, "RapidCADKind", "")
            if kind in {"profile", "path", "loft", "sweep"} and (
                shape is None or shape.isNull()
            ):
                raise ValueError(f"Native {kind} {obj.Name} has no geometry.")
            if kind in {"loft", "sweep"}:
                if obj.Solid and len(shape.Solids) != 1:
                    raise ValueError(f"Solid {kind} must produce exactly one solid.")
                if not obj.Solid and (shape.Solids or not shape.Shells):
                    raise ValueError(
                        f"Shell {kind} must produce a shell without solids."
                    )
                if obj.Solid and shape.Volume <= 0:
                    raise ValueError(f"Solid {kind} must have positive volume.")
            if kind == "loft":
                from .spline_geometry import sketch_wire, validate_loft_sections

                for section in obj.Sections:
                    wire = sketch_wire(section)
                    if obj.Solid and not wire.isClosed():
                        raise ValueError("Solid loft requires closed sections.")
                validate_loft_sections(obj.Sections)
            elif kind == "sweep":
                from .spline_geometry import validate_sweep_inputs

                if len(obj.Sections) != 1 or obj.Spine[0] is None:
                    raise ValueError(
                        "A sweep requires one linked section and a linked spine."
                    )
                validate_sweep_inputs(obj.Sections[0], obj.Spine[0])

    def commit(self) -> None:
        self.document.commitTransaction()
        if self._undo_mode == 0:
            self.document.UndoMode = 0

    def abort(self) -> None:
        try:
            self.document.abortTransaction()
            self.document.Meta = self._meta
            self.document.recompute()
        finally:
            if self._undo_mode == 0:
                self.document.UndoMode = 0
