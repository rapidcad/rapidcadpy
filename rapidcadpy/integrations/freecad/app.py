"""
FreeCAD App – top-level document manager and App implementation.
"""

import os
import sys
from typing import Any, Optional, Tuple, Union

from ...app import App
from ...cad_objects import CadDocument
from .cad_adapter import FreeCADAdapter

VectorLike = Union[Tuple[float, float, float], Tuple[float, float]]


def ensure_freecad_python_path() -> str | None:
    """Add FreeCAD module directory to ``sys.path`` when discoverable."""
    candidates = [
        os.environ.get("FREECAD_LIB_PATH", "").strip(),
        "/Applications/FreeCAD.app/Contents/Resources/lib",
        "/usr/lib/freecad/lib",
        "/usr/lib64/freecad/lib",
    ]

    seen: set[str] = set()
    for candidate in candidates:
        if not candidate:
            continue

        normalized = os.path.abspath(candidate)
        if normalized in seen or not os.path.exists(normalized):
            continue
        seen.add(normalized)

        os.environ.setdefault("FREECAD_LIB_PATH", normalized)
        if normalized not in sys.path:
            sys.path.insert(0, normalized)
        return normalized

    return None


class FreeCADApp(App):
    """
    FreeCAD implementation of the App base class.

    Creates and owns a headless FreeCAD document. Document-backed shapes retain
    backend-neutral references to their native FreeCAD features, while
    operations unsupported by native history can still use standalone
    ``Part.Shape`` geometry.
    """

    def __init__(
        self,
        doc_name: str = "RapidCADPy_Doc",
        silent_geometry_failures: bool = False,
        *,
        instance_id: Optional[str] = None,
    ):
        super().__init__(silent_geometry_failures=silent_geometry_failures)
        self._gui_connection = None
        if doc_name == "attach":
            from .gui_connection import FreeCADGuiConnection

            self._gui_connection = FreeCADGuiConnection.attach(instance_id=instance_id)
            self._fc_doc = None
            self._cad_adapter = FreeCADAdapter()
            self._cad_document = None
            self._feature_counter = 0
            return

        ensure_freecad_python_path()
        import FreeCAD

        self._fc_doc = FreeCAD.newDocument(doc_name)
        self._cad_adapter = FreeCADAdapter()
        self._cad_document = self._wrap_document(self._fc_doc)
        self._feature_counter = 0

    @classmethod
    def from_document(
        cls,
        document: Any,
        silent_geometry_failures: bool = False,
    ) -> "FreeCADApp":
        """Bind to an existing native document without creating a scratch one."""
        instance = cls.__new__(cls)
        App.__init__(
            instance,
            silent_geometry_failures=silent_geometry_failures,
        )
        instance._gui_connection = None
        instance._fc_doc = document
        instance._cad_adapter = FreeCADAdapter()
        instance._cad_document = instance._wrap_document(document)
        instance._feature_counter = 0
        return instance

    @staticmethod
    def list_instances():
        """List running FreeCAD GUIs with an active RapidCADPy connector."""
        from .gui_connection import FreeCADGuiConnection

        return FreeCADGuiConnection.list_instances()

    @property
    def is_remote(self) -> bool:
        """Return whether this app is attached through the GUI bridge."""
        return self._gui_connection is not None

    @property
    def connection(self):
        """Return the remote GUI connection for ``FreeCADApp('attach')``."""
        return self._gui_connection

    def call(self, method: str, **params):
        """Call a RapidCADPy session method in an attached FreeCAD GUI."""
        if self._gui_connection is None:
            raise RuntimeError("This FreeCADApp is not attached to a GUI bridge.")
        return self._gui_connection.call(method, params)

    def get_doc(self):
        """Get the underlying FreeCAD document."""
        if self._fc_doc is None:
            raise RuntimeError(
                "A remotely attached FreeCAD document has no in-process native "
                "handle. Use call() or CadSession instead."
            )
        return self._fc_doc

    @property
    def cad_document(self) -> CadDocument:
        """Get the backend-neutral reference to the live FreeCAD document."""
        if self._cad_document is None:
            raise RuntimeError(
                "Remote FreeCADApp documents are represented in the bridge process."
            )
        return self._cad_document

    def get_cad_document(self) -> CadDocument:
        """Compatibility method for code that cannot use properties."""
        return self.cad_document

    @property
    def feature_executor(self):
        """Return the history-preserving executor for semantic features."""

        from .feature_executor import FreeCADFeatureExecutor

        return FreeCADFeatureExecutor()

    def bind_document(self, document) -> CadDocument:
        """Replace the active native document and refresh its public binding."""
        self._fc_doc = document
        self._cad_document = self._wrap_document(document)
        self._feature_counter = 0
        self._shapes.clear()
        self._workplanes.clear()
        return self._cad_document

    def _wrap_document(self, document) -> CadDocument:
        return CadDocument(
            backend=self._cad_adapter.backend_name,
            native_handle=document,
            adapter=self._cad_adapter,
            name=str(getattr(document, "Name", "")),
            label=str(getattr(document, "Label", "")),
            file_name=str(getattr(document, "FileName", "")),
        )

    def get_next_feature_index(self) -> int:
        """Get next feature index for naming."""
        self._feature_counter += 1
        return self._feature_counter

    # ------------------------------------------------------------------
    # Abstract property implementations
    # ------------------------------------------------------------------

    @property
    def sketch_3d(self):
        """Entry point for building 3D path sketches (wires)."""
        from .sketch3d import FreeCADSketch3D

        return FreeCADSketch3D(self)

    @property
    def workplane_class(self):
        from .workplane import FreeCADWorkplane

        return FreeCADWorkplane

    @property
    def sketch_class(self):
        from .sketch2d import FreeCADSketch2D

        return FreeCADSketch2D

    # ------------------------------------------------------------------
    # Document helpers
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Close the current document and open a fresh one."""
        import FreeCAD

        FreeCAD.closeDocument(self._fc_doc.Name)
        self.bind_document(FreeCAD.newDocument("RapidCADPy_Doc"))

    def _make_feature_name(self, prefix: str, index: int) -> str:
        return f"{prefix}{index}"

    # ------------------------------------------------------------------
    # Bulk export helpers
    # ------------------------------------------------------------------

    def _upsert_doc_shapes(self, shapes, feature_name_prefix: str = "Shape"):
        doc = self._fc_doc
        existing = {obj.Name: obj for obj in doc.Objects}
        features = []

        for i, shape in enumerate(shapes):
            if not hasattr(shape, "obj"):
                continue

            feature_name = (
                feature_name_prefix if len(shapes) == 1 else f"{feature_name_prefix}{i}"
            )
            feature = existing.get(feature_name)
            if feature is None:
                feature = doc.addObject("Part::Feature", feature_name)
            feature.Label = feature_name
            feature.Shape = shape.obj
            features.append(feature)

        if not features:
            raise ValueError("No valid shapes to export")

        doc.recompute()
        return features

    def to_step(self, file_name: str) -> None:
        """Export all registered shapes to a single STEP file."""
        if not self._shapes:
            raise ValueError("No shapes to export")

        if len(self._shapes) == 1:
            self._shapes[0].to_step(file_name)
            return

        import Part

        features = self._upsert_doc_shapes(self._shapes, feature_name_prefix="Shape")
        Part.export(features, file_name)

    def to_stl(self, file_name: str) -> None:
        """Export all registered shapes to a single STL file."""
        if not self._shapes:
            raise ValueError("No shapes to export")

        if len(self._shapes) == 1:
            self._shapes[0].to_stl(file_name)
        else:
            # Fuse all shapes into one compound and export
            combined = self._shapes[0].obj
            for s in self._shapes[1:]:
                if hasattr(s, "obj"):
                    combined = combined.fuse(s.obj)
            combined.exportStl(file_name)

    def to_fcstd(
        self,
        file_name: str,
        shapes=None,
        feature_name_prefix: str = "Shape",
        allow_direct_geometry: bool = False,
    ) -> None:
        """Save the document in FreeCAD's native .FCStd format.

        When all target shapes belong to the same native document, save that
        document and preserve its complete feature tree. Baked ``Part::Feature``
        export is available only through explicit direct-geometry mode.

        Args:
            file_name: Destination path (should end in ``.FCStd``).
            shapes: Optional explicit shapes to write instead of the full app registry.
            feature_name_prefix: Base object name to use in the document tree.
        """
        target_shapes = self._shapes if shapes is None else shapes

        native_documents = [getattr(shape, "document", None) for shape in target_shapes]
        if target_shapes and all(document is not None for document in native_documents):
            first_document = native_documents[0]
            if all(
                document.native_handle is first_document.native_handle
                for document in native_documents[1:]
            ):
                first_document.save(file_name)
                return

        if not allow_direct_geometry:
            from .errors import FreeCADNativeFeatureError

            raise FreeCADNativeFeatureError(
                "FCStd export cannot preserve one native feature tree because the "
                "requested shapes are unbacked or belong to different documents. "
                "Refusing to bake them as Part::Feature objects. Pass "
                "allow_direct_geometry=True only when history loss is intentional."
            )

        if not target_shapes:
            raise ValueError("No shapes to export")

        self._upsert_doc_shapes(target_shapes, feature_name_prefix=feature_name_prefix)
        self._fc_doc.saveAs(file_name)
