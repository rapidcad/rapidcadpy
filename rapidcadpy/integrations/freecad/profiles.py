"""Persistent native sketch profiles and linked loft mutations."""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Callable
from typing import Any
from uuid import uuid4

from ...cad_objects import CadObject
from ...feature_updates import LoftFeatureUpdate, feature_update_from_dict
from ...modeling import (
    CadPath,
    CadProfile,
    LoftDefinition,
    ModelingRequest,
    PathDefinition,
    ProfileDefinition,
    SweepDefinition,
    modeling_request_from_dict,
)
from ...operation_result import CadOperationResult, CadOperationSupport
from ...primitives import Arc, Circle, Line
from .mutations import FreeCADMutationBackend
from .spline_geometry import (
    TRANSITIONS,
    definition_support,
    native_geometry,
    sketch_wire,
    validate_loft_sections,
    validate_sweep_inputs,
)


def document_id(document: Any, *, create: bool = False) -> str | None:
    metadata = dict(getattr(document, "Meta", {}))
    identity = metadata.get("RapidCADDocumentId")
    if not identity and create:
        identity = f"document_{uuid4().hex}"
        document.Meta = {**metadata, "RapidCADDocumentId": identity}
    return identity


def tag_object(native: Any, kind: str, identity: str | None = None) -> str:
    identity = identity or f"{kind}_{uuid4().hex}"
    for name, value in (("RapidCADObjectId", identity), ("RapidCADKind", kind)):
        if name not in native.PropertiesList:
            native.addProperty("App::PropertyString", name, "RapidCAD")
        setattr(native, name, value)
    return identity


def hydrate_profiles(session: Any, document: Any) -> None:
    """Recover stable references entirely from saved native document metadata."""
    identity = document_id(document)
    session.cad_document.id = identity
    session.profile_references.clear()
    session.path_references.clear()
    for native in getattr(document, "Objects", ()):
        kind = getattr(native, "RapidCADKind", "")
        object_id = getattr(native, "RapidCADObjectId", None)
        if not object_id or object_id not in session.objects:
            continue
        semantic = session.objects[object_id]
        if kind in {"profile", "path"}:
            if not identity:
                raise ValueError("Persistent profiles require a saved document identity.")
            reference = (
                CadProfile(identity, object_id)
                if kind == "profile"
                else CadPath(identity, object_id)
            )
            registry = session.profile_references if kind == "profile" else session.path_references
            registry[object_id] = reference
            semantic.type = kind
            session.runtime_objects[object_id].semantic_type = kind
            semantic.metadata[kind] = reference.to_dict()
            constrained = bool(getattr(native, "ConstraintCount", 0))
            update_operation = "update_profile" if kind == "profile" else "update_path"
            semantic.metadata["operation_support"] = {
                update_operation: CadOperationSupport(
                    update_operation,
                    "unsupported" if constrained else "supported",
                    "native_feature",
                    "Constrained sketches require constraint-aware editing."
                    if constrained
                    else "Updates the same native sketch; dependent features recompute.",
                ).to_dict()
            }
        elif kind in {"loft", "sweep"}:
            semantic.metadata["feature_kind"] = kind
            semantic.type = "solid" if native.Solid else "shell"
            session.runtime_objects[object_id].semantic_type = semantic.type
            semantic.metadata["profile_ids"] = [
                session.cad_document.object_id(section.Name) for section in native.Sections
            ]
            if kind == "sweep":
                semantic.metadata["path_id"] = session.cad_document.object_id(native.Spine[0].Name)
                semantic.metadata["orientation"] = "frenet" if native.Frenet else "corrected_frenet"
                semantic.metadata["transition"] = next(
                    key for key, value in TRANSITIONS.items() if value == native.Transition
                )
            else:
                semantic.metadata["alignment"] = "automatic"
            semantic.metadata["operation_support"] = {
                "edit_sections": CadOperationSupport(
                    "edit_sections",
                    "supported",
                    "native_dependent_feature",
                    f"Edit the referenced profiles to recompute this native {kind}.",
                ).to_dict()
            }
            semantic.metadata["operation_support"]["update_feature"] = CadOperationSupport(
                "update_feature",
                "conditional",
                "native_dependent_feature",
                "Supports declared loft/sweep parameters only; resulting geometry and dependencies must be valid.",
            ).to_dict()
            if kind == "sweep":
                semantic.metadata["operation_support"]["edit_path"] = CadOperationSupport(
                    "edit_path",
                    "supported",
                    "native_dependent_feature",
                    "Edit the linked planar sketch to recompute this native sweep.",
                ).to_dict()
    valid_ids = set(session.profile_references) | set(session.path_references)
    session._profile_cache = {
        key: value for key, value in session._profile_cache.items() if value in valid_ids
    }


class FreeCADProfileOperations:
    inspect_support = staticmethod(definition_support)

    def __init__(self, session: Any) -> None:
        self.session = session
        self.document = session.cad_document.native_handle

    def _refresh(self) -> None:
        # Transient coordinate frames and their pending geometry are independent
        # of the persistent native profiles being hydrated.
        session = self.session
        workplanes = {k: obj for k, obj in session.objects.items() if obj.type == "workplane"}
        handles = {k: session.runtime_objects[k] for k in workplanes}
        active_wp, active_wp_id = session.active_workplane, session.active_workplane_id
        active_shape = session.active_shape_id
        history = list(session.operations)
        session._hydrate_freecad_document(self.document, None, source_tool="mutation")
        session.objects.update(workplanes)
        session.runtime_objects.update(handles)
        session.active_workplane, session.active_workplane_id = active_wp, active_wp_id
        session.active_shape_id = active_shape if active_shape in session.objects else None
        session.operations = history

    def _run(
        self,
        operation: str,
        expected_revision: str | None,
        validate: Callable[[], None],
        mutate: Callable[[], dict[str, Any]],
    ) -> dict[str, Any]:
        reason = {
            "update_feature": "Only declared loft/sweep parameters are editable; dependencies and resulting geometry must remain valid.",
            "loft": "Kernel alignment is automatic; closed single-edge sections require consistent winding, parallel frames and corresponding seams.",
            "sweep": "Requires a closed section at the start of an open planar spine, with its normal parallel to the initial tangent.",
            "update_profile": "Replacing geometry requires an unconstrained native sketch and valid dependent features.",
            "update_path": "Replacing geometry requires an unconstrained planar native sketch and valid dependent features.",
        }.get(
            operation,
            "Native sketches and their linked features retain editable history.",
        )
        return self.session.mutation_coordinator.run(
            refresh=self._refresh,
            backend=FreeCADMutationBackend(self.document),
            support=CadOperationSupport(
                operation,
                "conditional"
                if operation in {"update_profile", "update_path", "update_feature", "loft", "sweep"}
                else "supported",
                "native_dependent_feature"
                if operation in {"loft", "sweep", "update_feature"}
                else "native_feature",
                reason,
            ),
            expected_revision=expected_revision,
            validate_inputs=validate,
            mutate=mutate,
        ).to_dict()

    def apply_modeling(self, definition: ModelingRequest, expected_revision: str) -> dict[str, Any]:
        support = definition_support(definition)
        if support.status == "unsupported":
            return CadOperationResult.failure(
                support.reason,
                error_code="cad_operation_not_supported",
                support=support,
                document_revision=self.session.document_revision,
            ).to_dict()
        references = (
            definition.profiles
            if isinstance(definition, LoftDefinition)
            else (definition.profile, definition.path)
            if isinstance(definition, SweepDefinition)
            else ()
        )
        if any(ref.document_id != document_id(self.document) for ref in references):
            return CadOperationResult.failure(
                "Modeling inputs do not belong to the active document.",
                error_code="invalid_cad_request",
                support=support,
                document_revision=self.session.document_revision,
            ).to_dict()
        if isinstance(definition, LoftDefinition):
            return self.loft(
                [ref.object_id for ref in definition.profiles],
                None,
                definition.make_solid,
                definition.ruled,
                expected_revision,
                alignment=definition.alignment,
            )
        if isinstance(definition, SweepDefinition):
            return self.sweep(
                definition.profile.object_id,
                definition.path.object_id,
                definition.make_solid,
                definition.orientation,
                definition.transition,
                expected_revision,
            )
        kind = "profile" if isinstance(definition, ProfileDefinition) else "path"
        geometry, placement = [], None

        def validate() -> None:
            nonlocal geometry, placement
            geometry, placement = native_geometry(definition)

        def mutate() -> dict[str, Any]:
            native = self.document.addObject(
                "Sketcher::SketchObject",
                "RapidCADProfile" if kind == "profile" else "RapidCADPath",
            )
            native.Label = definition.name
            native.Placement = placement
            native.addGeometry(geometry, False)
            identity = tag_object(native, kind)
            self.session.cad_document.bind_object_id(native.Name, identity)
            reference_type = CadProfile if kind == "profile" else CadPath
            return {
                "object_id": identity,
                kind: reference_type(document_id(self.document, create=True), identity).to_dict(),
            }

        return self._run(f"create_{kind}", expected_revision, validate, mutate)

    def update_path(
        self, path_id: str, definition: dict[str, Any], expected_revision: str | None
    ) -> dict[str, Any]:
        geometry, placement = [], None

        def validate() -> None:
            nonlocal geometry, placement
            request = modeling_request_from_dict(definition)
            if not isinstance(request, PathDefinition):
                raise TypeError("update_path requires a PathDefinition.")
            native = self._resolve(path_id, "path")
            if native.TypeId != "Sketcher::SketchObject" or native.ConstraintCount:
                raise ValueError("Path updates require an unconstrained planar native sketch.")
            geometry, placement = native_geometry(request)

        def mutate() -> dict[str, Any]:
            native = self._resolve(path_id, "path")
            for index in reversed(range(native.GeometryCount)):
                native.delGeometry(index)
            native.addGeometry(geometry, False)
            native.Placement = placement
            self.session._profile_cache = {
                key: value for key, value in self.session._profile_cache.items() if value != path_id
            }
            return {
                "object_id": path_id,
                "path": CadPath(document_id(self.document), path_id).to_dict(),
            }

        return self._run("update_path", expected_revision, validate, mutate)

    _single_wire = staticmethod(sketch_wire)

    def sweep(
        self,
        profile_id: str,
        path_id: str,
        make_solid: bool,
        orientation: str,
        transition: str,
        expected_revision: str | None,
    ) -> dict[str, Any]:
        def validate() -> None:
            if type(make_solid) is not bool:
                raise ValueError("make_solid must be a boolean.")
            if orientation not in {"frenet", "corrected_frenet"} or transition not in TRANSITIONS:
                raise ValueError("Unsupported sweep orientation or corner transition.")
            profile, path = self._resolve(profile_id), self._resolve(path_id, "path")
            validate_sweep_inputs(profile, path)

        def mutate() -> dict[str, Any]:
            native = self.document.addObject("Part::Sweep", "RapidCADSweep")
            native.Sections = [self._resolve(profile_id)]
            native.Spine = (self._resolve(path_id, "path"), [])
            native.Solid = make_solid
            native.Frenet = orientation == "frenet"
            native.Transition = TRANSITIONS[transition]
            identity = tag_object(native, "sweep")
            self.session.cad_document.bind_object_id(native.Name, identity)
            self.session.active_shape_id = identity
            return {
                "object_id": identity,
                "profile_ids": [profile_id],
                "path_id": path_id,
                "orientation": orientation,
                "transition": transition,
            }

        return self._run("sweep", expected_revision, validate, mutate)

    def _workplane(self, workplane_id: str) -> Any:
        semantic = self.session.objects.get(workplane_id)
        if semantic is None or semantic.type != "workplane":
            raise ValueError(f"Unknown workplane {workplane_id!r}.")
        workplane = self.session.runtime_objects[workplane_id]
        if getattr(workplane, "app", None) is not self.session.app:
            raise ValueError("Workplane belongs to a different CAD application/document.")
        self._primitives(workplane)
        return workplane

    @staticmethod
    def _primitives(workplane: Any) -> list[Any]:
        primitives = list(getattr(workplane, "_pending_shapes", ()))
        loops = getattr(workplane, "_accumulated_loops", ())
        if len(loops) > 1 or (loops and primitives):
            raise ValueError("Persistent profiles currently require one boundary loop.")
        if not primitives:
            primitives = list(loops[-1]) if loops else []
        if not primitives:
            raise ValueError("Profile has no sketch geometry.")
        if any(not isinstance(p, (Line, Circle, Arc)) for p in primitives):
            raise NotImplementedError(
                "Only existing line, arc and circle primitives are supported at this stage."
            )
        return copy.deepcopy(primitives)

    def _fingerprint(self, workplane: Any) -> str:
        payload = {
            "geometry": [
                {"kind": type(p).__name__, **vars(p)} for p in self._primitives(workplane)
            ],
            "frame": [list(workplane._to_3d(*p)) for p in ((0, 0), (1, 0), (0, 1))],
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

    def _capture(self, workplane_id: str, kind: str = "profile") -> tuple[str, Any]:
        from .sketch2d import FreeCADSketch2D

        workplane = self._workplane(workplane_id)
        fingerprint = self._fingerprint(workplane)
        cache_key = f"{kind}:{workplane_id}:{fingerprint}"
        cached = self.session._profile_cache.get(cache_key)
        if cached:
            native = self._resolve(cached, kind)
            return cached, native
        sketch = FreeCADSketch2D(
            primitives=self._primitives(workplane),
            workplane=workplane,
            app=self.session.app,
        )
        native = sketch._create_editable_sketch(
            self.document, "RapidCADProfile" if kind == "profile" else "RapidCADPath"
        )
        identity = tag_object(native, kind)
        document_id(self.document, create=True)
        self.session.cad_document.bind_object_id(native.Name, identity)
        self.session._profile_cache[cache_key] = identity
        return identity, native

    def _resolve(self, identity: str, kind: str = "profile") -> Any:
        wrapper = self.session.runtime_objects.get(identity)
        if isinstance(wrapper, CadObject):
            native = self.document.getObject(wrapper.native_name)
        else:
            # Newly captured objects exist before the end-of-transaction hydration.
            native = next(
                (
                    obj
                    for obj in self.document.Objects
                    if getattr(obj, "RapidCADObjectId", "") == identity
                ),
                None,
            )
        if native is None or getattr(native, "RapidCADKind", "") != kind:
            raise ValueError(f"Unknown persistent {kind} {identity!r} in the active document.")
        if native.Document is not self.document:
            raise ValueError("Profile belongs to another document.")
        return native

    def create_profile(
        self, workplane_id: str, expected_revision: str | None, *, kind: str = "profile"
    ) -> dict[str, Any]:
        def mutate() -> dict[str, Any]:
            identity, _ = self._capture(workplane_id, kind)
            reference = (
                CadProfile(document_id(self.document), identity)
                if kind == "profile"
                else CadPath(document_id(self.document), identity)
            )
            return {"object_id": identity, kind: reference.to_dict()}

        return self._run(
            f"create_{kind}",
            expected_revision,
            lambda: self._workplane(workplane_id),
            mutate,
        )

    def update_profile(
        self,
        profile_id: str,
        workplane_id: str | None,
        expected_revision: str | None,
        definition: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        geometry, placement = [], None

        def validate() -> None:
            nonlocal geometry, placement
            native = self._resolve(profile_id)
            if native.TypeId != "Sketcher::SketchObject" or native.ConstraintCount:
                raise ValueError(
                    "Cannot replace constrained sketch geometry; use constraint-aware editing."
                )
            if (definition is None) == (workplane_id is None):
                raise ValueError("Specify a profile definition or workplane_id, exclusively.")
            if definition is not None:
                request = modeling_request_from_dict(definition)
                if not isinstance(request, ProfileDefinition):
                    raise TypeError("update_profile requires a ProfileDefinition.")
                geometry, placement = native_geometry(request)
            else:
                self._workplane(workplane_id)

        def mutate() -> dict[str, Any]:
            from .sketch2d import FreeCADSketch2D

            native = self._resolve(profile_id)
            if definition is None:
                workplane = self._workplane(workplane_id)
                temporary = FreeCADSketch2D(
                    self._primitives(workplane), workplane, self.session.app
                )._create_editable_sketch(self.document, "RapidCADProfileUpdate")
                new_geometry, new_placement = temporary.Geometry, temporary.Placement
            else:
                temporary = None
                new_geometry, new_placement = geometry, placement
            for index in reversed(range(native.GeometryCount)):
                native.delGeometry(index)
            native.addGeometry(new_geometry, False)
            native.Placement = new_placement
            if temporary is not None:
                self.document.removeObject(temporary.Name)
            self.session._profile_cache = {
                k: v for k, v in self.session._profile_cache.items() if v != profile_id
            }
            return {
                "object_id": profile_id,
                "profile": CadProfile(document_id(self.document), profile_id).to_dict(),
            }

        return self._run("update_profile", expected_revision, validate, mutate)

    def update_feature(
        self, feature_id: str, parameters: dict[str, Any], expected_revision: str | None
    ) -> dict[str, Any]:
        update, native = None, None

        def validate() -> None:
            nonlocal update, native
            update = feature_update_from_dict(parameters)
            native = self._resolve(feature_id, update.kind)
            expected_type = "Part::Loft" if update.kind == "loft" else "Part::Sweep"
            if native.TypeId != expected_type:
                raise ValueError("Feature kind does not match the native editable feature.")
            if isinstance(update, LoftFeatureUpdate):
                if update.alignment not in {None, "automatic"}:
                    raise NotImplementedError("Part::Loft cannot preserve exact winding/seams.")
                for identity in update.profile_ids or ():
                    self._resolve(identity)
            else:
                if update.profile_id is not None:
                    self._resolve(update.profile_id)
                if update.path_id is not None:
                    self._resolve(update.path_id, "path")

        def mutate() -> dict[str, Any]:
            if isinstance(update, LoftFeatureUpdate):
                if update.profile_ids is not None:
                    native.Sections = [self._resolve(identity) for identity in update.profile_ids]
                if update.ruled is not None:
                    native.Ruled = update.ruled
            else:
                if update.profile_id is not None:
                    native.Sections = [self._resolve(update.profile_id)]
                if update.path_id is not None:
                    native.Spine = (self._resolve(update.path_id, "path"), [])
                if update.orientation is not None:
                    native.Frenet = update.orientation == "frenet"
                if update.transition is not None:
                    native.Transition = TRANSITIONS[update.transition]
            if update.make_solid is not None:
                native.Solid = update.make_solid
            return {"object_id": feature_id, "feature_kind": update.kind}

        return self._run("update_feature", expected_revision, validate, mutate)

    def inspect_capabilities(self) -> dict[str, Any]:
        from .spline_geometry import modeling_capabilities

        return {
            "ok": True,
            "document_revision": self.session.document_revision,
            **modeling_capabilities(),
        }

    def inspect_geometry(self, object_ids: list[str] | None = None) -> dict[str, Any]:
        self._refresh()
        identities = (
            object_ids
            if object_ids is not None
            else [key for key, obj in self.session.objects.items() if obj.type != "workplane"]
        )
        reports = []
        for identity in identities:
            if identity not in self.session.objects or identity not in self.session.runtime_objects:
                raise ValueError(f"Unknown object {identity!r}.")
            wrapper = self.session.runtime_objects[identity]
            native = getattr(wrapper, "native_handle", None)
            if native is None:
                continue
            shape = getattr(native, "Shape", None)
            errors = [
                str(status)
                for status in native.State
                if str(status).lower() in {"invalid", "error"}
            ]
            if errors:
                status = getattr(native, "getStatusString", lambda: "")()
                if status:
                    errors.append(str(status))
            has_geometry = shape is not None and not shape.isNull()
            if shape is not None and not has_geometry:
                errors.append("Native feature contains empty geometry.")
            if has_geometry and not shape.isValid():
                errors.append("Native geometry is invalid.")
            signature = self.session._shape_signature(shape) if has_geometry else {}
            bbox = signature.get("bbox", {})
            reports.append(
                {
                    "object_id": identity,
                    "object_type": self.session.objects[identity].type,
                    "feature_kind": self.session.objects[identity].metadata.get("feature_kind"),
                    "geometry_valid": not errors if shape is not None else None,
                    "errors": errors,
                    "pending_recompute": "Touched" in native.State,
                    "dimensions": {
                        "unit": "mm",
                        "x": bbox.get("x_length"),
                        "y": bbox.get("y_length"),
                        "z": bbox.get("z_length"),
                    },
                    "geometry": signature,
                    "solid_count": len(shape.Solids) if has_geometry else 0,
                    "dependency_ids": [
                        self.session.cad_document.object_id(obj.Name) for obj in native.OutList
                    ],
                    "dependent_ids": [
                        self.session.cad_document.object_id(obj.Name) for obj in native.InList
                    ],
                    "operation_support": self.session.objects[identity].metadata.get(
                        "operation_support", {}
                    ),
                }
            )
        return {
            "ok": True,
            "document_revision": self.session.document_revision,
            "geometry_valid": all(not item["errors"] for item in reports),
            "has_model_geometry": any(
                item["solid_count"] or item["geometry"].get("face_count", 0) for item in reports
            ),
            "objects": reports,
        }

    def loft(
        self,
        profile_ids: list[str] | None,
        workplane_ids: list[str] | None,
        make_solid: bool,
        ruled: bool,
        expected_revision: str | None,
        *,
        alignment: str = "automatic",
    ) -> dict[str, Any]:
        identifiers = profile_ids if profile_ids is not None else workplane_ids

        def validate() -> None:
            if alignment != "automatic":
                raise NotImplementedError(
                    "Part::Loft requires automatic kernel winding/seam matching."
                )
            if (profile_ids is None) == (workplane_ids is None):
                raise ValueError("Specify profile_ids or profile_workplane_ids, exclusively.")
            if type(make_solid) is not bool or type(ruled) is not bool:
                raise ValueError("make_solid and ruled must be booleans.")
            if (
                identifiers is None
                or len(identifiers) < 2
                or len(set(identifiers)) != len(identifiers)
            ):
                raise ValueError("Loft requires at least two distinct, ordered profiles.")
            for identity in identifiers:
                self._resolve(identity) if profile_ids is not None else self._workplane(identity)

        def mutate() -> dict[str, Any]:
            ids, sections = [], []
            for identity in identifiers:
                if profile_ids is not None:
                    ids.append(identity)
                    sections.append(self._resolve(identity))
                else:
                    captured_id, native = self._capture(identity)
                    ids.append(captured_id)
                    sections.append(native)
            self.document.recompute()
            for section in sections:
                self._single_wire(section)
            validate_loft_sections(sections)
            if make_solid and any(not obj.Shape.isClosed() for obj in sections):
                raise ValueError("Solid loft requires closed section profiles.")
            native = self.document.addObject("Part::Loft", "RapidCADLoft")
            native.Sections = sections
            native.Solid, native.Ruled, native.Closed = make_solid, ruled, False
            identity = tag_object(native, "loft")
            self.session.cad_document.bind_object_id(native.Name, identity)
            self.session.active_shape_id = identity
            return {
                "object_id": identity,
                "active_shape_id": identity,
                "profile_ids": ids,
                "alignment": alignment,
            }

        return self._run("loft", expected_revision, validate, mutate)


def loft_workplanes(app: Any, workplanes: list[Any], *, make_solid: bool, ruled: bool) -> Any:
    """Keep the fluent API as a convenience wrapper over persistent mutations."""
    from ...cad_session import CadSession, SemanticObject
    from .errors import FreeCADNativeFeatureError
    from .shape import FreeCADShape

    session = getattr(app, "_persistent_profile_session", None)
    if session is None or session.cad_document.native_handle is not app.get_doc():
        session = CadSession(execution_mode="embedded")
        session.app = app
        session.backend_name = "freecad"
        session.cad_document = app.cad_document
        app._persistent_profile_session = session
    ids = []
    for workplane in workplanes:
        existing = next(
            (key for key, handle in session.runtime_objects.items() if handle is workplane),
            None,
        )
        if existing is None:
            existing = session._new_id("workplane")
            session.runtime_objects[existing] = workplane
            session.objects[existing] = SemanticObject(
                existing, "workplane", "Fluent workplane", "fluent"
            )
        ids.append(existing)
    result = session.loft(ids, make_solid=make_solid, ruled=ruled)
    if not result["ok"]:
        raise FreeCADNativeFeatureError(result["error"])
    native = session.runtime_objects[result["object_id"]].native_handle
    return FreeCADShape(native.Shape, app, doc=app.get_doc(), current_feature=native)
