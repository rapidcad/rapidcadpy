"""Backend-neutral mutation lifecycle and rollback of the session mirror."""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Callable
from typing import Any, Protocol

from .operation_result import CadOperationResult, CadOperationSupport


def object_snapshot(objects: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Exclude hydration provenance and transient workplanes from native state."""
    return {
        key: {
            k: v for k, v in obj.to_dict().items() if k not in {"source_op", "source", "confidence"}
        }
        for key, obj in objects.items()
        if obj.type != "workplane"
    }


def document_revision(objects: dict[str, Any], parameters: dict[str, Any]) -> str:
    payload = {
        "objects": object_snapshot(objects),
        "parameters": {key: p.to_dict() for key, p in parameters.items()},
    }
    return (
        "sha256:"
        + hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
    )


class MutationBackend(Protocol):
    def begin(self, name: str) -> None: ...
    def recompute(self) -> None: ...
    def validate(self) -> None: ...
    def commit(self) -> None: ...
    def abort(self) -> None: ...


class SessionMutationState:
    """Snapshot only Python-owned state; never deepcopy CAD kernel handles."""

    _DEEP = (
        "objects",
        "operations",
        "geometry_signatures",
        "document",
        "_counters",
        "profile_references",
        "path_references",
        "_profile_cache",
    )
    _SHALLOW = ("runtime_objects", "_shape_wrappers", "parameters", "drawings")
    _VALUES = (
        "active_workplane",
        "active_workplane_id",
        "active_shape_id",
        "document_revision",
        "cad_document",
        "current_drawing_id",
    )

    def __init__(self, session: Any, refresh: Callable[[], None]) -> None:
        self.session = session
        self.refresh = refresh

    def snapshot(self) -> dict[str, Any]:
        session = self.session
        saved = {name: copy.deepcopy(getattr(session, name)) for name in self._DEEP}
        saved.update({name: dict(getattr(session, name)) for name in self._SHALLOW})
        saved.update({name: getattr(session, name) for name in self._VALUES})
        document = session.cad_document
        saved["document_state"] = (
            {
                key: copy.deepcopy(value)
                for key, value in vars(document).items()
                if key not in {"native_handle", "adapter"}
            }
            if document
            else None
        )
        saved["workplanes"] = []
        for key, obj in session.objects.items():
            if obj.type != "workplane":
                continue
            workplane = session.runtime_objects[key]
            saved["workplanes"].append(
                (
                    workplane,
                    {
                        name: copy.deepcopy(getattr(workplane, name))
                        for name in (
                            "_pending_shapes",
                            "_accumulated_loops",
                            "_current_position",
                            "_loop_start",
                            "_extruded_sketches",
                        )
                        if hasattr(workplane, name)
                    },
                )
            )
        saved["app_state"] = {
            name: copy.copy(getattr(session.app, name))
            for name in ("_workplanes", "_shapes", "_feature_counter")
            if hasattr(session.app, name)
        }
        return saved

    def restore(self, saved: dict[str, Any]) -> None:
        for name in (*self._DEEP, *self._SHALLOW, *self._VALUES):
            setattr(self.session, name, saved[name])
        if saved["document_state"] is not None:
            for name, value in saved["document_state"].items():
                setattr(self.session.cad_document, name, value)
        for workplane, state in saved["workplanes"]:
            for name, value in state.items():
                setattr(workplane, name, value)
        for name, value in saved["app_state"].items():
            setattr(self.session.app, name, value)


class MutationCoordinator:
    """Coordinate revision checks, native transactions, and session rollback."""

    def __init__(self, session: Any) -> None:
        self.session = session

    def revision(self) -> str:
        """Recalculate and publish the session's current document revision."""
        revision = document_revision(self.session.objects, self.session.parameters)
        self.session.document_revision = revision
        if self.session.cad_document is not None:
            self.session.cad_document.revision = revision
        self.session.document["revision"] = revision
        return revision

    def run(
        self,
        *,
        refresh: Callable[[], None],
        backend: MutationBackend,
        support: CadOperationSupport,
        expected_revision: str | None,
        validate_inputs: Callable[[], None],
        mutate: Callable[[], dict[str, Any]],
    ) -> CadOperationResult:
        """Run one mutation against the session owned by this coordinator."""
        return run_mutation(
            state=SessionMutationState(self.session, refresh),
            backend=backend,
            support=support,
            expected_revision=expected_revision,
            validate_inputs=validate_inputs,
            mutate=mutate,
        )


def run_mutation(
    *,
    state: SessionMutationState,
    backend: MutationBackend,
    support: CadOperationSupport,
    expected_revision: str | None,
    validate_inputs: Callable[[], None],
    mutate: Callable[[], dict[str, Any]],
) -> CadOperationResult:
    """Commit only after native validation, hydration and response serialization."""
    saved = state.snapshot()
    begun = False
    try:
        state.refresh()  # Detect external edits before accepting a revision.
        current_revision = state.session.document_revision
        if expected_revision is not None and expected_revision != current_revision:
            state.restore(saved)
            return CadOperationResult.failure(
                f"Document revision mismatch: expected {expected_revision}, current {current_revision}.",
                error_code="document_revision_mismatch",
                document_revision=current_revision,
                support=support,
            )
        validate_inputs()
        before = object_snapshot(state.session.objects)
        backend.begin(support.operation)
        begun = True
        data = mutate()
        backend.recompute()
        backend.validate()
        state.refresh()
        after = object_snapshot(state.session.objects)
        object_id = data.get("object_id")
        if object_id in after:
            data["geometry"] = after[object_id].get("geometry", {})
            data["object_type"] = after[object_id]["type"]
        data["operation_id"] = state.session._record(
            support.operation,
            {k: v for k, v in data.items() if k != "geometry"},
            sorted(after.keys() - before.keys()),
            f"Completed {support.operation}",
        )
        result = CadOperationResult.success(
            summary=f"Completed {support.operation}",
            document_revision=state.session.document_revision,
            created_object_ids=tuple(sorted(after.keys() - before.keys())),
            removed_object_ids=tuple(sorted(before.keys() - after.keys())),
            changed_object_ids=tuple(
                sorted(k for k in before.keys() & after.keys() if before[k] != after[k])
            ),
            support=support,
            data=data,
        )
        json.dumps(result.to_dict(), allow_nan=False)
        backend.commit()
        return result
    except Exception as exc:  # noqa: BLE001 - all failures must restore both states
        rollback_error = None
        if begun:
            try:
                backend.abort()
            except Exception as abort_exc:  # noqa: BLE001 - report failed rollback explicitly
                rollback_error = str(abort_exc)
        state.restore(saved)
        unsupported = isinstance(exc, NotImplementedError)
        if unsupported:
            support = CadOperationSupport(support.operation, "unsupported", "none", str(exc))
        return CadOperationResult.failure(
            f"{support.operation} failed: {type(exc).__name__}: {exc}",
            error_code="cad_rollback_failed"
            if rollback_error
            else "cad_operation_not_supported"
            if unsupported
            else "cad_operation_failed",
            document_revision=state.session.document_revision,
            support=support,
            warnings=(
                f"Native rollback failed: {rollback_error}. Refresh the document before continuing.",
            )
            if rollback_error
            else (),
        )
