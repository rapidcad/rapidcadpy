"""Serializable public results for live RapidCAD operations.

This module deliberately contains no native CAD handles.  It is the stable
boundary consumed by agent runtimes while :mod:`rapidcadpy.cad_objects` keeps
live native references inside RapidCADPy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Literal, Optional


SupportStatus = Literal["supported", "conditional", "unsupported"]


@dataclass(frozen=True)
class CadOperationSupport:
    """Operation-level editability information for a CAD mutation."""

    operation: str
    status: SupportStatus
    mode: str
    reason: str

    def to_dict(self) -> Dict[str, str]:
        return {
            "operation": self.operation,
            "status": self.status,
            "mode": self.mode,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class CadOperationResult:
    """Backend-neutral, serializable result of one public CAD operation.

    ``data`` contains operation-specific serializable fields such as
    ``object_id`` or ``geometry``.  Native handles must never be placed there.
    """

    ok: bool
    summary: Optional[str] = None
    document_revision: Optional[str] = None
    created_object_ids: tuple[str, ...] = ()
    changed_object_ids: tuple[str, ...] = ()
    removed_object_ids: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    support: Optional[CadOperationSupport] = None
    error: Optional[str] = None
    error_code: Optional[str] = None
    data: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "CadOperationResult":
        """Recover the public result envelope received through a CAD connector."""
        reserved = {
            "ok", "summary", "document_revision", "created_object_ids",
            "changed_object_ids", "removed_object_ids", "warnings", "support",
            "support_status", "error", "error_code",
        }
        raw_support = payload.get("support")
        return cls(
            ok=bool(payload.get("ok")), summary=payload.get("summary"),
            document_revision=payload.get("document_revision"),
            created_object_ids=tuple(payload.get("created_object_ids", ())),
            changed_object_ids=tuple(payload.get("changed_object_ids", ())),
            removed_object_ids=tuple(payload.get("removed_object_ids", ())),
            warnings=tuple(payload.get("warnings", ())),
            support=CadOperationSupport(**raw_support) if raw_support else None,
            error=payload.get("error"), error_code=payload.get("error_code"),
            data={key: value for key, value in payload.items() if key not in reserved},
        )

    def to_dict(self) -> Dict[str, Any]:
        reserved = {
            "ok",
            "summary",
            "document_revision",
            "created_object_ids",
            "changed_object_ids",
            "removed_object_ids",
            "warnings",
            "support",
            "support_status",
            "error",
            "error_code",
        }
        collision = reserved.intersection(self.data)
        if collision:
            joined = ", ".join(sorted(collision))
            raise ValueError(f"CadOperationResult data contains reserved keys: {joined}.")

        result: Dict[str, Any] = {
            "ok": self.ok,
            "document_revision": self.document_revision,
            "created_object_ids": list(self.created_object_ids),
            "changed_object_ids": list(self.changed_object_ids),
            "removed_object_ids": list(self.removed_object_ids),
            "warnings": list(self.warnings),
            "support": self.support.to_dict() if self.support else None,
            "support_status": self.support.status if self.support else None,
            **self.data,
        }
        if self.summary is not None:
            result["summary"] = self.summary
        if self.error is not None:
            result["error"] = self.error
        if self.error_code is not None:
            result["error_code"] = self.error_code
        return result

    @classmethod
    def success(
        cls, summary: Optional[str] = None, **kwargs: Any
    ) -> "CadOperationResult":
        return cls(ok=True, summary=summary, **kwargs)

    @classmethod
    def failure(
        cls,
        error: str,
        *,
        error_code: str = "cad_operation_failed",
        **kwargs: Any,
    ) -> "CadOperationResult":
        return cls(ok=False, error=error, error_code=error_code, **kwargs)
