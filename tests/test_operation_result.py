"""Tests for the public serializable CAD operation result contract."""

from __future__ import annotations

import pytest
from rapidcadpy.operation_result import CadOperationResult, CadOperationSupport


def test_operation_result_serializes_revision_diffs_and_support() -> None:
    result = CadOperationResult.success(
        summary="Created native extrusion.",
        document_revision="rev-42",
        created_object_ids=["shape-1"],
        changed_object_ids=["sketch-1"],
        warnings=["Sketch remains editable."],
        support=CadOperationSupport(
            operation="extrude",
            status="supported",
            mode="native_feature",
            reason="Created a native feature.",
        ),
        data={"shape_id": "shape-1"},
    ).to_dict()

    assert result == {
        "ok": True,
        "summary": "Created native extrusion.",
        "document_revision": "rev-42",
        "created_object_ids": ["shape-1"],
        "changed_object_ids": ["sketch-1"],
        "removed_object_ids": [],
        "warnings": ["Sketch remains editable."],
        "support": {
            "operation": "extrude",
            "status": "supported",
            "mode": "native_feature",
            "reason": "Created a native feature.",
        },
        "support_status": "supported",
        "shape_id": "shape-1",
    }
    assert CadOperationResult.from_dict(result).to_dict() == result


def test_operation_result_rejects_reserved_data_keys() -> None:
    with pytest.raises(ValueError, match="reserved keys"):
        CadOperationResult.success(data={"ok": "not allowed"}).to_dict()
