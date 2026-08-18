"""Backend-neutral records for the RapidCADPy session's semantic mirror."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class SemanticObject:
    id: str
    type: str
    label: str
    source_op: str
    source: str = "cad_operation"
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "label": self.label,
            "source_op": self.source_op,
            "source": self.source,
            "confidence": self.confidence,
            **self.metadata,
        }


@dataclass
class OperationRecord:
    id: str
    tool: str
    args: Dict[str, Any]
    outputs: list[str]
    summary: str
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "tool": self.tool,
            "args": self.args,
            "outputs": self.outputs,
            "summary": self.summary,
            "timestamp": self.timestamp,
        }
