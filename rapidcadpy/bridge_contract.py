"""Versioned live connector contract, independent of a CAD runtime."""

from __future__ import annotations

from typing import Any

CONTRACT_NAME = "rapidcad.live"
CONTRACT_MAJOR = 1
CONTRACT_MINOR = 0
VERSIONED_OPERATIONS = frozenset(
    {
        "create_profile",
        "update_profile",
        "create_path",
        "update_path",
        "loft",
        "sweep",
        "update_feature",
        "inspect_capabilities",
        "inspect_geometry",
        "apply_modeling",
        "inspect_modeling_support",
    }
)
CAPABILITY_OPERATIONS = {
    "profile.create": "create_profile",
    "profile.update": "update_profile",
    "path.create": "create_path",
    "path.update": "update_path",
    "feature.loft": "loft",
    "feature.sweep": "sweep",
    "feature.update": "update_feature",
    "capability.inspect": "inspect_capabilities",
    "geometry.inspect": "inspect_geometry",
}


def bridge_contract(operations: list[str] | None = None) -> dict[str, Any]:
    return {
        "name": CONTRACT_NAME,
        "major": CONTRACT_MAJOR,
        "minor": CONTRACT_MINOR,
        "operations": sorted(
            VERSIONED_OPERATIONS if operations is None else operations
        ),
    }


def contract_supports(payload: Any, operation: str) -> bool:
    return (
        isinstance(payload, dict)
        and payload.get("name") == CONTRACT_NAME
        and type(payload.get("major")) is int
        and payload["major"] == CONTRACT_MAJOR
        and type(payload.get("minor")) is int
        and payload["minor"] >= CONTRACT_MINOR
        and isinstance(payload.get("operations"), list)
        and operation in payload["operations"]
    )


def advertised_capabilities(
    capabilities: tuple[str, ...], payload: Any
) -> tuple[str, ...]:
    return tuple(
        capability
        for capability in capabilities
        if capability not in CAPABILITY_OPERATIONS
        or contract_supports(payload, CAPABILITY_OPERATIONS[capability])
    )


def negotiate_contract(
    client: Any, server: dict[str, Any] | None = None
) -> dict[str, Any]:
    server = bridge_contract() if server is None else server
    required = client.get("operations") if isinstance(client, dict) else None
    version_compatible = (
        isinstance(client, dict)
        and isinstance(server, dict)
        and type(client.get("minor")) is int
        and type(server.get("minor")) is int
        and server["minor"] >= client["minor"]
    )
    if (
        not version_compatible
        or not isinstance(required, list)
        or not required
        or any(
            not isinstance(op, str)
            or not contract_supports(client, op)
            or not contract_supports(server, op)
            for op in required
        )
    ):
        return {
            "ok": False,
            "error_code": "cad_bridge_incompatible",
            "error": "The live connector contract is missing, incompatible, or lacks required operations. Update the connector and reconnect.",
            "bridge_contract": server,
            "required_bridge_contract": bridge_contract(
                required
                if isinstance(required, list)
                and all(isinstance(op, str) for op in required)
                else []
            ),
        }
    return {"ok": True, "bridge_contract": server, "negotiated_operations": required}
