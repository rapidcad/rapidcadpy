"""Backend-neutral, bounded edits to native loft and sweep parameters."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from .modeling import ModelingDefinition, _identifier


@dataclass(frozen=True)
class LoftFeatureUpdate(ModelingDefinition):
    profile_ids: tuple[str, ...] | None = None
    make_solid: bool | None = None
    ruled: bool | None = None
    alignment: Literal["automatic", "preserve"] | None = None
    kind: Literal["loft"] = field(default="loft", init=False)

    def __post_init__(self) -> None:
        if self.profile_ids is not None:
            object.__setattr__(self, "profile_ids", tuple(self.profile_ids))
            if len(self.profile_ids) < 2 or len(set(self.profile_ids)) != len(
                self.profile_ids
            ):
                raise ValueError(
                    "Loft requires at least two distinct ordered profiles."
                )
            for value in self.profile_ids:
                _identifier(value, "profile_id")
        if self.alignment is not None and self.alignment not in {
            "automatic",
            "preserve",
        }:
            raise ValueError("Unknown loft alignment.")
        _validate_update(self)


@dataclass(frozen=True)
class SweepFeatureUpdate(ModelingDefinition):
    profile_id: str | None = None
    path_id: str | None = None
    make_solid: bool | None = None
    orientation: Literal["frenet", "corrected_frenet"] | None = None
    transition: Literal["transformed", "right", "round"] | None = None
    kind: Literal["sweep"] = field(default="sweep", init=False)

    def __post_init__(self) -> None:
        for name in ("profile_id", "path_id"):
            if getattr(self, name) is not None:
                _identifier(getattr(self, name), name)
        if self.orientation is not None and self.orientation not in {
            "frenet",
            "corrected_frenet",
        }:
            raise ValueError("Unknown sweep orientation.")
        if self.transition is not None and self.transition not in {
            "transformed",
            "right",
            "round",
        }:
            raise ValueError("Unknown sweep transition.")
        _validate_update(self)


def _validate_update(update: Any) -> None:
    values = {key: value for key, value in vars(update).items() if key != "kind"}
    if not any(value is not None for value in values.values()):
        raise ValueError("At least one feature parameter must be supplied.")
    for name in ("make_solid", "ruled"):
        value = values.get(name)
        if value is not None and type(value) is not bool:
            raise TypeError(f"{name} must be a boolean.")


def feature_update_from_dict(
    payload: dict[str, Any],
) -> LoftFeatureUpdate | SweepFeatureUpdate:
    data = dict(payload)
    kind = data.pop("kind", None)
    if kind == "loft":
        return LoftFeatureUpdate(**data)
    if kind == "sweep":
        return SweepFeatureUpdate(**data)
    raise ValueError("Feature update kind must be loft or sweep.")
