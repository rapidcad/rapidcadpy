"""Exact, backend-neutral assembly collision checks for RapidCADPy shapes."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Iterable, Mapping, Protocol


class _Solid(Protocol):
    def volume(self) -> float: ...

    def intersection_volume(self, other: "_Solid") -> float: ...


@dataclass(frozen=True)
class Interference:
    """One unintended solid/solid interference."""

    first_id: str
    second_id: str
    volume: float
    #: The volume a *legitimate* overlap of this pair would occupy -- the local
    #: scale the overlap should be judged against.  ``0.0`` when the caller
    #: supplied no reference for either solid.
    reference_volume: float = 0.0

    @property
    def overlap_fraction(self) -> float:
        """Overlap as a share of the local reference; 0.0 without one."""
        if self.reference_volume <= 0.0:
            return 0.0
        return self.volume / self.reference_volume

    def to_dict(self) -> dict[str, float | str]:
        return {
            "first_id": self.first_id,
            "second_id": self.second_id,
            "volume": self.volume,
            "reference_volume": self.reference_volume,
            "overlap_fraction": self.overlap_fraction,
        }


@dataclass(frozen=True)
class AssemblyFitReport:
    """Serializable geometry-quality result suitable for an RL reward."""

    solid_count: int
    total_solid_volume: float
    unintended_interferences: tuple[Interference, ...]
    query_errors: tuple[str, ...]
    overlap_tolerance: float
    #: True when the caller supplied local reference volumes, so
    #: :attr:`local_goodness_of_fit` is a real measurement rather than a
    #: fallback to :attr:`goodness_of_fit`.
    locally_normalized: bool = False

    @property
    def interference_volume(self) -> float:
        return sum(item.volume for item in self.unintended_interferences)

    @property
    def collision_free(self) -> bool:
        return not self.unintended_interferences and not self.query_errors

    @property
    def goodness_of_fit(self) -> float:
        """A 0--1 dense score; collision-free assemblies score exactly 1.0.

        Overlap is divided by the *total* assembly volume, so the score dilutes
        as the assembly grows: the same clash between two beams matters less
        the more unrelated beams surround it.  That makes this a poor training
        signal for an assembly built up over many steps.  Prefer
        :attr:`local_goodness_of_fit`, and keep this one for consumers that
        already depend on its scale.
        """
        if self.solid_count == 0 or self.query_errors:
            return 0.0
        overlap_fraction = self.interference_volume / max(
            self.total_solid_volume, 1e-12
        )
        return max(0.0, 1.0 - min(1.0, overlap_fraction))

    @property
    def overlap_excess(self) -> float:
        """Total overlap measured against the local reference, clamped to 1.0.

        Falls back to the whole-assembly ratio when no reference volumes were
        supplied, so the value is never silently zero.
        """
        if self.solid_count == 0:
            return 0.0
        if not self.locally_normalized:
            return min(
                1.0, self.interference_volume / max(self.total_solid_volume, 1e-12)
            )
        return min(
            1.0, sum(item.overlap_fraction for item in self.unintended_interferences)
        )

    @property
    def local_goodness_of_fit(self) -> float:
        """A 0--1 score that judges each clash against its own local scale.

        Two beams meeting at a corner overlap by a fraction of one member's
        joint region; two beams crossing at mid-span overlap by a large share
        of it.  Dividing by that local volume keeps those two cases far apart
        no matter how big the surrounding assembly gets.
        """
        if self.solid_count == 0 or self.query_errors:
            return 0.0
        return max(0.0, 1.0 - self.overlap_excess)

    def to_dict(self) -> dict[str, object]:
        return {
            "solid_count": self.solid_count,
            "total_solid_volume": self.total_solid_volume,
            "collision_free": self.collision_free,
            "unintended_collision_count": len(self.unintended_interferences),
            "interference_volume": self.interference_volume,
            "overlap_tolerance": self.overlap_tolerance,
            "goodness_of_fit": self.goodness_of_fit,
            "local_goodness_of_fit": self.local_goodness_of_fit,
            "overlap_excess": self.overlap_excess,
            "locally_normalized": self.locally_normalized,
            "interferences": [item.to_dict() for item in self.unintended_interferences],
            "query_errors": list(self.query_errors),
        }


class AssemblyFitEvaluator:
    """Evaluate solid interference without changing the model.

    ``allowed_pairs`` is for deliberately overlapping pairs: a bolt and its
    mating component, or two structural members that share a joint.  That last
    case is the common one and it is not optional -- any two beams meeting at
    an angle interpenetrate near the joint, so without the exception a
    correctly built corner is scored as a clash and the highest-scoring
    assembly is a pile of disconnected members.

    ``reference_volumes`` maps a solid id to the volume a legitimate overlap
    involving it would occupy -- for a beam, roughly its cross-section times
    its depth.  Supplying it switches :attr:`AssemblyFitReport.local_goodness_of_fit`
    from a whole-assembly ratio, which dilutes as the assembly grows, to a
    per-clash ratio that does not.
    """

    def __init__(self, overlap_tolerance: float = 1e-3) -> None:
        if overlap_tolerance < 0:
            raise ValueError("overlap_tolerance must be non-negative")
        self.overlap_tolerance = float(overlap_tolerance)

    def evaluate(
        self,
        solids: Mapping[str, _Solid],
        *,
        allowed_pairs: Iterable[tuple[str, str]] = (),
        reference_volumes: Mapping[str, float] | None = None,
    ) -> AssemblyFitReport:
        normalized_allowed = {
            frozenset((str(first), str(second)))
            for first, second in allowed_pairs
            if first != second
        }
        references = {
            str(key): float(value)
            for key, value in (reference_volumes or {}).items()
            if float(value) > 0.0
        }
        errors: list[str] = []
        valid_solids: dict[str, _Solid] = {}
        total_volume = 0.0
        for solid_id, solid in solids.items():
            try:
                volume = max(0.0, float(solid.volume()))
            except Exception as exc:
                errors.append(f"{solid_id}: cannot measure volume ({exc})")
                continue
            if volume <= self.overlap_tolerance:
                errors.append(f"{solid_id}: non-positive solid volume")
                continue
            valid_solids[str(solid_id)] = solid
            total_volume += volume

        interferences: list[Interference] = []
        for (first_id, first), (second_id, second) in combinations(
            valid_solids.items(), 2
        ):
            if frozenset((first_id, second_id)) in normalized_allowed:
                continue
            try:
                overlap = max(0.0, float(first.intersection_volume(second)))
            except Exception as exc:
                errors.append(
                    f"{first_id}/{second_id}: cannot measure intersection ({exc})"
                )
                continue
            if overlap > self.overlap_tolerance:
                interferences.append(
                    Interference(
                        first_id,
                        second_id,
                        overlap,
                        # The smaller member sets the scale: an overlap that
                        # would swallow the slimmer of the two is total, however
                        # large the other one is.
                        reference_volume=_pair_reference(
                            references, first_id, second_id
                        ),
                    )
                )

        return AssemblyFitReport(
            solid_count=len(valid_solids),
            total_solid_volume=total_volume,
            unintended_interferences=tuple(interferences),
            query_errors=tuple(errors),
            overlap_tolerance=self.overlap_tolerance,
            locally_normalized=bool(references),
        )


def _pair_reference(
    references: Mapping[str, float], first_id: str, second_id: str
) -> float:
    """The local overlap scale for one pair, or 0.0 if neither side has one."""
    candidates = [
        references[solid_id]
        for solid_id in (first_id, second_id)
        if solid_id in references
    ]
    return min(candidates) if candidates else 0.0
