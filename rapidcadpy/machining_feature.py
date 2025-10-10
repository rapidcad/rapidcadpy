from abc import ABC
from dataclasses import dataclass
from typing import Any, Dict, Literal

from .feature import Feature


@dataclass
class MachiningFeature(Feature, ABC):
    """
    Abstract base class for machining features.

    Machining features represent subtractive operations like holes, slots, and cuts.
    """

    pass


@dataclass
class CounterSunkHole(MachiningFeature):
    """
    A counter-sunk hole feature.

    Represents a hole with a conical countersink at the top.
    """

    diameter: float = 5.0
    depth: float = 10.0
    countersink_diameter: float = 8.0
    countersink_angle: float = 90.0  # degrees

    def to_json(self) -> Dict[str, Any]:
        """Convert to JSON representation."""
        return {
            "type": "CounterSunkHole",
            "id": str(self.id),
            "name": self.name,
            "sketch_plane": self.sketch_plane.to_json(),
            "diameter": self.diameter,
            "depth": self.depth,
            "countersink_diameter": self.countersink_diameter,
            "countersink_angle": self.countersink_angle,
        }

    def to_python(self, index: int = 0) -> str:
        """Generate Python code to recreate this feature."""
        return f"""countersunk_hole_{index} = CounterSunkHole(
    diameter={self.diameter},
    depth={self.depth},
    countersink_diameter={self.countersink_diameter},
    countersink_angle={self.countersink_angle},
    sketch_plane={self.sketch_plane.to_python() if hasattr(self.sketch_plane, 'to_python') else 'Plane()'},
    name="{self.name}"
)"""


@dataclass
class ParallelKeyway(MachiningFeature):
    """
    A parallel keyway feature.

    Represents a rectangular slot cut parallel to an axis.
    """

    width: float = 5.0
    depth: float = 3.0
    length: float = 20.0
    orientation: Literal["horizontal", "vertical"] = "horizontal"

    def to_json(self) -> Dict[str, Any]:
        """Convert to JSON representation."""
        return {
            "type": "ParallelKeyway",
            "id": str(self.id),
            "name": self.name,
            "sketch_plane": self.sketch_plane.to_json(),
            "width": self.width,
            "depth": self.depth,
            "length": self.length,
            "orientation": self.orientation,
        }

    def to_python(self, index: int = 0) -> str:
        """Generate Python code to recreate this feature."""
        return f"""parallel_keyway_{index} = ParallelKeyway(
    width={self.width},
    depth={self.depth},
    length={self.length},
    orientation="{self.orientation}",
    sketch_plane={self.sketch_plane.to_python() if hasattr(self.sketch_plane, 'to_python') else 'Plane()'},
    name="{self.name}"
)"""
