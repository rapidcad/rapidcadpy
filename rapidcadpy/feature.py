import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, Optional

if TYPE_CHECKING:
    from .workplane import Workplane


@dataclass
class Feature(ABC):
    """
    Abstract base class for all CAD features.

    Features represent operations that can be applied to create or modify 3D geometry.
    Each feature is localized on a specific workplane.
    """

    sketch_plane: Optional["Workplane"] = field(default=None)
    id: Optional[uuid.UUID] = field(default_factory=uuid.uuid4)
    name: str = "Feature"

    def __post_init__(self):
        if self.sketch_plane is None:
            from .workplane import Workplane

            self.sketch_plane = Workplane.xy_plane()

    @abstractmethod
    def to_json(self) -> Dict[str, Any]:
        """
        Convert the feature to JSON representation.

        Returns:
            Dict[str, Any]: JSON representation of the feature
        """
        pass

    @abstractmethod
    def to_python(self, index: int = 0) -> str:
        """
        Generate Python code to recreate this feature.

        Args:
            index: Index for variable naming

        Returns:
            str: Python code string
        """
        pass

    def __post_init__(self):
        if self.id is None:
            self.id = uuid.uuid4()
