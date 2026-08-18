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

    Features represent serializable modelling intent.  Native CAD application
    work is performed by a backend ``FeatureExecutor``; feature definitions do
    not import or branch on any CAD backend.
    """

    sketch_plane: Optional["Workplane"] = field(default=None)
    id: Optional[uuid.UUID] = field(default_factory=uuid.uuid4)
    name: str = "Feature"

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

    def __post_init__(self) -> None:
        if self.id is None:
            self.id = uuid.uuid4()

    def apply(
        self,
        executor: "FeatureExecutor",
        target: "CadObject",
        expected_revision: Optional[str],
    ) -> "FeatureResult":
        """Apply this intent through a selected backend executor.

        This small convenience method deliberately delegates all native work to
        the executor, keeping feature definitions backend-neutral.
        """

        return executor.apply(self, target, expected_revision)


if TYPE_CHECKING:
    from .cad_objects import CadObject
    from .feature_executor import FeatureExecutor, FeatureResult
