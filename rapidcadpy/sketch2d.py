"""
Sketch2D module - Represents a 2D sketch face that can be extruded into a 3D shape.
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, List, Optional

from rapidcadpy.primitives import Line
from rapidcadpy.shape import Shape

if TYPE_CHECKING:
    from rapidcadpy.app import App


class Sketch2D(ABC):
    """
    Abstract base class representing a 2D sketch with a constructed face.

    A Sketch2D is created when a workplane's close() method is called,
    and it holds the constructed face that can be extruded into a 3D shape.

    This represents a finalized sketch that is ready for 3D operations.
    """

    def __init__(
        self, primitives: List["Line"], workplane: Any, app: Optional["App"] = None
    ):
        """
        Initialize a sketch with a list of 2D primitives.

        Args:
            primitives: The 2D primitives that make up the sketch
            workplane: The workplane this sketch was created on
            app: Optional app instance for tracking
        """
        self._primitives = primitives
        self._workplane = workplane
        self.app = app

    @property
    def workplane(self) -> Any:
        """Get the workplane this sketch was created on."""
        return self._workplane

    @abstractmethod
    def extrude(
        self,
        distance: float,
        operation: str = "NewBodyFeatureOperation",
        symmetric: bool = False,
    ) -> Shape:
        """
        Extrude the sketch face along the workplane's normal direction.

        Args:
            distance: Distance to extrude (can be negative)
            operation: Operation type (for future compatibility)

        Returns:
            Shape: The extruded 3D shape
        """
        ...

    @abstractmethod
    def pipe(self, diameter: float): ...

    @abstractmethod
    def sweep(
        self,
        profile: "Sketch2D",
        make_solid: bool = True,
        is_frenet: bool = True,
        transition_mode: str = "right",
    ): ...

    @abstractmethod
    def to_png(
        self,
        file_name: Optional[str] = None,
        width: int = 800,
        height: int = 600,
        margin: float = 0.1,
    ) -> None:
        """
        Render the sketch to a PNG image.

        Args:
            file_name: Path to save the PNG file. If None, displays in a UI window instead.
            width: Image width in pixels (default: 800)
            height: Image height in pixels (default: 600)
            margin: Margin around the sketch as a fraction of size (default: 0.1)
        """
        ...
