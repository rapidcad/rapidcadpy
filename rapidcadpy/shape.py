from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from rapidcadpy.app import App


class Shape(ABC):
    def __init__(self, obj, app: Optional["App"]) -> None:
        self.obj = obj
        self.app = app

    @abstractmethod
    def to_stl(self, file_name: str) -> None:
        pass

    @abstractmethod
    def to_step(self, file_name: str) -> None:
        pass

    @abstractmethod
    def to_png(
        self,
        file_name: str,
        view: str = "iso",
        width: int = 800,
        height: int = 600,
        backend: str = "auto",
    ) -> None:
        pass


    @abstractmethod
    def cut(self, other: "Shape") -> "Shape":
        pass

    @abstractmethod
    def union(self, other: "Shape") -> "Shape":
        pass
