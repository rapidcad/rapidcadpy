import math
import uuid
from typing import Optional, Tuple, Union

import numpy as np


class Vector(np.ndarray):
    def __new__(cls, x: float, y: float, z: float = 0) -> "Vector":
        return np.asarray([x, y, z]).view(cls)

    def __eq__(self, other: object) -> bool:
        return np.allclose(self, other)

    def normalize(self):
        return self / np.linalg.norm(self)

    def get_2d(self):
        return np.array((self[0], self[1]))

    @property
    def x(self):
        return self[0]

    @property
    def y(self):
        return self[1]

    @property
    def z(self):
        return self[2]

    def to_json(self):
        return {
            "x": float(self.x),
            "y": float(self.y),
            "z": float(self.z),
        }

    @staticmethod
    def from_json(json_data):
        return Vector(json_data["x"], json_data["y"], json_data["z"])


VectorLike = Union[Tuple[float, float], Tuple[float, float, float], Vector]


class Vertex(np.ndarray):
    def __new__(
        cls,
        x: float,
        y: float,
        id: Optional[uuid.UUID] = None,
        name: str = "unnamed_vertex",
    ):
        obj = np.asarray([float(x), float(y)])
        obj = np.round(obj, 6).view(cls)
        return obj

    def __init__(
        self,
        x: float,
        y: float,
        id: Optional[uuid.UUID] = None,
        name: str = "unnamed_vertex",
    ):
        # Assign the extra attributes
        self.id = id if id else uuid.uuid4()
        self.name = name

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.id = getattr(obj, "id", None)
        self.name = getattr(obj, "name", "unnamed_vertex")

    def __str__(self):
        return f"Vertex(x={self.x}, y={self.y})"

    @property
    def x(self):
        return float(np.round(self[0], 6))

    @property
    def y(self):
        return float(np.round(self[1], 6))

    def to_json(self):
        return {
            "x": round(float(self.x), 6),
            "y": round(float(self.y), 6),
        }

    @staticmethod
    def from_json(json_data):
        return Vertex(json_data["x"], json_data["y"])

    def __eq__(self, other):
        tolerance = 1e-9
        if not isinstance(other, Vertex):
            return math.isclose(self.x, other[0], abs_tol=tolerance) and math.isclose(
                self.y, other[1], abs_tol=tolerance
            )
        return math.isclose(self.x, other.x, abs_tol=tolerance) and math.isclose(
            self.y, other.y, abs_tol=tolerance
        )

    def __hash__(self):
        return hash((round(self.x, 6), round(self.y, 6)))

    def to_python(self):
        return f"Vertex(x={self.x}, y={self.y})"

    def round(self, decimals=0):
        self[0] = round(self[0], decimals)
        self[1] = round(self[1], decimals)
