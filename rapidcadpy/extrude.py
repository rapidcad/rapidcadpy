from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class Extrude:
    extent_one: float
    extent_two: Optional[float] = 0.0
    extent_type: str = "OneSideFeatureExtentType"
    operation: str = "NewBodyFeatureOperation"
    direction: int = 1
    symmetric: Optional[bool] = False
    taper_angle_one: Optional[float] = 0.0
    taper_angle_two: Optional[float] = 0.0
    name: Optional[str] = None

    def __post_init__(self):
        self.extent_one = round(self.extent_one, 6)

    def to_json(self) -> Dict[str, Any]:
        return {
            "extent_one": self.extent_one,
            "extent_two": self.extent_two,
            "symmetric": self.symmetric,
            "taper_angle_one": self.taper_angle_one,
            "taper_angle_two": self.taper_angle_two,
            "extent_type": self.extent_type,
            "operation": self.operation,
        }

    @classmethod
    def from_tensor(cls, tensor) -> "Extrude":
        # Define constants locally since we can't import from models
        EXTENT_TYPE = [
            "OneSideFeatureExtentType",
            "TwoSidesFeatureExtentType",
            "SymmetricFeatureExtentType",
        ]
        EXTRUDE_OPERATIONS = [
            "NewBodyFeatureOperation",
            "JoinFeatureOperation",
            "CutFeatureOperation",
            "IntersectFeatureOperation",
        ]

        return cls(
            extent_one=float(tensor[0]),
            extent_two=float(tensor[1]),
            symmetric=bool(tensor[2]),
            taper_angle_one=float(tensor[3]),
            taper_angle_two=float(tensor[4]),
            extent_type=EXTENT_TYPE[int(tensor[5])],
            operation=EXTRUDE_OPERATIONS[int(tensor[6])],
        )

    @staticmethod
    def from_json(json):
        return Extrude(
            extent_one=json["extent_one"],
            extent_two=json["extent_two"],
            symmetric=json["symmetric"],
            taper_angle_one=json["taper_angle_one"],
            taper_angle_two=json["taper_angle_two"],
            extent_type=json["extent_type"],
            operation=json["operation"],
        )

    def to_python(self, index: Optional[int] = 0):
        return f"Extrude(extent_one={self.extent_one}, extent_two={self.extent_two}, symmetric={self.symmetric}, taper_angle_one={self.taper_angle_one}, taper_angle_two={self.taper_angle_two})"

    def __eq__(self, other):
        return (
            self.extent_one == other.extent_one
            and self.extent_two == other.extent_two
            and self.symmetric == other.symmetric
            and self.taper_angle_one == other.taper_angle_one
            and self.taper_angle_two == other.taper_angle_two
        )

    def transform(self, scale):
        self.extent_one *= scale
        self.extent_two *= scale
        return self

    def numericalize(self, n=256):
        n -= 1
        self.extent_one = round(self.extent_one * n)
        self.extent_two = round(self.extent_two * n)
        self.taper_angle_one = round(self.taper_angle_one * n)
        self.taper_angle_two = round(self.taper_angle_two * n)
        return self

    def denumericalize(self, n):
        self.extent_one = self.extent_one / n
        self.extent_two = self.extent_two / n
        self.taper_angle_one = self.taper_angle_one / n
        self.taper_angle_two = self.taper_angle_two / n
        return self
