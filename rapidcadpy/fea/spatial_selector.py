from typing import Dict
from attr import dataclass


@dataclass
class SpatialSelector:
    """Represents a spatial region selector from the load case."""

    id: str
    type: str  # 'box_2d', 'box_3d', 'sphere', etc.
    query: Dict[str, float]

    def to_location_string(self) -> str | tuple | dict:
        """
        Convert spatial selector to RapidCadPy location string, coordinate tuple, or box dict.

        For simple box selectors with a thin dimension, we map to standard locations like
        'x_min', 'x_max', 'top', 'bottom', etc.

        For point selectors with single coordinates (x, y, z), returns a tuple.

        For volume/face selectors (boxes with all ranges), returns the query dict directly
        for range-based node selection.
        """
        # Check if this is a point selector (single coordinates x, y, z)
        if all(k in self.query for k in ["x", "y", "z"]) and not any(
            k in self.query
            for k in ["x_min", "x_max", "y_min", "y_max", "z_min", "z_max"]
        ):
            # Point location - return as tuple for point loads
            return (self.query["x"], self.query["y"], self.query["z"])

        # Extract bounds
        x_min = self.query.get("x_min", float("-inf"))
        x_max = self.query.get("x_max", float("inf"))
        y_min = self.query.get("y_min", float("-inf"))
        y_max = self.query.get("y_max", float("inf"))
        z_min = self.query.get("z_min", float("-inf"))
        z_max = self.query.get("z_max", float("inf"))

        x_range = x_max - x_min
        y_range = y_max - y_min
        z_range = z_max - z_min

        if self.type == "box_2d":
            # Check if this is selecting a thin slice (a boundary)
            # If x range is very small, it's selecting x_min or x_max face
            if x_range < 1.0:
                if x_min < 1.0:
                    return "x_min"
                else:
                    return "x_max"

            # If y range is very small, it's selecting y_min or y_max face
            if y_range < 1.0:
                if y_min < 1.0:
                    return "y_min"
                else:
                    return "y_max"

        elif self.type == "box_3d":
            # Keep full range selectors for 3D boxes.
            # Using absolute thresholds (e.g. <1.0) to collapse into face names
            # breaks sub-mm models where all dimensions are <1.0.
            # Returning the query dict lets downstream node selection use exact ranges.
            return self.query

        # For thicker selection boxes, return the query dict directly
        # This allows range-based node selection in boundary_conditions.py
        return self.query
