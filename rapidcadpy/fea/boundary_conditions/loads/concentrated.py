"""
ConcentratedLoad — full force applied to every node in a selection.
"""

from typing import Optional, Tuple, Union, Literal, Dict, Any

from .base import Load


class ConcentratedLoad(Load):
    """
    Concentrated force applied independently to each node in the selection.

    Unlike :class:`~rapidcadpy.fea.boundary_conditions.loads.distributed.DistributedLoad`,
    which divides the total force by the node count, this class applies
    ``force`` in full to **every** node found.  This matches the Abaqus
    ``*CLOAD`` semantics for a node set: each node receives the full magnitude.

    Use when you have a known per-node force (e.g. from analytical
    calculations), not a known resultant over a face.
    """

    def __init__(
        self,
        location: Union[str, Dict[str, Any], Tuple[float, float, float]],
        force: Union[float, Tuple[float, float, float]],
        direction: Optional[Literal["x", "y", "z", "-x", "-y", "-z"]] = None,
        tolerance: float = 1,
    ):
        """
        Args:
            location: String name, bounding-box dict, or ``(x, y, z)`` point.
            force: Force in N applied to **each** node — scalar or vector.
            direction: Required when ``force`` is scalar.
            tolerance: Node-selection tolerance multiplier (× mesh_size, mm).
        """
        self.location = location
        self.force = force
        self.direction = direction
        self.tolerance = tolerance

    def apply(self, model, nodes, elements, geometry_info, mesh_size: float):
        """Apply concentrated load to the model."""
        from ...utils import find_nodes_in_box

        bbox = geometry_info["bounding_box"]
        load_nodes = None

        if isinstance(self.location, str):
            loc = self.location.lower()
            if loc in ["x_min", "end_1"]:
                load_nodes = find_nodes_in_box(
                    nodes,
                    xmin=bbox["xmin"],
                    xmax=bbox["xmin"],
                    tolerance=self.tolerance * mesh_size,
                )
            elif loc in ["x_max", "end_2"]:
                load_nodes = find_nodes_in_box(
                    nodes,
                    xmin=bbox["xmax"],
                    xmax=bbox["xmax"],
                    tolerance=self.tolerance * mesh_size,
                )
            elif loc in ["y_min"]:
                load_nodes = find_nodes_in_box(
                    nodes,
                    ymin=bbox["ymin"],
                    ymax=bbox["ymin"],
                    tolerance=self.tolerance * mesh_size,
                )
            elif loc in ["y_max"]:
                load_nodes = find_nodes_in_box(
                    nodes,
                    ymin=bbox["ymax"],
                    ymax=bbox["ymax"],
                    tolerance=self.tolerance * mesh_size,
                )
            elif loc in ["z_min", "bottom"]:
                load_nodes = find_nodes_in_box(
                    nodes,
                    zmin=bbox["zmin"],
                    zmax=bbox["zmin"],
                    tolerance=self.tolerance * mesh_size,
                )
            elif loc in ["z_max", "top"]:
                load_nodes = find_nodes_in_box(
                    nodes,
                    zmin=bbox["zmax"],
                    zmax=bbox["zmax"],
                    tolerance=self.tolerance * mesh_size,
                )
            else:
                raise ValueError(f"Unknown location: {self.location!r}")

        elif isinstance(self.location, dict):
            load_nodes = find_nodes_in_box(
                nodes,
                xmin=self.location.get("x_min"),
                xmax=self.location.get("x_max"),
                ymin=self.location.get("y_min"),
                ymax=self.location.get("y_max"),
                zmin=self.location.get("z_min"),
                zmax=self.location.get("z_max"),
                tolerance=self.tolerance * mesh_size,
            )
        elif isinstance(self.location, (tuple, list)) and len(self.location) == 3:
            x, y, z = self.location
            load_nodes = find_nodes_in_box(
                nodes,
                xmin=x,
                xmax=x,
                ymin=y,
                ymax=y,
                zmin=z,
                zmax=z,
                tolerance=self.tolerance,
            )
        else:
            raise ValueError(f"Unsupported location type: {type(self.location)}")

        if load_nodes is None or len(load_nodes) == 0:
            print(f"Warning: No nodes found at location: {self.location}")
            return 0

        if isinstance(self.force, (int, float)):
            if self.direction is None:
                raise ValueError(
                    "direction must be specified for scalar force on ConcentratedLoad"
                )
            direction = self.direction.lower()
            dir_map = {"x": 0, "y": 1, "z": 2, "-x": 0, "-y": 1, "-z": 2}
            if direction not in dir_map:
                raise ValueError(f"Invalid direction: {direction!r}")
            dir_idx = dir_map[direction]
            sign = -1 if direction.startswith("-") else 1
            model.forces[load_nodes, dir_idx] += sign * self.force
        else:
            fx, fy, fz = self.force
            model.forces[load_nodes, 0] += fx
            model.forces[load_nodes, 1] += fy
            model.forces[load_nodes, 2] += fz

        return len(load_nodes)
