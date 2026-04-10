"""
PointLoad — concentrated force at a specific location.
"""

from typing import Optional, Tuple, Union, Literal

from .base import Load


class PointLoad(Load):
    """
    Concentrated load applied to the node(s) closest to a specified point.

    Node selection uses a box search around the target point.  If no nodes are
    found the search degrades gracefully through two fallbacks:

    1. Standard mesh-scaled tolerance search.
    2. Nearest-node fallback (always returns at least one node).

    When multiple nodes are found the total force is distributed equally among
    them so that the resultant equals ``force``.
    """

    def __init__(
        self,
        point: Union[Tuple[float, float, float], dict],
        force: Union[float, Tuple[float, float, float]],
        direction: Optional[Literal["x", "y", "z", "-x", "-y", "-z"]] = None,
        tolerance: float = 1,
        search_radius: Optional[Tuple[float, float, float]] = None,
    ):
        """
        Args:
            point: ``(x, y, z)`` coordinate tuple/list, or a dict with ``x``,
                ``y``, ``z`` keys (and optionally ``rx``, ``ry``, ``rz`` for
                anisotropic search radii).
            force: Force magnitude in N (scalar) or vector ``(Fx, Fy, Fz)``.
            direction: Required when ``force`` is scalar.
            tolerance: Node-selection tolerance multiplier (× mesh_size, mm).
            search_radius: Optional ``(rx, ry, rz)`` anisotropic radii for
                ellipsoidal node selection (mm).
        """
        self.point = point
        self.force = force
        self.direction = direction
        self.tolerance = tolerance
        self.search_radius = search_radius

    def __str__(self) -> str:
        return (
            f"PointLoad(point={self.point}, force={self.force}, "
            f"direction={self.direction}, tolerance={self.tolerance}, "
            f"search_radius={self.search_radius})"
        )

    def apply(self, model, nodes, elements, geometry_info, mesh_size: float):
        """Apply point load to the model."""
        from ...utils import find_nodes_in_box
        import torch

        point = self.point

        if isinstance(point, dict):
            x = point.get("x", (point.get("x_min", 0) + point.get("x_max", 0)) / 2)
            y = point.get("y", (point.get("y_min", 0) + point.get("y_max", 0)) / 2)
            z = point.get("z", (point.get("z_min", 0) + point.get("z_max", 0)) / 2)
            if self.search_radius is None and all(
                k in point for k in ["rx", "ry", "rz"]
            ):
                self.search_radius = (
                    float(point["rx"]),
                    float(point["ry"]),
                    float(point["rz"]),
                )
        elif isinstance(point, (tuple, list)):
            x, y, z = point[0], point[1], point[2]
        else:
            raise ValueError(
                f"Invalid point format: {self.point!r}. Expected tuple/list or dict."
            )

        if self.search_radius is not None:
            rx, ry, rz = [max(float(v), 1e-9) for v in self.search_radius]
            candidate_nodes = find_nodes_in_box(
                nodes,
                xmin=x - rx,
                xmax=x + rx,
                ymin=y - ry,
                ymax=y + ry,
                zmin=z - rz,
                zmax=z + rz,
                tolerance=0,
            )
            load_nodes = candidate_nodes

            # Fallback 1: mesh-scaled tolerance
            if len(load_nodes) == 0:
                load_nodes = find_nodes_in_box(
                    nodes,
                    xmin=x,
                    xmax=x,
                    ymin=y,
                    ymax=y,
                    zmin=z,
                    zmax=z,
                    tolerance=self.tolerance * mesh_size,
                )
            # Fallback 2: nearest node
            if len(load_nodes) == 0:
                point_tensor = torch.tensor(
                    [x, y, z], dtype=nodes.dtype, device=nodes.device
                )
                distances = torch.norm(nodes - point_tensor, dim=1)
                nearest = int(torch.argmin(distances).item())
                load_nodes = torch.tensor(
                    [nearest], dtype=torch.long, device=nodes.device
                )
        else:
            load_nodes = find_nodes_in_box(
                nodes,
                xmin=x,
                xmax=x,
                ymin=y,
                ymax=y,
                zmin=z,
                zmax=z,
                tolerance=self.tolerance * mesh_size,
            )

        if len(load_nodes) == 0:
            print(f"Warning: No nodes found near point: {self.point}")
            return 0

        force_array = (
            model.forces
            if hasattr(model, "forces")
            else model.f_ext if hasattr(model, "f_ext") else None
        )
        if force_array is None:
            raise AttributeError("Model must expose either 'forces' or 'f_ext'")

        if isinstance(self.force, (int, float)):
            if self.direction is None:
                raise ValueError("direction must be specified for scalar force")
            direction = self.direction.lower()
            dir_map = {"x": 0, "y": 1, "z": 2, "-x": 0, "-y": 1, "-z": 2}
            if direction not in dir_map:
                raise ValueError(f"Invalid direction: {direction!r}")
            dir_idx = dir_map[direction]
            sign = -1 if direction.startswith("-") else 1
            force_per_node = self.force / len(load_nodes)
            force_array[load_nodes, dir_idx] = sign * force_per_node
        else:
            fx, fy, fz = self.force
            force_array[load_nodes, 0] = fx / len(load_nodes)
            force_array[load_nodes, 1] = fy / len(load_nodes)
            force_array[load_nodes, 2] = fz / len(load_nodes)

        return len(load_nodes)
