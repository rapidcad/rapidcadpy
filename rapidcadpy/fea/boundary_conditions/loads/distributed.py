"""
DistributedLoad — total force spread evenly across a surface.
"""

from typing import Optional, Tuple, Union, Literal

from .base import Load


class DistributedLoad(Load):
    """
    Distributed load over a surface.

    Applies a total force distributed evenly across all nodes on the specified
    surface.  The force is divided equally: ``force_per_node = F_total / n_nodes``.

    Use this when you know the resultant force on a face (e.g. 1000 N on the top
    face) and want each surface node to carry an equal share.  For per-unit-area
    pressure use :class:`~rapidcadpy.fea.boundary_conditions.loads.pressure.PressureLoad`.
    """

    def __init__(
        self,
        location: Union[str, dict],
        force: Union[float, Tuple[float, float, float]],
        direction: Optional[Literal["x", "y", "z", "normal", "-x", "-y", "-z"]] = None,
        tolerance: float = 1,
    ):
        """
        Args:
            location: Surface identifier string (``'top'``, ``'bottom'``, ``'x_min'``,
                ``'x_max'``, ``'y_min'``, ``'y_max'``, ``'z_min'``, ``'z_max'``) or
                a bounding-box dict with ``x_min/max``, ``y_min/max``, ``z_min/max`` keys.
            force: Total force magnitude in N (scalar) or vector ``(Fx, Fy, Fz)``.
            direction: Force direction for scalar ``force``.  If ``None`` the surface
                normal direction is used as the default.
            tolerance: Node-selection tolerance multiplier (× mesh_size, mm).
        """
        self.location = location
        self.force = force
        self.direction = direction
        self.tolerance = tolerance

    def __repr__(self) -> str:
        return (
            f"DistributedLoad(location={self.location!r}, "
            f"force={self.force!r}, direction={self.direction!r}, "
            f"tolerance={self.tolerance!r})"
        )

    def apply(self, model, nodes, elements, geometry_info, mesh_size: float):
        """Apply distributed load to the model."""
        from ...utils import find_nodes_in_box
        import torch  # noqa: F401

        bbox = geometry_info["bounding_box"]

        if isinstance(self.location, dict):
            surface_tolerance = 0.1 * mesh_size
            load_nodes = find_nodes_in_box(
                nodes,
                xmin=self.location.get("x_min"),
                xmax=self.location.get("x_max"),
                ymin=self.location.get("y_min"),
                ymax=self.location.get("y_max"),
                zmin=self.location.get("z_min"),
                zmax=self.location.get("z_max"),
                tolerance=surface_tolerance,
            )
            x_range = abs(self.location.get("x_max", 0) - self.location.get("x_min", 0))
            y_range = abs(self.location.get("y_max", 0) - self.location.get("y_min", 0))
            z_range = abs(self.location.get("z_max", 0) - self.location.get("z_min", 0))
            if z_range < x_range and z_range < y_range:
                default_dir = "z"
            elif y_range < x_range:
                default_dir = "y"
            else:
                default_dir = "x"

        elif isinstance(self.location, str):
            loc = self.location.lower()
            if loc in ["top", "z_max"]:
                load_nodes = find_nodes_in_box(
                    nodes, zmin=bbox["zmax"], zmax=bbox["zmax"],
                    tolerance=self.tolerance * mesh_size,
                )
                default_dir = "z"
            elif loc in ["bottom", "z_min"]:
                load_nodes = find_nodes_in_box(
                    nodes, zmin=bbox["zmin"], zmax=bbox["zmin"],
                    tolerance=self.tolerance * mesh_size,
                )
                default_dir = "z"
            elif loc in ["x_min"]:
                load_nodes = find_nodes_in_box(
                    nodes, xmin=bbox["xmin"], xmax=bbox["xmin"],
                    tolerance=self.tolerance * mesh_size,
                )
                default_dir = "x"
            elif loc in ["x_max"]:
                load_nodes = find_nodes_in_box(
                    nodes, xmin=bbox["xmax"], xmax=bbox["xmax"],
                    tolerance=self.tolerance * mesh_size,
                )
                default_dir = "x"
            elif loc in ["y_min"]:
                load_nodes = find_nodes_in_box(
                    nodes, ymin=bbox["ymin"], ymax=bbox["ymin"],
                    tolerance=self.tolerance * mesh_size,
                )
                default_dir = "y"
            elif loc in ["y_max"]:
                load_nodes = find_nodes_in_box(
                    nodes, ymin=bbox["ymax"], ymax=bbox["ymax"],
                    tolerance=self.tolerance * mesh_size,
                )
                default_dir = "y"
            else:
                raise ValueError(f"Unknown location: {self.location!r}")
        else:
            raise ValueError(f"Unsupported location type: {type(self.location)}")

        if len(load_nodes) == 0:
            raise ValueError(f"No nodes found at location: {self.location}")

        if isinstance(self.force, (int, float)):
            direction = self.direction or default_dir
            force_per_node = self.force / len(load_nodes)
            dir_map = {
                "x": 0, "y": 1, "z": 2,
                "-x": 0, "-y": 1, "-z": 2,
                "+x": 0, "+y": 1, "+z": 2,
            }
            direction = direction.lower()
            if direction not in dir_map:
                raise ValueError(f"Invalid direction: {direction!r}")
            dir_idx = dir_map[direction]
            sign = -1 if direction.startswith("-") else 1
            model.forces[load_nodes, dir_idx] = sign * force_per_node
        else:
            fx, fy, fz = self.force
            model.forces[load_nodes, 0] = fx / len(load_nodes)
            model.forces[load_nodes, 1] = fy / len(load_nodes)
            model.forces[load_nodes, 2] = fz / len(load_nodes)

        return len(load_nodes)
