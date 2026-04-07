"""
LinearDistributedLoad — load that varies linearly (triangular) along an axis.

Useful for hydrostatic pressure where load intensity grows with water depth.
"""

from typing import Union, Dict, Any

from .base import Load


class LinearDistributedLoad(Load):
    """
    Distributed load whose intensity varies linearly along a spatial axis.

    The total resultant force ``F`` is distributed among the selected nodes
    with node weights ``w_i`` that vary linearly along ``axis``:

    - ``profile='triangular_invert'``: ``w_i ∝ (coord_max − coord_i)``
      — maximum intensity at the minimum coordinate (e.g. deep-water pressure
      is highest at the bottom of a submerged wall).
    - ``profile='triangular'``: ``w_i ∝ (coord_i − coord_min)``
      — maximum intensity at the maximum coordinate.

    The weights are normalised so that ``Σ w_i = 1``, preserving the
    specified total force regardless of mesh density.
    """

    def __init__(
        self,
        location: Union[str, Dict[str, Any]],
        force: float,
        direction: str,
        axis: str = "z",
        profile: str = "triangular_invert",
        tolerance: float = 1,
    ):
        """
        Args:
            location: Surface identifier string or bounding-box dict.
            force: Total force resultant magnitude (N).
            direction: Force vector direction (``'x'``, ``'y'``, ``'z'``,
                ``'-x'``, ``'-y'``, ``'-z'``).
            axis: Spatial axis along which the intensity varies.
            profile: ``'triangular_invert'`` (max at min coord) or
                ``'triangular'`` (max at max coord).
            tolerance: Node-selection tolerance multiplier (× mesh_size, mm).
        """
        self.location = location
        self.force = force
        self.direction = direction
        self.axis = axis.lower()
        self.profile = profile
        self.tolerance = tolerance

    def apply(self, model, nodes, elements, geometry_info, mesh_size: float):
        """Apply linearly distributed load to the model."""
        from ...utils import find_nodes_in_box
        import torch

        bbox = geometry_info["bounding_box"]
        load_nodes = None

        if isinstance(self.location, dict):
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
        elif isinstance(self.location, str):
            loc = self.location.lower()
            if loc in ["x_min", "face_left"]:
                load_nodes = find_nodes_in_box(
                    nodes, xmin=bbox["xmin"], xmax=bbox["xmin"],
                    tolerance=self.tolerance * mesh_size,
                )
            elif loc in ["x_max", "face_right"]:
                load_nodes = find_nodes_in_box(
                    nodes, xmin=bbox["xmax"], xmax=bbox["xmax"],
                    tolerance=self.tolerance * mesh_size,
                )
            elif loc in ["y_min"]:
                load_nodes = find_nodes_in_box(
                    nodes, ymin=bbox["ymin"], ymax=bbox["ymin"],
                    tolerance=self.tolerance * mesh_size,
                )
            elif loc in ["face_front", "y_max"]:
                load_nodes = find_nodes_in_box(
                    nodes, ymin=bbox["ymax"], ymax=bbox["ymax"],
                    tolerance=self.tolerance * mesh_size,
                )
            elif loc in ["z_min", "bottom"]:
                load_nodes = find_nodes_in_box(
                    nodes, zmin=bbox["zmin"], zmax=bbox["zmin"],
                    tolerance=self.tolerance * mesh_size,
                )
            elif loc in ["z_max", "top"]:
                load_nodes = find_nodes_in_box(
                    nodes, zmin=bbox["zmax"], zmax=bbox["zmax"],
                    tolerance=self.tolerance * mesh_size,
                )
            else:
                pass  # load_nodes stays None

        if load_nodes is None or len(load_nodes) == 0:
            print(f"Warning: No nodes found for linear distributed load at: {self.location}")
            return 0

        selected_xyz = nodes[load_nodes]
        axis_idx = {"x": 0, "y": 1, "z": 2}[self.axis]
        coords = selected_xyz[:, axis_idx]

        c_min = coords.min()
        c_max = coords.max()
        range_val = c_max - c_min

        if range_val < 1e-6:
            weights = torch.ones_like(coords)
        elif self.profile == "triangular_invert":
            weights = c_max - coords
        else:
            weights = coords - c_min

        total_weight = weights.sum()
        if total_weight == 0:
            return 0

        normalized_weights = weights / total_weight

        dir_map = {"x": 0, "y": 1, "z": 2, "-x": 0, "-y": 1, "-z": 2}
        sign = -1 if self.direction.startswith("-") else 1
        dir_idx = dir_map[self.direction.lower()]
        force_values = float(self.force) * sign * normalized_weights
        model.forces[load_nodes, dir_idx] += force_values

        return len(load_nodes)
