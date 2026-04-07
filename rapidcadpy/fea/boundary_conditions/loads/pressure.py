"""
PressureLoad — radial pressure on a circular surface (bore or pin face).
"""

from typing import Tuple, Literal

from .base import Load


class PressureLoad(Load):
    """
    Radial pressure load on a circular bore or face.

    Selects all nodes within ``radius`` of ``center`` on the plane perpendicular
    to ``normal_axis``, then applies radially directed forces whose magnitude
    equals ``pressure × (πr²) / n_nodes``.

    Common uses:

    - Hydraulic cylinder pressure on a circular piston face.
    - Bolt-hole bearing load (pin pushing on a cylindrical bore).
    - Circular die or punch contact.
    """

    def __init__(
        self,
        center: Tuple[float, float, float],
        radius: float,
        pressure: float,
        direction: Literal["inward", "outward"] = "inward",
        normal_axis: Literal["x", "y", "z"] = "z",
        tolerance: float = 1,
    ):
        """
        Args:
            center: ``(x, y, z)`` of the circular area centre (mm).
            radius: Radius of the circular area (mm).
            pressure: Pressure magnitude (MPa = N/mm²).
            direction: ``'inward'`` (compression, toward centre) or
                ``'outward'`` (tension, away from centre).
            normal_axis: Axis perpendicular to the face (``'x'``, ``'y'``, ``'z'``).
            tolerance: Node-selection tolerance multiplier (× mesh_size, mm).
        """
        self.center = center
        self.radius = radius
        self.pressure = pressure
        self.direction = direction.lower()
        self.normal_axis = normal_axis.lower()
        self.tolerance = tolerance

    def apply(self, model, nodes, elements, geometry_info, mesh_size: float):
        """Apply circular pressure load to the model."""
        from ...utils import find_nodes_in_box
        import torch
        import math

        cx, cy, cz = self.center
        bbox = geometry_info["bounding_box"]

        axis_map = {"x": 0, "y": 1, "z": 2}
        normal_idx = axis_map[self.normal_axis]

        bbox_args = {}
        for i, coord in enumerate([cx, cy, cz]):
            axis_name = ["x", "y", "z"][i]
            if i == normal_idx:
                bbox_args[f"{axis_name}min"] = bbox[f"{axis_name}min"]
                bbox_args[f"{axis_name}max"] = bbox[f"{axis_name}max"]
            else:
                bbox_args[f"{axis_name}min"] = coord - self.radius
                bbox_args[f"{axis_name}max"] = coord + self.radius

        candidate_nodes = find_nodes_in_box(
            nodes, **bbox_args, tolerance=self.tolerance * mesh_size
        )

        if len(candidate_nodes) == 0:
            print(
                f"Warning: No nodes found near circular pressure centre: {self.center}"
            )
            return 0

        candidate_positions = nodes[candidate_nodes]
        center_tensor = torch.tensor(
            self.center, dtype=nodes.dtype, device=nodes.device
        )

        plane_diff = candidate_positions - center_tensor
        plane_diff[:, normal_idx] = 0
        distances = torch.norm(plane_diff, dim=1)
        within_radius = distances <= self.radius
        load_nodes = candidate_nodes[within_radius]

        if len(load_nodes) == 0:
            print(f"Warning: No nodes within radius {self.radius} at {self.center}")
            return 0

        load_positions = nodes[load_nodes]
        radial_vectors = load_positions - center_tensor
        radial_vectors[:, normal_idx] = 0

        radial_distances = torch.norm(radial_vectors, dim=1, keepdim=True)
        radial_distances = torch.clamp(radial_distances, min=1e-9)
        radial_directions = radial_vectors / radial_distances

        total_area = math.pi * self.radius * self.radius
        area_per_node = total_area / len(load_nodes)
        force_magnitude = self.pressure * area_per_node

        if self.direction == "inward":
            force_magnitude = -force_magnitude
        elif self.direction != "outward":
            raise ValueError(
                f"Invalid direction: {self.direction!r}. Use 'inward' or 'outward'."
            )

        force_vectors = force_magnitude * radial_directions
        model.forces[load_nodes, 0] += force_vectors[:, 0]
        model.forces[load_nodes, 1] += force_vectors[:, 1]
        model.forces[load_nodes, 2] += force_vectors[:, 2]

        return len(load_nodes)
