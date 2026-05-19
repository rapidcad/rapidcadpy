"""
MomentLoad — applied torque via equivalent force couple on a circular cross-section.
"""

from typing import Tuple, Literal, Optional

from .base import Load


class MomentLoad(Load):
    """
    Applied torque (moment) about a specified axis using a force couple.

    Selects nodes on a circular cross-section perpendicular to the axis at a
    given center location, then applies equal and opposite tangential forces
    to create the desired moment:

    .. math::
        M = F \\times 2r

    where *F* is the force magnitude and *r* is the radius of the cross-section.

    Common uses:

    - Torsional loading on shafts and rods.
    - Twisting moments on beam structures.
    - Motor/propeller torque application on mounts.
    """

    def __init__(
        self,
        center: Tuple[float, float, float],
        axis: Literal["x", "y", "z"],
        radius: float,
        magnitude_nm: float,
        direction: Literal["cw", "ccw"] = "cw",
        tolerance: float = 1,
    ):
        """
        Args:
            center: ``(x, y, z)`` center of the circular cross-section (mm).
            axis: Rotation axis (``'x'``, ``'y'``, or ``'z'``).
            radius: Radius of the circular cross-section (mm).
            magnitude_nm: Torque magnitude (N·mm).
            direction: ``'cw'`` (clockwise looking along +axis) or ``'ccw'``.
            tolerance: Node-selection tolerance multiplier (× mesh_size, mm).
        """
        self.center = tuple(float(v) for v in center)
        self.axis = axis.lower()
        self.radius = float(radius)
        self.magnitude_nm = float(magnitude_nm)
        self.direction = direction.lower()
        self.tolerance = tolerance

        if self.axis not in ("x", "y", "z"):
            raise ValueError(f"axis must be 'x', 'y', or 'z', got {self.axis!r}")
        if self.direction not in ("cw", "ccw"):
            raise ValueError(f"direction must be 'cw' or 'ccw', got {self.direction!r}")

    def __repr__(self) -> str:
        return (
            f"MomentLoad(center={self.center}, axis={self.axis!r}, "
            f"radius={self.radius}, magnitude_nm={self.magnitude_nm}, "
            f"direction={self.direction!r})"
        )

    def apply(self, model, nodes, elements, geometry_info, mesh_size: float):
        """Apply torque as a force couple to the model."""
        from ...utils import find_nodes_in_box
        import torch
        import math

        cx, cy, cz = self.center
        bbox = geometry_info["bounding_box"]

        axis_map = {"x": 0, "y": 1, "z": 2}
        axis_idx = axis_map[self.axis]

        # Build bounding box: cross-section perpendicular to axis
        bbox_args = {}
        for i, coord in enumerate([cx, cy, cz]):
            axis_name = ["x", "y", "z"][i]
            if i == axis_idx:
                # Search along the axis (thick slice at center)
                bbox_args[f"{axis_name}min"] = bbox[f"{axis_name}min"]
                bbox_args[f"{axis_name}max"] = bbox[f"{axis_name}max"]
            else:
                # Search perpendicular to axis within radius
                bbox_args[f"{axis_name}min"] = coord - self.radius
                bbox_args[f"{axis_name}max"] = coord + self.radius

        candidate_nodes = find_nodes_in_box(
            nodes, **bbox_args, tolerance=self.tolerance * mesh_size
        )

        if len(candidate_nodes) == 0:
            print(f"Warning: No nodes found for moment load at center: {self.center}")
            return 0

        candidate_positions = nodes[candidate_nodes]
        center_tensor = torch.tensor(
            self.center, dtype=nodes.dtype, device=nodes.device
        )

        # Filter to nodes within radius of axis
        plane_diff = candidate_positions - center_tensor
        plane_diff[:, axis_idx] = 0  # Project onto perpendicular plane
        distances = torch.norm(plane_diff, dim=1)
        within_radius = distances <= self.radius
        load_nodes = candidate_nodes[within_radius]

        if len(load_nodes) == 0:
            print(f"Warning: No nodes within radius {self.radius} for moment load")
            return 0

        load_positions = nodes[load_nodes]

        # Compute tangential force direction for each node
        # Create two opposite force lines to form a couple
        radial_vectors = load_positions - center_tensor
        radial_vectors[:, axis_idx] = 0  # Project onto perpendicular plane

        radial_distances = torch.norm(radial_vectors, dim=1, keepdim=True)
        radial_distances = torch.clamp(radial_distances, min=1e-9)
        radial_directions = radial_vectors / radial_distances

        # Tangential direction: perpendicular to both axis and radial vector
        # For z-axis: tangent = (-radial_y, radial_x, 0)
        # For x-axis: tangent = (0, -radial_z, radial_y)
        # For y-axis: tangent = (radial_z, 0, -radial_x)
        tangent_directions = torch.zeros_like(radial_vectors)

        if self.axis == "z":
            tangent_directions[:, 0] = -radial_directions[:, 1]
            tangent_directions[:, 1] = radial_directions[:, 0]
        elif self.axis == "x":
            tangent_directions[:, 1] = -radial_directions[:, 2]
            tangent_directions[:, 2] = radial_directions[:, 1]
        elif self.axis == "y":
            tangent_directions[:, 0] = radial_directions[:, 2]
            tangent_directions[:, 2] = -radial_directions[:, 0]

        # Apply direction (cw/ccw)
        if self.direction == "ccw":
            tangent_directions = -tangent_directions

        # Distribute total moment among nodes
        # M = F * 2r, so F_per_node = M / (2r * n_nodes)
        force_per_node = self.magnitude_nm / (2.0 * self.radius * len(load_nodes))

        force_vectors = force_per_node * tangent_directions
        model.forces[load_nodes, 0] += force_vectors[:, 0]
        model.forces[load_nodes, 1] += force_vectors[:, 1]
        model.forces[load_nodes, 2] += force_vectors[:, 2]

        return len(load_nodes)
