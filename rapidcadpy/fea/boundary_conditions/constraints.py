"""
Structural constraints (boundary conditions) for FEA analysis.

This module provides the :class:`BoundaryCondition` abstract base class and
concrete constraint implementations:

- :class:`FixedConstraint`      — zero-displacement at a surface or point.
- :class:`CylindricalConstraint` — fix nodes on a cylindrical bore/surface.
- :class:`PinnedConstraint`     — alias for a fully-fixed constraint.
- :class:`RollerConstraint`     — constrain a single translational DOF.
"""

from abc import ABC, abstractmethod
from typing import Optional, Tuple, Union, Literal, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ....cad_types import Vector


class BoundaryCondition(ABC):
    """Abstract base class for all structural constraints."""

    @abstractmethod
    def apply(self, model, nodes, elements, geometry_info):
        """
        Apply this boundary condition to the FEA model.

        Args:
            model: FEA solver model object.
            nodes: Mesh nodes tensor (n_nodes, 3).
            elements: Mesh elements tensor (n_elements, nodes_per_element).
            geometry_info: Dict with ``bounding_box`` and geometry information.
        """
        pass


class FixedConstraint(BoundaryCondition):
    """
    Fix nodes in specified degrees of freedom.

    Prevents all movement (translations) at the specified location by setting
    the corresponding rows of ``model.constraints`` to True.
    """

    def __init__(
        self,
        location: Union[str, dict, Tuple[float, float, float]],
        dofs: Tuple[bool, bool, bool] = (True, True, True),
        tolerance: float = 1,
        node_coords: Optional[np.ndarray] = None,
    ):
        """
        Create a fixed constraint.

        Args:
            location: One of:

                - A string identifier: ``'end_1'``/``'x_min'``, ``'end_2'``/``'x_max'``,
                  ``'y_min'``, ``'y_max'``, ``'bottom'``/``'z_min'``, ``'top'``/``'z_max'``.
                - A bounding-box dict with ``x_min``, ``x_max``, ``y_min``, ``y_max``,
                  ``z_min``, ``z_max`` keys (any subset).
                - An ``(x, y, z)`` coordinate tuple for a point constraint.

            dofs: Which translational DOFs to constrain — ``(X, Y, Z)``.
            tolerance: Node-selection tolerance multiplier (× mesh_size, mm).
            node_coords: Optional ``(N, 3)`` array of *exact* constraint node
                coordinates.  When supplied the bounding-box ``location`` is
                ignored and instead the nearest mesh node to each supplied
                coordinate is selected (within ``tolerance × mesh_size``).
                This produces precise results when the constraint comes from a
                named node-set (NSET) whose members are spatially scattered.
        """
        self.location = location
        self.dofs = dofs
        self.tolerance = tolerance
        self.node_coords = node_coords

    def __repr__(self) -> str:
        return (
            f"FixedConstraint(location={self.location!r}, "
            f"dofs={self.dofs!r}, tolerance={self.tolerance!r})"
        )

    def apply(self, model, nodes, elements, geometry_info, mesh_size: float):
        """Apply fixed constraint to the model."""
        from ..utils import find_nodes_in_box
        import torch  # noqa: F401 — kept for symmetry; torch ops via find_nodes_in_box

        bbox = geometry_info["bounding_box"]

        # ── Fast-path: exact node coordinates supplied (e.g. from NSET) ──────
        if self.node_coords is not None and len(self.node_coords) > 0:
            import torch as _torch

            ref = _torch.tensor(
                self.node_coords, dtype=nodes.dtype, device=nodes.device
            )  # (M, 3)
            # For each reference point find the closest mesh node within tolerance
            tol = self.tolerance * mesh_size
            matched: list = []
            for pt in ref:
                dists = _torch.norm(nodes - pt, dim=1)
                closest_idx = int(dists.argmin().item())
                if float(dists[closest_idx].item()) <= tol:
                    matched.append(closest_idx)
                else:
                    # Tolerance too tight — fall back to search within 2× tol
                    within = (dists <= 2 * tol).nonzero(as_tuple=False).squeeze(1)
                    if within.numel() > 0:
                        matched.append(int(within[0].item()))

            if matched:
                constrained_nodes = _torch.tensor(
                    list(dict.fromkeys(matched)),  # unique, order-preserving
                    dtype=_torch.long,
                    device=nodes.device,
                )
                for i, constrain in enumerate(self.dofs):
                    if constrain:
                        model.constraints[constrained_nodes, i] = True
                return len(constrained_nodes)
            # Fall through to bounding-box if nothing matched
        # ── Bounding-box / string / point path (original logic) ───────────────
        if isinstance(self.location, dict):
            constrained_nodes = find_nodes_in_box(
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
            if loc in ["end_1", "x_min"]:
                constrained_nodes = find_nodes_in_box(
                    nodes,
                    xmin=bbox["xmin"],
                    xmax=bbox["xmin"],
                    tolerance=self.tolerance * mesh_size,
                )
            elif loc in ["end_2", "x_max"]:
                constrained_nodes = find_nodes_in_box(
                    nodes,
                    xmin=bbox["xmax"],
                    xmax=bbox["xmax"],
                    tolerance=self.tolerance * mesh_size,
                )
            elif loc in ["y_min"]:
                constrained_nodes = find_nodes_in_box(
                    nodes,
                    ymin=bbox["ymin"],
                    ymax=bbox["ymin"],
                    tolerance=self.tolerance * mesh_size,
                )
            elif loc in ["y_max"]:
                constrained_nodes = find_nodes_in_box(
                    nodes,
                    ymin=bbox["ymax"],
                    ymax=bbox["ymax"],
                    tolerance=self.tolerance * mesh_size,
                )
            elif loc in ["bottom", "z_min"]:
                constrained_nodes = find_nodes_in_box(
                    nodes,
                    zmin=bbox["zmin"],
                    zmax=bbox["zmin"],
                    tolerance=self.tolerance * mesh_size,
                )
            elif loc in ["top", "z_max"]:
                constrained_nodes = find_nodes_in_box(
                    nodes,
                    zmin=bbox["zmax"],
                    zmax=bbox["zmax"],
                    tolerance=self.tolerance * mesh_size,
                )
            else:
                raise ValueError(f"Unknown location: {self.location!r}")
        else:
            x, y, z = self.location
            constrained_nodes = find_nodes_in_box(
                nodes,
                xmin=x,
                xmax=x,
                ymin=y,
                ymax=y,
                zmin=z,
                zmax=z,
                tolerance=self.tolerance,
            )

        for i, constrain in enumerate(self.dofs):
            if constrain:
                model.constraints[constrained_nodes, i] = True

        return len(constrained_nodes)


class CylindricalConstraint(BoundaryCondition):
    """
    Fix nodes on a cylindrical surface — bolt holes, pins, bearings, bushings.

    Selects all nodes within ``radius`` of the cylinder axis over the full
    part extent along the normal axis, then constrains the specified DOFs.
    """

    def __init__(
        self,
        center: Tuple[float, float, float],
        radius: float,
        normal_axis: Literal["x", "y", "z"] = "z",
        dofs: Tuple[bool, bool, bool] = (True, True, True),
        tolerance: float = 1,
    ):
        """
        Create a cylindrical constraint.

        Args:
            center: ``(x, y, z)`` of the cylinder axis centre point (mm).
            radius: Cylinder bore radius (mm).
            normal_axis: Axis along which the cylinder extends (``'x'``, ``'y'``, ``'z'``).
            dofs: DOFs to constrain — ``(X, Y, Z)``.
            tolerance: Node-selection tolerance multiplier (× mesh_size, mm).
        """
        self.center = center
        self.radius = radius
        self.normal_axis = normal_axis.lower()
        self.dofs = dofs
        self.tolerance = tolerance

    def apply(self, model, nodes, elements, geometry_info, mesh_size: float):
        """Apply cylindrical constraint to the model."""
        from ..utils import find_nodes_in_box
        import torch

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
            nodes,
            **bbox_args,
            tolerance=self.tolerance * mesh_size,
        )

        if len(candidate_nodes) == 0:
            print(
                f"Warning: No nodes found near cylindrical constraint center: {self.center}"
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
        constrained_nodes = candidate_nodes[within_radius]

        if len(constrained_nodes) == 0:
            print(
                f"Warning: No nodes found within cylindrical radius {self.radius} at {self.center}"
            )
            return 0

        for i, constrain in enumerate(self.dofs):
            if constrain:
                model.constraints[constrained_nodes, i] = True

        return len(constrained_nodes)


class PinnedConstraint(BoundaryCondition):
    """
    Pin support — restricts all translations; delegates to :class:`FixedConstraint`.
    """

    def __init__(
        self,
        location: Union[str, Tuple[float, float, float]],
        tolerance: float = 1,
    ):
        """
        Args:
            location: Location identifier or coordinates (same as :class:`FixedConstraint`).
            tolerance: Node-selection tolerance (mm).
        """
        self.location = location
        self.tolerance = tolerance
        self.dofs = (True, True, True)

    def apply(self, model, nodes, elements, geometry_info, mesh_size: float = 1.0):
        """Apply pinned constraint."""
        fixed = FixedConstraint(self.location, self.dofs, self.tolerance)
        return fixed.apply(model, nodes, elements, geometry_info, mesh_size)


class RollerConstraint(BoundaryCondition):
    """
    Roller support — constrains a single translational DOF, allowing free
    sliding in the other two directions.
    """

    def __init__(
        self,
        location: Union[str, Tuple[float, float, float]],
        direction: Literal["x", "y", "z"] = "z",
        tolerance: float = 1,
    ):
        """
        Args:
            location: Location identifier or coordinates.
            direction: Axis to constrain (``'x'``, ``'y'``, or ``'z'``).
            tolerance: Node-selection tolerance (mm).
        """
        self.location = location
        self.direction = direction.lower()
        self.tolerance = tolerance

    def apply(self, model, nodes, elements, geometry_info, mesh_size: float = 1.0):
        """Apply roller constraint."""
        dof_map = {"x": 0, "y": 1, "z": 2}
        dof_idx = dof_map[self.direction]
        dofs = [False, False, False]
        dofs[dof_idx] = True
        fixed = FixedConstraint(self.location, tuple(dofs), self.tolerance)
        return fixed.apply(model, nodes, elements, geometry_info, mesh_size)
