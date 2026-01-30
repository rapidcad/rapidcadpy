"""
Boundary conditions for FEA analysis.

This module provides loads and constraints that can be applied to FEA models.
"""

from abc import ABC, abstractmethod
from typing import Tuple, Optional, Union, Literal, TYPE_CHECKING, Dict, Any

if TYPE_CHECKING:
    from ...cad_types import Vector


def visualize_boundary_conditions(
    model, nodes, elements, window_size=(1400, 700), interactive=True
):
    """
    Visualize boundary conditions (constraints and loads) on a mesh.

    This function creates an interactive 3D visualization showing:
    - The mesh in semi-transparent blue
    - Fixed/constrained nodes as red spheres
    - Loaded nodes as green spheres
    - Force vectors as green arrows

    Args:
        model: FEA solver model object with constraints and forces attributes
        nodes: Mesh nodes tensor (n_nodes, 3)
        elements: Mesh elements tensor (n_elements, nodes_per_element)
        window_size: Window size as (width, height). Default: (1400, 700)
        jupyter_backend: PyVista backend for Jupyter notebooks ('static', 'panel', etc.)
                        Use None (default) for interactive window, 'static' for Jupyter notebooks
        filename: If provided, save the visualization to this file path (e.g., 'boundary_conditions.png')

    Returns:
        str: Path to saved file if filename is provided, otherwise None

    Example:
        >>> from .fea.boundary_conditions import visualize_boundary_conditions
        >>> # After setting up model with constraints and forces
        >>> visualize_boundary_conditions(model, nodes, elements)

        >>> # For Jupyter notebooks
        >>> visualize_boundary_conditions(model, nodes, elements, jupyter_backend='static')

        >>> # Save to file
        >>> visualize_boundary_conditions(model, nodes, elements, filename='bc_plot.png')
    """
    try:
        import pyvista as pv
        import numpy as np
    except ImportError:
        raise ImportError(
            "PyVista is required for boundary condition visualization. "
            "Install it with: pip install pyvista"
        )

    # Create PyVista mesh for visualization
    points = nodes.cpu().numpy()
    cells = elements.cpu().numpy()

    # Create VTK cells (prepend count of nodes per element)
    nodes_per_elem = cells.shape[1]
    vtk_cells = np.column_stack([np.full(len(cells), nodes_per_elem), cells]).ravel()

    # Determine cell type based on nodes per element
    if nodes_per_elem == 4:
        celltypes = np.full(len(cells), pv.CellType.TETRA)
    elif nodes_per_elem == 8:
        celltypes = np.full(len(cells), pv.CellType.HEXAHEDRON)
    elif nodes_per_elem == 10:
        celltypes = np.full(len(cells), pv.CellType.QUADRATIC_TETRA)
    elif nodes_per_elem == 20:
        celltypes = np.full(len(cells), pv.CellType.QUADRATIC_HEXAHEDRON)
    else:
        raise ValueError(f"Unsupported element type with {nodes_per_elem} nodes")

    pv_mesh = pv.UnstructuredGrid(vtk_cells, celltypes, points)

    # Create plotter - use off_screen if saving to file
    off_screen = not interactive
    plotter = pv.Plotter(window_size=window_size, off_screen=off_screen)

    # Add the main mesh (semi-transparent)
    plotter.add_mesh(
        pv_mesh,
        color="lightblue",
        opacity=0.3,
        show_edges=True,
        edge_color="gray",
        line_width=0.5,
    )

    # Visualize FIXED NODES (constraints)
    constrained_mask = model.constraints.any(dim=1).cpu().numpy()
    constrained_nodes = nodes[constrained_mask].cpu().numpy()

    if len(constrained_nodes) > 0:
        # Add fixed nodes as red spheres
        fixed_points = pv.PolyData(constrained_nodes)
        plotter.add_mesh(
            fixed_points,
            color="red",
            point_size=15,
            render_points_as_spheres=True,
            label="Fixed Nodes",
        )
        print(f"✓ Visualizing {len(constrained_nodes)} constrained nodes (RED)")

    # Visualize LOADED NODES (forces)
    force_mask = (model.forces.abs() > 1e-10).any(dim=1).cpu().numpy()
    loaded_nodes = nodes[force_mask].cpu().numpy()
    force_vectors = model.forces[force_mask].cpu().numpy()

    if len(loaded_nodes) > 0:
        # Add loaded nodes as green spheres
        load_points = pv.PolyData(loaded_nodes)
        plotter.add_mesh(
            load_points,
            color="green",
            point_size=15,
            render_points_as_spheres=True,
            label="Loaded Nodes",
        )

        # Add force arrows
        # Scale arrows for visibility
        arrow_scale = (nodes[:, 0].max() - nodes[:, 0].min()).item() * 0.1
        force_magnitude = np.linalg.norm(force_vectors, axis=1, keepdims=True)
        force_directions = force_vectors / (force_magnitude + 1e-10)
        scaled_vectors = force_directions * arrow_scale

        plotter.add_arrows(
            loaded_nodes,
            scaled_vectors,
            mag=1.0,
            color="darkgreen",
            label="Force Vectors",
        )
        print(
            f"✓ Visualizing {len(loaded_nodes)} loaded nodes (GREEN) with force arrows"
        )

    # Add legend and labels
    plotter.add_legend()
    plotter.add_text("Boundary Conditions", position="upper_edge", font_size=12)
    plotter.add_text(
        f"Red = Fixed (Constraints)\nGreen = Loaded (Forces)",
        position="lower_left",
        font_size=10,
    )
    plotter.add_axes()
    plotter.camera_position = "iso"

    return plotter


class BoundaryCondition(ABC):
    """Base class for all boundary conditions (constraints)"""

    @abstractmethod
    def apply(self, model, nodes, elements, geometry_info):
        """
        Apply this boundary condition to the FEA model.

        Args:
            model: FEA solver model object
            nodes: Mesh nodes tensor
            elements: Mesh elements tensor
            geometry_info: Dict with bounding box and geometry information
        """
        pass


class FixedConstraint(BoundaryCondition):
    """
    Fix nodes in specified degrees of freedom.

    Prevents all movement (translations and rotations) at the specified location.
    """

    def __init__(
        self,
        location: Union[str, Tuple[float, float, float]],
        dofs: Tuple[bool, bool, bool] = (True, True, True),
        tolerance: float = 1,
    ):
        """
        Create a fixed constraint.

        Args:
            location: Location identifier ('end_1', 'end_2', 'top', 'bottom', 'x_min',
                     'x_max', 'y_min', 'y_max', 'z_min', 'z_max') or (x, y, z) coordinates
            dofs: Degrees of freedom to constrain (X, Y, Z)
            tolerance: Distance tolerance for node selection (mm)
        """
        self.location = location
        self.dofs = dofs
        self.tolerance = tolerance

    def apply(self, model, nodes, elements, geometry_info, mesh_size: float):
        """Apply fixed constraint to the model"""
        from .utils import find_nodes_in_box
        import torch

        bbox = geometry_info["bounding_box"]

        # Determine nodes based on location
        if isinstance(self.location, dict):
            # Box selector with x_min/max, y_min/max, z_min/max
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
                raise ValueError(f"Unknown location: {self.location}")
        else:
            # Point location
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

        # Apply constraints
        for i, constrain in enumerate(self.dofs):
            if constrain:
                model.constraints[constrained_nodes, i] = True

        return len(constrained_nodes)


class CylindricalConstraint(BoundaryCondition):
    """
    Cylindrical constraint - fixes nodes on a cylindrical surface.

    Useful for:
    - Bolt holes constrained by a bolt shaft
    - Pin holes with inserted pins
    - Bearing surfaces
    - Bushings and sleeves
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
            center: (x, y, z) coordinates of the cylinder center
            radius: Radius of the cylindrical surface (mm)
            normal_axis: Axis along which the cylinder extends ('x', 'y', or 'z')
            dofs: Degrees of freedom to constrain (X, Y, Z)
            tolerance: Distance tolerance for node selection (mm)
        """
        self.center = center
        self.radius = radius
        self.normal_axis = normal_axis.lower()
        self.dofs = dofs
        self.tolerance = tolerance

    def apply(self, model, nodes, elements, geometry_info, mesh_size: float):
        """Apply cylindrical constraint to the model"""
        from .utils import find_nodes_in_box
        import torch

        cx, cy, cz = self.center
        bbox = geometry_info["bounding_box"]

        # Determine which axes form the circular plane
        axis_map = {"x": 0, "y": 1, "z": 2}
        normal_idx = axis_map[self.normal_axis]

        # For cylindrical constraints, we want nodes along the entire
        # length of the cylinder, not just at one plane. Use part bbox for normal axis.
        bbox_args = {}
        for i, coord in enumerate([cx, cy, cz]):
            axis_name = ["x", "y", "z"][i]
            if i == normal_idx:
                # Use full part extent along cylinder axis
                bbox_args[f"{axis_name}min"] = bbox[f"{axis_name}min"]
                bbox_args[f"{axis_name}max"] = bbox[f"{axis_name}max"]
            else:
                # Constrain other axes to cylinder center ± radius
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

        # Filter nodes within circular radius
        candidate_positions = nodes[candidate_nodes]

        # Calculate distance from center in the circular plane
        center_tensor = torch.tensor(
            self.center, dtype=nodes.dtype, device=nodes.device
        )

        # Distance in the plane perpendicular to normal axis
        plane_diff = candidate_positions - center_tensor
        plane_diff[:, normal_idx] = 0  # Ignore normal axis component

        distances = torch.norm(plane_diff, dim=1)

        # Select nodes within radius
        within_radius = distances <= self.radius
        constrained_nodes = candidate_nodes[within_radius]

        if len(constrained_nodes) == 0:
            print(
                f"Warning: No nodes found within cylindrical radius {self.radius} at {self.center}"
            )
            return 0

        # Apply constraints
        for i, constrain in enumerate(self.dofs):
            if constrain:
                model.constraints[constrained_nodes, i] = True

        return len(constrained_nodes)


class PinnedConstraint(BoundaryCondition):
    """
    Pin support - restricts all translations but allows rotations.

    Equivalent to FixedConstraint with all DOFs constrained.
    """

    def __init__(
        self, location: Union[str, Tuple[float, float, float]], tolerance: float = 1
    ):
        """
        Create a pinned constraint.

        Args:
            location: Location identifier or coordinates
            tolerance: Distance tolerance for node selection (mm)
        """
        self.location = location
        self.tolerance = tolerance
        self.dofs = (True, True, True)

    def apply(self, model, nodes, elements, geometry_info):
        """Apply pinned constraint"""
        fixed = FixedConstraint(self.location, self.dofs, self.tolerance)
        return fixed.apply(model, nodes, elements, geometry_info)


class RollerConstraint(BoundaryCondition):
    """
    Roller support - restricts movement in one direction only.

    Allows sliding in tangential directions while preventing normal movement.
    """

    def __init__(
        self,
        location: Union[str, Tuple[float, float, float]],
        direction: Literal["x", "y", "z"] = "z",
        tolerance: float = 1,
    ):
        """
        Create a roller constraint.

        Args:
            location: Location identifier or coordinates
            direction: Direction to constrain ('x', 'y', or 'z')
            tolerance: Distance tolerance for node selection (mm)
        """
        self.location = location
        self.direction = direction.lower()
        self.tolerance = tolerance

    def apply(self, model, nodes, elements, geometry_info):
        """Apply roller constraint"""
        # Constrain only the specified direction
        dof_map = {"x": 0, "y": 1, "z": 2}
        dof_idx = dof_map[self.direction]

        dofs = [False, False, False]
        dofs[dof_idx] = True

        fixed = FixedConstraint(self.location, tuple(dofs), self.tolerance)
        return fixed.apply(model, nodes, elements, geometry_info)


class Load(ABC):
    """Base class for all loads"""

    @abstractmethod
    def apply(self, model, nodes, elements, geometry_info):
        """
        Apply this load to the FEA model.

        Args:
            model: FEA solver model object
            nodes: Mesh nodes tensor
            elements: Mesh elements tensor
            geometry_info: Dict with bounding box and geometry information
        """
        pass


class DistributedLoad(Load):
    """
    Distributed load over a surface.

    Applies a total force distributed evenly across all nodes on the specified surface.
    """

    def __init__(
        self,
        location: str,
        force: Union[float, Tuple[float, float, float]],
        direction: Optional[Literal["x", "y", "z", "normal"]] = None,
        tolerance: float = 1,
    ):
        """
        Create a distributed load.

        Args:
            location: Surface identifier ('top', 'bottom', 'x_min', 'x_max', etc.)
            force: Total force magnitude (N) or force vector (Fx, Fy, Fz)
            direction: Force direction ('x', 'y', 'z', 'normal'). If None and force is scalar,
                      defaults to normal direction of the surface
            tolerance: Distance tolerance for node selection (mm)
        """
        self.location = location
        self.force = force
        self.direction = direction
        self.tolerance = tolerance

    def apply(self, model, nodes, elements, geometry_info, mesh_size: float):
        """Apply distributed load to the model"""
        from .utils import find_nodes_in_box
        import torch

        bbox = geometry_info["bounding_box"]

        # Handle dict location (box selector)
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
            # Determine default direction from which dimension is thin
            x_range = abs(self.location.get("x_max", 0) - self.location.get("x_min", 0))
            y_range = abs(self.location.get("y_max", 0) - self.location.get("y_min", 0))
            z_range = abs(self.location.get("z_max", 0) - self.location.get("z_min", 0))

            # Default direction is along the thinnest dimension
            if z_range < x_range and z_range < y_range:
                default_dir = "z"
            elif y_range < x_range:
                default_dir = "y"
            else:
                default_dir = "x"

        elif isinstance(self.location, str):
            loc = self.location.lower()

            # Find nodes on the specified surface
            if loc in ["top", "z_max"]:
                load_nodes = find_nodes_in_box(
                    nodes,
                    zmin=bbox["zmax"],
                    zmax=bbox["zmax"],
                    tolerance=self.tolerance * mesh_size,
                )
                default_dir = "z"
            elif loc in ["bottom", "z_min"]:
                load_nodes = find_nodes_in_box(
                    nodes,
                    zmin=bbox["zmin"],
                    zmax=bbox["zmin"],
                    tolerance=self.tolerance * mesh_size,
                )
                default_dir = "z"
            elif loc in ["x_min"]:
                load_nodes = find_nodes_in_box(
                    nodes,
                    xmin=bbox["xmin"],
                    xmax=bbox["xmin"],
                    tolerance=self.tolerance * mesh_size,
                )
                default_dir = "x"
            elif loc in ["x_max"]:
                load_nodes = find_nodes_in_box(
                    nodes,
                    xmin=bbox["xmax"],
                    xmax=bbox["xmax"],
                    tolerance=self.tolerance * mesh_size,
                )
                default_dir = "x"
            elif loc in ["y_min"]:
                load_nodes = find_nodes_in_box(
                    nodes,
                    ymin=bbox["ymin"],
                    ymax=bbox["ymin"],
                    tolerance=self.tolerance * mesh_size,
                )
                default_dir = "y"
            elif loc in ["y_max"]:
                load_nodes = find_nodes_in_box(
                    nodes,
                    ymin=bbox["ymax"],
                    ymax=bbox["ymax"],
                    tolerance=self.tolerance * mesh_size,
                )
                default_dir = "y"
            else:
                raise ValueError(f"Unknown location: {self.location}")
        else:
            raise ValueError(f"Unsupported location type: {type(self.location)}")

        if len(load_nodes) == 0:
            raise ValueError(f"No nodes found at location: {self.location}")

        # Calculate force per node
        if isinstance(self.force, (int, float)):
            # Scalar force - apply in specified or default direction
            direction = self.direction or default_dir
            force_per_node = self.force / len(load_nodes)

            dir_map = {
                "x": 0,
                "y": 1,
                "z": 2,
                "-x": 0,
                "-y": 1,
                "-z": 2,
                "+x": 0,
                "+y": 1,
                "+z": 2,
            }
            direction = direction.lower()

            if direction not in dir_map:
                raise ValueError(f"Invalid direction: {direction}")

            dir_idx = dir_map[direction]
            sign = -1 if direction.startswith("-") else 1

            model.forces[load_nodes, dir_idx] = sign * force_per_node
        else:
            # Vector force
            fx, fy, fz = self.force
            force_per_node_x = fx / len(load_nodes)
            force_per_node_y = fy / len(load_nodes)
            force_per_node_z = fz / len(load_nodes)

            model.forces[load_nodes, 0] = force_per_node_x
            model.forces[load_nodes, 1] = force_per_node_y
            model.forces[load_nodes, 2] = force_per_node_z

        return len(load_nodes)


class PointLoad(Load):
    """
    Concentrated load at a point.

    Applies a force to the nearest node(s) to the specified point.
    """

    def __init__(
        self,
        point: Tuple[float, float, float],
        force: Union[float, Tuple[float, float, float]],
        direction: Optional[Literal["x", "y", "z"]] = None,
        tolerance: float = 1,
    ):
        """
        Create a point load.

        Args:
            point: (x, y, z) coordinates of load application point
            force: Force magnitude (N) or force vector (Fx, Fy, Fz)
            direction: Force direction if force is scalar ('x', 'y', 'z')
            tolerance: Distance tolerance for node selection (mm)
        """
        self.point = point
        self.force = force
        self.direction = direction
        self.tolerance = tolerance

    def __str__(self):
        return (
            f"PointLoad(point={self.point}, force={self.force}, "
            f"direction={self.direction}, tolerance={self.tolerance})"
        )

    def apply(self, model, nodes, elements, geometry_info, mesh_size):
        """Apply point load to the model"""
        from .utils import find_nodes_in_box

        x, y, z = self.point

        # Find nodes near the point
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

        # Apply force
        if isinstance(self.force, (int, float)):
            # Scalar force
            if self.direction is None:
                raise ValueError("direction must be specified for scalar force")

            dir_map = {"x": 0, "y": 1, "z": 2}
            dir_idx = dir_map[self.direction.lower()]

            # Distribute force among found nodes
            force_per_node = self.force / len(load_nodes)
            model.forces[load_nodes, dir_idx] = force_per_node
        else:
            # Vector force
            fx, fy, fz = self.force
            force_per_node_x = fx / len(load_nodes)
            force_per_node_y = fy / len(load_nodes)
            force_per_node_z = fz / len(load_nodes)

            model.forces[load_nodes, 0] = force_per_node_x
            model.forces[load_nodes, 1] = force_per_node_y
            model.forces[load_nodes, 2] = force_per_node_z

        return len(load_nodes)


class PressureLoad(Load):
    """
    Circular pressure load on a circular surface.

    Applies radial pressure (inward or outward) on a circular area defined by
    center and radius. Forces are applied radially from/to the center point.

    Common use cases:
    - Hydraulic cylinder pressure
    - Bolt hole compression
    - Pin bearing loads
    - Circular die pressure
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
        Create a circular pressure load.

        Args:
            center: (x, y, z) coordinates of the circular area center
            radius: Radius of the circular area (mm)
            pressure: Pressure magnitude (MPa or N/mm²)
            direction: 'inward' (compression, toward center) or 'outward' (tension, away from center)
            normal_axis: Axis perpendicular to the circular face ('x', 'y', or 'z')
            tolerance: Distance tolerance for node selection along normal axis (mm)
        """
        self.center = center
        self.radius = radius
        self.pressure = pressure
        self.direction = direction.lower()
        self.normal_axis = normal_axis.lower()
        self.tolerance = tolerance

    def apply(self, model, nodes, elements, geometry_info, mesh_size: float):
        """Apply circular pressure load to the model"""
        from .utils import find_nodes_in_box
        import torch
        import math

        cx, cy, cz = self.center
        bbox = geometry_info["bounding_box"]

        # Determine which axes form the circular plane
        axis_map = {"x": 0, "y": 1, "z": 2}
        normal_idx = axis_map[self.normal_axis]

        # For cylindrical pressure loads (holes, pins), we want nodes along the entire
        # length of the cylinder, not just at one plane. Use part bbox for normal axis.
        bbox_args = {}
        for i, coord in enumerate([cx, cy, cz]):
            axis_name = ["x", "y", "z"][i]
            if i == normal_idx:
                # Use full part extent along cylinder axis
                bbox_args[f"{axis_name}min"] = bbox[f"{axis_name}min"]
                bbox_args[f"{axis_name}max"] = bbox[f"{axis_name}max"]
            else:
                # Constrain other axes to cylinder center ± radius
                bbox_args[f"{axis_name}min"] = coord - self.radius
                bbox_args[f"{axis_name}max"] = coord + self.radius

        candidate_nodes = find_nodes_in_box(
            nodes,
            **bbox_args,
            tolerance=self.tolerance * mesh_size,
        )

        if len(candidate_nodes) == 0:
            print(
                f"Warning: No nodes found near circular pressure center: {self.center}"
            )
            return 0

        # Filter nodes within circular radius
        candidate_positions = nodes[candidate_nodes]

        # Calculate distance from center in the circular plane
        center_tensor = torch.tensor(
            self.center, dtype=nodes.dtype, device=nodes.device
        )

        # Distance in the plane perpendicular to normal axis
        plane_diff = candidate_positions - center_tensor
        plane_diff[:, normal_idx] = 0  # Ignore normal axis component

        distances = torch.norm(plane_diff, dim=1)

        # Select nodes within radius
        within_radius = distances <= self.radius
        load_nodes = candidate_nodes[within_radius]

        if len(load_nodes) == 0:
            print(
                f"Warning: No nodes found within circular radius {self.radius} at {self.center}"
            )
            return 0

        # Calculate radial directions for each node
        load_positions = nodes[load_nodes]
        radial_vectors = load_positions - center_tensor
        radial_vectors[:, normal_idx] = 0  # Project onto plane

        # Normalize radial vectors
        radial_distances = torch.norm(radial_vectors, dim=1, keepdim=True)
        # Avoid division by zero for nodes at exact center
        radial_distances = torch.clamp(radial_distances, min=1e-9)
        radial_directions = radial_vectors / radial_distances

        # Estimate surface area element per node
        # Approximate as total circular area divided by number of nodes
        total_area = math.pi * self.radius * self.radius
        area_per_node = total_area / len(load_nodes)

        # Force per node = pressure * area_per_node
        force_magnitude = self.pressure * area_per_node

        # Apply direction (inward = toward center, outward = away from center)
        if self.direction == "inward":
            force_magnitude = -force_magnitude
        elif self.direction != "outward":
            raise ValueError(
                f"Invalid direction: {self.direction}. Use 'inward' or 'outward'"
            )

        # Apply forces in radial directions
        force_vectors = force_magnitude * radial_directions

        model.forces[load_nodes, 0] += force_vectors[:, 0]
        model.forces[load_nodes, 1] += force_vectors[:, 1]
        model.forces[load_nodes, 2] += force_vectors[:, 2]

        return len(load_nodes)


class ConcentratedLoad(Load):
    """
    Concentrated force applied directly to each node in a selection.

    Applies the full specified force magnitude to EACH node found in the location.
    This corresponds to the Abaqus *CLOAD command applied to a node or node set.
    """

    def __init__(
        self,
        location: Union[str, Dict[str, Any], Tuple[float, float, float]],
        force: Union[float, Tuple[float, float, float]],
        direction: Optional[Literal["x", "y", "z"]] = None,
        tolerance: float = 1,
    ):
        """
        Create a concentrated load.

        Args:
            location: Location identifier, box dict, or (x, y, z) coordinate
            force: Force magnitude (N) or vector (Fx, Fy, Fz) to apply to EACH node
            direction: Force direction ('x', 'y', 'z') if force is scalar
            tolerance: Distance tolerance for node selection (mm)
        """
        self.location = location
        self.force = force
        self.direction = direction
        self.tolerance = tolerance

    def apply(self, model, nodes, elements, geometry_info, mesh_size: float):
        """Apply concentrated load to the model"""
        from .utils import find_nodes_in_box

        bbox = geometry_info["bounding_box"]
        load_nodes = None

        # Determine nodes based on location
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
                raise ValueError(f"Unknown location: {self.location}")

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
                tolerance=self.tolerance,  # Use absolute tolerance for point
            )
        else:
            raise ValueError(f"Unsupported location type: {type(self.location)}")

        if load_nodes is None or len(load_nodes) == 0:
            print(f"Warning: No nodes found at location: {self.location}")
            return 0

        # Apply force to EACH node
        if isinstance(self.force, (int, float)):
            if self.direction is None:
                raise ValueError(
                    "direction must be specified for scalar force on ConcentratedLoad"
                )

            direction = self.direction.lower()
            dir_map = {"x": 0, "y": 1, "z": 2, "-x": 0, "-y": 1, "-z": 2}

            # Check for valid direction
            if direction not in dir_map:
                raise ValueError(f"Invalid direction: {direction}")

            dir_idx = dir_map[direction]
            sign = -1 if direction.startswith("-") else 1

            # Direct assignment of full force to each node
            # Use += to allow superposition of loads
            model.forces[load_nodes, dir_idx] += sign * self.force

        else:
            fx, fy, fz = self.force
            model.forces[load_nodes, 0] += fx
            model.forces[load_nodes, 1] += fy
            model.forces[load_nodes, 2] += fz

        return len(load_nodes)


class LinearDistributedLoad(Load):
    """
    Distributed load that varies linearly along an axis.
    Useful for hydrostatic pressure (triangular loads).

    The Total Force is distributed among nodes based on weights w_i.
    If profile='triangular_invert': w_i = (coord_max - coord_i)  [Max at min coord, 0 at max coord]
    If profile='triangular':        w_i = (coord_i - coord_min)  [0 at min coord, Max at max coord]
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
        Create a linearly distributed load.

        Args:
            location: Surface identifier or box dict
            force: Total force magnitude (N)
            direction: Direction of the force vector ('x', 'y', 'z', '-x', etc.)
            axis: Axis along which the load varies ('x', 'y', 'z')
            profile: 'triangular_invert' (max at min coord, e.g. deep water)
                     or 'triangular' (max at max coord).
            tolerance: Node selection tolerance
        """
        self.location = location
        self.force = force
        self.direction = direction
        self.axis = axis.lower()
        self.profile = profile
        self.tolerance = tolerance

    def apply(self, model, nodes, elements, geometry_info, mesh_size: float):
        from .utils import find_nodes_in_box
        import torch

        bbox = geometry_info["bounding_box"]
        load_nodes = None

        # Reuse finding logic (simplified for brevity, ideally share code with DistributedLoad)
        # Assuming location is a dict for custom geometry usually
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
            # ... support basic string locations if needed ...
            # For now implementing the dict path which is most common for specific faces
            # Fallback to DistributedLoad's logic could be done if refactored, but copying key parts:
            loc = self.location.lower()
            if loc in ["x_min", "face_left"]:
                load_nodes = find_nodes_in_box(
                    nodes,
                    xmin=bbox["xmin"],
                    xmax=bbox["xmin"],
                    tolerance=self.tolerance * mesh_size,
                )
            elif loc in ["x_max", "face_right"]:
                load_nodes = find_nodes_in_box(
                    nodes,
                    xmin=bbox["xmax"],
                    xmax=bbox["xmax"],
                    tolerance=self.tolerance * mesh_size,
                )
            elif loc in ["face_front", "y_max"]:
                load_nodes = find_nodes_in_box(
                    nodes,
                    ymin=bbox["ymax"],
                    ymax=bbox["ymax"],
                    tolerance=self.tolerance * mesh_size,
                )
            # ... etc ...
            else:
                # Try general surface check
                pass

            if load_nodes is None:
                # Fallback similar to DistributedLoad
                # (If user provides string location logic not fully copied here)
                # Ideally we should refactor find_nodes to a helper method on a base class or util
                pass

        if load_nodes is None or len(load_nodes) == 0:
            print(f"Warning: No nodes found for gradient load at: {self.location}")
            return 0

        # Calculate weights
        selected_xyz = nodes[load_nodes]
        axis_idx = {"x": 0, "y": 1, "z": 2}[self.axis]
        coords = selected_xyz[:, axis_idx]

        c_min = coords.min()
        c_max = coords.max()
        range_val = c_max - c_min

        if range_val < 1e-6:
            # Degenerate to uniform if no variation along axis
            weights = torch.ones_like(coords)
        else:
            if self.profile == "triangular_invert":  # High at min, 0 at max
                weights = c_max - coords
            else:  # High at max, 0 at min
                weights = coords - c_min

        # Normalize
        total_weight = weights.sum()
        if total_weight == 0:
            return 0

        normalized_weights = weights / total_weight

        # Apply forces
        dir_map = {"x": 0, "y": 1, "z": 2, "-x": 0, "-y": 1, "-z": 2}
        sign = -1 if self.direction.startswith("-") else 1
        dir_idx = dir_map[self.direction.lower()]

        force_values = float(self.force) * sign * normalized_weights

        # Add to model forces
        model.forces[load_nodes, dir_idx] += force_values

        return len(load_nodes)
