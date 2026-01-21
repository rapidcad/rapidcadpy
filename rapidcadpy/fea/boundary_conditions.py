"""
Boundary conditions for FEA analysis.

This module provides loads and constraints that can be applied to FEA models.
"""

from abc import ABC, abstractmethod
from typing import Tuple, Optional, Union, Literal, TYPE_CHECKING

if TYPE_CHECKING:
    from rapidcadpy.cad_types import Vector


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
        >>> from rapidcadpy.fea.boundary_conditions import visualize_boundary_conditions
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
        from rapidcadpy.fea.utils import find_nodes_in_box
        import torch

        bbox = geometry_info["bounding_box"]

        # Determine nodes based on location
        if isinstance(self.location, str):
            loc = self.location.lower()

            if loc in ["end_1", "x_min"]:
                constrained_nodes = find_nodes_in_box(
                    nodes,
                    xmin=bbox["xmin"],
                    xmax=bbox["xmin"],
                    tolerance=self.tolerance*mesh_size,
                )
            elif loc in ["end_2", "x_max"]:
                constrained_nodes = find_nodes_in_box(
                    nodes,
                    xmin=bbox["xmax"],
                    xmax=bbox["xmax"],
                    tolerance=self.tolerance*mesh_size,
                )
            elif loc in ["y_min"]:
                constrained_nodes = find_nodes_in_box(
                    nodes,
                    ymin=bbox["ymin"],
                    ymax=bbox["ymin"],
                    tolerance=self.tolerance*mesh_size,
                )
            elif loc in ["y_max"]:
                constrained_nodes = find_nodes_in_box(
                    nodes,
                    ymin=bbox["ymax"],
                    ymax=bbox["ymax"],
                    tolerance=self.tolerance*mesh_size,
                )
            elif loc in ["bottom", "z_min"]:
                constrained_nodes = find_nodes_in_box(
                    nodes,
                    zmin=bbox["zmin"],
                    zmax=bbox["zmin"],
                    tolerance=self.tolerance*mesh_size,
                )
            elif loc in ["top", "z_max"]:
                constrained_nodes = find_nodes_in_box(
                    nodes,
                    zmin=bbox["zmax"],
                    zmax=bbox["zmax"],
                    tolerance=self.tolerance*mesh_size,
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

    def apply(self, model, nodes, elements, geometry_info, mesh_size: float ):
        """Apply distributed load to the model"""
        from rapidcadpy.fea.utils import find_nodes_in_box
        import torch

        bbox = geometry_info["bounding_box"]
        loc = self.location.lower()

        # Find nodes on the specified surface
        if loc in ["top", "z_max"]:
            load_nodes = find_nodes_in_box(
                nodes, zmin=bbox["zmax"], zmax=bbox["zmax"], tolerance=self.tolerance * mesh_size
            )
            default_dir = "z"
        elif loc in ["bottom", "z_min"]:
            load_nodes = find_nodes_in_box(
                nodes, zmin=bbox["zmin"], zmax=bbox["zmin"], tolerance=self.tolerance * mesh_size
            )
            default_dir = "z"
        elif loc in ["x_min"]:
            load_nodes = find_nodes_in_box(
                nodes, xmin=bbox["xmin"], xmax=bbox["xmin"], tolerance=self.tolerance * mesh_size
            )
            default_dir = "x"
        elif loc in ["x_max"]:
            load_nodes = find_nodes_in_box(
                nodes, xmin=bbox["xmax"], xmax=bbox["xmax"], tolerance=self.tolerance * mesh_size
            )
            default_dir = "x"
        elif loc in ["y_min"]:
            load_nodes = find_nodes_in_box(
                nodes, ymin=bbox["ymin"], ymax=bbox["ymin"], tolerance=self.tolerance * mesh_size
            )
            default_dir = "y"
        elif loc in ["y_max"]:
            load_nodes = find_nodes_in_box(
                nodes, ymin=bbox["ymax"], ymax=bbox["ymax"], tolerance=self.tolerance * mesh_size
            )
            default_dir = "y"
        else:
            raise ValueError(f"Unknown location: {self.location}")

        if len(load_nodes) == 0:
            raise ValueError(f"No nodes found at location: {self.location}")

        # Calculate force per node
        if isinstance(self.force, (int, float)):
            # Scalar force - apply in specified or default direction
            direction = self.direction or default_dir
            force_per_node = self.force / len(load_nodes)

            dir_map = {"x": 0, "y": 1, "z": 2}
            dir_idx = dir_map[direction]

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
        from rapidcadpy.fea.utils import find_nodes_in_box

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
            tolerance=self.tolerance*mesh_size,
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
    Pressure load on a surface.

    Applies pressure (force per unit area) to a surface.
    Note: This is a simplified implementation that converts pressure to total force.
    """

    def __init__(self, location: str, pressure: float, tolerance: float = 1):
        """
        Create a pressure load.

        Args:
            location: Surface identifier ('top', 'bottom', etc.)
            pressure: Pressure magnitude (MPa, negative for compression)
            tolerance: Distance tolerance for node selection (mm)
        """
        self.location = location
        self.pressure = pressure
        self.tolerance = tolerance

    def apply(self, model, nodes, elements, geometry_info):
        """Apply pressure load to the model"""
        # This is a simplified implementation
        # In practice, we'd need to calculate surface area and convert pressure to force
        # For now, we'll treat it as a distributed load with arbitrary scaling

        # Estimate total force from pressure (simplified)
        # This would need proper surface area calculation in production
        area_estimate = 100.0  # mm² - placeholder
        total_force = self.pressure * area_estimate

        # Use DistributedLoad to apply
        dist_load = DistributedLoad(
            self.location, total_force, tolerance=self.tolerance
        )
        return dist_load.apply(model, nodes, elements, geometry_info)
