"""
FEA base classes with dependency injection architecture.

This module provides:
- FEAKernel: Abstract base class for FEA solver backends
- FEAAnalyzer: Non-abstract analyzer that uses a FEAKernel via dependency injection
"""

from abc import ABC, abstractmethod
import traceback
from typing import List, Optional, Tuple, TYPE_CHECKING, Union
import numpy as np
import pyvista as pv

if TYPE_CHECKING:
    from ...shape import Shape
    from ..load_case import LoadCase
    from ..materials import MaterialProperties
    from ..boundary_conditions import Load, BoundaryCondition
    from ..results import FEAResults, OptimizationResult


class FEAKernel(ABC):
    """
    Abstract base class for FEA solver backends.

    Concrete implementations (e.g., TorchFEMKernel) provide specific
    FEA solver backends. The kernel handles meshing, solving, and
    result extraction for a specific solver.
    """

    def __str__(self) -> str:
        """String representation of the FEA kernel."""
        solver_name = self.get_solver_name()
        available = "✓ available" if self.is_available() else "✗ not available"
        return f"FEAKernel({solver_name}, {available})"

    @classmethod
    @abstractmethod
    def is_available(cls) -> bool:
        """
        Check if this kernel's dependencies are available.

        Returns:
            True if the kernel can be used, False otherwise
        """
        pass

    @classmethod
    @abstractmethod
    def get_solver_name(cls) -> str:
        """
        Get the name of the FEA solver backend.

        Returns:
            Name of the solver (e.g., "torch-fem", "CalculiX")
        """
        pass

    @abstractmethod
    def solve(
        self,
        shape: "Shape",
        material: "MaterialProperties",
        loads: List["Load"],
        constraints: List["BoundaryCondition"],
        mesh_size: float,
        element_type: str,
        verbose: bool = False,
    ) -> "FEAResults":
        """
        Run FEA analysis and return results.

        Args:
            shape: Shape to analyze
            material: Material properties
            loads: List of loads to apply
            constraints: List of boundary conditions to apply
            mesh_size: Target mesh element size (mm)
            element_type: Element type (solver-dependent)
            verbose: Print detailed progress information

        Returns:
            FEAResults object containing analysis results
        """
        pass

    def optimize(
        self,
        shape: "Shape",
        material: "MaterialProperties",
        loads: List["Load"],
        constraints: List["BoundaryCondition"],
        mesh_size: float,
        element_type: str,
        volume_fraction: float = 0.5,
        num_iterations: int = 100,
        penalization: float = 3.0,
        filter_radius: float = 0.0,
        move_limit: float = 0.2,
        rho_min: float = 1e-3,
        use_autograd: bool = False,
        verbose: bool = False,
    ) -> "OptimizationResult":
        """
        Run topology optimization using SIMP method.

        This is an optional method that kernels can implement for topology optimization.
        Not all kernels support optimization.

        Args:
            shape: Shape to optimize
            material: Material properties
            loads: List of loads to apply
            constraints: List of boundary conditions to apply
            mesh_size: Target mesh element size (mm)
            element_type: Element type (solver-dependent)
            volume_fraction: Target volume fraction (0 < v < 1)
            num_iterations: Number of optimization iterations
            penalization: SIMP penalization factor (p)
            filter_radius: Sensitivity filter radius (0 = no filtering)
            move_limit: Maximum change in density per iteration
            rho_min: Minimum density to avoid singularity
            use_autograd: Use automatic differentiation for sensitivities
            verbose: Print detailed progress information

        Returns:
            OptimizationResult object containing optimization results

        Raises:
            NotImplementedError: If the kernel does not support optimization
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support topology optimization"
        )
        pass

    def get_visualization_data(
        self,
        shape: Union["Shape", str],
        material: "MaterialProperties",
        loads: List["Load"],
        constraints: List["BoundaryCondition"],
        mesh_size: float,
        element_type: str,
        **kwargs,
    ) -> Tuple[pv.UnstructuredGrid, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Get mesh and boundary condition data for visualization.

        This method prepares the mesh and applies boundary conditions without solving,
        returning data needed for visualization of constraints and loads.

        Args:
            shape: Shape to visualize or path to STEP file
            material: Material properties
            loads: List of loads to apply
            constraints: List of boundary conditions to apply
            mesh_size: Target mesh element size (mm)
            element_type: Element type (solver-dependent)
            **kwargs: Additional keyword arguments for kernel-specific options

        Returns:
            Tuple containing:
            - pv_mesh: PyVista mesh for visualization
            - nodes: Node coordinates as numpy array (N, 3)
            - constraint_mask: Boolean mask for constrained nodes (N,)
            - force_mask: Boolean mask for loaded nodes (N,)
            - force_vectors: Force vectors for loaded nodes (M, 3) where M is number of loaded nodes

        Raises:
            NotImplementedError: If the kernel does not support visualization
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support visualization. "
            "Consider using solve() followed by results.show() instead."
        )


class FEAAnalyzer:
    """
    Non-abstract FEA analyzer that uses dependency injection.

    The analyzer orchestrates the FEA workflow using an injected
    FEAKernel for the actual solver operations.
    """

    def __init__(
        self,
        shape: Union["Shape", str],
        material: Optional["MaterialProperties"] = None,
        kernel: str = "torch-fem",
        mesh_size: float = 2.0,
        element_type: str = "tet4",
        loads: Optional[List["Load"]] = None,
        constraints: Optional[List["BoundaryCondition"]] = None,
        device: str = "auto",
        mesher: str = "netgen",
        load_case: Optional["LoadCase"] = None,
    ):
        """
        Initialize FEA analyzer with dependency injection.

        Args:
            shape: Shape to analyze or path to STEP file
            material: Material properties (ignored if load_case is provided)
            kernel: FEAKernel implementation to use for solving
            mesh_size: Target mesh element size (mm)
            element_type: Element type (solver-dependent)
            device: Device to use ('cpu', 'cuda', or 'auto'). Only for torch-fem kernel.
            mesher: Mesher to use ('netgen' or 'gmsh'). Only for torch-fem kernel.
            load_case: Optional object containing material, loads, and constraints
        """
        self.shape = shape
        if load_case is None:
            from ..load_case import LoadCase
            from ..materials import Material

            self.load_case = LoadCase(
                material=material if material is not None else Material.STEEL,
                loads=list(loads) if loads is not None else [],
                constraints=list(constraints) if constraints is not None else [],
            )
        else:
            self.load_case = load_case
        if kernel == "torch-fem":
            from .torch_fem_kernel import TorchFEMKernel

            self.kernel = TorchFEMKernel(device=device, mesher=mesher)
        self.mesh_size = mesh_size
        self.element_type = element_type

    @property
    def material(self) -> "MaterialProperties":
        """Backward-compatible accessor for `self.load_case.material`."""
        return self.load_case.material

    @material.setter
    def material(self, value: "MaterialProperties") -> None:
        self.load_case.material = value

    @property
    def loads(self) -> List["Load"]:
        """Backward-compatible accessor for `self.load_case.loads`."""
        return self.load_case.loads

    @loads.setter
    def loads(self, value: List["Load"]) -> None:
        self.load_case.loads = value

    @property
    def constraints(self) -> List["BoundaryCondition"]:
        """Backward-compatible accessor for `self.load_case.constraints`."""
        return self.load_case.constraints

    @constraints.setter
    def constraints(self, value: List["BoundaryCondition"]) -> None:
        self.load_case.constraints = value

    def __str__(self) -> str:
        """String representation of the FEA analyzer."""
        solver = self.kernel.get_solver_name()
        num_loads = len(self.load_case.loads)
        num_constraints = len(self.load_case.constraints)
        return (
            f"FEAAnalyzer(\n"
            f"  solver={solver},\n"
            f"  material={self.load_case.material.name},\n"
            f"  mesh_size={self.mesh_size}mm,\n"
            f"  element_type={self.element_type},\n"
            f"  loads={[str(load) for load in self.load_case.loads]},\n"
            f"  constraints={[str(constraint) for constraint in self.load_case.constraints]}\n"
            f")"
        )

    def add_load(self, load: "Load") -> "FEAAnalyzer":
        """
        Add a load to the analysis (fluent interface).

        Args:
            load: Load object to apply

        Returns:
            Self for method chaining
        """
        self.load_case.loads.append(load)
        return self

    def add_constraint(self, constraint: "BoundaryCondition") -> "FEAAnalyzer":
        """
        Add a constraint to the analysis (fluent interface).

        Args:
            constraint: Boundary condition object to apply

        Returns:
            Self for method chaining
        """
        self.load_case.constraints.append(constraint)
        return self

    def solve(self) -> "FEAResults":
        """
        Run FEA analysis and return results.

        Delegates to the injected kernel for actual solving.

        Args:
            verbose: Print detailed progress information

        Returns:
            FEAResults object containing analysis results
        """
        try:
            return self.kernel.solve(
                shape=self.shape,
                material=self.load_case.material,
                loads=self.load_case.loads,
                constraints=self.load_case.constraints,
                mesh_size=self.mesh_size,
                element_type=self.element_type,
            )
        except ZeroDivisionError:
            raise ValueError(
                "Error applying boundary conditions: No nodes found for one of the loads. "
                "Please check that your load locations match the geometry and mesh size."
            ) from None

    def optimize(
        self,
        volume_fraction: float = 0.5,
        num_iterations: int = 100,
        penalization: float = 3.0,
        filter_radius: float = 0.0,
        move_limit: float = 0.2,
        rho_min: float = 1e-3,
        use_autograd: bool = False,
    ) -> "OptimizationResult":
        """
        Run topology optimization using SIMP method.

        Delegates to the injected kernel for optimization.

        Args:
            volume_fraction: Target volume fraction (0 < v < 1), default 0.5
            num_iterations: Number of optimization iterations, default 100
            penalization: SIMP penalization factor (p), default 3.0
            filter_radius: Sensitivity filter radius (0 = no filtering), default 0.0
            move_limit: Maximum change in density per iteration, default 0.2
            rho_min: Minimum density to avoid singularity, default 1e-3
            use_autograd: Use automatic differentiation for sensitivities, default False
            verbose: Print detailed progress information

        Returns:
            OptimizationResult object containing optimization results
        """
        try:
            return self.kernel.optimize(
                shape=self.shape,
                material=self.load_case.material,
                loads=self.load_case.loads,
                constraints=self.load_case.constraints,
                mesh_size=self.mesh_size,
                element_type=self.element_type,
                volume_fraction=volume_fraction,
                num_iterations=num_iterations,
                penalization=penalization,
                filter_radius=filter_radius,
                move_limit=move_limit,
                rho_min=rho_min,
                use_autograd=use_autograd,
            )
        except ZeroDivisionError:
            raise ValueError(
                "Error applying boundary conditions: No nodes found for one of the loads. "
                "Please check that your load locations match the geometry and mesh size."
            ) from None

    def get_solver_name(self) -> str:
        """
        Get the name of the FEA solver backend.

        Returns:
            Name of the solver (e.g., "torch-fem", "CalculiX")
        """
        return self.kernel.get_solver_name()

    def validate_connectivity(self) -> bool:
        """
        Check if loaded and constrained nodes are connected via the mesh.

        This validates that forces can be transmitted from loaded regions
        to constrained regions through the mesh structure. Disconnected
        geometry or floating parts will return False.

        Returns:
            True if loaded and constrained nodes are in the same connected
            component of the mesh, False otherwise.

        Note:
            - Returns True if there are no loads or no constraints
            - Requires the kernel to support visualization data extraction
            - This creates a temporary mesh to check connectivity
        """
        from collections import deque

        # If no loads or constraints, consider it valid
        if len(self.load_case.loads) == 0 or len(self.load_case.constraints) == 0:
            return True

        try:
            # Get mesh and boundary condition data
            pv_mesh, nodes, constraint_mask, force_mask, force_vectors = (
                self.kernel.get_visualization_data(
                    shape=self.shape,
                    material=self.load_case.material,
                    loads=self.load_case.loads,
                    constraints=self.load_case.constraints,
                    mesh_size=self.mesh_size,
                    element_type=self.element_type,
                )
            )

            # Find constrained and loaded nodes
            constrained_nodes = set(np.where(constraint_mask)[0])
            loaded_nodes = set(np.where(force_mask)[0])

            if len(constrained_nodes) == 0 or len(loaded_nodes) == 0:
                return True

            # Extract element connectivity from PyVista mesh
            cells = pv_mesh.cells
            cell_types = pv_mesh.celltypes

            # Build adjacency list from elements
            n_nodes = nodes.shape[0]
            adjacency = [set() for _ in range(n_nodes)]

            # Parse VTK cell array
            idx = 0
            for cell_type in cell_types:
                n_points = cells[idx]
                elem_nodes = cells[idx + 1 : idx + 1 + n_points]

                # Connect all pairs of nodes in this element
                for i in range(len(elem_nodes)):
                    for j in range(i + 1, len(elem_nodes)):
                        adjacency[elem_nodes[i]].add(elem_nodes[j])
                        adjacency[elem_nodes[j]].add(elem_nodes[i])

                idx += 1 + n_points

            # Find connected component containing constrained nodes using BFS
            visited = set()
            queue = deque(constrained_nodes)
            visited.update(constrained_nodes)

            while queue:
                node = queue.popleft()
                for neighbor in adjacency[node]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)

            # Check if all loaded nodes are in the same component
            disconnected_loads = loaded_nodes - visited

            return len(disconnected_loads) == 0

        except Exception as e:
            # If we can't check connectivity, assume it's invalid
            print(f"⚠ Warning: Could not validate connectivity: {e}")
            return False

    def show(
        self,
        interactive: bool = True,
        window_size: Tuple[int, int] = (1400, 700),
        filename: Optional[str] = None,
        show_legend: bool = True,
        display: str = "conditions",
        camera_position: str = "iso",
    ) -> None:
        """
        Visualize constraints, loads, and mesh of the analyzed shape.

        This method creates a visualization showing:
        - The mesh (semi-transparent light blue)
        - Fixed/constrained nodes (red spheres)
        - Loaded nodes (green spheres with force arrows)

        Args:
            interactive: Use interactive viewer. If False or filename is set,
                        uses off-screen rendering. Default: True
            window_size: Window dimensions (width, height). Default: (1400, 700)
            filename: Optional path to save the plot. If set, saves to file
                     instead of showing interactively.
            show_legend: Whether to show the legend. Default: True
            display: What to display. 'conditions' (default) shows mesh, loads,
                    and constraints. 'mesh' shows only the mesh.
            camera_position: Camera view angle ('iso', 'x', 'y', 'z', 'xy', 'xz', 'yz').
                           Default: 'iso' (isometric view)

        Note: This method requires the kernel to support visualization data extraction.
        """
        if display not in ["conditions", "mesh"]:
            raise ValueError("display must be either 'conditions' or 'mesh'")

        # Configure for headless/non-interactive mode if needed
        if filename:
            interactive = False
            pv.start_xvfb()  # Start virtual framebuffer for headless environments
            pv.OFF_SCREEN = True

        # Get visualization data from kernel
        try:
            pv_mesh, nodes, constraint_mask, force_mask, force_vectors = (
                self.kernel.get_visualization_data(
                    shape=self.shape,
                    material=self.load_case.material,
                    loads=self.load_case.loads,
                    constraints=self.load_case.constraints,
                    mesh_size=self.mesh_size,
                    element_type=self.element_type,
                    with_conditions=(display == "conditions"),
                )
            )
        except Exception as e:
            print(f"Could not show debug mesh: {e}")
            raise ValueError(
                f"Error applying boundary conditions: {e} \n {traceback.format_exc()} "
            ) from None

        # Visualize boundary conditions
        plotter = pv.Plotter(window_size=list(window_size), off_screen=not interactive)

        # Add the main mesh (semi-transparent)
        plotter.add_mesh(
            pv_mesh,
            color="lightblue",
            opacity=0.3,
            show_edges=True,
            edge_color="gray",
            line_width=0.5,
        )

        if display == "conditions":
            has_legend_entries = False
            # Visualize FIXED NODES (constraints)
            constrained_nodes = nodes[constraint_mask]

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
                has_legend_entries = True
            # Visualize LOADED NODES (forces)
            loaded_nodes = nodes[force_mask]

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
                has_legend_entries = True

                # Add force arrows
                # Scale arrows for visibility
                arrow_scale = (nodes[:, 0].max() - nodes[:, 0].min()) * 0.05
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
                has_legend_entries = True

            # Add legend and labels
            if show_legend and has_legend_entries:
                plotter.add_legend()
                plotter.add_text(
                    "Boundary Conditions", position="upper_edge", font_size=12
                )
                plotter.add_text(
                    "Red = Fixed (Constraints)\nGreen = Loaded (Forces)",
                    position="lower_left",
                    font_size=10,
                )
        else:
            # Display mode is 'mesh'
            plotter.add_text("Mesh Visualization", position="upper_edge", font_size=12)

        plotter.add_axes()

        # Set camera position
        camera_position = camera_position.lower()
        if camera_position == "iso":
            plotter.camera_position = "iso"
        elif camera_position in ["x", "yz"]:
            plotter.view_yz()
        elif camera_position in ["y", "xz"]:
            plotter.view_xz()
        elif camera_position in ["z", "xy"]:
            plotter.view_xy()
        else:
            # Default to iso if invalid
            plotter.camera_position = "iso"

        # Show or save
        if filename:
            # Save to file
            plotter.screenshot(filename)
            print(f"✓ Saved boundary condition visualization to: {filename}")
            plotter.close()
        else:
            # Show interactively
            if interactive:
                plotter.show()
            else:
                plotter.show(jupyter_backend="static")

    def export_gltf(
        self,
        filepath: str,
        display: str = "conditions",
    ) -> dict:
        """
        Export visualization data as glTF for web rendering.

        Args:
            filepath: Path to save the glTF file (without extension)
            display: What to display. 'conditions' (default) shows mesh, loads,
                    and constraints. 'mesh' shows only the mesh.

        Returns:
            Dictionary containing paths and metadata for frontend rendering
        """
        import trimesh

        if display not in ["conditions", "mesh"]:
            raise ValueError("display must be either 'conditions' or 'mesh'")

        # Get visualization data from kernel
        pv_mesh, nodes, constraint_mask, force_mask, force_vectors = (
            self.kernel.get_visualization_data(
                shape=self.shape,
                material=self.load_case.material,
                loads=self.load_case.loads,
                constraints=self.load_case.constraints,
                mesh_size=self.mesh_size,
                element_type=self.element_type,
                with_conditions=(display == "conditions"),
            )
        )

        # Extract surface from volume mesh for glTF export
        surface_mesh = pv_mesh.extract_surface()

        # Convert PyVista surface mesh to trimesh for glTF export
        faces = surface_mesh.faces.reshape(-1, 4)[:, 1:]  # Remove the count prefix
        vertices = surface_mesh.points
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

        # Export main mesh as glTF
        mesh_path = f"{filepath}_mesh.gltf"
        mesh.export(mesh_path)

        # Prepare boundary condition data
        result = {
            "mesh": mesh_path,
            "bounds": {
                "min": nodes.min(axis=0).tolist(),
                "max": nodes.max(axis=0).tolist(),
            },
        }

        if display == "conditions":
            # Export constraint points
            constrained_nodes = nodes[constraint_mask]
            loaded_nodes = nodes[force_mask]

            result["constraints"] = (
                constrained_nodes.tolist() if len(constrained_nodes) > 0 else []
            )
            result["loads"] = loaded_nodes.tolist() if len(loaded_nodes) > 0 else []
            result["force_vectors"] = (
                force_vectors.tolist() if len(force_vectors) > 0 else []
            )

        return result
