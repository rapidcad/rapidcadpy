"""
FEA base classes with dependency injection architecture.

This module provides:
- FEAKernel: Abstract base class for FEA solver backends
- FEAAnalyzer: Non-abstract analyzer that uses a FEAKernel via dependency injection
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, TYPE_CHECKING
import numpy as np
import pyvista as pv

if TYPE_CHECKING:
    from rapidcadpy.shape import Shape
    from rapidcadpy.fea.materials import MaterialProperties
    from rapidcadpy.fea.boundary_conditions import Load, BoundaryCondition
    from rapidcadpy.fea.results import FEAResults, OptimizationResult


class FEAKernel(ABC):
    """
    Abstract base class for FEA solver backends.

    Concrete implementations (e.g., TorchFEMKernel) provide specific
    FEA solver backends. The kernel handles meshing, solving, and
    result extraction for a specific solver.
    """

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
        shape: "Shape",
        material: "MaterialProperties",
        loads: List["Load"],
        constraints: List["BoundaryCondition"],
        mesh_size: float,
        element_type: str,
    ) -> Tuple[pv.UnstructuredGrid, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Get mesh and boundary condition data for visualization.

        This method prepares the mesh and applies boundary conditions without solving,
        returning data needed for visualization of constraints and loads.

        Args:
            shape: Shape to visualize
            material: Material properties
            loads: List of loads to apply
            constraints: List of boundary conditions to apply
            mesh_size: Target mesh element size (mm)
            element_type: Element type (solver-dependent)

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
        shape: "Shape",
        material: "MaterialProperties",
        kernel: str,
        mesh_size: float = 2.0,
        element_type: str = "tet4",
        loads: Optional[List["Load"]] = None,
        constraints: Optional[List["BoundaryCondition"]] = None,
    ):
        """
        Initialize FEA analyzer with dependency injection.

        Args:
            shape: Shape to analyze
            material: Material properties
            kernel: FEAKernel implementation to use for solving
            mesh_size: Target mesh element size (mm)
            element_type: Element type (solver-dependent)
        """
        self.shape = shape
        self.material = material
        if kernel=="torch-fem":
            from rapidcadpy.fea.kernels.torch_fem_kernel import TorchFEMKernel

            self.kernel = TorchFEMKernel()
        self.mesh_size = mesh_size
        self.element_type = element_type
        if loads is not None:
            self.loads = loads
        else:
            self.loads: List["Load"] = []
        if constraints is not None:
            self.constraints = constraints
        else:
            self.constraints: List["BoundaryCondition"] = []

    def add_load(self, load: "Load") -> "FEAAnalyzer":
        """
        Add a load to the analysis (fluent interface).

        Args:
            load: Load object to apply

        Returns:
            Self for method chaining
        """
        self.loads.append(load)
        return self

    def add_constraint(self, constraint: "BoundaryCondition") -> "FEAAnalyzer":
        """
        Add a constraint to the analysis (fluent interface).

        Args:
            constraint: Boundary condition object to apply

        Returns:
            Self for method chaining
        """
        self.constraints.append(constraint)
        return self

    def solve(self, verbose: bool = False) -> "FEAResults":
        """
        Run FEA analysis and return results.

        Delegates to the injected kernel for actual solving.

        Args:
            verbose: Print detailed progress information

        Returns:
            FEAResults object containing analysis results
        """
        return self.kernel.solve(
            shape=self.shape,
            material=self.material,
            loads=self.loads,
            constraints=self.constraints,
            mesh_size=self.mesh_size,
            element_type=self.element_type,
            verbose=verbose,
        )

    def optimize(
        self,
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
        return self.kernel.optimize(
            shape=self.shape,
            material=self.material,
            loads=self.loads,
            constraints=self.constraints,
            mesh_size=self.mesh_size,
            element_type=self.element_type,
            volume_fraction=volume_fraction,
            num_iterations=num_iterations,
            penalization=penalization,
            filter_radius=filter_radius,
            move_limit=move_limit,
            rho_min=rho_min,
            use_autograd=use_autograd,
            verbose=verbose,
        )

    def get_solver_name(self) -> str:
        """
        Get the name of the FEA solver backend.

        Returns:
            Name of the solver (e.g., "torch-fem", "CalculiX")
        """
        return self.kernel.get_solver_name()

    def plot(self) -> None:
        """
        Visualize constraints, loads, and mesh of the analyzed shape.

        This method creates a visualization showing:
        - The mesh (semi-transparent light blue)
        - Fixed/constrained nodes (red spheres)
        - Loaded nodes (green spheres with force arrows)

        Note: This method requires the kernel to support visualization data extraction.
        """
        # Get visualization data from kernel
        pv_mesh, nodes, constraint_mask, force_mask, force_vectors = (
            self.kernel.get_visualization_data(
                shape=self.shape,
                material=self.material,
                loads=self.loads,
                constraints=self.constraints,
                mesh_size=self.mesh_size,
                element_type=self.element_type,
            )
        )

        # Visualize boundary conditions
        plotter = pv.Plotter(window_size=[1400, 700])

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
            print(f"✓ Visualizing {len(constrained_nodes)} constrained nodes (RED)")

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

            # Add force arrows
            # Scale arrows for visibility
            arrow_scale = (nodes[:, 0].max() - nodes[:, 0].min()) * 0.1
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
            "Red = Fixed (Constraints)\nGreen = Loaded (Forces)",
            position="lower_left",
            font_size=10,
        )
        plotter.add_axes()
        plotter.camera_position = "iso"

        # Show
        plotter.show(jupyter_backend="static")
