"""
FEA base classes with dependency injection architecture.

This module provides:
- FEAKernel: Abstract base class for FEA solver backends
- FEAAnalyzer: Non-abstract analyzer that uses a FEAKernel via dependency injection
"""

from abc import ABC, abstractmethod
import os
import sys
import threading
import traceback
from typing import Any, List, Optional, Tuple, TYPE_CHECKING, Union
import numpy as np
import pyvista as pv

from ..boundary_conditions import FixedConstraint, PointLoad  #
import logging

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from ...shape import Shape
    from ..load_case.load_case import LoadCase
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
        shape: Union["Shape", str, Any],
        material: Optional["MaterialProperties"] = None,
        kernel: str = "torch-fem",
        mesh_size: float = 2.0,
        element_type: str = "tet4",
        device: str = "auto",
        mesher: "Union[MesherBase, str]" = "netgen",
        load_case: Optional["LoadCase"] = None,
    ):
        """
        Initialize FEA analyzer with dependency injection.

        Args:
            shape: Shape to analyze, path to STEP file (str), or meshio.Mesh
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
            from ..load_case.load_case import LoadCase
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
            f"  load_case={str(self.load_case)}\n"
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

    def get_bc_node_coverage(self) -> dict:
        """
        Return raw mesh-node counts for loaded and constrained zones.

        These raw counts should be normalised by the maximum possible counts
        (obtainable via :meth:`compute_max_bc_node_counts`, called once at
        dataset-creation time and stored in ``load_case.max_n_loaded_nodes`` /
        ``load_case.max_n_constraint_nodes``).  A ratio of 1.0 means the
        generated geometry fully covers the intended load / support area.

        Returns:
            Dict with keys:
                n_loaded_nodes     (int) – nodes in all load zones combined
                n_constraint_nodes (int) – nodes in all constraint zones combined
                n_total_nodes      (int) – total FE mesh nodes (for reference)

        Notes:
            - Returns all-zero dict with ``error`` key when meshing fails.
            - Requires the kernel to support ``get_visualization_data()``.
        """
        zero = {
            "n_loaded_nodes": 0,
            "n_constraint_nodes": 0,
            "n_total_nodes": 0,
        }
        try:
            if hasattr(self.kernel, "get_bc_node_masks"):
                nodes, constraint_mask, force_mask = self.kernel.get_bc_node_masks(
                    shape=self.shape,
                    material=self.load_case.material,
                    loads=self.load_case.loads,
                    constraints=self.load_case.constraints,
                    mesh_size=self.mesh_size,
                    element_type=self.element_type,
                )
            else:
                _pv_mesh, nodes, constraint_mask, force_mask, _ = (
                    self.kernel.get_visualization_data(
                        shape=self.shape,
                        material=self.load_case.material,
                        loads=self.load_case.loads,
                        constraints=self.load_case.constraints,
                        mesh_size=self.mesh_size,
                        element_type=self.element_type,
                        with_conditions=True,
                    )
                )
            n_total = int(nodes.shape[0])
            if n_total == 0:
                return {**zero, "error": "Empty mesh"}

            return {
                "n_loaded_nodes": int(np.sum(force_mask)),
                "n_constraint_nodes": int(np.sum(constraint_mask)),
                "n_total_nodes": n_total,
            }
        except Exception as e:
            return {**zero, "error": str(e)}

    def compute_max_bc_node_counts(self) -> dict:
        """
        Compute the theoretical maximum loaded/constrained node counts.

        Meshes a solid box that fills the entire design domain (from
        ``load_case.bounds``) and counts how many nodes fall in each BC zone.
        This represents coverage = 1.0 — achievable only when the generated
        part completely fills every load and constraint region.

        Call this **once per load case at dataset-creation time** and store
        the results::

            counts = fea.compute_max_bc_node_counts()
            load_case.max_n_loaded_nodes     = counts["max_n_loaded_nodes"]
            load_case.max_n_constraint_nodes = counts["max_n_constraint_nodes"]

        Returns:
            Dict with keys:
                max_n_loaded_nodes     (int)
                max_n_constraint_nodes (int)
            Plus ``error`` key when the computation fails.
        """
        import os
        import tempfile

        try:
            import cadquery as cq
        except ImportError:
            return {
                "max_n_loaded_nodes": 0,
                "max_n_constraint_nodes": 0,
                "error": "cadquery not available",
            }

        bounds = self.load_case.bounds
        if not bounds:
            return {
                "max_n_loaded_nodes": 0,
                "max_n_constraint_nodes": 0,
                "error": "No bounds in load_case",
            }

        try:
            x_min = bounds.get("x_min", 0.0)
            x_max = bounds.get("x_max", 100.0)
            y_min = bounds.get("y_min", 0.0)
            y_max = bounds.get("y_max", 100.0)
            z_min = bounds.get("z_min", 0.0)
            z_max = bounds.get("z_max", 100.0)
            dx = max(x_max - x_min, 1e-6)
            dy = max(y_max - y_min, 1e-6)
            dz = max(z_max - z_min, 1e-6)

            box_shape = (
                cq.Workplane("XY")
                .box(dx, dy, dz, centered=False)
                .translate((x_min, y_min, z_min))
            )

            tmp_fd, tmp_path = tempfile.mkstemp(suffix=".step")
            os.close(tmp_fd)
            try:
                cq.exporters.export(box_shape, tmp_path)

                _, nodes, constraint_mask, force_mask, _ = (
                    self.kernel.get_visualization_data(
                        shape=tmp_path,
                        material=self.load_case.material,
                        loads=self.load_case.loads,
                        constraints=self.load_case.constraints,
                        mesh_size=self.mesh_size,
                        element_type=self.element_type,
                        with_conditions=True,
                    )
                )
                return {
                    "max_n_loaded_nodes": int(np.sum(force_mask)),
                    "max_n_constraint_nodes": int(np.sum(constraint_mask)),
                }
            finally:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

        except Exception as e:
            return {
                "max_n_loaded_nodes": 0,
                "max_n_constraint_nodes": 0,
                "error": str(e),
            }

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
        show_grid: bool = True,
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
                    and constraints. 'mesh' shows only the mesh. 'design space'
                    shows design bounds and overlays mesh when available.
                    'debug' shows a simplified view for debugging.
            camera_position: Camera view angle ('iso', 'x', 'y', 'z', 'xy', 'xz', 'yz').
                           Default: 'iso' (isometric view)

        Note: This method requires the kernel to support visualization data extraction.
        """
        display = display.lower().strip()
        if display == "design_space":
            display = "design space"

        if display not in ["conditions", "mesh", "design space", "debug"]:
            raise ValueError(
                "display must be 'conditions', 'mesh', 'design space', or 'debug'"
            )

        def _extract_bounds(
            bounds: Optional[dict],
        ) -> Optional[Tuple[float, float, float, float, float, float]]:
            if not isinstance(bounds, dict):
                return None
            keys = ("x_min", "x_max", "y_min", "y_max", "z_min", "z_max")
            if not all(k in bounds for k in keys):
                return None
            try:
                x_min = float(bounds["x_min"])
                x_max = float(bounds["x_max"])
                y_min = float(bounds["y_min"])
                y_max = float(bounds["y_max"])
                z_min = float(bounds["z_min"])
                z_max = float(bounds["z_max"])
            except (TypeError, ValueError):
                return None

            if x_max < x_min or y_max < y_min or z_max < z_min:
                return None
            return (x_min, x_max, y_min, y_max, z_min, z_max)

        def _add_design_space_box(plotter: pv.Plotter, bounds: Optional[dict]) -> bool:
            extracted = _extract_bounds(bounds)
            if extracted is None:
                return False

            design_box = pv.Box(bounds=list(extracted))
            plotter.add_mesh(
                design_box,
                style="wireframe",
                color="green",
                line_width=3,
                opacity=1.0,
            )
            return True

        # ── Debug mode: pure matplotlib, no meshing needed ───────────────────────
        if display == "debug":
            import matplotlib

            running_on_main_thread = (
                threading.current_thread() is threading.main_thread()
            )
            headless_env = (
                sys.platform.startswith("linux")
                and not os.environ.get("DISPLAY")
                and not os.environ.get("WAYLAND_DISPLAY")
            )
            use_non_gui_backend = (
                bool(filename)
                or (not interactive)
                or (not running_on_main_thread)
                or headless_env
            )

            if use_non_gui_backend:
                try:
                    matplotlib.use("Agg", force=True)
                except Exception:
                    pass

            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
            from ..boundary_conditions import (
                FixedConstraint,
                PointLoad,
                DistributedLoad,
            )
            from ..boundary_conditions import (
                FixedConstraint,
                PointLoad,
                DistributedLoad,
            )

            load_case = self.load_case

            # Bounds come from load_case.bounds (set by the parser / get_fea_analyzer).
            # Fall back to _get_geometry_bounds(self.shape) if not set.
            geometry_bounds = load_case.bounds
            if not geometry_bounds:
                geometry_bounds = FEAAnalyzer._get_geometry_bounds(self.shape)
            if not geometry_bounds:
                geometry_bounds = {
                    "x_min": 0,
                    "x_max": 1,
                    "y_min": 0,
                    "y_max": 1,
                    "z_min": 0,
                    "z_max": 1,
                    "x_min": 0,
                    "x_max": 1,
                    "y_min": 0,
                    "y_max": 1,
                    "z_min": 0,
                    "z_max": 1,
                }

            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection="3d")

            def _draw_bbox(bounds: dict, color: str, label=None, alpha: float = 0.08):
                if not bounds:
                    return
                x0 = bounds.get("x_min", 0)
                x1 = bounds.get("x_max", 0)
                y0 = bounds.get("y_min", 0)
                y1 = bounds.get("y_max", 0)
                z0 = bounds.get("z_min", 0)
                z1 = bounds.get("z_max", 0)
                ax.bar3d(
                    x0,
                    y0,
                    z0,
                    max(x1 - x0, 1e-9),
                    max(y1 - y0, 1e-9),
                    max(z1 - z0, 1e-9),
                    alpha=alpha,
                    color=color,
                    shade=False,
                )
                if label:
                    ax.text(x0, y0, z1, label, color=color, fontsize=8)

            # Draw design-domain bounding box in blue
            _draw_bbox(geometry_bounds, "tab:blue", "Design domain", alpha=0.06)

            span = max(
                geometry_bounds["x_max"] - geometry_bounds["x_min"],
                geometry_bounds["y_max"] - geometry_bounds["y_min"],
                geometry_bounds["z_max"] - geometry_bounds["z_min"],
                1e-6,
            )

            # ── Constraints in red ────────────────────────────────────────────
            for i, bc in enumerate(load_case.constraints):
                label = f"BC {i+1}"
                if isinstance(bc, FixedConstraint):
                    loc = bc.location
                    if isinstance(loc, dict):
                        _draw_bbox(loc, "tab:red", label, alpha=0.15)
                    elif isinstance(loc, (tuple, list)) and len(loc) == 3:
                        ax.scatter(
                            loc[0],
                            loc[1],
                            loc[2],
                            color="tab:red",
                            s=100,
                            label=label if i == 0 else None,
                        )
                    # String locations ('x_min', 'top', …) — skip,
                    # exact face coords only known after meshing.
                elif hasattr(bc, "center") and hasattr(bc, "radius"):
                    # CylindricalConstraint
                    cx_b, cy_b, cz_b = bc.center
                    ax.scatter(
                        cx_b,
                        cy_b,
                        cz_b,
                        color="tab:red",
                        s=100,
                        label=label if i == 0 else None,
                    )
                    ax.scatter(
                        cx_b,
                        cy_b,
                        cz_b,
                        color="tab:red",
                        s=100,
                        label=label if i == 0 else None,
                    )

            # ── Loads in green + force arrows ─────────────────────────────────
            for i, ld in enumerate(load_case.loads):
                label = f"Load {i+1}"
                lx = ly = lz = None

                if isinstance(ld, PointLoad):
                    pt = ld.point
                    if isinstance(pt, dict):
                        lx = pt.get("x", (pt.get("x_min", 0) + pt.get("x_max", 0)) / 2)
                        ly = pt.get("y", (pt.get("y_min", 0) + pt.get("y_max", 0)) / 2)
                        lz = pt.get("z", (pt.get("z_min", 0) + pt.get("z_max", 0)) / 2)
                    elif isinstance(pt, (tuple, list)) and len(pt) == 3:
                        lx, ly, lz = float(pt[0]), float(pt[1]), float(pt[2])
                    if lx is not None:
                        ax.scatter(
                            lx,
                            ly,
                            lz,
                            color="tab:green",
                            s=100,
                            label=label if i == 0 else None,
                        )
                        ax.scatter(
                            lx,
                            ly,
                            lz,
                            color="tab:green",
                            s=100,
                            label=label if i == 0 else None,
                        )

                elif isinstance(ld, DistributedLoad) and isinstance(ld.location, dict):
                    loc = ld.location
                    _draw_bbox(loc, "tab:green", label, alpha=0.15)
                    lx = (loc.get("x_min", 0) + loc.get("x_max", 0)) / 2
                    ly = (loc.get("y_min", 0) + loc.get("y_max", 0)) / 2
                    lz = (loc.get("z_min", 0) + loc.get("z_max", 0)) / 2

                # Force arrow from centroid
                if lx is not None:
                    force = ld.force
                    if isinstance(force, (int, float)):
                        direction = str(getattr(ld, "direction", "z") or "z").lower()
                        _axis_map = {
                            "x": (1, 0, 0),
                            "-x": (-1, 0, 0),
                            "y": (0, 1, 0),
                            "-y": (0, -1, 0),
                            "z": (0, 0, 1),
                            "-z": (0, 0, -1),
                            "x": (1, 0, 0),
                            "-x": (-1, 0, 0),
                            "y": (0, 1, 0),
                            "-y": (0, -1, 0),
                            "z": (0, 0, 1),
                            "-z": (0, 0, -1),
                        }
                        vx, vy, vz = _axis_map.get(direction, (0, 0, 1))
                    else:
                        vx, vy, vz = float(force[0]), float(force[1]), float(force[2])
                        fmag = (vx**2 + vy**2 + vz**2) ** 0.5 + 1e-10
                        vx, vy, vz = vx / fmag, vy / fmag, vz / fmag
                    ax.quiver(
                        lx, ly, lz, vx, vy, vz, length=span * 0.15, color="darkgreen"
                    )
                    ax.quiver(
                        lx, ly, lz, vx, vy, vz, length=span * 0.15, color="darkgreen"
                    )

            ax.set_title("FEA Debug: Design domain + BC/Load Regions")
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")
            if not show_grid:
                ax.grid(False)
                ax.set_axis_off()
            if show_legend:
                ax.legend(fontsize=8, loc="upper right")
            plt.tight_layout()

            if filename:
                fig.savefig(filename, dpi=150)
                print(f"✓ Saved debug visualization to: {filename}")
            elif use_non_gui_backend:
                print(
                    "⚠ Debug visualization running in non-GUI mode "
                    "(headless/non-main thread); skipping plt.show()."
                )
            else:
                plt.show()
            plt.close(fig)
            return

        # ── conditions / mesh modes: requires meshing ─────────────────────────
        # Configure for headless/non-interactive mode if needed
        if filename:
            interactive = False
            pv.OFF_SCREEN = True

        # On macOS, vtkCocoaRenderWindow requires the main thread to create
        # NSWindow even when off_screen=True.  Ensure VTK uses its true
        # framebuffer-only code path (set before Plotter construction).
        os.environ["VTK_DEFAULT_RENDER_WINDOW_OFFSCREEN"] = "1"

        plotter = pv.Plotter(window_size=list(window_size), off_screen=True)
        logger.debug(
            f"Renderer {plotter.renderer.GetClassName()} initialized for visualization."
        )

        # Add design-space box (wireframe) if bounds are available
        has_design_space = _add_design_space_box(plotter, self.load_case.bounds)

        pv_mesh = None
        nodes = None
        constraint_mask = None
        force_mask = None
        force_vectors = None

        if display in ["conditions", "mesh", "design space"]:
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
                raise ValueError(
                    f"Error applying boundary conditions: {e} \n {traceback.format_exc()} "
                )

        # Add the main mesh (semi-transparent) when available
        if pv_mesh is not None:
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
            logger.info(
                f"Saving visualization to: {filename} using renderer {plotter.renderer.GetClassName()}"
            )
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

    def _get_geometry_bounds(cad: Any) -> dict:
        """Get bounding box of a CadQuery Workplane."""
        try:
            cq_obj = cad.val() if hasattr(cad, "val") else cad
            bb = cq_obj.BoundingBox()
            return {
                "x_min": bb.xmin,
                "x_max": bb.xmax,
                "y_min": bb.ymin,
                "y_max": bb.ymax,
                "z_min": bb.zmin,
                "z_max": bb.zmax,
            }
        except Exception as e:
            logger.warning(f"Bounding box calculation failed: {e}")
            return None

