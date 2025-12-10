"""
FEA results container and analysis methods.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple
import torch
import numpy as np


@dataclass
class FEAResults:
    """Container for FEA analysis results"""

    # Input parameters
    material: "MaterialProperties"
    mesh_size: float
    element_type: str

    # Mesh data
    nodes: torch.Tensor
    elements: torch.Tensor

    # Solution data
    displacement: torch.Tensor  # Nodal displacements
    stress: torch.Tensor  # Element stresses (6 components or 3x3 tensor)
    von_mises_stress: torch.Tensor  # Von Mises stress per element

    # Geometry info
    bounding_box: Dict[str, float]
    volume: float
    mass: float

    # Boundary conditions (optional, for visualization)
    model: Optional[Any] = None  # FEA solver model with constraints and forces

    # Visualization
    _plotter: Optional[Any] = None

    @property
    def max_displacement(self) -> float:
        """Maximum displacement magnitude (mm)"""
        return torch.norm(self.displacement, dim=1).max().item()

    @property
    def max_stress(self) -> float:
        """Maximum von Mises stress (MPa)"""
        return self.von_mises_stress.max().item()

    @property
    def min_stress(self) -> float:
        """Minimum von Mises stress (MPa)"""
        return self.von_mises_stress.min().item()

    @property
    def mean_stress(self) -> float:
        """Mean von Mises stress (MPa)"""
        return self.von_mises_stress.mean().item()

    def safety_factor(self, yield_strength: Optional[float] = None) -> float:
        """
        Calculate factor of safety based on yield strength.

        Args:
            yield_strength: Material yield strength (MPa).
                          If None, uses material.yield_strength

        Returns:
            Factor of safety (yield_strength / max_stress)
        """
        strength = yield_strength or self.material.yield_strength
        if strength is None:
            raise ValueError(
                "No yield strength specified. Provide yield_strength argument "
                "or use a material with defined yield_strength."
            )

        return strength / self.max_stress

    def show(
        self,
        display: str = "both",
        displacement_scale: float = 10.0,
        interactive: bool = True,
        window_size: Tuple[int, int] = (1600, 600),
    ):
        """
        Display FEA results interactively or save to file.

        Args:
            display: What to display - 'stress', 'displacement', 'both', or 'conditions'
            displacement_scale: Scale factor for displacement visualization
            interactive: Use interactive viewer (vs. off-screen). Default: True
            window_size: Window dimensions (width, height)

        Returns:
            PyVista plotter object (or None for 'conditions' mode)
        """
        try:
            import pyvista as pv
        except ImportError:
            raise ImportError(
                "PyVista is required for visualization. "
                "Install with: pip install rapidcadpy[fea]"
            )

        # Special handling for 'conditions' display mode
        if display == "conditions":
            if self.model is None:
                raise ValueError(
                    "Cannot display boundary conditions: model not available. "
                    "The FEA solver model must be stored in results.model to visualize conditions."
                )
            from rapidcadpy.fea.boundary_conditions import visualize_boundary_conditions

            visualize_boundary_conditions(
                self.model,
                self.nodes,
                self.elements,
                window_size=window_size,
                jupyter_backend=None if interactive else "static",
            )
            return None  # visualize_boundary_conditions handles display directly

        # Create PyVista mesh
        points = self.nodes.cpu().numpy()
        cells = self.elements.cpu().numpy()

        # Determine cell type
        cell_type_map = {
            "tet4": pv.CellType.TETRA,
            "tet10": pv.CellType.QUADRATIC_TETRA,
            "hex8": pv.CellType.HEXAHEDRON,
            "hex20": pv.CellType.QUADRATIC_HEXAHEDRON,
        }

        nodes_per_elem_map = {"tet4": 4, "tet10": 10, "hex8": 8, "hex20": 20}

        cell_type = cell_type_map.get(self.element_type, pv.CellType.TETRA)
        nodes_per_elem = nodes_per_elem_map.get(self.element_type, 4)

        # Create VTK cells
        vtk_cells = np.column_stack(
            [np.full(len(cells), nodes_per_elem), cells]
        ).ravel()
        celltypes = np.full(len(cells), cell_type)

        mesh = pv.UnstructuredGrid(vtk_cells, celltypes, points)

        # Add data to mesh
        mesh.point_data["displacement"] = self.displacement.cpu().numpy()
        mesh.point_data["displacement_magnitude"] = (
            torch.norm(self.displacement, dim=1).cpu().numpy()
        )
        mesh.cell_data["von_mises_stress"] = self.von_mises_stress.cpu().numpy()

        # Create plotter
        if display == "both":
            plotter = pv.Plotter(
                shape=(1, 2), window_size=list(window_size), off_screen=not interactive
            )

            # Left: Stress
            plotter.subplot(0, 0)
            plotter.add_text("Von Mises Stress", font_size=12, position="upper_edge")
            plotter.add_mesh(
                mesh,
                scalars="von_mises_stress",
                cmap="turbo",
                show_edges=False,
                scalar_bar_args={"title": "Stress (MPa)", "vertical": True},
            )
            plotter.camera_position = "iso"
            plotter.add_axes()

            # Right: Displacement
            plotter.subplot(0, 1)
            plotter.add_text(
                f"Displacement ({displacement_scale:.0f}x scaled)",
                font_size=12,
                position="upper_edge",
            )
            warped = mesh.warp_by_vector("displacement", factor=displacement_scale)
            plotter.add_mesh(
                warped,
                scalars="displacement_magnitude",
                cmap="viridis",
                show_edges=False,
                scalar_bar_args={"title": "Displacement (mm)", "vertical": True},
            )
            plotter.camera_position = "iso"
            plotter.add_axes()

        else:
            plotter = pv.Plotter(
                window_size=list(window_size), off_screen=not interactive
            )

            if display == "stress":
                plotter.add_text(
                    "Von Mises Stress", font_size=12, position="upper_edge"
                )
                plotter.add_mesh(
                    mesh,
                    scalars="von_mises_stress",
                    cmap="turbo",
                    show_edges=False,
                    scalar_bar_args={"title": "Stress (MPa)", "vertical": True},
                )
            elif display == "displacement":
                plotter.add_text(
                    f"Displacement ({displacement_scale:.0f}x scaled)",
                    font_size=12,
                    position="upper_edge",
                )
                warped = mesh.warp_by_vector("displacement", factor=displacement_scale)
                plotter.add_mesh(
                    warped,
                    scalars="displacement_magnitude",
                    cmap="viridis",
                    show_edges=False,
                    scalar_bar_args={"title": "Displacement (mm)", "vertical": True},
                )
            else:
                raise ValueError(f"Invalid display mode: {display}")

            plotter.camera_position = "iso"
            plotter.add_axes()

        if interactive:
            plotter.show()

        # Store mesh and plotter for later access
        self._plotter = plotter
        plotter.mesh = mesh

        return plotter

    def plot(
        self,
        path: str,
        display: str = "both",
        displacement_scale: float = 10.0,
        window_size: Tuple[int, int] = (1600, 600),
        transparent_background: bool = False,
    ) -> str:
        """
        Save FEA visualization to an image file.

        This method renders the same visualization as show() but saves it
        to a file instead of displaying interactively.

        Args:
            path: File path to save the image (e.g., 'results.png', 'stress.jpg')
            display: What to display - 'stress', 'displacement', 'both', or 'conditions'
            displacement_scale: Scale factor for displacement visualization
            window_size: Size of the rendered image (width, height) in pixels
            transparent_background: If True, save with transparent background

        Returns:
            The path to the saved image file

        Example:
            >>> results = shape.run_fea(...)
            >>> results.plot('stress_analysis.png', display='stress')
            >>> results.plot('full_results.png', display='both', window_size=(2400, 1200))
        """
        # Call show() with interactive=False to set up the plotter
        plotter = self.show(
            display=display,
            displacement_scale=displacement_scale,
            interactive=False,
            window_size=window_size,
        )

        # Save the screenshot
        plotter.screenshot(path, transparent_background=transparent_background)

        return path

    def to_vtk(self, filename: str):
        """
        Export results to VTK format.

        Args:
            filename: Output VTK file path
        """
        from rapidcadpy.fea.utils import export_to_vtk

        # Prepare stress components for export
        if len(self.stress.shape) == 3:
            # Full stress tensor (N x 3 x 3)
            stress_data = {
                "von_mises_stress": self.von_mises_stress,
                "sigma_xx": self.stress[:, 0, 0],
                "sigma_yy": self.stress[:, 1, 1],
                "sigma_zz": self.stress[:, 2, 2],
                "sigma_xy": self.stress[:, 0, 1],
                "sigma_xz": self.stress[:, 0, 2],
                "sigma_yz": self.stress[:, 1, 2],
            }
        else:
            # Just von Mises
            stress_data = {"von_mises_stress": self.von_mises_stress}

        export_to_vtk(
            filename,
            nodes=self.nodes,
            elements=self.elements,
            element_type=self.element_type,
            point_data={
                "displacement": self.displacement,
                "displacement_magnitude": torch.norm(
                    self.displacement, dim=1
                ).unsqueeze(-1),
            },
            cell_data=stress_data,
        )

    def summary(self) -> str:
        """
        Generate text summary of results.

        Returns:
            Formatted summary string
        """
        lines = [
            "=" * 80,
            "FEA ANALYSIS RESULTS SUMMARY",
            "=" * 80,
            f"\nMaterial: {self.material.name}",
            f"  E = {self.material.E:,.0f} MPa",
            f"  ν = {self.material.nu}",
            f"  Density = {self.material.density} g/cm³",
            f"\nMesh:",
            f"  Nodes: {self.nodes.shape[0]:,}",
            f"  Elements: {self.elements.shape[0]:,}",
            f"  Element type: {self.element_type}",
            f"  Mesh size: {self.mesh_size} mm",
            f"\nGeometry:",
            f"  Volume: {self.volume:.2f} mm³",
            f"  Mass: {self.mass:.4f} kg ({self.mass*1000:.2f} g)",
            f"\nResults:",
            f"  Max displacement: {self.max_displacement:.6e} mm",
            f"  Max von Mises stress: {self.max_stress:.2f} MPa",
            f"  Mean von Mises stress: {self.mean_stress:.2f} MPa",
        ]

        if self.material.yield_strength:
            sf = self.safety_factor()
            status = "SAFE" if sf > 1.5 else "WARNING" if sf > 1.0 else "FAILURE"
            lines.extend(
                [
                    f"\nSafety Analysis:",
                    f"  Yield strength: {self.material.yield_strength} MPa",
                    f"  Factor of safety: {sf:.2f}",
                    f"  Status: {status}",
                ]
            )

        lines.append("=" * 80)

        return "\n".join(lines)

    def __repr__(self):
        return (
            f"FEAResults(material={self.material.name}, "
            f"max_stress={self.max_stress:.1f} MPa, "
            f"max_displacement={self.max_displacement:.2e} mm)"
        )


@dataclass
class OptimizationResult:
    """Container for topology optimization results"""

    # Input parameters
    material: "MaterialProperties"
    mesh_size: float
    element_type: str
    volume_fraction: float
    num_iterations: int
    penalization: float

    # Mesh data
    nodes: torch.Tensor
    elements: torch.Tensor

    # Optimization results
    final_density: torch.Tensor  # Final element densities
    density_history: list  # History of density tensors
    compliance_history: list  # History of compliance values

    # Geometry info
    bounding_box: Dict[str, float]
    volume: float
    mass: float

    # FEA model (for visualization)
    model: Optional[Any] = None

    # Visualization
    _plotter: Optional[Any] = None

    @property
    def final_compliance(self) -> float:
        """Final compliance value"""
        return self.compliance_history[-1] if self.compliance_history else 0.0

    @property
    def final_volume_fraction(self) -> float:
        """Final achieved volume fraction"""
        return self.final_density.sum().item() / len(self.final_density)

    @property
    def convergence_ratio(self) -> float:
        """Ratio of final to initial compliance (lower is better)"""
        if len(self.compliance_history) < 2:
            return 1.0
        return self.compliance_history[-1] / self.compliance_history[0]

    def get_solid_elements(self, threshold: float = 0.5) -> torch.Tensor:
        """
        Get indices of elements that are considered solid.

        Args:
            threshold: Density threshold for considering an element solid

        Returns:
            Tensor of element indices where density > threshold
        """
        return torch.where(self.final_density > threshold)[0]

    def show(
        self,
        display: str = "density",
        threshold: float = 0.0,
        interactive: bool = True,
        window_size: Tuple[int, int] = (1200, 800),
        show_colorbar: bool = True,
    ):
        """
        Display optimization results interactively.

        Args:
            display: What to display - 'density', 'binary', 'convergence', 'solid'
            threshold: Minimum density to display (for 'density' and 'binary' modes)
            interactive: Use interactive viewer. Default: True
            window_size: Window dimensions (width, height)
            show_colorbar: Show colorbar for density. Default: True

        Returns:
            PyVista plotter object (for 'density'/'binary') or matplotlib figure (for 'convergence')
        """
        if display == "convergence":
            return self._show_convergence()
        elif display == "solid":
            return self._show_solid(threshold=threshold)
        else:
            return self._show_density(
                display, threshold, interactive, window_size, show_colorbar
            )

    def _show_convergence(self):
        """Plot compliance convergence history."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError(
                "matplotlib is required for convergence plotting. "
                "Install with: pip install matplotlib"
            )

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(self.compliance_history, "b-", linewidth=2)
        ax.set_xlabel("Iteration", fontsize=12)
        ax.set_ylabel("Compliance", fontsize=12)
        ax.set_title("Topology Optimization Convergence", fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.set_yscale("log")
        plt.tight_layout()
        plt.show()
        return fig

    def _show_density(
        self,
        display: str,
        threshold: float,
        interactive: bool,
        window_size: Tuple[int, int],
        show_colorbar: bool,
    ):
        """Display density distribution on mesh."""
        try:
            import pyvista as pv
        except ImportError:
            raise ImportError(
                "PyVista is required for visualization. "
                "Install with: pip install pyvista"
            )

        # Convert to numpy for PyVista
        nodes_np = self.nodes.detach().cpu().numpy()
        elements_np = self.elements.detach().cpu().numpy()
        density_np = self.final_density.detach().cpu().numpy()

        # Filter elements by threshold
        if threshold > 0:
            mask = density_np > threshold
            elements_np = elements_np[mask]
            density_np = density_np[mask]

        if len(elements_np) == 0:
            print(f"⚠ No elements above threshold {threshold}")
            return None

        # Create unstructured grid
        n_cells = len(elements_np)
        cell_types = np.full(n_cells, pv.CellType.TETRA, dtype=np.uint8)

        # PyVista expects cells in format: [n_points, p1, p2, p3, p4, ...]
        cells = np.hstack(
            [np.full((n_cells, 1), 4, dtype=np.int64), elements_np]
        ).ravel()

        mesh = pv.UnstructuredGrid(cells, cell_types, nodes_np)

        if display == "binary":
            # Binary display: solid elements only
            mesh.cell_data["density"] = np.ones(n_cells)
            cmap = "gray"
            clim = [0, 1]
        else:
            # Continuous density display
            mesh.cell_data["density"] = density_np
            cmap = "viridis"
            clim = [0, 1]

        # Create plotter
        plotter = pv.Plotter(window_size=window_size, off_screen=not interactive)
        plotter.add_mesh(
            mesh,
            scalars="density",
            cmap=cmap,
            clim=clim,
            show_scalar_bar=show_colorbar,
            scalar_bar_args={"title": "Density", "vertical": True},
        )

        plotter.add_axes()
        plotter.set_background("white")

        if interactive:
            plotter.show()

        self._plotter = plotter
        return plotter

    def plot(
        self,
        density: Optional[torch.Tensor] = None,
        cmap: str = "gray_r",
        show_bcs: bool = False,
        linewidth: float = 0,
        iteration: Optional[int] = None,
        **kwargs,
    ):
        """
        Plot optimization result using torch-fem's built-in plotting.

        This method uses the torch-fem model's native plot function for
        efficient visualization of the density distribution.

        Args:
            density: Custom density tensor to plot. If None, uses final_density
                    or density from specified iteration.
            cmap: Colormap name (default: "gray_r" for topology optimization)
            show_bcs: Whether to show boundary conditions (default: False)
            linewidth: Line width for mesh edges (default: 0, no edges)
            iteration: If specified, plot density from this iteration in history.
                      Use -1 for final, 0 for initial, etc.
            **kwargs: Additional arguments passed to model.plot()

        Returns:
            The plot object from torch-fem

        Example:
            >>> result.plot()  # Plot final density
            >>> result.plot(iteration=0)  # Plot initial density
            >>> result.plot(iteration=50)  # Plot density at iteration 50
            >>> result.plot(cmap="viridis", linewidth=0.1)  # Custom colormap with edges
        """
        if self.model is None:
            raise ValueError(
                "No FEA model available for plotting. "
                "The model may not have been stored during optimization."
            )

        # Determine which density to plot
        if density is not None:
            plot_density = density
        elif iteration is not None:
            if iteration < 0:
                iteration = len(self.density_history) + iteration
            if iteration < 0 or iteration >= len(self.density_history):
                raise IndexError(
                    f"Iteration {iteration} out of range. "
                    f"History has {len(self.density_history)} entries."
                )
            plot_density = self.density_history[iteration]
        else:
            plot_density = self.final_density

        # Use torch-fem's built-in plot function
        return self.model.plot(
            element_property=plot_density,
            cmap=cmap,
            bcs=show_bcs,
            linewidth=linewidth,
            **kwargs,
        )

    def plot_iteration(self, iteration: int, **kwargs):
        """
        Convenience method to plot density at a specific iteration.

        Args:
            iteration: Iteration number (0-indexed, or negative for reverse indexing)
            **kwargs: Additional arguments passed to plot()

        Returns:
            The plot object from torch-fem
        """
        return self.plot(iteration=iteration, **kwargs)

    def _show_solid(
        self,
        threshold: float = 0.5,
        cmap: str = "viridis",
        show_edges: bool = False,
        opacity: float = 1.0,
        window_size: Tuple[int, int] = (1200, 800),
        interactive: bool = True,
        background: str = "white",
    ):
        """
        Display only the solid (dense) material above a threshold.

        This creates a clean visualization showing only elements with density
        above the threshold, useful for visualizing the final optimized shape.

        Args:
            threshold: Minimum density to display (default: 0.5)
            cmap: Colormap for density values (default: "viridis")
            show_edges: Show mesh edges (default: False)
            opacity: Mesh opacity 0-1 (default: 1.0)
            window_size: Window dimensions (width, height)
            interactive: Use interactive viewer (default: True)
            background: Background color (default: "white")

        Returns:
            PyVista plotter object

        Example:
            >>> result.show_solid(threshold=0.3)  # Show elements > 30% density
            >>> result.show_solid(threshold=0.7, cmap="gray_r")  # High density only
        """
        try:
            import pyvista as pv
        except ImportError:
            raise ImportError(
                "PyVista is required for visualization. "
                "Install with: pip install pyvista"
            )

        # Convert to numpy
        nodes_np = self.nodes.detach().cpu().numpy()
        elements_np = self.elements.detach().cpu().numpy()
        density_np = self.final_density.detach().cpu().numpy()

        # Filter elements by threshold
        mask = density_np >= threshold
        solid_elements = elements_np[mask]
        solid_density = density_np[mask]

        if len(solid_elements) == 0:
            print(f"⚠ No elements with density >= {threshold}")
            return None

        print(f"Showing {len(solid_elements)} elements with density >= {threshold}")
        print(f"  ({100 * len(solid_elements) / len(elements_np):.1f}% of total)")

        # Create unstructured grid for solid elements only
        n_cells = len(solid_elements)
        cell_types = np.full(n_cells, pv.CellType.TETRA, dtype=np.uint8)
        cells = np.hstack(
            [np.full((n_cells, 1), 4, dtype=np.int64), solid_elements]
        ).ravel()

        mesh = pv.UnstructuredGrid(cells, cell_types, nodes_np)
        mesh.cell_data["density"] = solid_density

        # Create plotter
        plotter = pv.Plotter(window_size=window_size, off_screen=not interactive)
        plotter.set_background(background)

        plotter.add_mesh(
            mesh,
            scalars="density",
            cmap=cmap,
            clim=[threshold, 1.0],
            show_edges=show_edges,
            opacity=opacity,
            scalar_bar_args={
                "title": "Density",
                "vertical": True,
                "position_x": 0.85,
            },
        )

        plotter.add_axes()
        plotter.camera_position = "iso"

        if interactive:
            plotter.show()

        self._plotter = plotter
        return plotter

    def export_mesh(
        self,
        filename: str,
        threshold: float = 0.0,
    ):
        """
        Export the optimization result mesh to VTU format.

        This uses torchfem's export_mesh function to create a VTU file
        that can be opened in ParaView or other visualization tools.

        Args:
            filename: Output file path (should end in .vtu)
            threshold: Only export elements with density > threshold (default: 0.0, all)

        Example:
            >>> result.export_mesh("topology_result.vtu")
            >>> result.export_mesh("solid_only.vtu", threshold=0.5)
        """
        from torchfem.io import export_mesh

        if self.model is None:
            raise ValueError(
                "No FEA model available for export. "
                "The model must be stored during optimization."
            )

        density = self.final_density

        # Apply threshold if specified
        if threshold > 0:
            # Create a copy with zeroed-out low-density elements
            density = density.clone()
            density[density < threshold] = 0.0

        export_mesh(
            self.model,
            filename,
            elem_data={"density": density},
        )
        print(f"Mesh exported to {filename}")

    def plot_animation(
        self,
        output_path: Optional[str] = None,
        fps: int = 10,
        cmap: str = "gray_r",
        show_bcs: bool = False,
        skip_iterations: int = 1,
    ):
        """
        Create an animation of the optimization history.

        Args:
            output_path: Path to save animation (e.g., "optimization.gif").
                        If None, displays interactively.
            fps: Frames per second for animation
            cmap: Colormap name
            show_bcs: Whether to show boundary conditions
            skip_iterations: Plot every nth iteration (default: 1, all iterations)

        Returns:
            Animation object or None if saved to file
        """
        try:
            import matplotlib.pyplot as plt
            from matplotlib.animation import FuncAnimation
        except ImportError:
            raise ImportError(
                "matplotlib is required for animation. "
                "Install with: pip install matplotlib"
            )

        if self.model is None:
            raise ValueError("No FEA model available for animation.")

        # Select iterations to animate
        iterations = list(range(0, len(self.density_history), skip_iterations))
        if iterations[-1] != len(self.density_history) - 1:
            iterations.append(len(self.density_history) - 1)

        fig, ax = plt.subplots(figsize=(10, 8))

        def update(frame_idx):
            ax.clear()
            iteration = iterations[frame_idx]
            density = self.density_history[iteration]
            compliance = (
                self.compliance_history[iteration]
                if iteration < len(self.compliance_history)
                else self.compliance_history[-1]
            )

            # Use model's plot function (this may need adjustment based on torch-fem's API)
            self.model.plot(
                element_property=density,
                cmap=cmap,
                bcs=show_bcs,
                linewidth=0,
                ax=ax,
            )
            ax.set_title(f"Iteration {iteration} | Compliance: {compliance:.4e}")
            return (ax,)

        anim = FuncAnimation(
            fig,
            update,
            frames=len(iterations),
            interval=1000 // fps,
            blit=False,
        )

        if output_path:
            anim.save(output_path, fps=fps)
            print(f"Animation saved to {output_path}")
            plt.close(fig)
            return None
        else:
            plt.show()
            return anim

    def summary(self) -> str:
        """
        Generate a formatted summary of optimization results.

        Returns:
            Multi-line string with optimization summary
        """
        lines = []
        lines.append("=" * 80)
        lines.append("TOPOLOGY OPTIMIZATION RESULTS")
        lines.append("=" * 80)

        lines.append("\nOPTIMIZATION PARAMETERS:")
        lines.append("-" * 40)
        lines.append(f"  Target volume fraction: {self.volume_fraction:.2%}")
        lines.append(f"  Iterations: {self.num_iterations}")
        lines.append(f"  Penalization (p): {self.penalization}")
        lines.append(f"  Mesh size: {self.mesh_size} mm")

        lines.append("\nRESULTS:")
        lines.append("-" * 40)
        lines.append(f"  Final compliance: {self.final_compliance:.4e}")
        lines.append(f"  Final volume fraction: {self.final_volume_fraction:.2%}")
        lines.append(f"  Convergence ratio: {self.convergence_ratio:.4f}")

        # Element statistics
        solid_50 = (self.final_density > 0.5).sum().item()
        solid_90 = (self.final_density > 0.9).sum().item()
        total = len(self.final_density)

        lines.append("\nELEMENT STATISTICS:")
        lines.append("-" * 40)
        lines.append(f"  Total elements: {total}")
        lines.append(
            f"  Elements > 50% density: {solid_50} ({100*solid_50/total:.1f}%)"
        )
        lines.append(
            f"  Elements > 90% density: {solid_90} ({100*solid_90/total:.1f}%)"
        )

        lines.append("\nGEOMETRY:")
        lines.append("-" * 40)
        lines.append(f"  Original volume: {self.volume:.2f} mm³")
        lines.append(f"  Original mass: {self.mass*1000:.2f} g")

        lines.append("=" * 80)

        return "\n".join(lines)

    def __repr__(self):
        return (
            f"OptimizationResult(volume_fraction={self.final_volume_fraction:.2%}, "
            f"compliance={self.final_compliance:.4e}, "
            f"iterations={self.num_iterations})"
        )
