"""
torch-fem based FEA kernel for OpenCASCADE shapes.
"""

import os
import numpy as np
from numpy import ndarray
from pyvista import UnstructuredGrid
import torch
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, List, Tuple, Optional

from torchfem import Solid

from rapidcadpy.fea.boundary_conditions import BoundaryCondition, Load
from rapidcadpy.fea.materials import MaterialProperties
from rapidcadpy.fea.mesher import MesherBase, NetgenMesher
from rapidcadpy.fea.utils import get_geometry_properties
from rapidcadpy.fea.kernels.base import FEAKernel
from rapidcadpy.fea.results import FEAResults, OptimizationResult
from torchfem.materials import IsotropicElasticity3D

if TYPE_CHECKING:
    from rapidcadpy.shape import Shape


class TorchFEMKernel(FEAKernel):
    """
    torch-fem based FEA kernel.

    This implementation uses torch-fem as the FEA solver backend,
    converting OCC shapes to STEP files, then meshing with Gmsh/Netgen.
    """

    def __init__(self, device: str = "auto", mesher: Optional[MesherBase] = None, num_threads: int = 0):
        """
        Initialize torch-fem kernel.
        
        Args:
            device: Device to use for computations ('cpu', 'cuda', or 'auto').
                    'auto' will use CUDA if available, otherwise CPU.
            mesher: Mesher instance to use for geometry meshing. 
                   If None, uses NetgenMesher by default.
            num_threads: Number of threads for meshing (0 = auto-detect).
                        Only used if mesher is None.
        """
        if not self.is_available():
            raise ImportError(
                "torch-fem dependencies not available. "
                "Install with: pip install rapidcadpy[fea]"
            )
        torch.set_default_dtype(torch.float64)
        
        # Set mesher
        if mesher is None:
            self.mesher = NetgenMesher(num_threads=num_threads)
        else:
            self.mesher = mesher
        
        # Set device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        
        # Set default device for all tensor operations
        torch.set_default_device(self.device)
        
        print(f"TorchFEMKernel initialized with device: {self.device}")

    @classmethod
    def is_available(cls) -> bool:
        """Check if torch-fem dependencies are available"""
        try:
            import torch
            import torchfem

            return True
        except ImportError:
            return False

    @classmethod
    def get_solver_name(cls) -> str:
        return "torch-fem"

    def solve(
        self,
        shape,
        material,
        loads: List,
        constraints: List,
        mesh_size: float,
        element_type: str,
    ) -> FEAResults:
        """
        Run FEA analysis using torch-fem.

        Args:
            shape: Shape to analyze
            material: Material properties
            loads: List of loads to apply
            constraints: List of boundary conditions to apply
            mesh_size: Target mesh element size (mm)
            element_type: Element type

        Returns:
            FEAResults object containing analysis results
        """

        # Step 1: Export mesh
        with tempfile.NamedTemporaryFile(suffix=".step", delete=False) as tmp:
            step_path = tmp.name
        shape.to_step(step_path)
        torch.set_default_dtype(torch.float64)
        nodes, elements = self.mesher.generate_mesh(
            step_path,
            mesh_size=mesh_size,
            element_type="tet4",
            dim=3,
        )

        elements = elements[:, [1, 0, 2, 3]]

        # Convert nodes to float64 and move to device
        nodes = nodes.to(torch.float64).to(self.device)
        elements = elements.to(self.device)

        # Step 2: Create FEM model
        from torchfem import Solid

        material_fem = material.to_torchfem()
        model = Solid(
            nodes,
            elements,
            IsotropicElasticity3D(
                E=210000.0, nu=0.3  # Young's modulus in MPa  # Poisson's ratio
            ),
        )

        if os.getenv("RCADPY_VERBOSE") == "1":
            print(f"✓ FEM model created")
            print(f"  Degrees of freedom: {model.n_dofs}")
            print(f"  Nodes: {model.n_nod}")
            print(f"  Elements: {model.n_elem}")

        # Step 3: Apply boundary conditions
        self._apply_boundary_conditions(
            model, nodes, elements, loads, constraints
        )

        solution = model.solve()

        # Get geo props
        geo_props = get_geometry_properties(step_file_path=step_path)

        # Step 5: Extract results
        results = self._extract_results(
            solution,
            nodes,
            elements,
            model,
            material,
            mesh_size,
            element_type,
            geo_props,
        )

        return results

    def _apply_boundary_conditions(
        self,
        model,
        nodes: torch.Tensor,
        elements: torch.Tensor,
        loads: List,
        constraints: List,
    ) -> None:
        """
        Apply loads and constraints to torch-fem model.

        Args:
            model: torchfem.Solid model
            nodes: Mesh nodes
            elements: Mesh elements
            loads: List of loads to apply
            constraints: List of boundary conditions to apply
        """

        from rapidcadpy.fea.utils import get_geometry_info

        geometry_info = get_geometry_info(nodes)

        if os.getenv("RCADPY_VERBOSE") == "1":
            bbox = geometry_info["bounding_box"]
            print(
                f"  Bounding box: X[{bbox['xmin']:.2f}, {bbox['xmax']:.2f}], "
                f"Y[{bbox['ymin']:.2f}, {bbox['ymax']:.2f}], "
                f"Z[{bbox['zmin']:.2f}, {bbox['zmax']:.2f}]"
            )

        total_constrained = 0
        total_loaded = 0

        # Apply constraints
        for constraint in constraints:
            num_nodes = constraint.apply(model, nodes, elements, geometry_info)
            if num_nodes is not None:
                total_constrained += num_nodes
            if os.getenv("RCADPY_VERBOSE") == "1":
                print(
                    f"  ✓ Applied {constraint.__class__.__name__} ({num_nodes} nodes)"
                )

        # Apply loads
        for load in loads:
            num_nodes = load.apply(model, nodes, elements, geometry_info)
            if num_nodes is not None:
                total_loaded += num_nodes
            if os.getenv("RCADPY_VERBOSE") == "1":
                print(f"  ✓ Applied {load.__class__.__name__} ({num_nodes} nodes)")

        # Warn if no constraints or loads
        if total_constrained == 0:
            print(
                "  ⚠ WARNING: No nodes were constrained! Model may be under-constrained."
            )
        if total_loaded == 0:
            print("  ⚠ WARNING: No loads were applied!")

    def _extract_results(
        self,
        solution: tuple,
        nodes: torch.Tensor,
        elements: torch.Tensor,
        model,
        material,
        mesh_size: float,
        element_type: str,
        geo_props: dict,
    ) -> FEAResults:
        """
        Extract results from torch-fem solution.

        Args:
            solution: Tuple of (u, f, sigma, F, alpha) from model.solve()
            nodes: Mesh nodes
            elements: Mesh elements
            model: FEA solver model (for boundary condition visualization)
            material: Material properties
            mesh_size: Mesh size used
            element_type: Element type used
            geo_props: Geometry properties dict

        Returns:
            FEAResults object
        """
        from rapidcadpy.fea.utils import calculate_von_mises, get_geometry_info

        u, f, sigma, F, alpha = solution

        # Calculate von Mises stress
        von_mises = calculate_von_mises(sigma)

        # Get geometry info
        geometry_info = get_geometry_info(nodes)

        # Create results object
        results = FEAResults(
            material=material,
            mesh_size=mesh_size,
            element_type=element_type,
            nodes=nodes,
            elements=elements,
            displacement=u,
            stress=sigma,
            von_mises_stress=von_mises,
            bounding_box=geometry_info["bounding_box"],
            volume=geo_props["volume_mm3"],
            mass=geo_props["mass_kg"],
            model=model,  # Store model for boundary condition visualization
        )

        return results

    def optimize(
        self,
        shape,
        material,
        loads: List,
        constraints: List,
        mesh_size: float,
        element_type: str,
        volume_fraction: float = 0.5,
        num_iterations: int = 100,
        penalization: float = 3.0,
        filter_radius: float = 0.0,
        move_limit: float = 0.2,
        rho_min: float = 1e-3,
        use_autograd: bool = False,
    ) -> "OptimizationResult":
        """
        Run topology optimization using SIMP (Solid Isotropic Material with Penalization).

        This method optimizes the material distribution to minimize compliance
        (maximize stiffness) subject to a volume constraint.

        Args:
            shape: Shape to optimize
            material: Material properties
            loads: List of loads to apply
            constraints: List of boundary conditions to apply
            mesh_size: Target mesh element size (mm)
            element_type: Element type
            volume_fraction: Target volume fraction (0 < v < 1), default 0.5
            num_iterations: Number of optimization iterations, default 100
            penalization: SIMP penalization factor (p), default 3.0
            filter_radius: Sensitivity filter radius (0 = no filtering), default 0.0
            move_limit: Maximum change in density per iteration, default 0.2
            rho_min: Minimum density to avoid singularity, default 1e-3
            use_autograd: Use automatic differentiation for sensitivities, default False

        Returns:
            OptimizationResult object containing optimization history and final design
        """
        from scipy.optimize import bisect
        from tqdm import tqdm

        if os.getenv("RCADPY_VERBOSE") == "1":
            print("=" * 80)
            print(f"TOPOLOGY OPTIMIZATION ({self.get_solver_name()})")
            print("=" * 80)
            print(f"  Volume fraction: {volume_fraction}")
            print(f"  Iterations: {num_iterations}")
            print(f"  Penalization (p): {penalization}")
            print(f"  Filter radius: {filter_radius}")
            print(f"  Move limit: {move_limit}")

        import time
        mesh_time = 0
        
        # Step 1: Create mesh and model
        torch.set_default_dtype(torch.float64)
        if hasattr(shape, "to_step"):  # Check if it's a Shape (duck typing)
            with tempfile.NamedTemporaryFile(suffix=".step", delete=False) as tmp:
                step_path = tmp.name
            shape.to_step(step_path)
            
            t0 = time.perf_counter()
            nodes, elements = self.mesher.generate_mesh(
                step_path,
                mesh_size=mesh_size,
                element_type="tet4",
                dim=3,
            )
            mesh_time = time.perf_counter() - t0

            elements = elements[:, [1, 0, 2, 3]]
            nodes = nodes.to(torch.float64)

            # Step 2: Create FEM model
            

            material_fem = material.to_torchfem()
            model = Solid(
                nodes,
                elements,
                IsotropicElasticity3D(E=210000.0, nu=0.3),
            )
        elif isinstance(shape, Solid):
            model = shape
            nodes = shape.nodes
            elements = shape.elements[:, [1, 0, 2, 3]]
            nodes = nodes.to(torch.float64)
            model.nodes = model.nodes.to(torch.float64)
            model.elements = model.elements[:, [1, 0, 2, 3]]

        if os.getenv("RCADPY_VERBOSE") == "1":
            print(f"\n✓ FEM model created")
            print(f"  Elements: {model.n_elem}")
            print(f"  Nodes: {model.n_nod}")
            print(f"  DOFs: {model.n_dofs}")

        # Step 3: Apply boundary conditions
        self._apply_boundary_conditions(
            model, nodes, elements, loads, constraints
        )

        # Step 4: Initialize optimization variables
        n_elem = model.n_elem
        rho_max = torch.ones(n_elem, dtype=torch.float64, device=self.device)
        rho_min_tensor = torch.full((n_elem,), rho_min, dtype=torch.float64, device=self.device)

        # Initial uniform density to satisfy volume constraint
        rho_0 = torch.full(
            (n_elem,), volume_fraction, dtype=torch.float64, requires_grad=use_autograd, device=self.device
        )
        V_0 = volume_fraction * n_elem  # Target volume

        rho_history = [rho_0]
        compliance_history = []
        move = move_limit

        # Compute element stiffness matrix for sensitivity calculation
        # k0() is a method in torchfem that returns element stiffness matrices
        k0 = self._get_element_stiffness(model)

        # Store original material stiffness tensor for SIMP interpolation
        C0 = model.material.C.clone()

        # Build filter matrix if needed
        if filter_radius > 0.0:
            H = self._build_filter_matrix(nodes, elements, filter_radius)
            if os.getenv("RCADPY_VERBOSE") == "1":
                nnz = H._nnz()
                total = n_elem * n_elem
                sparsity = 100 * (1 - nnz / total)
                print(f"  Filter matrix: {n_elem}×{n_elem}, {nnz:,} non-zeros ({sparsity:.2f}% sparse)")
        else:
            H = None

        if os.getenv("RCADPY_VERBOSE") == "1":
            print(f"\n{'─'*80}")
            print("Starting optimization iterations...")
            print(f"{'─'*80}")

        # Step 5: Optimization loop
        iterator = (
            tqdm(range(num_iterations), desc="Optimizing")
            if os.getenv("RCADPY_VERBOSE") == "1"
            else range(num_iterations)
        )

        import time
        solve_time = 0
        sensitivity_time = 0
        update_time = 0

        for k in iterator:
            rho_k = rho_history[k]

            # Adjust material stiffness with SIMP penalization
            # C = rho^p * C0 (element-wise scaling of material stiffness tensor)
            model.material.C = torch.einsum("n,nijkl->nijkl", rho_k**penalization, C0)

            # Compute solution
            # Use iterative solver (minres) with relaxed tolerance for speed
            # For topology optimization, high precision isn't needed in early iterations
            t0 = time.perf_counter()
            u_k, f_k, _, _, _ = model.solve(method="minres", rtol=1e-4)
            solve_time += time.perf_counter() - t0

            # Evaluation of compliance
            compliance = torch.inner(f_k.ravel(), u_k.ravel())
            compliance_history.append(compliance.item())

            t0 = time.perf_counter()
            if use_autograd:
                # Compute sensitivity via automatic differentiation
                sensitivity = torch.autograd.grad(compliance, rho_k)[0]
            else:
                # Compute analytical sensitivities
                u_j = u_k[elements].reshape(model.n_elem, -1)
                w_k = torch.einsum("...i, ...ij, ...j", u_j, k0, u_j)
                sensitivity = -penalization * rho_k ** (penalization - 1.0) * w_k

            # Filter sensitivities (if filter radius provided)
            if H is not None:
                # Use sparse matrix-vector multiplication
                numerator = torch.sparse.mm(H, (rho_k * sensitivity).unsqueeze(1)).squeeze(1)
                denominator = torch.sparse.sum(H, dim=1).to_dense() * rho_k
                sensitivity = numerator / denominator
            sensitivity_time += time.perf_counter() - t0

            # Optimality criteria update
            t0 = time.perf_counter()
            def make_step(mu):
                G_k = -sensitivity / mu
                upper = torch.min(rho_max, (1 + move) * rho_k)
                lower = torch.max(rho_min_tensor, (1 - move) * rho_k)
                rho_trial = G_k**0.5 * rho_k
                return torch.maximum(torch.minimum(rho_trial, upper), lower)

            # Constraint function for bisection
            def g(mu):
                rho_new = make_step(mu)
                return (rho_new.sum() - V_0).item()

            # Find the Lagrange multiplier via bisection
            with torch.no_grad():
                mu = bisect(g, 1e-10, 100.0)

            # Update density
            rho_new = make_step(mu)
            if use_autograd:
                rho_new = rho_new.detach().requires_grad_(True)
            rho_history.append(rho_new)
            update_time += time.perf_counter() - t0

            if os.getenv("RCADPY_VERBOSE") == "1" and (k + 1) % 10 == 0:
                tqdm.write(
                    f"  Iter {k+1}: Compliance = {compliance.item():.4e}, "
                    f"Volume = {rho_new.sum().item()/n_elem:.4f}"
                )

        if os.getenv("RCADPY_VERBOSE") == "1":
            total_time = mesh_time + solve_time + sensitivity_time + update_time
            print(f"\n  Timing breakdown:")
            print(f"    Meshing:     {mesh_time:.2f}s ({100*mesh_time/total_time:.1f}%)")
            print(f"    Solve:       {solve_time:.2f}s ({100*solve_time/total_time:.1f}%)")
            print(f"    Sensitivity: {sensitivity_time:.2f}s ({100*sensitivity_time/total_time:.1f}%)")
            print(f"    Update:      {update_time:.2f}s ({100*update_time/total_time:.1f}%)")

        # Get geometry properties
        if hasattr(shape, "volume"):  # If shape has volume method, use it
            volume = shape.volume()
            mass = volume * material.density / 1e9  # mm³ to m³
            geo_props = {"volume_mm3": volume, "mass_kg": mass}
        else:
            geo_props = get_geometry_properties(step_file_path=step_path)

        # Create optimization result
        result = OptimizationResult(
            material=material,
            mesh_size=mesh_size,
            element_type=element_type,
            nodes=nodes,
            elements=elements,
            final_density=rho_history[-1],
            density_history=rho_history,
            compliance_history=compliance_history,
            volume_fraction=volume_fraction,
            num_iterations=num_iterations,
            penalization=penalization,
            model=model,
            bounding_box=self._get_bounding_box(nodes),
            volume=geo_props["volume_mm3"],
            mass=geo_props["mass_kg"],
        )

        if os.getenv("RCADPY_VERBOSE") == "1":
            print(f"\n{'='*80}")
            print("OPTIMIZATION COMPLETE")
            print(f"{'='*80}")
            print(f"  Final compliance: {compliance_history[-1]:.4e}")
            print(f"  Final volume fraction: {rho_history[-1].sum().item()/n_elem:.4f}")

        return result

    def _get_element_stiffness(self, model) -> torch.Tensor:
        """
        Get the element stiffness matrix k0 from the model.

        Args:
            model: torchfem.Solid model

        Returns:
            Element stiffness matrix tensor
        """
        # torchfem's k0() is a method that returns the element stiffness matrices
        if hasattr(model, "k0") and callable(model.k0):
            return model.k0()
        elif hasattr(model, "k0"):
            return model.k0
        elif hasattr(model, "Ke"):
            return model.Ke
        else:
            raise AttributeError(
                "Could not find element stiffness matrix in model. "
                "Expected model.k0() method or model.Ke attribute."
            )

    def _build_filter_matrix(
        self, nodes: torch.Tensor, elements: torch.Tensor, radius: float
    ) -> torch.Tensor:
        """
        Build the sensitivity filter matrix H using sparse operations.

        The filter averages sensitivities over neighboring elements within
        the specified radius to avoid checkerboard patterns.

        Uses scipy's cKDTree for efficient neighbor search, avoiding O(n²) memory.

        Args:
            nodes: Mesh nodes
            elements: Mesh elements
            radius: Filter radius

        Returns:
            Sparse filter matrix H (COO format for memory efficiency)
        """
        from scipy.spatial import cKDTree

        # Compute element centroids
        centroids = nodes[elements].mean(dim=1).cpu().numpy()
        n_elem = elements.shape[0]

        # Use KD-tree for efficient neighbor search - O(n log n) instead of O(n²)
        tree = cKDTree(centroids)

        # Find all pairs within radius
        pairs = tree.query_pairs(radius, output_type='ndarray')

        # Build sparse filter matrix
        # Initialize with diagonal (self-weight = radius)
        row_indices = list(range(n_elem))
        col_indices = list(range(n_elem))
        values = [radius] * n_elem

        # Add neighbor weights (symmetric)
        for i, j in pairs:
            dist = np.linalg.norm(centroids[i] - centroids[j])
            weight = radius - dist
            # Add both (i,j) and (j,i) since filter is symmetric
            row_indices.extend([i, j])
            col_indices.extend([j, i])
            values.extend([weight, weight])

        # Create sparse tensor - KEEP IT SPARSE!
        indices = torch.tensor([row_indices, col_indices], dtype=torch.long, device=self.device)
        values_tensor = torch.tensor(values, dtype=torch.float64, device=self.device)
        H_sparse = torch.sparse_coo_tensor(indices, values_tensor, (n_elem, n_elem))

        # Coalesce to combine duplicate entries
        return H_sparse.coalesce()

    def _get_bounding_box(self, nodes: torch.Tensor) -> dict:
        """Get bounding box from nodes."""
        return {
            "xmin": nodes[:, 0].min().item(),
            "xmax": nodes[:, 0].max().item(),
            "ymin": nodes[:, 1].min().item(),
            "ymax": nodes[:, 1].max().item(),
            "zmin": nodes[:, 2].min().item(),
            "zmax": nodes[:, 2].max().item(),
        }

    def get_visualization_data(
        self,
        shape: "Shape",
        material: "MaterialProperties",
        loads: List["Load"],
        constraints: List["BoundaryCondition"],
        mesh_size: float,
        element_type: str,
    ) -> Tuple[UnstructuredGrid, ndarray, ndarray, ndarray, ndarray]:
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
            - pv_mesh: PyVista UnstructuredGrid mesh for visualization
            - nodes: Node coordinates as numpy array (N, 3)
            - constraint_mask: Boolean mask for constrained nodes (N,)
            - force_mask: Boolean mask for loaded nodes (N,)
            - force_vectors: Force vectors for loaded nodes (M, 3)
        """
        import numpy as np
        import pyvista as pv

        # Step 1: Export shape and create mesh
        with tempfile.NamedTemporaryFile(suffix=".step", delete=False) as tmp:
            step_path = tmp.name
        shape.to_step(step_path)
        torch.set_default_dtype(torch.float64)

        nodes, elements = self.mesher.generate_mesh(
            step_path,
            mesh_size=mesh_size,
            element_type="tet4",
            dim=3,
            verbose=False,
        )

        elements = elements[:, [1, 0, 2, 3]]
        nodes = nodes.to(torch.float64)

        # Step 2: Create FEM model
        from torchfem import Solid

        model = Solid(
            nodes,
            elements,
            IsotropicElasticity3D(E=210000.0, nu=0.3),
        )

        # Step 3: Apply boundary conditions
        self._apply_boundary_conditions(
            model, nodes, elements, loads, constraints
        )

        # Step 4: Create PyVista mesh
        nodes_np = nodes.cpu().numpy()
        elements_np = elements.cpu().numpy()

        # Create VTK cell array for tetrahedra
        n_cells = elements_np.shape[0]
        cells = np.hstack(
            [np.full((n_cells, 1), 4, dtype=np.int64), elements_np]  # 4 nodes per tet
        ).ravel()

        # Cell types: VTK_TETRA = 10
        cell_types = np.full(n_cells, 10, dtype=np.uint8)

        pv_mesh = pv.UnstructuredGrid(cells, cell_types, nodes_np)

        # Step 5: Extract constraint and force masks
        # Find nodes with any constraint
        constraint_mask = model.constraints.any(dim=1).cpu().numpy()

        # Find nodes with non-zero forces
        force_mask = (model.forces.abs() > 1e-10).any(dim=1).cpu().numpy()
        force_vectors = model.forces[force_mask].cpu().numpy()

        return pv_mesh, nodes_np, constraint_mask, force_mask, force_vectors
