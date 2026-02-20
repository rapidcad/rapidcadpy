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
from typing import TYPE_CHECKING, List, Tuple, Optional, Union

from torchfem import Solid

from ...fea.mesher.netgen_mesher import NetgenMesher, MesherBase

from ..boundary_conditions import BoundaryCondition, Load
from ..materials import MaterialProperties
from ..utils import get_geometry_properties
from .base import FEAKernel
from ..results import FEAResults, OptimizationResult
from torchfem.materials import IsotropicElasticity3D

if TYPE_CHECKING:
    from ...shape import Shape


class TorchFEMKernel(FEAKernel):
    """
    torch-fem based FEA kernel.

    This implementation uses torch-fem as the FEA solver backend,
    converting OCC shapes to STEP files, then meshing with Gmsh/Netgen.
    """

    def __init__(
        self,
        device: str = "auto",
        mesher: Optional[Union[MesherBase, str]] = None,
        num_threads: int = 0,
    ):
        """
        Initialize torch-fem kernel.

        Args:
            device: Device to use for computations ('cpu', 'cuda', or 'auto').
                    'auto' will use CUDA if available, otherwise CPU.
            mesher: Mesher to use. Can be:
                   - MesherBase instance (custom mesher)
                   - String: 'netgen', 'gmsh', or 'netgen-subprocess' (will instantiate the appropriate mesher)
                   - None: uses NetgenMesher by default
            num_threads: Number of threads for meshing (0 = auto-detect).
                        Only used if mesher is None or a string.
        """
        if not self.is_available():
            raise ImportError(
                "torch-fem dependencies not available. "
                "Install with: pip install rapidcadpy[fea]"
            )
        torch.set_default_dtype(torch.float64)

        # Set mesher
        if mesher is None or mesher == "netgen":
            self.mesher = NetgenMesher(num_threads=num_threads)
        elif isinstance(mesher, str):
            if mesher == "gmsh-subprocess":
                from ..mesher import GmshSubprocessMesher

                self.mesher = GmshSubprocessMesher(num_threads=num_threads)
            elif mesher == "netgen-subprocess":
                from ..mesher import NetgenSubprocessMesher

                self.mesher = NetgenSubprocessMesher(num_threads=num_threads)
            elif mesher == "gmsh-isolated":
                from ..mesher import IsolatedGmshMesher

                self.mesher = IsolatedGmshMesher(num_threads=num_threads)
            else:
                raise ValueError(
                    f"Unknown mesher: {mesher}. Supported: 'netgen', 'gmsh', 'netgen-subprocess', 'gmsh-isolated"
                )
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
        shape: Union["Shape", str],
        material,
        loads: List,
        constraints: List,
        mesh_size: float,
        element_type: str,
    ) -> FEAResults:
        """
        Run FEA analysis using torch-fem.

        Args:
            shape: Shape to analyze or path to STEP file
            material: Material properties
            loads: List of loads to apply
            constraints: List of boundary conditions to apply
            mesh_size: Target mesh element size (mm)
            element_type: Element type

        Returns:
            FEAResults object containing analysis results
        """

        # Step 1: Export mesh or use provided STEP file
        if isinstance(shape, str):
            # shape is a path to a STEP file
            step_path = shape
        else:
            # shape is a Shape object, export to STEP
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

        # elements = elements[:, [1, 0, 2, 3]]

        # Convert nodes to float64 and move to device
        nodes = nodes.to(torch.float64).to(self.device)
        elements = elements.to(self.device)

        # Step 2: Create FEM model
        from torchfem import Solid

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
            model, nodes, elements, loads, constraints, mesh_size=mesh_size
        )

        # ── Diagnostic dump: pre-solve inputs (enabled by RCADPY_DUMP_INPUTS=1)
        dump_call_id = None
        if os.getenv("RCADPY_DUMP_INPUTS") == "1":
            dump_call_id = self._dump_solve_inputs(
                model, nodes, elements, loads, constraints, mesh_size, step_path
            )
        # ───────────────────────────────────────────────────────────────────────

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

        # ── Diagnostic dump: post-solve results ────────────────────────────────
        if os.getenv("RCADPY_DUMP_INPUTS") == "1" and dump_call_id is not None:
            self._dump_solve_results(dump_call_id, results)
        # ───────────────────────────────────────────────────────────────────────

        return results

    # ------------------------------------------------------------------
    def _dump_solve_inputs(
        self,
        model,
        nodes: torch.Tensor,
        elements: torch.Tensor,
        loads: List,
        constraints: List,
        mesh_size: float,
        step_path,
    ) -> str:
        """Dump all inputs to model.solve() to a JSON-Lines file for analysis.

        Enable by setting environment variable::

            RCADPY_DUMP_INPUTS=1

        Files are written to the directory specified by RCADPY_DUMP_DIR
        (default: /tmp/fea_input_dumps).  Each call appends one record to
        ``<dump_dir>/fea_inputs.jsonl`` and returns the call_id string so
        that _dump_solve_results() can write a companion record.
        """
        import json, datetime, threading, hashlib, time

        dump_dir = Path(os.getenv("RCADPY_DUMP_DIR", "tmp/fea_input_dumps"))
        dump_dir.mkdir(parents=True, exist_ok=True)

        call_id = f"{int(time.time()*1000)}_{threading.get_ident()}"
        dump_file = dump_dir / "fea_inputs.jsonl"

        # ── nodes ──────────────────────────────────────────────────────
        npa = nodes.numpy() if not nodes.is_cuda else nodes.cpu().numpy()
        nodes_info = {
            "shape": list(npa.shape),
            "n_nodes": int(npa.shape[0]),
            "min_xyz": npa.min(axis=0).tolist(),
            "max_xyz": npa.max(axis=0).tolist(),
            "mean_xyz": npa.mean(axis=0).tolist(),
            "hash_md5": hashlib.md5(npa.tobytes()).hexdigest(),
        }

        # ── elements ───────────────────────────────────────────────────
        epa = elements.numpy() if not elements.is_cuda else elements.cpu().numpy()
        elements_info = {
            "shape": list(epa.shape),
            "n_elements": int(epa.shape[0]),
            "hash_md5": hashlib.md5(epa.tobytes()).hexdigest(),
        }

        # ── forces (model.forces) ─────────────────────────────────────
        try:
            fpa = (
                model.forces.numpy()
                if not model.forces.is_cuda
                else model.forces.cpu().numpy()
            )
            nz_mask = np.abs(fpa) > 1e-30
            nz_indices = np.argwhere(nz_mask)
            forces_info = {
                "shape": list(fpa.shape),
                "nonzero_count": int(nz_mask.sum()),
                "total_force_per_dof": fpa.sum(axis=0).tolist(),
                "nonzero_entries": [
                    {"node": int(r), "dof": int(c), "value": float(fpa[r, c])}
                    for r, c in nz_indices[:200]  # cap at 200 entries
                ],
                "hash_md5": hashlib.md5(fpa.tobytes()).hexdigest(),
            }
        except Exception as exc:
            forces_info = {"error": str(exc)}

        # ── constraints (model.constraints) ───────────────────────────
        try:
            cpa = (
                model.constraints.numpy()
                if not model.constraints.is_cuda
                else model.constraints.cpu().numpy()
            )
            constrained_mask = cpa.astype(bool)
            cz_indices = np.argwhere(constrained_mask)
            constraints_info = {
                "shape": list(cpa.shape),
                "constrained_dof_count": int(constrained_mask.sum()),
                "constrained_entries": [
                    {"node": int(r), "dof": int(c)}
                    for r, c in cz_indices[:500]  # cap at 500 entries
                ],
                "hash_md5": hashlib.md5(cpa.tobytes()).hexdigest(),
            }
        except Exception as exc:
            constraints_info = {"error": str(exc)}

        # ── BC objects ────────────────────────────────────────────────
        def _bc_repr(bc_list):
            out = []
            for bc in bc_list:
                try:
                    entry = {"type": type(bc).__name__}
                    entry.update(vars(bc))
                    out.append(entry)
                except Exception:
                    out.append({"type": str(type(bc)), "repr": repr(bc)})
            return out

        # ── assemble record ───────────────────────────────────────────
        record = {
            "call_id": call_id,
            "record_type": "inputs",
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "thread_id": threading.get_ident(),
            "step_path": str(step_path),
            "mesh_size": float(mesh_size),
            "nodes": nodes_info,
            "elements": elements_info,
            "forces": forces_info,
            "constraints": constraints_info,
            "load_objects": _bc_repr(loads),
            "constraint_objects": _bc_repr(constraints),
        }

        line = json.dumps(record, default=str)
        with open(dump_file, "a") as fh:
            fh.write(line + "\n")

        print(
            f"[DUMP-IN  {call_id}] "
            f"nodes={nodes_info['n_nodes']} "
            f"elems={elements_info['n_elements']} "
            f"force_nz={forces_info.get('nonzero_count','?')} "
            f"constr_dofs={constraints_info.get('constrained_dof_count','?')} "
            f"mesh={mesh_size:.4f} "
            f"node_hash={nodes_info['hash_md5'][:8]} "
            f"force_hash={forces_info.get('hash_md5','?')[:8]} "
            f"constr_hash={constraints_info.get('hash_md5','?')[:8]}"
        )
        return call_id

    # ------------------------------------------------------------------
    def _dump_solve_results(self, call_id: str, results) -> None:
        """Append a results record to fea_inputs.jsonl keyed by call_id.

        Works with both FEAResults dataclass objects and plain dicts.
        Wrapped in a broad try/except so it never crashes the training loop.
        """
        import json

        try:
            dump_dir = Path(os.getenv("RCADPY_DUMP_DIR", "tmp/fea_input_dumps"))
            dump_file = dump_dir / "fea_inputs.jsonl"

            # ── Extract von Mises stress from FEAResults or dict ──────────────
            try:
                # FEAResults dataclass path (preferred)
                vm_tensor = results.von_mises_stress
                vm = vm_tensor.detach().cpu().numpy().ravel()
                stress_info = {
                    "n_elements": int(vm.size),
                    "max": float(vm.max()),
                    "min": float(vm.min()),
                    "mean": float(vm.mean()),
                    "p50": float(np.percentile(vm, 50)),
                    "p95": float(np.percentile(vm, 95)),
                    "p99": float(np.percentile(vm, 99)),
                    # derived convenience fields
                    "max_stress": float(results.max_stress),
                    "min_stress": float(results.min_stress),
                    "mean_stress": float(results.mean_stress),
                    "stress_p95": float(results.stress_p95),
                    "stress_p99": float(results.stress_p99),
                    "safety_factor": (
                        float(results.safety_factor)
                        if hasattr(results, "safety_factor")
                        else None
                    ),
                    "safety_factor_p95": (
                        float(results.safety_factor_p95)
                        if hasattr(results, "safety_factor_p95")
                        else None
                    ),
                }
            except AttributeError:
                # Plain-dict fallback
                vm_raw = results.get("von_mises_stress", np.array([]))
                if hasattr(vm_raw, "detach"):
                    vm_raw = vm_raw.detach().cpu().numpy()
                vm = np.array(vm_raw).ravel()
                stress_info = {
                    "n_elements": int(vm.size),
                    "max": float(vm.max()) if vm.size > 0 else None,
                    "min": float(vm.min()) if vm.size > 0 else None,
                    "mean": float(vm.mean()) if vm.size > 0 else None,
                    "p50": float(np.percentile(vm, 50)) if vm.size > 0 else None,
                    "p95": float(np.percentile(vm, 95)) if vm.size > 0 else None,
                    "p99": float(np.percentile(vm, 99)) if vm.size > 0 else None,
                }

            record = {
                "call_id": call_id,
                "record_type": "results",
                "stress_stats": stress_info,
            }

            line = json.dumps(record, default=str)
            with open(dump_file, "a") as fh:
                fh.write(line + "\n")

            max_v = stress_info.get("max")
            p95_v = stress_info.get("p95")
            p99_v = stress_info.get("p99")
            print(
                f"[DUMP-OUT {call_id}] "
                f"max_stress={max_v:.4f}  p95={p95_v:.4f}  p99={p99_v:.4f}"
            )
        except Exception as exc:
            print(f"[DUMP-OUT {call_id}] ERROR (non-fatal): {exc}")

    # ------------------------------------------------------------------
    def _apply_boundary_conditions(
        self,
        model,
        nodes: torch.Tensor,
        elements: torch.Tensor,
        loads: List,
        constraints: List,
        mesh_size: float,
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

        from ..utils import get_geometry_info

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
            num_nodes = constraint.apply(
                model, nodes, elements, geometry_info=geometry_info, mesh_size=mesh_size
            )
            if num_nodes is not None:
                total_constrained += num_nodes
            if os.getenv("RCADPY_VERBOSE") == "1":
                print(
                    f"  ✓ Applied {constraint.__class__.__name__} ({num_nodes} nodes)"
                )

        # Apply loads
        for load in loads:
            num_nodes = load.apply(
                model, nodes, elements, geometry_info, mesh_size=mesh_size
            )
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

        # Check connectivity between loaded and constrained nodes
        if total_constrained > 0 and total_loaded > 0:
            self._validate_connectivity(model, nodes, elements)

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
        from ..utils import calculate_von_mises, get_geometry_info

        u, f, sigma, F, alpha = solution

        # Calculate von Mises stress
        von_mises = calculate_von_mises(sigma)
        von_mises_np = von_mises.detach().cpu().numpy()
        stress_p95 = float(np.percentile(von_mises_np, 95))
        stress_p99 = float(np.percentile(von_mises_np, 99))

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
            max_stress_p95=stress_p95,
            max_stress_p99=stress_p99,
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
        if isinstance(shape, str):
            # shape is a path to a STEP file
            step_path = shape
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
        elif hasattr(shape, "to_step"):  # Check if it's a Shape (duck typing)
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
            model, nodes, elements, loads, constraints, mesh_size=mesh_size
        )

        # Step 4: Initialize optimization variables
        n_elem = model.n_elem
        rho_max = torch.ones(n_elem, dtype=torch.float64, device=self.device)
        rho_min_tensor = torch.full(
            (n_elem,), rho_min, dtype=torch.float64, device=self.device
        )

        # Initial uniform density to satisfy volume constraint
        rho_0 = torch.full(
            (n_elem,),
            volume_fraction,
            dtype=torch.float64,
            requires_grad=use_autograd,
            device=self.device,
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
                print(
                    f"  Filter matrix: {n_elem}×{n_elem}, {nnz:,} non-zeros ({sparsity:.2f}% sparse)"
                )
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
                numerator = torch.sparse.mm(
                    H, (rho_k * sensitivity).unsqueeze(1)
                ).squeeze(1)
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
            print(
                f"    Meshing:     {mesh_time:.2f}s ({100*mesh_time/total_time:.1f}%)"
            )
            print(
                f"    Solve:       {solve_time:.2f}s ({100*solve_time/total_time:.1f}%)"
            )
            print(
                f"    Sensitivity: {sensitivity_time:.2f}s ({100*sensitivity_time/total_time:.1f}%)"
            )
            print(
                f"    Update:      {update_time:.2f}s ({100*update_time/total_time:.1f}%)"
            )

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
        pairs = tree.query_pairs(radius, output_type="ndarray")

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
        indices = torch.tensor(
            [row_indices, col_indices], dtype=torch.long, device=self.device
        )
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

    def _validate_connectivity(
        self,
        model,
        nodes: torch.Tensor,
        elements: torch.Tensor,
    ) -> bool:
        """
        Validate that loaded and constrained nodes are connected via the mesh.

        Uses BFS to find connected components and checks if all loaded and
        constrained nodes belong to the same component.

        Args:
            model: torchfem.Solid model with applied boundary conditions
            nodes: Mesh nodes
            elements: Mesh elements

        Raises:
            RuntimeError: If loaded and constrained nodes are disconnected
        """
        from collections import deque

        # Find constrained and loaded nodes
        constraint_mask = model.constraints.any(dim=1).cpu().numpy()
        force_mask = (model.forces.abs() > 1e-10).any(dim=1).cpu().numpy()

        constrained_nodes = set(np.where(constraint_mask)[0])
        loaded_nodes = set(np.where(force_mask)[0])

        if len(constrained_nodes) == 0 or len(loaded_nodes) == 0:
            return True  # Nothing to check

        # Build adjacency list from elements
        n_nodes = nodes.shape[0]
        adjacency = [set() for _ in range(n_nodes)]

        elements_np = elements.cpu().numpy()
        for elem in elements_np:
            # For each element, connect all pairs of nodes
            for i in range(len(elem)):
                for j in range(i + 1, len(elem)):
                    adjacency[elem[i]].add(elem[j])
                    adjacency[elem[j]].add(elem[i])

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

        if disconnected_loads:
            return False

        return True

    def get_visualization_data(
        self,
        shape: Union["Shape", str],
        material: "MaterialProperties",
        loads: List["Load"],
        constraints: List["BoundaryCondition"],
        mesh_size: float,
        element_type: str,
        with_conditions: bool = True,
    ) -> Tuple[UnstructuredGrid, ndarray, ndarray, ndarray, ndarray]:
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

        # Step 1: Export shape or use provided STEP file
        if isinstance(shape, str):
            # shape is a path to a STEP file
            step_path = shape
        else:
            # shape is a Shape object, export to STEP
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
        nodes = nodes.to(torch.float64)

        # Optimization: If no loads/constraints, skip FEM model creation
        if not with_conditions:
            nodes_np = nodes.cpu().numpy()
            elements_np = elements.cpu().numpy()

            # Create VTK cell array for tetrahedra
            n_cells = elements_np.shape[0]
            cells = np.hstack(
                [
                    np.full((n_cells, 1), 4, dtype=np.int64),
                    elements_np,
                ]  # 4 nodes per tet
            ).ravel()

            # Cell types: VTK_TETRA = 10
            cell_types = np.full(n_cells, 10, dtype=np.uint8)

            pv_mesh = pv.UnstructuredGrid(cells, cell_types, nodes_np)

            # Return empty masks/vectors
            n_nodes = nodes_np.shape[0]
            constraint_mask = np.zeros(n_nodes, dtype=bool)
            force_mask = np.zeros(n_nodes, dtype=bool)
            force_vectors = np.zeros((0, 3))

            return pv_mesh, nodes_np, constraint_mask, force_mask, force_vectors

        # Step 2: Create FEM model
        from torchfem import Solid

        model = Solid(
            nodes,
            elements,
            IsotropicElasticity3D(E=210000.0, nu=0.3),
        )

        # Step 3: Apply boundary conditions
        self._apply_boundary_conditions(
            model, nodes, elements, loads, constraints, mesh_size=mesh_size
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
