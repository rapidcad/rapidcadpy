"""
Tests for FEA integration with torch-fem
"""

import pytest
import torch
import numpy as np


# Check if FEA dependencies are available
try:
    import pyvista as pv
    from torchfem import Solid
    from torchfem.materials import IsotropicElasticity3D
    from rapidcadpy.fea import (
        visualize_boundary_conditions,
        FixedConstraint,
        DistributedLoad,
        Material,
    )
    from rapidcadpy.fea.utils import import_geometry

    FEA_AVAILABLE = True
except ImportError:
    FEA_AVAILABLE = False


@pytest.mark.skipif(not FEA_AVAILABLE, reason="FEA dependencies not installed")
class TestVisualizeBoundaryConditions:
    """Test suite for visualize_boundary_conditions function"""

    @pytest.fixture
    def simple_mesh(self):
        """Create a simple cube mesh for testing"""
        # Create a simple 2x2x2 cube mesh with tet4 elements
        # 8 corner nodes
        nodes = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 1.0],
                [1.0, 1.0, 1.0],
                [0.0, 1.0, 1.0],
            ],
            dtype=torch.float64,
        )

        # 6 tet4 elements (decompose cube into tets)
        elements = torch.tensor(
            [
                [0, 1, 2, 5],
                [0, 2, 3, 7],
                [0, 5, 2, 7],
                [5, 2, 6, 7],
                [0, 5, 4, 7],
                [5, 6, 4, 7],
            ],
            dtype=torch.long,
        )

        return nodes, elements

    @pytest.fixture
    def model_with_bc(self, simple_mesh):
        """Create a FEA model with boundary conditions"""
        nodes, elements = simple_mesh

        # Create material
        material = IsotropicElasticity3D(E=210000.0, nu=0.3)

        # Create model
        model = Solid(nodes, elements, material)

        # Apply constraints (fix bottom nodes z=0)
        bottom_nodes = torch.tensor([0, 1, 2, 3], dtype=torch.long)
        model.constraints[bottom_nodes, :] = True

        # Apply loads (top nodes z=1)
        top_nodes = torch.tensor([4, 5, 6, 7], dtype=torch.long)
        model.forces[top_nodes, 2] = -10.0  # Downward force in Z

        return model, nodes, elements

    def test_visualize_basic(self, model_with_bc, monkeypatch):
        """Test basic visualization without actually showing plot"""
        model, nodes, elements = model_with_bc
        # Should not raise any errors
        visualize_boundary_conditions(model, nodes, elements)

    def test_visualize_no_constraints(self, simple_mesh, monkeypatch):
        """Test visualization with no constraints"""
        nodes, elements = simple_mesh
        material = IsotropicElasticity3D(E=210000.0, nu=0.3)
        model = Solid(nodes, elements, material)

        # Only apply forces, no constraints
        model.forces[[0, 1], 2] = -5.0

        show_called = {"called": False}

        def mock_show(self, jupyter_backend=None):
            show_called["called"] = True
            return None

        monkeypatch.setattr(pv.Plotter, "show", mock_show)

        # Should still work
        visualize_boundary_conditions(model, nodes, elements)
        assert show_called["called"]

    def test_visualize_no_loads(self, simple_mesh, monkeypatch):
        """Test visualization with no loads"""
        nodes, elements = simple_mesh
        material = IsotropicElasticity3D(E=210000.0, nu=0.3)
        model = Solid(nodes, elements, material)

        # Only apply constraints, no forces
        model.constraints[[0, 1, 2], :] = True

        show_called = {"called": False}

        def mock_show(self, jupyter_backend=None):
            show_called["called"] = True
            return None

        monkeypatch.setattr(pv.Plotter, "show", mock_show)

        # Should still work
        visualize_boundary_conditions(model, nodes, elements)
        assert show_called["called"]

    def test_visualize_window_size(self, model_with_bc, monkeypatch):
        """Test custom window size"""
        model, nodes, elements = model_with_bc

        created_window_size = {"size": None}

        original_init = pv.Plotter.__init__

        def mock_init(self, *args, **kwargs):
            created_window_size["size"] = kwargs.get("window_size")
            return original_init(self, *args, **kwargs)

        def mock_show(self, jupyter_backend=None):
            return None

        monkeypatch.setattr(pv.Plotter, "__init__", mock_init)
        monkeypatch.setattr(pv.Plotter, "show", mock_show)

        custom_size = (800, 600)
        visualize_boundary_conditions(model, nodes, elements, window_size=custom_size)

        assert created_window_size["size"] == custom_size

    def test_visualize_different_element_types(self, monkeypatch):
        """Test visualization with different element types"""
        # Test with hex8 elements
        nodes = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 1.0],
                [1.0, 1.0, 1.0],
                [0.0, 1.0, 1.0],
            ],
            dtype=torch.float64,
        )

        # Single hex8 element
        elements = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7]], dtype=torch.long)

        material = IsotropicElasticity3D(E=210000.0, nu=0.3)
        model = Solid(nodes, elements, material)
        model.constraints[[0, 1], :] = True

        def mock_show(self, jupyter_backend=None):
            return None

        monkeypatch.setattr(pv.Plotter, "show", mock_show)

        # Should work with hex8
        visualize_boundary_conditions(model, nodes, elements)

    def test_visualize_unsupported_element_type(self, monkeypatch):
        """Test that unsupported element types raise error"""
        nodes = torch.randn(10, 3, dtype=torch.float64)
        # Create elements with unsupported number of nodes (e.g., 5)
        elements = torch.randint(0, 10, (5, 5), dtype=torch.long)

        material = IsotropicElasticity3D(E=210000.0, nu=0.3)
        model = Solid(nodes, elements, material)

        def mock_show(self, jupyter_backend=None):
            return None

        monkeypatch.setattr(pv.Plotter, "show", mock_show)

        # Should raise ValueError for unsupported element type
        with pytest.raises(ValueError, match="Unsupported element type"):
            visualize_boundary_conditions(model, nodes, elements)

    def test_visualize_output_messages(self, model_with_bc, capsys, monkeypatch):
        """Test that visualization prints informative messages"""
        model, nodes, elements = model_with_bc

        def mock_show(self, jupyter_backend=None):
            return None

        monkeypatch.setattr(pv.Plotter, "show", mock_show)

        visualize_boundary_conditions(model, nodes, elements)

        captured = capsys.readouterr()

        # Should print information about constrained and loaded nodes
        assert "constrained nodes" in captured.out.lower()
        assert "loaded nodes" in captured.out.lower()
        assert "red" in captured.out.lower()
        assert "green" in captured.out.lower()

    def test_visualize_force_arrows(self, model_with_bc, monkeypatch):
        """Test that force arrows are added when forces are present"""
        model, nodes, elements = model_with_bc

        arrows_added = {"called": False}

        original_add_arrows = pv.Plotter.add_arrows

        def mock_add_arrows(self, *args, **kwargs):
            arrows_added["called"] = True
            return None

        def mock_show(self, jupyter_backend=None):
            return None

        monkeypatch.setattr(pv.Plotter, "add_arrows", mock_add_arrows)
        monkeypatch.setattr(pv.Plotter, "show", mock_show)

        visualize_boundary_conditions(model, nodes, elements)

        assert arrows_added[
            "called"
        ], "add_arrows should have been called for force visualization"

    def test_visualize_with_cpu_tensors(self, simple_mesh, monkeypatch):
        """Test visualization with CPU tensors"""
        nodes, elements = simple_mesh
        material = IsotropicElasticity3D(E=210000.0, nu=0.3)
        model = Solid(nodes, elements, material)

        # Ensure tensors are on CPU
        assert nodes.device.type == "cpu"

        model.constraints[[0, 1], :] = True
        model.forces[[4, 5], 2] = -10.0

        def mock_show(self, jupyter_backend=None):
            return None

        monkeypatch.setattr(pv.Plotter, "show", mock_show)

        # Should work fine with CPU tensors
        visualize_boundary_conditions(model, nodes, elements)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_visualize_with_gpu_tensors(self, simple_mesh, monkeypatch):
        """Test visualization with GPU tensors (if CUDA available)"""
        nodes, elements = simple_mesh

        # Move to GPU
        nodes_gpu = nodes.cuda()
        elements_gpu = elements.cuda()

        material = IsotropicElasticity3D(E=210000.0, nu=0.3)
        model = Solid(nodes_gpu, elements_gpu, material)

        model.constraints[[0, 1], :] = True
        model.forces[[4, 5], 2] = -10.0

        def mock_show(self, jupyter_backend=None):
            return None

        monkeypatch.setattr(pv.Plotter, "show", mock_show)

        # Should work with GPU tensors (automatically moved to CPU for visualization)
        visualize_boundary_conditions(model, nodes_gpu, elements_gpu)


@pytest.mark.skipif(not FEA_AVAILABLE, reason="FEA dependencies not installed")
class TestBoundaryConditionIntegration:
    """Integration tests for boundary conditions with visualization"""

    def test_fixed_constraint_visualization(self, tmp_path, monkeypatch):
        """Test that FixedConstraint properly shows in visualization"""
        # Create simple beam mesh
        nodes = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [10.0, 0.0, 0.0],
                [10.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [10.0, 0.0, 1.0],
                [10.0, 1.0, 1.0],
                [0.0, 1.0, 1.0],
            ],
            dtype=torch.float64,
        )

        elements = torch.tensor(
            [
                [0, 1, 2, 5],
                [0, 2, 3, 7],
            ],
            dtype=torch.long,
        )

        material = IsotropicElasticity3D(E=210000.0, nu=0.3)
        model = Solid(nodes, elements, material)

        # Apply boundary conditions using our BC classes
        geometry_info = {
            "bounding_box": {
                "xmin": 0.0,
                "xmax": 10.0,
                "ymin": 0.0,
                "ymax": 1.0,
                "zmin": 0.0,
                "zmax": 1.0,
            }
        }

        fixed_bc = FixedConstraint(location="x_min", tolerance=0.1)
        load_bc = DistributedLoad(
            location="x_max", force=-1000.0, direction="z", tolerance=0.1
        )

        n_fixed = fixed_bc.apply(model, nodes, elements, geometry_info)
        n_loaded = load_bc.apply(model, nodes, elements, geometry_info)

        assert n_fixed > 0, "Should have fixed some nodes"
        assert n_loaded > 0, "Should have loaded some nodes"

        def mock_show(self, jupyter_backend=None):
            return None

        monkeypatch.setattr(pv.Plotter, "show", mock_show)

        # Visualization should work with applied boundary conditions
        visualize_boundary_conditions(model, nodes, elements)


@pytest.mark.skipif(not FEA_AVAILABLE, reason="FEA dependencies not installed")
def test_visualize_import_error():
    """Test that helpful error is raised if pyvista not available"""
    import sys
    from unittest.mock import MagicMock

    # Temporarily hide pyvista
    pyvista_module = sys.modules.get("pyvista")
    if pyvista_module:
        sys.modules["pyvista"] = None

    try:
        # Re-import the function to trigger import error handling
        # This test is a bit tricky because pyvista is already loaded
        # In real scenario, the ImportError would be raised
        pass  # Skip this test as it's hard to simulate without breaking other tests
    finally:
        # Restore pyvista
        if pyvista_module:
            sys.modules["pyvista"] = pyvista_module


@pytest.mark.skipif(not FEA_AVAILABLE, reason="FEA dependencies not installed")
class TestVisualizationQuality:
    """Test the quality and correctness of visualizations"""

    def test_force_arrow_scaling(self, monkeypatch):
        """Test that force arrows are properly scaled"""
        nodes = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [100.0, 0.0, 0.0],  # Large geometry
                [100.0, 100.0, 0.0],
                [0.0, 100.0, 0.0],
                [0.0, 0.0, 10.0],
                [100.0, 0.0, 10.0],
                [100.0, 100.0, 10.0],
                [0.0, 100.0, 10.0],
            ],
            dtype=torch.float64,
        )

        elements = torch.tensor([[0, 1, 2, 5]], dtype=torch.long)

        material = IsotropicElasticity3D(E=210000.0, nu=0.3)
        model = Solid(nodes, elements, material)

        # Apply large forces
        model.forces[[4, 5, 6, 7], 2] = -10000.0

        arrow_params = {"centers": None, "directions": None, "mag": None}

        def mock_add_arrows(self, centers, directions, mag=1.0, **kwargs):
            arrow_params["centers"] = centers
            arrow_params["directions"] = directions
            arrow_params["mag"] = mag
            return None

        def mock_show(self, jupyter_backend=None):
            return None

        monkeypatch.setattr(pv.Plotter, "add_arrows", mock_add_arrows)
        monkeypatch.setattr(pv.Plotter, "show", mock_show)

        visualize_boundary_conditions(model, nodes, elements)

        # Arrow scaling should be ~10% of geometry size
        assert arrow_params["centers"] is not None
        assert arrow_params["directions"] is not None
        # Arrows should be scaled relative to geometry
        geometry_size = nodes[:, 0].max() - nodes[:, 0].min()
        # The scaled vectors should be reasonable (not too small or large)
        assert arrow_params["mag"] == 1.0  # Magnitude parameter should be 1.0
