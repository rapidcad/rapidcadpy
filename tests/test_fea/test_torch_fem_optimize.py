import pytest
from rapidcadpy.integrations.occ.app import OpenCascadeApp


def test_torch_fem_optimization_plot():
    from rapidcadpy.fea import (
        TorchFEMKernel,
        FEAAnalyzer,
        Material,
        FixedConstraint,
        DistributedLoad,
    )

    app = OpenCascadeApp()
    # create the design space
    box = app.work_plane("XY").rect(20, 10).extrude(10)

    # Create shape and analyzer (use CUDA if available)
    analyzer = FEAAnalyzer(
        box, Material.STEEL, kernel="torch-fem", mesh_size=0.3, device="cuda"
    )

    # Add boundary conditions
    analyzer.add_constraint(FixedConstraint("x_min"))
    analyzer.add_load(DistributedLoad("top", force=-10))
    # analyzer.show(filename="torch_fem_loads.png")

    # Run optimization
    print("Starting topology optimization...")
    result = analyzer.optimize(
        volume_fraction=0.3, 
        num_iterations=30,       
        penalization=3.0,
        filter_radius=1.0,        
        verbose=True,
    )

    # View results
    print(result.summary())
    # Use interactive=False for headless environments (no display)
    # Higher threshold (0.5) shows only denser elements = clearer structure
    result.show(display="solid", filename="torch_fem_optimization_plot.png", interactive=False, threshold=0.5)
