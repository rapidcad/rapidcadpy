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

    # Create shape and analyzer
    kernel = TorchFEMKernel()
    analyzer = FEAAnalyzer(box, Material.STEEL, kernel="torch-fem", mesh_size=1)

    # Add boundary conditions
    analyzer.add_constraint(FixedConstraint("x_min"))
    analyzer.add_load(DistributedLoad("top", force=-1000))

    # Run optimization
    print("Starting topology optimization...")
    result = analyzer.optimize(
        volume_fraction=0.4,
        num_iterations=20,
        penalization=3.0,
        filter_radius=0,
    )

    # View results
    print(result.summary())
    result.show(display="solid")
