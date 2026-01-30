"""
Example: Visualizing Boundary Conditions with file output

This example demonstrates the new FEAAnalyzer.show() method with support
for headless environments and file output.
"""

from rapidcadpy import Sketch
from rapidcadpy.fea import (
    FEAAnalyzer,
    Material,
    FixedConstraint,
    DistributedLoad,
)


def example_interactive_visualization():
    """Example showing boundary conditions interactively (requires display)."""

    # Create a simple beam
    sketch = Sketch("XY")
    sketch.add_rectangle(0, 0, 20, 5)
    beam = sketch.extrude(3)

    # Create analyzer
    analyzer = FEAAnalyzer(
        beam, Material.STEEL, kernel="torch-fem", mesh_size=1.0, device="cpu"
    )

    # Add boundary conditions
    analyzer.add_constraint(FixedConstraint("x_min"))
    analyzer.add_load(DistributedLoad("x_max", force=-100))

    # Show interactively (requires display)
    analyzer.show(interactive=True)


def example_headless_visualization():
    """Example saving boundary conditions to file (headless mode)."""

    # Create a simple beam
    sketch = Sketch("XY")
    sketch.add_rectangle(0, 0, 20, 5)
    beam = sketch.extrude(3)

    # Create analyzer
    analyzer = FEAAnalyzer(
        beam, Material.STEEL, kernel="torch-fem", mesh_size=1.0, device="cpu"
    )

    # Add boundary conditions
    analyzer.add_constraint(FixedConstraint("x_min"))
    analyzer.add_load(DistributedLoad("x_max", force=-100))

    # Save to file (headless mode, no display needed)
    analyzer.show(interactive=False, filename="boundary_conditions.png")
    print("Boundary condition visualization saved to boundary_conditions.png")


def example_custom_window_size():
    """Example with custom window size."""

    # Create a simple beam
    sketch = Sketch("XY")
    sketch.add_rectangle(0, 0, 20, 5)
    beam = sketch.extrude(3)

    # Create analyzer
    analyzer = FEAAnalyzer(
        beam, Material.STEEL, kernel="torch-fem", mesh_size=1.0, device="cpu"
    )

    # Add boundary conditions
    analyzer.add_constraint(FixedConstraint("x_min"))
    analyzer.add_load(DistributedLoad("x_max", force=-100))

    # Save with custom size
    analyzer.show(window_size=(1920, 1080), filename="boundary_conditions_hd.png")


if __name__ == "__main__":
    print("=" * 80)
    print("Boundary Condition Visualization Examples")
    print("=" * 80)

    print("\n1. Headless Visualization (saving to file):")
    print("-" * 80)
    example_headless_visualization()

    print("\n2. Custom Window Size:")
    print("-" * 80)
    example_custom_window_size()

    # Uncomment to test interactive mode (requires display)
    # print("\n3. Interactive Visualization:")
    # print("-" * 80)
    # example_interactive_visualization()
