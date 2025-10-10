"""
Example usage of the new CadQuery-like fluent API for rapidcadpy.

This demonstrates the syntax you requested and shows how to use Inventor as the backend.
"""

import rapidcadpy as rc


def main():
    print("=== RapidCADpy Fluent API Demo ===")

    # Example 1: Your requested syntax - Circle extrusion
    print("\n1. Creating a circle using your requested syntax:")
    wp_sketch0 = rc.Workplane(
        rc.Plane(rc.Vector(-0.2, 0.06, 0), rc.Vector(1, 0, 0), rc.Vector(0, 0, 1))
    )
    loop0 = wp_sketch0.moveTo(0.01, 0).circle(0.01)
    solid0 = wp_sketch0.add(loop0).extrude(0.75)
    solid = solid0

    print(f"✓ Circle solid created with {len(solid.construction_history)} features")
    solid.to_step("circle_example.step")
    print("✓ Exported to circle_example.step")

    # Example 2: Rectangle with different operations
    print("\n2. Creating a rectangle:")
    wp = rc.Workplane(
        rc.Plane(rc.Vector(0, 0, 0), rc.Vector(1, 0, 0), rc.Vector(0, 1, 0))
    )
    rect_solid = wp.moveTo(-5, -5).rect(10, 10).extrude(2)

    print("✓ Rectangle solid created")
    rect_solid.to_step("rectangle_example.step")
    print("✓ Exported to rectangle_example.step")

    # Example 3: Complex shape with lines
    print("\n3. Creating a complex shape with lines:")
    wp2 = rc.Workplane(
        rc.Plane(rc.Vector(0, 0, 0), rc.Vector(1, 0, 0), rc.Vector(0, 1, 0))
    )
    line_solid = (
        wp2.moveTo(0, 0).line(3, 0).line(0, 3).line(-3, 0).line(0, -3).extrude(1)
    )

    print("✓ Line-based solid created")
    line_solid.to_step("lines_example.step")
    print("✓ Exported to lines_example.step")

    # Example 4: Demonstrate Inventor backend (if Inventor is available)
    print("\n4. Inventor backend integration:")
    print("The to_inventor() method is now available on all CAD objects:")
    print("Usage: solid.to_inventor(inventor_app, part_doc, 'filename.ipt')")

    # The new fluent API supports:
    print("\n=== Available Fluent API Methods ===")
    print("• Workplane(plane) - Create a workplane")
    print("• .moveTo(x, y) - Move to position")
    print("• .circle(radius) - Create circle")
    print("• .rect(width, height) - Create rectangle")
    print("• .line(x, y) / .lineTo(x, y) - Create line")
    print("• .add(other_workplane) - Combine workplanes")
    print("• .extrude(distance) - Extrude to 3D")
    print("• CAD.to_inventor(app, doc, filename) - Export to Inventor")

    return solid, rect_solid, line_solid


if __name__ == "__main__":
    solids = main()
    print("\n✅ All examples completed successfully!")
    print(
        "The CadQuery-like syntax is now fully functional with Inventor backend support!"
    )
