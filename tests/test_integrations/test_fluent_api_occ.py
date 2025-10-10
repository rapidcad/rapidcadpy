"""
Unit tests for the fluent API with OCC backend.

This module tests the fluent API integration with the OCC backend,
including creating simple geometries and exporting to STEP format.
"""

import pytest

import rapidcadpy as rc
from rapidcadpy.integrations.occ.app import OpenCascadeApp


class TestFluentAPIOCCBackend:
    """Test cases for the fluent API with OCC backend."""

    @pytest.mark.skipif(
        not pytest.importorskip("OCP", reason="OCP not available"),
        reason="OCP not available",
    )
    def test_simple_cube_creation_and_export(self):
        # Create a simple cube using fluent API
        app = OpenCascadeApp()
        workplane = app.work_plane("XY")
        cube = workplane.rect(10, 10).extrude(10)
        cube.to_stl("test_simple_cube_occ.stl")

    # @pytest.mark.skipif(
    #     not pytest.importorskip("OCC", reason="OCC not available"),
    #     reason="OCC not available",
    # )
    # def test_circle_extrusion_step_export(self):
    #     """Test creating a cylindrical shape and exporting to STEP."""
    #     if not self.occ_available:
    #         pytest.skip("OCC backend not available")
    #
    #     # Create a cylinder using your exact requested syntax
    #     wp_sketch0 = rc.Workplane(
    #         rc.Plane(rc.Vector(-0.2, 0.06, 0), rc.Vector(1, 0, 0), rc.Vector(0, 0, 1))
    #     )
    #     loop0 = wp_sketch0.moveTo(0.01, 0).circle(0.01)
    #     solid0 = wp_sketch0.add(loop0).extrude(0.75)
    #     solid = solid0
    #
    #     # Verify the CAD object
    #     assert isinstance(solid, rc.Cad)
    #     assert len(solid.construction_history) == 1
    #
    #     feature = solid.construction_history[0]
    #     assert feature.extrude.extent_one == 0.75
    #
    #     # Verify it's a circle
    #     sketch = feature.sketch[0]
    #     assert len(sketch.outer_wire.edges) == 1
    #     assert isinstance(sketch.outer_wire.edges[0], rc.Circle)
    #
    #     # Test export to STEP
    #     with tempfile.NamedTemporaryFile(
    #         suffix=".step", delete=False, dir="."
    #     ) as tmp_file:
    #         step_filename = tmp_file.name
    #
    #     try:
    #         solid.export(step_filename)
    #         assert os.path.exists(step_filename)
    #         assert os.path.getsize(step_filename) > 0
    #
    #         print(f"✓ Successfully exported cylinder to {step_filename}")
    #
    #     finally:
    #         if os.path.exists(step_filename):
    #             os.unlink(step_filename)
    #
    # @pytest.mark.skipif(
    #     not pytest.importorskip("OCC", reason="OCC not available"),
    #     reason="OCC not available",
    # )
    # def test_backend_switching_with_export(self):
    #     """Test switching between backends and exporting."""
    #     if not self.occ_available:
    #         pytest.skip("OCC backend not available")
    #
    #     # Create a model
    #     wp = rc.Workplane(
    #         rc.Plane(rc.Vector(0, 0, 0), rc.Vector(1, 0, 0), rc.Vector(0, 1, 0))
    #     )
    #     model = wp.circle(5).extrude(3)
    #
    #     # Test backend switching syntax
    #     original_backend = rc.backend
    #
    #     # Switch to OCC explicitly
    #     rc.backend = "occ"
    #     assert rc.backend == "occ"
    #
    #     # Export using OCC backend
    #     with tempfile.NamedTemporaryFile(
    #         suffix=".step", delete=False, dir="."
    #     ) as tmp_file:
    #         step_filename = tmp_file.name
    #
    #     try:
    #         model.export(step_filename)
    #         assert os.path.exists(step_filename)
    #         print("✓ Successfully exported using OCC backend")
    #
    #     finally:
    #         if os.path.exists(step_filename):
    #             os.unlink(step_filename)
    #
    #         # Restore original backend
    #         if original_backend:
    #             rc.backend = original_backend
    #
    # @pytest.mark.skipif(
    #     not pytest.importorskip("OCC", reason="OCC not available"),
    #     reason="OCC not available",
    # )
    # def test_complex_shape_with_lines(self):
    #     """Test creating a complex shape with multiple lines."""
    #     if not self.occ_available:
    #         pytest.skip("OCC backend not available")
    #
    #     # Create a complex shape using lines
    #     wp = rc.Workplane(
    #         rc.Plane(rc.Vector(0, 0, 0), rc.Vector(1, 0, 0), rc.Vector(0, 1, 0))
    #     )
    #     shape = wp.moveTo(0, 0).line(3, 0).line(0, 3).line(-3, 0).line(0, -3).extrude(2)
    #
    #     # Verify the model
    #     assert isinstance(shape, rc.Cad)
    #     feature = shape.construction_history[0]
    #     sketch = feature.sketch[0]
    #
    #     # Should have 4 lines forming a square
    #     assert len(sketch.outer_wire.edges) == 4
    #     for edge in sketch.outer_wire.edges:
    #         assert isinstance(edge, rc.Line)
    #
    #     # Test export
    #     with tempfile.NamedTemporaryFile(
    #         suffix=".step", delete=False, dir="."
    #     ) as tmp_file:
    #         step_filename = tmp_file.name
    #
    #     try:
    #         shape.export(step_filename)
    #         assert os.path.exists(step_filename)
    #         print("✓ Successfully exported complex shape")
    #
    #     finally:
    #         if os.path.exists(step_filename):
    #             os.unlink(step_filename)
    #
    # @pytest.mark.skipif(
    #     not pytest.importorskip("OCC", reason="OCC not available"),
    #     reason="OCC not available",
    # )
    # def test_direct_occ_backend_method(self):
    #     """Test using the direct OCC backend methods."""
    #     if not self.occ_available:
    #         pytest.skip("OCC backend not available")
    #
    #     # Create a simple model
    #     wp = rc.Workplane(
    #         rc.Plane(rc.Vector(0, 0, 0), rc.Vector(1, 0, 0), rc.Vector(0, 1, 0))
    #     )
    #     model = wp.rect(4, 4).extrude(1)
    #
    #     # Test direct backend method
    #     with tempfile.NamedTemporaryFile(
    #         suffix=".step", delete=False, dir="."
    #     ) as tmp_file:
    #         step_filename = tmp_file.name
    #
    #     try:
    #         # Use the to_backend method directly
    #         model.to_backend("occ", step_filename)
    #         assert os.path.exists(step_filename)
    #         print("✓ Direct OCC backend method works")
    #
    #     finally:
    #         if os.path.exists(step_filename):
    #             os.unlink(step_filename)
    #
    # def test_occ_backend_availability_check(self):
    #     """Test checking OCC backend availability."""
    #     available_backends = rc.get_available_backends()
    #
    #     # Check that the backends dictionary contains OCC
    #     assert "occ" in available_backends
    #     assert "opencascade" in available_backends
    #
    #     # OCC should be an alias for opencascade
    #     assert available_backends["occ"] == available_backends["opencascade"]
    #
    #     print(f"OCC backend available: {available_backends['occ']}")
    #
    # def test_fluent_api_methods_exist(self):
    #     """Test that all fluent API methods are available."""
    #     # Create a simple model to test method availability
    #     wp = rc.Workplane(
    #         rc.Plane(rc.Vector(0, 0, 0), rc.Vector(1, 0, 0), rc.Vector(0, 1, 0))
    #     )
    #     model = wp.circle(1).extrude(1)
    #
    #     # Check that backend methods exist on CAD objects
    #     assert hasattr(model, "export")
    #     assert hasattr(model, "to_backend")
    #     assert hasattr(model, "to_inventor")
    #
    #     # Check that backend management functions exist
    #     assert callable(rc.set_backend)
    #     assert callable(rc.get_current_backend)
    #     assert callable(rc.get_available_backends)
    #
    #     # Check that the backend property exists
    #     assert hasattr(rc, "backend")
    #
    #     print("✓ All fluent API methods are available")


class TestOCCBackendFallback:
    """Test cases for when OCC backend is not available."""

    def test_graceful_degradation_when_occ_unavailable(self):
        """Test that the system works gracefully when OCC is not available."""
        # This test will always run regardless of OCC availability

        # Check that we can still create models with fluent API
        wp = rc.Workplane(
            rc.Plane(rc.Vector(0, 0, 0), rc.Vector(1, 0, 0), rc.Vector(0, 1, 0))
        )
        model = wp.circle(1).extrude(1)

        assert isinstance(model, rc.Cad)
        assert len(model.construction_history) == 1

        # Check that backend functions exist even if backends aren't available
        available = rc.get_available_backends()
        assert isinstance(available, dict)
        assert "occ" in available

        print("✓ System works gracefully even when backends are unavailable")


if __name__ == "__main__":
    # Simple test runner for manual testing
    test_instance = TestFluentAPIOCCBackend()
    test_instance.setup_occ_backend()

    print("=== Testing Fluent API with OCC Backend ===")

    try:
        test_instance.test_occ_backend_availability_check()
        test_instance.test_fluent_api_methods_exist()

        if test_instance.occ_available:
            print("\nOCC backend is available - running full tests...")
            test_instance.test_simple_cube_creation_and_export()
            test_instance.test_circle_extrusion_step_export()
            test_instance.test_complex_shape_with_lines()
            test_instance.test_direct_occ_backend_method()
            test_instance.test_backend_switching_with_export()
        else:
            print("\nOCC backend not available - running fallback tests...")
            TestOCCBackendFallback().test_graceful_degradation_when_occ_unavailable()

    except Exception as e:
        print(f"Test failed: {e}")
        import traceback

        traceback.print_exc()

    print("\n=== Tests completed ===")
