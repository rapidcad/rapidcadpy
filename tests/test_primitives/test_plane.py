import numpy.testing as npt

from rapidcadpy.cad_types import Vector
from rapidcadpy.workplane import Workplane


def test_workplane_coordinate_system():
    """Test the unified workplane coordinate system functionality."""
    # Test basic creation
    wp = Workplane()
    assert wp.origin.x == 0.0
    assert wp.origin.y == 0.0
    assert wp.origin.z == 0.0
    assert wp.normal.z == 1.0  # Default XY plane should have Z normal

    # Test factory methods
    wp_xy = Workplane.xy_plane((1, 2, 3))
    assert wp_xy.origin.x == 1.0
    assert wp_xy.origin.y == 2.0
    assert wp_xy.origin.z == 3.0
    assert wp_xy.normal.z == 1.0

    wp_xz = Workplane.xz_plane((1, 2, 3))
    assert wp_xz.normal.y == 1.0  # XZ plane should have Y normal

    wp_yz = Workplane.yz_plane((1, 2, 3))
    assert wp_yz.normal.x == 1.0  # YZ plane should have X normal


def test_workplane_translation():
    """Test workplane translation functionality."""
    wp = Workplane()
    translated = wp.translate_plane((5, 5, 5))

    # Original should be unchanged
    assert wp.origin.x == 0.0
    assert wp.origin.y == 0.0
    assert wp.origin.z == 0.0

    # Translated should be moved
    assert translated.origin.x == 5.0
    assert translated.origin.y == 5.0
    assert translated.origin.z == 5.0


def test_workplane_json_serialization():
    """Test workplane JSON serialization and deserialization."""
    wp = Workplane((1, 2, 3), (1, 0, 0), (0, 1, 0), (0, 0, 1))

    json_data = wp.to_json()
    wp_reconstructed = Workplane.from_json(json_data)

    assert wp_reconstructed.origin.x == wp.origin.x
    assert wp_reconstructed.origin.y == wp.origin.y
    assert wp_reconstructed.origin.z == wp.origin.z


def test_workplane_to_from_vector():
    """Test workplane vector serialization and deserialization."""
    origin = Vector(1.0, 2.0, 3.0)
    x_dir = Vector(1.0, 0.0, 0.0)
    y_dir = Vector(0.0, 1.0, 0.0)
    z_dir = Vector(0.0, 0.0, 1.0)

    # Test workplane with custom coordinate system
    wp = Workplane(origin, x_dir, y_dir, z_dir)

    # Test that the workplane maintains the coordinate system
    npt.assert_allclose(
        [wp.origin.x, wp.origin.y, wp.origin.z], [1.0, 2.0, 3.0], atol=1e-6
    )
    npt.assert_allclose(
        [wp.x_dir.x, wp.x_dir.y, wp.x_dir.z], [1.0, 0.0, 0.0], atol=1e-6
    )
    npt.assert_allclose(
        [wp.y_dir.x, wp.y_dir.y, wp.y_dir.z], [0.0, 1.0, 0.0], atol=1e-6
    )
    npt.assert_allclose(
        [wp.z_dir.x, wp.z_dir.y, wp.z_dir.z], [0.0, 0.0, 1.0], atol=1e-6
    )
