"""
Tests for FEAAnalyzer connectivity validation.
"""

import pytest
from rapidcadpy import OpenCascadeApp
from rapidcadpy.fea import FEAAnalyzer
from rapidcadpy.fea.materials import Material, MaterialProperties
from rapidcadpy.fea.boundary_conditions import DistributedLoad, DistributedLoad, FixedConstraint, Load



def test_connectivity_positive_single_box():
    """
    Test connectivity validation with a single connected box.
    
    Loads and constraints are on opposite faces of the same box,
    so they should be connected.
    """
    # Create a single box
    app = OpenCascadeApp()
    wp = app.work_plane("XY")
    box = wp.rect(100, 100).extrude(50)
    
    # Create analyzer with constraint on one face, load on opposite face
    fea = FEAAnalyzer(
        shape=box,
        material=Material.STEEL,
        kernel="torch-fem",
        mesh_size=20.0,
        loads=[DistributedLoad(location="z_max", direction="z", force=1000)],
        constraints=[FixedConstraint(location="x_min")],
        mesher="netgen",
    )
    
    # Validate connectivity - should be True (connected)
    is_connected = fea.validate_connectivity()
    
    assert is_connected is True, "Single box should have connected loads and constraints"


def test_connectivity_negative_disconnected_boxes():
    """
    Test connectivity validation with disconnected geometry.
    
    Two separate boxes: constraint on first box, load on second box.
    They are not connected, so validation should fail.
    """
    # Create two separate boxes (not connected)
    app = OpenCascadeApp()
    wp = app.work_plane("XY")
    box1 = wp.rect(50, 50).extrude(50)
    box2 = wp.move_to(100, 0).rect(50, 50).extrude(50)
    
    # Combine them without fusing (creates disconnected geometry)
    combined = box1.union(box2)
    
    # Create analyzer with constraint on left box, load on right box
    fea = FEAAnalyzer(
        shape=combined,
        material=Material.STEEL,
        kernel="torch-fem",
        mesh_size=20.0,
        loads=[DistributedLoad(location="z_max", direction="z", force=1000)],
        constraints=[FixedConstraint(location="x_min")],
        mesher="netgen",
    )

    fea.show()
    
    # Validate connectivity - should be False (disconnected)
    is_connected = fea.validate_connectivity()
    
    assert is_connected is False, "Disconnected boxes should fail connectivity check"


def test_connectivity_positive_connected_geometry(steel_material):
    """
    Test connectivity validation with connected geometry.
    
    Two boxes that are fused together - loads and constraints should be connected.
    """
    # Create two boxes connected by fusion
    app = OpenCascadeApp()
    box1 = Box(app, width=50, length=50, height=50).translate(0, 0, 0)
    box2 = Box(app, width=50, length=50, height=50).translate(50, 0, 0)  # Adjacent
    
    # Fuse them together (creates single connected body)
    connected = box1.fuse(box2)
    
    # Create analyzer with constraint on one end, load on other end
    fea = FEAAnalyzer(
        shape=connected,
        material=steel_material,
        kernel="torch-fem",
        mesh_size=15.0,
        loads=[Load(location="x_max", direction=(-1, 0, 0), magnitude=1000)],
        constraints=[FixedConstraint(location="x_min")],
        mesher="gmsh-subprocess",
    )
    
    # Validate connectivity - should be True (connected)
    is_connected = fea.validate_connectivity()
    
    assert is_connected is True, "Fused boxes should have connected loads and constraints"


def test_connectivity_no_loads_or_constraints(steel_material):
    """
    Test connectivity validation with no loads or constraints.
    
    Should return True (trivially valid) when nothing to validate.
    """
    # Create a simple box
    app = OpenCascadeApp()
    box = Box(app, width=100, length=100, height=50)
    
    # Create analyzer with no loads or constraints
    fea = FEAAnalyzer(
        shape=box,
        material=steel_material,
        kernel="torch-fem",
        mesh_size=20.0,
        mesher="gmsh-subprocess",
    )
    
    # Validate connectivity - should be True (nothing to validate)
    is_connected = fea.validate_connectivity()
    
    assert is_connected is True, "Empty boundary conditions should be considered valid"
