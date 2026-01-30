#!/usr/bin/env python3
"""
Test script for deepcad_to_loadcase.py

This script tests the converter with available STEP files in the workspace.
"""

import sys
from pathlib import Path
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from deepcad_to_loadcase import (
    load_step_and_extract_faces,
    compute_shape_bbox,
    select_constraint_face,
    select_load_face,
    generate_loadcase_json,
)


def test_basic_workflow():
    """Test the basic workflow with a known STEP file."""

    # Check for available STEP files in workspace
    workspace_root = Path(__file__).parent.parent
    test_step_files = []

    # Look for test STEP files
    for pattern in ["*.step", "*.stp"]:
        test_step_files.extend(workspace_root.glob(pattern))

    if not test_step_files:
        print("No STEP files found in workspace root for testing.")
        print("Please ensure you have STEP files available.")
        return False

    # Use first available STEP file
    step_path = test_step_files[0]
    print(f"Testing with: {step_path.name}")
    print("=" * 60)

    # Step 1: Load and extract faces
    print("\n1. Loading STEP file and extracting faces...")
    shape, face_props_list = load_step_and_extract_faces(step_path)

    if not shape or not face_props_list:
        print("   ✗ Failed to load STEP or extract faces")
        return False

    print(f"   ✓ Found {len(face_props_list)} faces")
    for i, fp in enumerate(face_props_list[:5], 1):  # Show first 5
        print(
            f"      Face {i}: area={fp.area:.2f}mm², planar={fp.is_planar}, "
            f"normal=({fp.normal[0]:.2f}, {fp.normal[1]:.2f}, {fp.normal[2]:.2f})"
        )
    if len(face_props_list) > 5:
        print(f"      ... and {len(face_props_list) - 5} more faces")

    # Step 2: Compute part bbox
    print("\n2. Computing part bounding box...")
    part_bbox = compute_shape_bbox(shape)

    if not part_bbox:
        print("   ✗ Failed to compute part bbox")
        return False

    print(f"   ✓ Part bbox:")
    print(f"      X: [{part_bbox['x_min']:.2f}, {part_bbox['x_max']:.2f}] mm")
    print(f"      Y: [{part_bbox['y_min']:.2f}, {part_bbox['y_max']:.2f}] mm")
    print(f"      Z: [{part_bbox['z_min']:.2f}, {part_bbox['z_max']:.2f}] mm")

    # Step 3: Select constraint face
    print("\n3. Selecting constraint face...")
    constraint_face = select_constraint_face(face_props_list)

    if not constraint_face:
        print("   ✗ Failed to select constraint face")
        return False

    print(f"   ✓ Constraint face:")
    print(f"      Area: {constraint_face.area:.2f} mm²")
    print(f"      Planar: {constraint_face.is_planar}")
    print(
        f"      Normal: ({constraint_face.normal[0]:.3f}, "
        f"{constraint_face.normal[1]:.3f}, {constraint_face.normal[2]:.3f})"
    )
    print(
        f"      Center: ({constraint_face.center[0]:.2f}, "
        f"{constraint_face.center[1]:.2f}, {constraint_face.center[2]:.2f})"
    )

    # Step 4: Select load face
    print("\n4. Selecting load face...")
    load_face = select_load_face(constraint_face, face_props_list, part_bbox)

    if not load_face:
        print("   ✗ Failed to select load face")
        return False

    print(f"   ✓ Load face:")
    print(f"      Area: {load_face.area:.2f} mm²")
    print(f"      Planar: {load_face.is_planar}")
    print(
        f"      Normal: ({load_face.normal[0]:.3f}, "
        f"{load_face.normal[1]:.3f}, {load_face.normal[2]:.3f})"
    )
    print(
        f"      Center: ({load_face.center[0]:.2f}, "
        f"{load_face.center[1]:.2f}, {load_face.center[2]:.2f})"
    )

    # Step 5: Generate JSON
    print("\n5. Generating load-case JSON...")
    loadcase = generate_loadcase_json(
        step_path,
        shape,
        constraint_face,
        load_face,
        part_bbox,
        force_newtons=1000.0,
        face_tol=0.5,
        box_pad=1.0,
    )

    print(f"   ✓ Generated JSON structure")
    print(f"      Constraint selector: box_3d")
    cs = loadcase["spatial_selectors"][0]["query"]
    print(f"         X: [{cs['x_min']:.2f}, {cs['x_max']:.2f}]")
    print(f"         Y: [{cs['y_min']:.2f}, {cs['y_max']:.2f}]")
    print(f"         Z: [{cs['z_min']:.2f}, {cs['z_max']:.2f}]")

    print(f"      Load selector: box_3d")
    ls = loadcase["spatial_selectors"][1]["query"]
    print(f"         X: [{ls['x_min']:.2f}, {ls['x_max']:.2f}]")
    print(f"         Y: [{ls['y_min']:.2f}, {ls['y_max']:.2f}]")
    print(f"         Z: [{ls['z_min']:.2f}, {ls['z_max']:.2f}]")

    print(f"      Load direction: {loadcase['loads'][0]['direction']}")
    print(f"      Force magnitude: {loadcase['loads'][0]['magnitude_newtons']} N")

    # Step 6: Validate JSON structure
    print("\n6. Validating JSON structure...")

    # Check required fields
    required_keys = [
        "meta",
        "design_domain",
        "spatial_selectors",
        "boundary_conditions",
        "loads",
        "material",
    ]
    for key in required_keys:
        if key not in loadcase:
            print(f"   ✗ Missing required key: {key}")
            return False

    # Check bbox validity
    for bbox_dict in [part_bbox, cs, ls]:
        if bbox_dict["x_min"] >= bbox_dict["x_max"]:
            print(
                f"   ✗ Invalid X bounds: {bbox_dict['x_min']} >= {bbox_dict['x_max']}"
            )
            return False
        if bbox_dict["y_min"] >= bbox_dict["y_max"]:
            print(
                f"   ✗ Invalid Y bounds: {bbox_dict['y_min']} >= {bbox_dict['y_max']}"
            )
            return False
        if bbox_dict["z_min"] >= bbox_dict["z_max"]:
            print(
                f"   ✗ Invalid Z bounds: {bbox_dict['z_min']} >= {bbox_dict['z_max']}"
            )
            return False

    print(f"   ✓ All validations passed")

    # Step 7: Write test output
    output_path = Path(__file__).parent / "test_loadcase_output.json"
    with open(output_path, "w") as f:
        json.dump(loadcase, f, indent=2)

    print(f"\n7. Test output written to: {output_path.name}")

    print("\n" + "=" * 60)
    print("✓ All tests passed!")
    print("=" * 60)

    return True


def main():
    """Run tests."""
    print("DeepCAD to Load-Case Converter - Test Script")
    print("=" * 60)

    try:
        success = test_basic_workflow()

        if success:
            print("\n✓ Test completed successfully!")
            return 0
        else:
            print("\n✗ Test failed!")
            return 1

    except Exception as e:
        print(f"\n✗ Test failed with exception: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
