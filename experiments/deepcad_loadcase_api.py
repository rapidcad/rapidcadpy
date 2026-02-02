"""
Python API wrapper for deepcad_to_loadcase functionality.

This module provides a programmatic interface to the STEP-to-loadcase
converter, useful for integration into larger pipelines.

Example usage:
    from deepcad_loadcase_api import convert_step_to_loadcase, batch_convert

    # Single file
    result = convert_step_to_loadcase(
        'part.step',
        output_dir='./loadcases',
        force_newtons=1500
    )

    # Batch processing
    results = batch_convert(
        step_dir='./parts',
        output_dir='./loadcases',
        force_newtons=2000
    )
"""

from pathlib import Path
from typing import Dict, List, Optional, Any
import json

# Import from main module
import sys

sys.path.insert(0, str(Path(__file__).parent))

from deepcad_to_loadcase import (
    load_step_and_extract_faces,
    compute_shape_bbox,
    select_constraint_face,
    select_load_face,
    generate_loadcase_json,
    process_step_file,
)


class LoadCaseConfig:
    """Configuration for load case generation."""

    def __init__(
        self,
        force_newtons: float = 1000.0,
        face_tol: float = 0.5,
        box_pad: float = 1.0,
        prefer_opposing: float = 0.85,
    ):
        self.force_newtons = force_newtons
        self.face_tol = face_tol
        self.box_pad = box_pad
        self.prefer_opposing = prefer_opposing


def convert_step_to_loadcase(
    step_path: Path | str,
    output_dir: Optional[Path | str] = None,
    config: Optional[LoadCaseConfig] = None,
    return_dict: bool = False,
) -> Dict[str, Any]:
    """
    Convert a single STEP file to FEA load-case JSON.

    Args:
        step_path: Path to STEP file
        output_dir: Output directory for JSON (if None, returns dict only)
        config: LoadCaseConfig instance (uses defaults if None)
        return_dict: If True, returns the JSON dict even when writing to file

    Returns:
        Dictionary with conversion result:
        {
            'success': bool,
            'loadcase': dict,  # JSON structure if successful
            'json_path': Path,  # Output path if written
            'error': str  # Error message if failed
        }
    """
    step_path = Path(step_path)

    if not step_path.exists():
        return {"success": False, "error": f"STEP file not found: {step_path}"}

    config = config or LoadCaseConfig()

    try:
        # Load and extract faces
        shape, face_props_list = load_step_and_extract_faces(step_path)

        if not shape or not face_props_list:
            return {"success": False, "error": "No faces found or failed to load STEP"}

        # Compute part bbox
        part_bbox = compute_shape_bbox(shape)
        if not part_bbox:
            return {"success": False, "error": "Failed to compute part bounding box"}

        # Select faces
        constraint_face = select_constraint_face(face_props_list)
        if not constraint_face:
            return {"success": False, "error": "Failed to select constraint face"}

        load_face = select_load_face(
            constraint_face, face_props_list, part_bbox, config.prefer_opposing
        )
        if not load_face:
            return {"success": False, "error": "Failed to select load face"}

        # Generate JSON
        loadcase = generate_loadcase_json(
            step_path,
            shape,
            constraint_face,
            load_face,
            part_bbox,
            config.force_newtons,
            config.face_tol,
            config.box_pad,
        )

        result = {
            "success": True,
            "loadcase": loadcase,
            "constraint_area": constraint_face.area,
            "load_area": load_face.area,
            "load_direction": loadcase["loads"][0]["direction"],
        }

        # Write to file if output_dir specified
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            json_path = output_dir / f"{step_path.stem}.json"
            with open(json_path, "w") as f:
                json.dump(loadcase, f, indent=2)

            result["json_path"] = json_path

        # Remove loadcase from result if not requested
        if not return_dict and output_dir:
            result.pop("loadcase", None)

        return result

    except Exception as e:
        return {"success": False, "error": str(e)}


def batch_convert(
    step_dir: Path | str,
    output_dir: Path | str,
    config: Optional[LoadCaseConfig] = None,
    patterns: List[str] = None,
) -> Dict[str, Any]:
    """
    Batch convert STEP files in a directory.

    Args:
        step_dir: Directory containing STEP files
        output_dir: Output directory for JSON files
        config: LoadCaseConfig instance
        patterns: List of glob patterns (default: ['*.step', '*.stp'])

    Returns:
        Dictionary with batch results:
        {
            'total': int,
            'successful': int,
            'failed': int,
            'results': List[Dict],  # Individual file results
            'summary': Dict  # Statistics
        }
    """
    step_dir = Path(step_dir)
    output_dir = Path(output_dir)
    config = config or LoadCaseConfig()
    patterns = patterns or ["*.step", "*.stp", "*.STEP", "*.STP"]

    if not step_dir.exists():
        return {
            "total": 0,
            "successful": 0,
            "failed": 1,
            "error": f"Step directory not found: {step_dir}",
        }

    # Find all STEP files
    step_files = []
    for pattern in patterns:
        step_files.extend(step_dir.rglob(pattern))
    step_files = sorted(set(step_files))

    # Process each file
    results = []
    successful = 0
    failed = 0

    for step_path in step_files:
        result = convert_step_to_loadcase(
            step_path, output_dir, config, return_dict=False
        )

        result["step_file"] = step_path.name
        results.append(result)

        if result["success"]:
            successful += 1
        else:
            failed += 1

    # Compute summary statistics
    if successful > 0:
        constraint_areas = [r["constraint_area"] for r in results if r["success"]]
        load_areas = [r["load_area"] for r in results if r["success"]]

        summary = {
            "avg_constraint_area": sum(constraint_areas) / len(constraint_areas),
            "avg_load_area": sum(load_areas) / len(load_areas),
            "max_constraint_area": max(constraint_areas),
            "max_load_area": max(load_areas),
            "min_constraint_area": min(constraint_areas),
            "min_load_area": min(load_areas),
        }
    else:
        summary = {}

    return {
        "total": len(step_files),
        "successful": successful,
        "failed": failed,
        "results": results,
        "summary": summary,
        "output_dir": output_dir,
    }


def validate_loadcase(loadcase: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate a load-case JSON structure.

    Args:
        loadcase: Load-case dictionary

    Returns:
        Dictionary with validation results:
        {
            'valid': bool,
            'errors': List[str],
            'warnings': List[str]
        }
    """
    errors = []
    warnings = []

    # Check required top-level keys
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
            errors.append(f"Missing required key: {key}")

    if errors:
        return {"valid": False, "errors": errors, "warnings": warnings}

    # Validate bounding boxes
    def validate_bbox(bbox: Dict[str, float], name: str):
        if bbox["x_min"] >= bbox["x_max"]:
            errors.append(
                f"{name}: Invalid X bounds ({bbox['x_min']} >= {bbox['x_max']})"
            )
        if bbox["y_min"] >= bbox["y_max"]:
            errors.append(
                f"{name}: Invalid Y bounds ({bbox['y_min']} >= {bbox['y_max']})"
            )
        if bbox["z_min"] >= bbox["z_max"]:
            errors.append(
                f"{name}: Invalid Z bounds ({bbox['z_min']} >= {bbox['z_max']})"
            )

    # Validate design domain
    validate_bbox(loadcase["design_domain"]["bounds"], "Design domain")

    # Validate spatial selectors
    if len(loadcase["spatial_selectors"]) < 2:
        warnings.append("Expected at least 2 spatial selectors")

    for selector in loadcase["spatial_selectors"]:
        if "id" not in selector:
            errors.append(f"Selector missing 'id' field")
        if "type" not in selector or selector["type"] != "box_3d":
            errors.append(f"Selector must have type='box_3d'")
        if "query" in selector:
            validate_bbox(
                selector["query"], f"Selector {selector.get('id', 'unknown')}"
            )

    # Validate loads
    if not loadcase["loads"]:
        warnings.append("No loads defined")

    for load in loadcase["loads"]:
        if "direction" not in load:
            errors.append("Load missing 'direction' field")
        elif load["direction"] not in ["-x", "+x", "-y", "+y", "-z", "+z"]:
            warnings.append(f"Unusual load direction: {load['direction']}")

        if "magnitude_newtons" not in load:
            errors.append("Load missing 'magnitude_newtons' field")
        elif load["magnitude_newtons"] <= 0:
            warnings.append(f"Load magnitude is zero or negative")

    # Validate boundary conditions
    if not loadcase["boundary_conditions"]:
        warnings.append("No boundary conditions defined")

    return {"valid": len(errors) == 0, "errors": errors, "warnings": warnings}


# Example usage
if __name__ == "__main__":
    # Single file example
    print("Example 1: Convert single STEP file")
    result = convert_step_to_loadcase(
        "../test_count.step",
        output_dir="/tmp/api_test",
        config=LoadCaseConfig(force_newtons=1500),
        return_dict=True,
    )

    if result["success"]:
        print(f"  ✓ Success!")
        print(f"    Constraint area: {result['constraint_area']:.2f} mm²")
        print(f"    Load area: {result['load_area']:.2f} mm²")
        print(f"    Load direction: {result['load_direction']}")
        print(f"    Output: {result.get('json_path', 'N/A')}")

        # Validate
        validation = validate_loadcase(result["loadcase"])
        print(f"  Validation: {'✓ Valid' if validation['valid'] else '✗ Invalid'}")
        if validation["errors"]:
            print(f"    Errors: {validation['errors']}")
        if validation["warnings"]:
            print(f"    Warnings: {validation['warnings']}")
    else:
        print(f"  ✗ Failed: {result['error']}")

    print("\nExample 2: Batch convert")
    batch_result = batch_convert(
        step_dir="..",
        output_dir="/tmp/api_batch_test",
        config=LoadCaseConfig(force_newtons=2000),
    )

    print(f"  Total files: {batch_result['total']}")
    print(f"  Successful: {batch_result['successful']}")
    print(f"  Failed: {batch_result['failed']}")

    if batch_result["summary"]:
        print(
            f"  Average constraint area: {batch_result['summary']['avg_constraint_area']:.2f} mm²"
        )
        print(
            f"  Average load area: {batch_result['summary']['avg_load_area']:.2f} mm²"
        )
