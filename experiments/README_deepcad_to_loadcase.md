# DeepCAD to FEA Load-Case Converter

## Overview

`deepcad_to_loadcase.py` converts DeepCAD STEP files into FEA load-case JSON files with spatial selectors for boundary conditions and loads.

## Features

- **Automatic Face Selection**: Intelligently selects constraint (largest face) and load (opposing/perpendicular) faces
- **Spatial Selector Generation**: Creates axis-aligned bounding boxes (AABBs) for FEA region selection
- **Configurable Parameters**: Adjustable force magnitude, tolerances, and face selection thresholds
- **Batch Processing**: Processes entire directories of STEP files
- **Robust Error Handling**: Continues processing even if individual files fail
- **Summary Statistics**: Generates CSV with processing results

## Installation

Requires CadQuery and OCP:

```bash
pip install cadquery
```

## Usage

### Basic Usage

```bash
python experiments/deepcad_to_loadcase.py \
    --step_dir /path/to/step/files \
    --out_dir /path/to/output/json
```

### Full Options

```bash
python experiments/deepcad_to_loadcase.py \
    --step_dir /path/to/step/files \
    --out_dir /path/to/output/json \
    --force_newtons 1000 \
    --face_tol 0.5 \
    --box_pad 1.0 \
    --prefer_opposing 0.85 \
    --summary_csv /path/to/summary.csv
```

### Parameters

- `--step_dir`: Directory containing STEP files (required)
- `--out_dir`: Output directory for JSON files (required)
- `--force_newtons`: Total force magnitude in Newtons (default: 1000)
- `--face_tol`: Tolerance for face thickness in mm (default: 0.5)
- `--box_pad`: Padding for selector boxes in mm (default: 1.0)
- `--prefer_opposing`: Threshold for opposing face detection (default: 0.85)
- `--summary_csv`: Path for summary CSV (default: out_dir/summary.csv)

## Output Format

Each STEP file generates a JSON file with:

### Structure

```json
{
  "meta": {
    "problem_id": "DEEPCAD_AUTO",
    "description": "Auto-generated load case from DeepCAD STEP.",
    "analysis_type": "3d",
    "source_step": "<path>"
  },
  "design_domain": {
    "units": "mm",
    "bounds": {
      "x_min": 0.0, "x_max": 1000.0,
      "y_min": 0.0, "y_max": 200.0,
      "z_min": 0.0, "z_max": 600.0
    }
  },
  "spatial_selectors": [
    {
      "id": "constraint_region",
      "type": "box_3d",
      "query": { "x_min": ..., "x_max": ..., ... }
    },
    {
      "id": "load_region",
      "type": "box_3d",
      "query": { "x_min": ..., "x_max": ..., ... }
    }
  ],
  "boundary_conditions": [
    {
      "name": "fix_constraint",
      "region_id": "constraint_region",
      "type": "fixed_displacement",
      "dof_lock": { "x": true, "y": true, "z": true }
    }
  ],
  "loads": [
    {
      "name": "applied_load",
      "region_id": "load_region",
      "type": "distributed_force",
      "direction": "-z",
      "magnitude_newtons": 1000.0
    }
  ],
  "material": {
    "type": "isotropic",
    "name": "Aluminium_6061",
    "elastic_modulus_mpa": 69000,
    "poissons_ratio": 0.33,
    "density_g_cm3": 2.70
  }
}
```

## Algorithm Details

### Face Selection

1. **Constraint Face**:
   - Prefers planar faces when available
   - Selects face with largest area
   - Used as the "fixed" region

2. **Load Face**:
   - First priority: Opposing face (normal opposite to constraint)
   - Second priority: Perpendicular face (normal 90° to constraint)
   - Scoring factors:
     - Opposing score: `s_opp = max(0, -dot(n_load, n_fix))`
     - Perpendicular score: `s_perp = 1 - abs(dot)`
     - Distance score: normalized center distance
     - Area score: face area

### Selector Box Generation

1. Compute face AABB (axis-aligned bounding box)
2. Apply padding (`box_pad`) to all dimensions
3. Optionally enforce thickness along face normal:
   - For planar faces with dominant axis-aligned normals
   - Clamps box thickness to `face_tol` in normal direction

### Load Direction

- Load applied opposite to face normal (into surface)
- Encoded as dominant axis: `-x`, `+y`, `-z`, etc.
- Determined by largest component of negated normal vector

## Example Workflow

```bash
# Process DeepCAD STEP files
python experiments/deepcad_to_loadcase.py \
    --step_dir /mnt/data/deepcad_steps/0000 \
    --out_dir /mnt/data/deepcad_loadcases/0000 \
    --force_newtons 1500 \
    --face_tol 1.0

# Review summary
cat /mnt/data/deepcad_loadcases/0000/summary.csv
```

## Troubleshooting

### Common Issues

1. **No faces found**: STEP file may be empty or corrupted
2. **Failed to select load face**: Part may have only one face or unusual geometry
3. **Invalid bbox**: Check STEP file units and scale

### Validation

Check JSON output:
- All `x_min < x_max`, `y_min < y_max`, `z_min < z_max`
- Constraint and load regions should not overlap completely
- Load direction should be reasonable for geometry

## Implementation Notes

- Uses OCP (OpenCascade) for STEP geometry processing
- Deterministic: same input produces same output
- Units: assumes STEP files are in millimeters
- Material: fixed to Aluminium 6061 (can be modified in code)
- Coordinate frame: uses STEP file coordinate system

## Future Enhancements

- [ ] Support for multiple load cases per part
- [ ] Non-axis-aligned selectors (oriented bounding boxes)
- [ ] User-defined material properties
- [ ] Interactive face selection GUI
- [ ] Validation against mesh geometry
