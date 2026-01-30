# Quick Reference - deepcad_to_loadcase.py

## Installation
```bash
# Requires CadQuery (already installed in cadgpt environment)
conda activate cadgpt
```

## Basic Commands

### Single File
```bash
conda run -n cadgpt python experiments/deepcad_to_loadcase.py \
    --step_file <path_to_step_file> \
    --out_dir <output_directory>
```

### Directory (Batch)
```bash
conda run -n cadgpt python experiments/deepcad_to_loadcase.py \
    --step_dir <input_directory> \
    --out_dir <output_directory>
```

## All Options
```bash
--step_file PATH        # Single STEP file to convert
--step_dir PATH         # Input directory with STEP files (mutually exclusive with --step_file)
--out_dir PATH          # Output directory for JSON files (required)
--force_newtons FLOAT   # Force magnitude in Newtons (default: 1000)
--face_tol FLOAT        # Face thickness tolerance in mm (default: 0.5)
--box_pad FLOAT         # Selector box padding in mm (default: 1.0)
--prefer_opposing FLOAT # Opposing face threshold (default: 0.85)
--summary_csv PATH      # Custom summary CSV path (default: out_dir/summary.csv)
```

## Quick Examples

### Process single file
```bash
conda run -n cadgpt python experiments/deepcad_to_loadcase.py \
    --step_file /mnt/data/deepcad_step/0000/00000007_00001.step \
    --out_dir /mnt/data/deepcad_fea/0000
```

### Process current directory
```bash
conda run -n cadgpt python experiments/deepcad_to_loadcase.py \
    --step_dir . \
    --out_dir ./loadcases
```

### High force scenario
```bash
conda run -n cadgpt python experiments/deepcad_to_loadcase.py \
    --step_dir ./parts \
    --out_dir ./loadcases \
    --force_newtons 5000
```

### Tight tolerances
```bash
conda run -n cadgpt python experiments/deepcad_to_loadcase.py \
    --step_dir ./parts \
    --out_dir ./loadcases \
    --face_tol 0.1 \
    --box_pad 0.5
```

## Output Structure
```
out_dir/
├── part1.json          # Load-case JSON for part1.step
├── part2.json          # Load-case JSON for part2.step
├── ...
└── summary.csv         # Processing statistics
```

## JSON Schema
```json
{
  "meta": {
    "problem_id": "DEEPCAD_AUTO",
    "analysis_type": "3d",
    "source_step": "path/to/source.step"
  },
  "design_domain": {
    "units": "mm",
    "bounds": {"x_min": ..., "x_max": ..., ...}
  },
  "spatial_selectors": [
    {
      "id": "constraint_region",
      "type": "box_3d",
      "query": {"x_min": ..., "x_max": ..., ...}
    },
    {
      "id": "load_region",
      "type": "box_3d",
      "query": {"x_min": ..., "x_max": ..., ...}
    }
  ],
  "boundary_conditions": [
    {
      "name": "fix_constraint",
      "region_id": "constraint_region",
      "type": "fixed_displacement",
      "dof_lock": {"x": true, "y": true, "z": true}
    }
  ],
  "loads": [
    {
      "name": "applied_load",
      "region_id": "load_region",
      "type": "distributed_force",
      "direction": "-z",  # or +x, -y, etc.
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

## Algorithm Summary

1. **Load STEP** → Extract all faces using OCP
2. **Analyze Faces** → Compute area, AABB, normal, planarity
3. **Select Constraint** → Largest planar face (or largest face)
4. **Select Load** → Opposing face (or perpendicular if no opposing)
5. **Generate Selectors** → Expand AABBs with padding and thickness
6. **Encode Direction** → Determine dominant axis from load face normal
7. **Write JSON** → Output complete load-case file

## Face Selection Logic

### Constraint Face
- Prefers: Planar faces
- Metric: Maximum area
- Usage: Fixed displacement boundary condition

### Load Face
- Prefers: Opposing face (normal ≈ -n_constraint)
- Fallback: Perpendicular face (normal ⊥ n_constraint)
- Score: `s = orientation * distance * area`
- Usage: Distributed force application

## Validation Checks
- ✓ All bounding boxes have min < max
- ✓ Constraint and load regions are distinct
- ✓ Load direction is valid axis string
- ✓ All required JSON fields present

## Testing
```bash
# Run test suite
conda run -n cadgpt python experiments/test_deepcad_loadcase.py

# View test output
cat experiments/test_loadcase_output.json
```

## Troubleshooting

### No faces found
- Check STEP file is valid
- Verify units are in millimeters
- Ensure file is not empty or corrupted

### Failed to select load face
- Part may have unusual geometry
- Try adjusting --prefer_opposing threshold
- Check if part has at least 2 faces

### Invalid bbox (min >= max)
- Verify STEP file coordinate system
- Check for degenerate geometry
- Review face_tol and box_pad values

## Common Use Cases

### DeepCAD Dataset
```bash
conda run -n cadgpt python experiments/deepcad_to_loadcase.py \
    --step_dir /mnt/data/deepcad_steps/0000 \
    --out_dir /mnt/data/deepcad_loadcases/0000 \
    --force_newtons 1500
```

### Custom Parts Library
```bash
conda run -n cadgpt python experiments/deepcad_to_loadcase.py \
    --step_dir ~/cad_parts \
    --out_dir ~/fea_loadcases \
    --face_tol 1.0 \
    --box_pad 2.0
```

### High-Precision Analysis
```bash
conda run -n cadgpt python experiments/deepcad_to_loadcase.py \
    --step_dir ./precision_parts \
    --out_dir ./precision_loadcases \
    --face_tol 0.1 \
    --box_pad 0.5 \
    --prefer_opposing 0.95
```

## Performance
- ~1-5 seconds per STEP file (depends on complexity)
- Processes 11 test files in < 5 seconds
- Scales linearly with number of files

## Dependencies
- CadQuery 2.5.2+
- OCP (OpenCascade Python bindings)
- Python 3.8+
- NumPy (indirect dependency)

## Support Files
- `README_deepcad_to_loadcase.md` - Full documentation
- `test_deepcad_loadcase.py` - Test suite
- `example_usage.sh` - Shell script examples
- `IMPLEMENTATION_SUMMARY.md` - Technical details
