# Load Case Parser - Quick Usage Guide

## Overview
The `load_case_parser.py` script can load, inspect, and visualize FEA load case JSON files.

## Basic Usage

### View Load Case Information
```bash
conda run -n cadgpt python experiments/load_case_parser.py <path_to_json>
```

**Example:**
```bash
conda run -n cadgpt python experiments/load_case_parser.py /mnt/data/deepcad_fea/0000/00000007_00001.json
```

This displays:
- Problem ID and description
- Design domain bounds
- Spatial selectors
- Boundary conditions
- Loads
- Material properties

### Export Design Domain to STEP
```bash
conda run -n cadgpt python experiments/load_case_parser.py <path_to_json> --export-step <output.step>
```

**Example:**
```bash
conda run -n cadgpt python experiments/load_case_parser.py /mnt/data/deepcad_fea/0000/00000007_00001.json --export-step /tmp/domain.step
```

### Render Load Case to Image File
```bash
conda run -n cadgpt python experiments/load_case_parser.py <path_to_json> --show <output.png>
```

**Example:**
```bash
conda run -n cadgpt python experiments/load_case_parser.py /mnt/data/deepcad_fea/0000/00000007_00001.json --show /tmp/loadcase.png
```

This generates a rendered image showing:
- Mesh (semi-transparent)
- Fixed nodes (green spheres)
- Loaded nodes (red spheres)
- Force vectors (red arrows)

### Visualize Load Case Interactively
```bash
conda run -n cadgpt python experiments/load_case_parser.py <path_to_json> --visualize
```

**Example:**
```bash
conda run -n cadgpt python experiments/load_case_parser.py /mnt/data/deepcad_fea/0000/00000007_00001.json --visualize --mesher netgen
```

Options:
- `--mesher netgen` (default) - Use Netgen mesher
- `--mesher gmsh` - Use Gmsh mesher

## Complete Workflow

### 1. Generate Load Case from STEP
```bash
conda run -n cadgpt python experiments/deepcad_to_loadcase.py \
    --step_file /mnt/data/deepcad_step/0000/00000007_00001.step \
    --out_dir /mnt/data/deepcad_fea/0000
```

### 2. Inspect the Generated Load Case
```bash
conda run -n cadgpt python experiments/load_case_parser.py \
    /mnt/data/deepcad_fea/0000/00000007_00001.json
```

### 3. Visualize (Optional)
```bash
conda run -n cadgpt python experiments/load_case_parser.py \
    /mnt/data/deepcad_fea/0000/00000007_00001.json \
    --visualize
```

## Output Example

```
============================================================
Loading load case from: /mnt/data/deepcad_fea/0000/00000007_00001.json
============================================================

Problem ID: DEEPCAD_AUTO
Description: Auto-generated load case from DeepCAD STEP.
Analysis Type: 3d
Units: mm

------------------------------------------------------------
Design Domain:
------------------------------------------------------------
Legacy Box Bounds: {'x_min': -0.5, 'y_min': -0.5, 'z_min': 0.0, ...}

------------------------------------------------------------
Spatial Selectors: 2
------------------------------------------------------------
  constraint_region:
    Type: box_3d
    Query: {...}
  load_region:
    Type: box_3d
    Query: {...}

------------------------------------------------------------
Boundary Conditions: 1
------------------------------------------------------------
  fix_constraint:
    Type: fixed_displacement
    Region ID: constraint_region
    Locked DOFs: x, y, z

------------------------------------------------------------
Loads: 1
------------------------------------------------------------
  applied_load:
    Type: distributed_force
    Region ID: load_region
    Magnitude: 1000.0 N
    Direction: -z

------------------------------------------------------------
Material Properties:
------------------------------------------------------------
  Type: isotropic
  Elastic Modulus: 69000 MPa
  Poisson's Ratio: 0.33
  Density: 2.7 g/cm³

============================================================
✓ Load case parsed successfully!
============================================================
```

## Command Reference

```bash
# Help
python experiments/load_case_parser.py --help

# View load case
python experiments/load_case_parser.py <json_file>

# Export domain
python experiments/load_case_parser.py <json_file> --export-step <output.step>

# Render to image file
python experiments/load_case_parser.py <json_file> --show <output.png>

# Visualize interactively
python experiments/load_case_parser.py <json_file> --visualize

# Visualize with specific mesher
python experiments/load_case_parser.py <json_file> --visualize --mesher gmsh

# All options combined
python experiments/load_case_parser.py <json_file> \
    --export-step domain.step \
    --show render.png \
    --mesher netgen
```

## Tips

1. **Quick Inspection**: Use without flags to quickly check what's in a load case JSON
2. **Validation**: The parser validates the JSON structure and warns about issues
3. **Rendering**: Use `--show` to save a PNG/image of the mesh with constraints and loads
4. **Interactive View**: Use `--visualize` to explore the mesh interactively (rotate, zoom, pan)
5. **Export**: Use `--export-step` to get the design domain as a STEP file for CAD tools
6. **Troubleshooting**: If visualization fails, check that the load and constraint regions match the geometry bounds

## Related Tools

- `deepcad_to_loadcase.py` - Generate load cases from STEP files
- `deepcad_loadcase_api.py` - Programmatic API for load case generation
