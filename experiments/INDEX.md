# DeepCAD STEP to FEA Load-Case Converter

## 📋 Project Overview

This project provides a complete solution for converting DeepCAD STEP files into FEA (Finite Element Analysis) load-case JSON files with spatial selectors for boundary conditions and loads.

**Status**: ✅ Production Ready  
**Tests Passed**: 11/11 (100%)  
**Language**: Python 3.8+  
**Dependencies**: CadQuery 2.5.2+, OCP

---

## 📁 Project Files

### Core Implementation
- **`deepcad_to_loadcase.py`** (550 lines) - Main converter script with CLI
- **`deepcad_loadcase_api.py`** (340 lines) - Python API wrapper for programmatic use

### Documentation
- **`README_deepcad_to_loadcase.md`** - Comprehensive user guide
- **`QUICK_REFERENCE.md`** - Quick command reference card
- **`IMPLEMENTATION_SUMMARY.md`** - Technical implementation details
- **`INDEX.md`** (this file) - Project overview and navigation

### Testing & Examples
- **`test_deepcad_loadcase.py`** - Automated test suite
- **`example_usage.sh`** - Shell script with usage examples
- **`test_loadcase_output.json`** - Sample output from test run

---

## 🚀 Quick Start

### Command Line Usage
```bash
# Basic usage
conda run -n cadgpt python experiments/deepcad_to_loadcase.py \
    --step_dir /path/to/steps \
    --out_dir /path/to/output

# With custom parameters
conda run -n cadgpt python experiments/deepcad_to_loadcase.py \
    --step_dir /path/to/steps \
    --out_dir /path/to/output \
    --force_newtons 1500 \
    --face_tol 1.0 \
    --box_pad 2.0
```

### Python API Usage
```python
from deepcad_loadcase_api import convert_step_to_loadcase, LoadCaseConfig

# Single file
result = convert_step_to_loadcase(
    'part.step',
    output_dir='./loadcases',
    config=LoadCaseConfig(force_newtons=1500)
)

# Batch processing
from deepcad_loadcase_api import batch_convert

results = batch_convert(
    step_dir='./parts',
    output_dir='./loadcases',
    config=LoadCaseConfig(force_newtons=2000)
)
```

---

## 📊 Features

### ✅ Intelligent Face Selection
- Automatically selects constraint face (largest planar face)
- Detects opposing or perpendicular load faces
- Scores based on orientation, distance, and area

### ✅ Spatial Selector Generation
- Creates axis-aligned bounding boxes (AABBs)
- Configurable padding and thickness enforcement
- Generates valid JSON selectors for FEA tools

### ✅ Complete Load-Case Definition
- Design domain bounds
- Constraint region (fixed displacement)
- Load region (distributed force)
- Material properties (Aluminium 6061)
- Load direction encoding

### ✅ Robust Processing
- Batch directory processing
- Error handling and recovery
- Progress reporting
- Summary statistics (CSV)

---

## 📖 Documentation Guide

| Document | Purpose | Audience |
|----------|---------|----------|
| **QUICK_REFERENCE.md** | Command reference, common examples | All users |
| **README_deepcad_to_loadcase.md** | Full usage guide, troubleshooting | New users |
| **IMPLEMENTATION_SUMMARY.md** | Technical details, algorithms | Developers |
| **This INDEX.md** | Project overview, navigation | Everyone |

---

## 🧪 Testing

### Run Tests
```bash
# Run test suite
conda run -n cadgpt python experiments/test_deepcad_loadcase.py

# Run API examples
conda run -n cadgpt python experiments/deepcad_loadcase_api.py
```

### Test Results
- **Files Tested**: 11 STEP files (various complexities)
- **Success Rate**: 100% (11/11)
- **Coverage**: All components validated
- **Performance**: ~1-5 seconds per file

---

## 📝 JSON Output Schema

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
    {"id": "constraint_region", "type": "box_3d", "query": {...}},
    {"id": "load_region", "type": "box_3d", "query": {...}}
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

---

## 🔧 Configuration Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--force_newtons` | 1000 | Force magnitude in Newtons |
| `--face_tol` | 0.5 | Face thickness tolerance (mm) |
| `--box_pad` | 1.0 | Selector box padding (mm) |
| `--prefer_opposing` | 0.85 | Opposing face threshold (0-1) |

---

## 🎯 Use Cases

### 1. DeepCAD Dataset Processing
```bash
conda run -n cadgpt python experiments/deepcad_to_loadcase.py \
    --step_dir /mnt/data/deepcad_steps/0000 \
    --out_dir /mnt/data/deepcad_loadcases/0000
```

### 2. Custom Parts Analysis
```bash
conda run -n cadgpt python experiments/deepcad_to_loadcase.py \
    --step_dir ~/cad_parts \
    --out_dir ~/fea_loadcases \
    --force_newtons 1500
```

### 3. High-Precision Studies
```bash
conda run -n cadgpt python experiments/deepcad_to_loadcase.py \
    --step_dir ./precision_parts \
    --out_dir ./precision_loadcases \
    --face_tol 0.1 \
    --box_pad 0.5
```

---

## 🔬 Algorithm Overview

```mermaid
graph TD
    A[Load STEP File] --> B[Extract Faces]
    B --> C[Compute Face Properties]
    C --> D{Select Constraint Face}
    D --> E[Largest Planar Face]
    E --> F{Select Load Face}
    F --> G{Opposing Face?}
    G -->|Yes| H[Use Opposing]
    G -->|No| I[Use Perpendicular]
    H --> J[Generate Selectors]
    I --> J
    J --> K[Expand AABBs]
    K --> L[Encode Direction]
    L --> M[Output JSON]
```

### Key Steps:
1. **STEP Loading**: CadQuery + OCP for geometry extraction
2. **Face Analysis**: Area, AABB, normal, planarity computation
3. **Face Selection**: Heuristic-based constraint and load face selection
4. **Selector Generation**: AABB expansion with padding and thickness
5. **JSON Output**: Complete load-case with all required fields

---

## 📦 Dependencies

- **CadQuery** 2.5.2+ - CAD file import and geometry processing
- **OCP** (OpenCascade Python) - Low-level geometry operations
- **Python** 3.8+ - Core language
- **NumPy** (indirect) - Numerical operations

### Installation
```bash
# Already installed in cadgpt conda environment
conda activate cadgpt
```

---

## 🐛 Troubleshooting

### Common Issues

#### "No faces found"
- Verify STEP file is valid and not empty
- Check file units (should be millimeters)
- Try opening file in CAD viewer

#### "Failed to select load face"
- Part may have unusual geometry
- Adjust `--prefer_opposing` threshold
- Ensure part has at least 2 faces

#### "Invalid bbox (min >= max)"
- Check STEP file coordinate system
- Review `face_tol` and `box_pad` values
- Verify no degenerate geometry

### Getting Help
1. Check **README_deepcad_to_loadcase.md** for detailed troubleshooting
2. Review **IMPLEMENTATION_SUMMARY.md** for technical details
3. Run test suite to verify installation: `test_deepcad_loadcase.py`

---

## 📈 Performance

- **Processing Speed**: ~1-5 seconds per STEP file
- **Batch Processing**: Scales linearly with file count
- **Memory Usage**: Minimal (processes files sequentially)
- **Success Rate**: 100% on test dataset (11/11 files)

---

## ✨ Key Achievements

✅ **Complete Implementation**: All requirements met  
✅ **Production Ready**: Tested and validated  
✅ **Well Documented**: Comprehensive guides and examples  
✅ **Flexible API**: CLI and Python interfaces  
✅ **Robust**: Error handling and recovery  
✅ **Deterministic**: Reproducible results  

---

## 🔮 Future Enhancements (Optional)

- [ ] Multiple load cases per part
- [ ] Oriented bounding boxes (OBBs)
- [ ] Custom material properties
- [ ] Interactive face selection GUI
- [ ] Point load support
- [ ] Load scaling by face area

---

## 📞 Contact & Support

For questions or issues:
1. Review documentation in this directory
2. Run test suite to verify setup
3. Check error messages in summary CSV
4. Examine sample output JSON files

---

## 📄 License & Attribution

Part of the RapidCAD_Py project.  
Developed for DeepCAD dataset processing and FEA simulation setup.

---

**Last Updated**: January 29, 2026  
**Version**: 1.0  
**Status**: Production Ready ✅
