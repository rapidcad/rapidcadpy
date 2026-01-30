# Experiments Directory

This directory contains experimental scripts and tools for CAD analysis and processing.

## 🎯 DeepCAD to FEA Load-Case Converter (NEW)

### Main Files
- **`deepcad_to_loadcase.py`** - CLI tool for batch STEP-to-JSON conversion
- **`deepcad_loadcase_api.py`** - Python API for programmatic usage
- **`test_deepcad_loadcase.py`** - Test suite and validation
- **`example_usage.sh`** - Shell script with usage examples

### Documentation
- **`INDEX.md`** - Project overview and navigation ⭐ **START HERE**
- **`QUICK_REFERENCE.md`** - Quick command reference
- **`README_deepcad_to_loadcase.md`** - Comprehensive user guide
- **`IMPLEMENTATION_SUMMARY.md`** - Technical details

### Quick Start
```bash
# Convert STEP files to FEA load-case JSON
conda run -n cadgpt python experiments/deepcad_to_loadcase.py \
    --step_dir /path/to/steps \
    --out_dir /path/to/output

# Run tests
conda run -n cadgpt python experiments/test_deepcad_loadcase.py
```

### Features
✅ Automatic face selection (constraint & load)  
✅ Spatial selector generation (AABBs)  
✅ Complete FEA load-case JSON output  
✅ Batch directory processing  
✅ 100% success rate on test dataset  

---

## 📊 Other Experiments

### Element Counters
- **`count_elements.py`** - Count geometric elements in CAD files
- **`count_shaft_features.py`** - Analyze shaft features
- **`primitive_counter.py`** - Count primitive shapes

---

## 📖 Getting Started

### For DeepCAD Converter
1. Read **`INDEX.md`** for overview
2. Check **`QUICK_REFERENCE.md`** for commands
3. Run **`test_deepcad_loadcase.py`** to verify setup
4. See **`README_deepcad_to_loadcase.md`** for detailed guide

### Requirements
- CadQuery 2.5.2+ (installed in `cadgpt` conda environment)
- Python 3.8+
- OCP (OpenCascade Python bindings)

---

## 🧪 Testing

```bash
# Test DeepCAD converter
conda run -n cadgpt python experiments/test_deepcad_loadcase.py

# Test Python API
conda run -n cadgpt python experiments/deepcad_loadcase_api.py
```

---

## 📦 Output Examples

Sample output files:
- **`test_loadcase_output.json`** - Example FEA load-case JSON

---

## 🎓 Learning Resources

- Start with INDEX.md for project overview
- QUICK_REFERENCE.md for immediate usage
- README_deepcad_to_loadcase.md for in-depth guide
- IMPLEMENTATION_SUMMARY.md for technical details

---

**Last Updated**: January 29, 2026
