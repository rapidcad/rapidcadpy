# PyCadSeq

A new Python library under heavy development for CAD sequence processing and manipulation. Designed with a primary focus on seamless integration with industry-grade CAD software like AutoDesk Inventor, enabling professional-level parametric 3D modeling, sketches, and construction history management.

![Status](https://img.shields.io/badge/status-under%20heavy%20development-orange.svg)
![Focus](https://img.shields.io/badge/focus-AutoDesk%20Inventor%20Integration-blue.svg)

## 🚀 Features

- **Industry-Grade CAD Integration**: Deep integration with AutoDesk Inventor for professional CAD workflows
- **Parametric CAD Modeling**: Create and manipulate 3D models using construction sequences
- **Professional Sketch-based Design**: Work with 2D sketches containing lines, arcs, circles, and constraints
- **Multiple Export Formats**: Export to STEP, STL, and native CAD formats
- **Enterprise CAD System Support**: Built-in support for Autodesk Inventor and OpenCASCADE
- **Advanced Data Processing**: Parse and process CAD data from Fusion 360 Gallery and DeepCAD datasets
- **Professional Visualization**: Built-in plotting and 3D visualization capabilities
- **Comprehensive Constraint System**: Handle geometric constraints like coincidence, perpendicularity, and parallelism

> **Note**: This library is currently under heavy development. Features and APIs may change as we continue to add features.

## 📦 Installation

### From source
```bash
git clone https://github.com/rapidcad/rapidcadpy.git
cd rapidcadpy
pip install -e .
```


## 🛠️ Dependencies

### Core Dependencies
- Python 3.8+
- NumPy 1.20+
- PyTorch 1.10+
- Plotly 5.0+
- Matplotlib 3.5+

### Optional Dependencies
- **OpenCASCADE**: For 3D geometry operations and STEP/STL export
- **win32com** (Windows): For Autodesk Inventor integration
- **Build123d**: For advanced 3D modeling capabilities

## 🏁 Quick Start

### Creating a Simple CAD Model (App-based approach)

```python
from rapidcadpy.integrations.inventor import InventorApp

# Initialize Inventor application
app = InventorApp()
app.new_document()

# Create a workplane on the XY plane
wp = app.work_plane("XY")

# Create a simple cube
cube = wp.move_to(-5, -5).rect(10, 10).extrude(10)

# Export to STEP format
cube.export("my_cube.step")
```

### More Advanced Example: Arm with Cut

```python
app = InventorApp()
app.new_document()
wp = app.work_plane("XY")

arm_thick = 5

# Create an arm shape
arm = (
    wp.move_to(-4.5, 0)
    .line_to(-4.5, 20)
    .line_to(-8, 45)
    .three_point_arc((0, 53), (8, 45))
    .line_to(4.5, 20)
    .line_to(4.5, 0)
    .three_point_arc((0, -4.5), (-4.5, 0))
    .extrude(arm_thick)
)

# Create a hole and cut it from the arm
hole = wp.move_to(0, 45).circle(4).extrude(arm_thick)
arm.cut(hole)
```

### Exporting Models

```python
# Export to STEP format
cube.export("model.step")

# Export to STL format  
cube.export("model.stl")

# Export to Autodesk Inventor (Windows only)
cube.export("model.ipt")
```

### Loading from CAD Systems

```python
# Load from Autodesk Inventor file
cad = Cad.from_inventor("existing_model.ipt")

# Load from JSON (Fusion 360 Gallery format)
cad = Cad.from_json(json_data)
```

## 🎯 Core Concepts

### Construction Sequences
RapidCADpy models CAD objects as sequences of construction operations:
- **Sketches**: 2D profiles containing geometric primitives
- **Extrudes**: 3D operations that create solid bodies from sketches
- **Operations**: Join, Cut, Intersect operations for boolean modeling

### Geometric Primitives
- **Line**: Defined by start and end points
- **Circle**: Defined by center point and radius
- **Arc**: Defined by start, end, and mid points

### Constraints
- **Geometric Constraints**: Coincidence, perpendicular, parallel, horizontal, vertical
- **Dimensional Constraints**: Distance, angle, radius constraints

### Coordinate Systems
- **Sketch Planes**: Local 2D coordinate systems for sketches
- **Global Coordinates**: 3D world coordinate system
- **Transformations**: Automatic conversion between coordinate systems

## 🔧 Advanced Usage

### Custom Integrations
```python
from rapidcadpy.integrations import BaseIntegration

class CustomCADIntegration(BaseIntegration):
    def export(self, cad, filename, **kwargs):
        # Implement custom export logic
        pass
    
    def import_file(self, file_path):
        # Implement custom import logic
        pass
```

### Data Cleaning and Normalization
```python
# Apply data cleaning operations
cad.apply_data_cleaning(visualize_steps=True)

# Normalize to unit scale
cad.normalize()

# Round coordinates to specified precision
cad.round(decimals=6)
```

### Working with Graph Representations
```python
# Convert to graph format for machine learning
# graph_data = cad.to_graph_format(with_constraints=True)

# Numericalize for neural networks
cad.numericalize(n=256)

# Restore original values
cad.denumericalize(n=256)
```

## 🧪 Testing

Run the test suite:
```bash
pytest tests/
```

Run with coverage:
```bash
pytest --cov=rapidcadpy tests/
```

## 📚 Documentation

```bash
# Install documentation dependencies
cd docs
npm install

# Start the development server
npm run dev
# or
npm run build && npm run start

# Open in browser
open http://localhost:3000/docs  # Development server
```

The documentation is built with [Fuma Docs](https://fumadocs.vercel.app/) and Next.js.


## 📋 Project Structure

```
rapidcadpy/
├── rapidcadpy/              # Main package
│   ├── cad.py              # Core CAD class
│   ├── sketch.py           # 2D sketch handling
│   ├── primitive.py        # Geometric primitives
│   ├── extrude.py          # Extrude operations
│   ├── constraint.py       # Geometric constraints
│   ├── integrations/       # CAD system integrations
│   ├── f360gallery_processing/  # Fusion 360 data processing
│   └── onshape_processing/ # OnShape data processing
├── tests/                  # Test suite
├── examples/              # Example scripts
└── docs/                  # Documentation
```
