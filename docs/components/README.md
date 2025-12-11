# Python Documentation Components

This directory contains React components for documenting Python APIs in MDX files.

## Components

### PythonFunction

A component for documenting individual Python functions or methods.

**Props:**
- `name` (string): Function name
- `signature` (string): Full function signature
- `description` (string): Function description
- `params` (object): Parameter definitions with type, description, required, and default
- `returns` (string, optional): Return value description
- `returnType` (string, optional): Return type annotation
- `source` (string, optional): Link to source code

**Example:**
```tsx
<PythonFunction
  name="generate_mesh"
  signature="generate_mesh(filename, mesh_size=1.0, element_type='tet4')"
  description="Generate mesh from geometry file."
  params={{
    filename: {
      type: 'str',
      description: 'Path to geometry file',
      required: true,
    },
    mesh_size: {
      type: 'float',
      description: 'Maximum element size',
      required: false,
      default: '1.0',
    },
  }}
  returns="Tuple of (nodes, elements) as PyTorch tensors"
  returnType="Tuple[torch.Tensor, torch.Tensor]"
  source="https://github.com/..."
/>
```

### PythonClass

A component for documenting Python classes that can contain multiple `PythonFunction` components as children to document methods.

**Props:**
- `name` (string): Class name
- `description` (string): Class description
- `bases` (string[], optional): List of base classes
- `attributes` (object, optional): Class attributes with type, description, readonly, and default
- `source` (string, optional): Link to source code
- `children` (ReactNode, optional): Child components (typically PythonFunction components for methods)

**Example:**
```tsx
<PythonClass
  name="NetgenMesher"
  description="Netgen-based mesh generator."
  bases={['MesherBase']}
  attributes={{
    num_threads: {
      type: 'int',
      description: 'Number of threads for parallel meshing',
      default: 'auto-detected',
    },
  }}
  source="https://github.com/..."
>
  <PythonFunction
    name="__init__"
    signature="__init__(num_threads: int = 0)"
    description="Initialize the mesher."
    params={{
      num_threads: {
        type: 'int',
        description: 'Thread count (0 = auto)',
        default: '0',
      },
    }}
  />
  
  <PythonFunction
    name="generate_mesh"
    signature="generate_mesh(filename, mesh_size=1.0)"
    description="Generate mesh from file."
    params={{
      filename: {
        type: 'str',
        description: 'Path to geometry file',
        required: true,
      },
    }}
    returns="Mesh data"
    returnType="Tuple[torch.Tensor, torch.Tensor]"
  />
</PythonClass>
```

## Usage in MDX

```mdx
---
title: My API Documentation
---

import { PythonClass } from '@/components/python-class';
import { PythonFunction } from '@/components/python-function';

# My API

<PythonClass
  name="MyClass"
  description="Description of my class."
  bases={['BaseClass']}
>
  <PythonFunction
    name="my_method"
    signature="my_method(param1, param2='default')"
    description="Does something useful."
    params={{
      param1: {
        type: 'str',
        description: 'First parameter',
        required: true,
      },
      param2: {
        type: 'str',
        description: 'Second parameter',
        default: "'default'",
      },
    }}
    returns="Result of the operation"
    returnType="bool"
  />
</PythonClass>
```

## Styling

Both components use Fumadocs UI design tokens:
- `fd-card`: Card background
- `fd-border`: Border color
- `fd-primary`: Primary accent color
- `fd-foreground`: Primary text color
- `fd-muted-foreground`: Secondary text color
- `fd-muted`: Muted background

The components automatically adapt to light/dark themes.

## Features

### PythonFunction Features:
- Displays function signature prominently
- Shows parameter table with types, descriptions, and defaults
- Indicates required vs optional parameters
- Shows return type and description
- Optional source code link

### PythonClass Features:
- Shows class inheritance hierarchy
- Displays class attributes with types and defaults
- Contains multiple method documentations
- Separates attributes and methods into clear sections
- Handles empty states gracefully
- Optional source code link
