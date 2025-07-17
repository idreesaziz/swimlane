# Installation Guide for Swimlane

## Installation Options

### 1. Install from PyPI (Recommended)

```bash
pip install swimlane
```

### 2. Install directly from GitHub

```bash
pip install git+https://github.com/idreesaziz/swimlane.git
```

### 3. Install in development mode (for contributors)

Clone the repository:
```bash
git clone https://github.com/idreesaziz/swimlane.git
cd swimlane
```

Install in development mode:
```bash
pip install -e .
```

### 4. Install from a local copy

If you've downloaded the package or cloned it:
```bash
cd path/to/swimlane
pip install .
```

## Usage after Installation

After installation, you can use the package in two ways:

### As a command-line tool

```bash
# Basic usage
swimlane your_composition.swml output.mp4

# Specify a custom Blender executable path
swimlane your_composition.swml output.mp4 path/to/blender
```

### As a Python library

```python
from swimlane import SwimlansEngine

# Initialize the engine
engine = SwimlaneEngine('your_composition.swml', 'output.mp4')

# Render the video
engine.render()
```

## Uninstalling

```bash
pip uninstall swimlane
```
