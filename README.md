# SWML: The Swimlane Markup Language Engine

**swml** is a lightweight, JSON-based domain-specific language (DSL) and Python engine for describing and rendering layered, timeline-based video compositions.

## Features (v1.0)
- Declarative JSON format (`.swml`)
- Layer-based compositing with Z-ordering
- Resolution-independent positioning using a normalized coordinate system with anchors
- Flexible scaling and sizing of clips
- Support for video, image, and transparent sources (e.g., `.png`, `.webm`)
- Output to standard MP4 or transparency-enabled MOV/WebM formats

## Quick Start

**1. Prerequisites**
- Python 3.8+
- FFmpeg (must be installed and available in your system's PATH)

**2. Setup**
```bash
git clone https://github.com/idreesaziz/swimlane.git
cd swml
python -m venv venv
# Activate the virtual environment
# Windows: .\\venv\\Scripts\\activate
# macOS/Linux: source venv/bin/activate
pip install -r requirements.txt
```

**3. Run an Example**
Create your media files inside examples/media/. Then run the engine:

```bash
python engine.py examples/composition.swml output.mov
```

## Transform Model

SWML uses an "Explicit & Sequential" transform model to position, size, and rotate clips within a composition. The transform properties provide fine-grained control over how clips are displayed.

### Size

The `size` property controls the dimensions of a clip, with an explicit sequence of operations:

```json
"transform": {
  "size": {
    "pixels": [800, 600],  // Optional: First, resize the media to these dimensions
    "scale": [0.5, 0.5]    // Optional: Then, scale the result of the above operation
  }
}
```

Order of Operations:
1. If `size.pixels` is defined, the source media is resized to these exact pixel dimensions
2. If `size.scale` is defined, the result of step 1 is then scaled by this [x, y] factor

Examples:
- Scale an image to 50% of its original size: `"size": { "scale": [0.5, 0.5] }`
- Force an image to be 400x300px: `"size": { "pixels": [400, 300] }`
- Resize to 400x300px, then scale by 50%: `"size": { "pixels": [400, 300], "scale": [0.5, 0.5] }`

### Position

The `position` property defines where to place the clip within the composition:

```json
"transform": {
  "position": {
    // Choose ONE of these:
    "pixels": [100, 50],     // [X, Y] from top-left of the composition
    "cartesian": [0.0, 0.0]  // [X, Y] where [0,0] is the center, [-1,-1] is top-left
  }
}
```

### Anchor

The `anchor` property defines the reference point on the clip itself that will be placed at the position:

```json
"transform": {
  "anchor": {
    // Choose ONE of these:
    "pixels": [40, 30],      // [X, Y] from top-left of the clip itself
    "cartesian": [0.0, 0.0]  // [X, Y] where [0,0] is the center, [-1,-1] is top-left of clip
  }
}
```

### Rotation

Rotation is specified in degrees:

```json
"transform": {
  "rotation": 90  // In degrees
}
```

### Complete Example

```json
"clips": [
  {
    "id": "my_clip",
    "source_id": "video1",
    "start_time": 0, "end_time": 10,
    "transform": {
      "size": {
        "pixels": [400, 225]
      },
      "position": {
        "cartesian": [0.0, 0.0]
      },
      "anchor": {
        "cartesian": [0.0, 0.0]
      },
      "rotation": 45
    }
  }
]
```

This will:
1. Resize the clip to 400x225 pixels
2. Place the center of the clip at the center of the composition
3. Rotate the clip 45 degrees clockwise

## Important Note

This version of the SWML engine only supports the explicit transform model described above. All SWML files must use this format.