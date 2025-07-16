# swimlane Engine

**A declarative video rendering engine powered by Blender's Video Sequence Editor (VSE)**

swimlane Engine is a programmatic video editing tool that allows you to define video compositions using a structured JSON format called SWML (Swimlane Markup Language). The engine processes these definitions through Blender's Video Sequence Editor to produce rendered video files, enabling automated video creation workflows and template-based content generation.

## Use Cases

- **Content Creators**: Automate repetitive video editing tasks
- **Developers**: Build dynamic video generation into applications
- **Marketers**: Create template-based video campaigns
- **Educators**: Generate educational content programmatically
- **Social Media**: Batch-create videos with consistent formatting

## Key Features

- **Declarative**: Define your entire video in a clean JSON file
- **Blender-Powered**: Leverages professional-grade rendering capabilities
- **Layer-Based**: Intuitive track system with proper z-indexing
- **Rich Transitions**: Fade, wipe, and dissolve effects between clips
- **Audio Support**: Full audio mixing with volume and fade controls
- **Multiple Formats**: Export to MP4, MOV, WebM, and more

## Quick Start

### Prerequisites

1. **Python 3.6+** - [Download here](https://python.org)
2. **Blender 2.80+** - [Download here](https://blender.org)
3. **ffmpeg-python** - Install via pip

### Installation

#### Option 1: Install from PyPI (Recommended)

```bash
# Install the swimlane package
pip install swimlane
```

#### Option 2: Install from Source

```bash
# Clone the repository
git clone https://github.com/idreesaziz/swimlane.git
cd swimlane

# Install the package in development mode
pip install -e .
```

#### Option 3: Manual Installation

```bash
# Install the required Python library
pip install ffmpeg-python

# Ensure Blender is in your PATH, or note its location
# Test by running:
blender --version
```

### Your First Video

Let's create a simple 10-second video with a background and logo:

**Step 1**: Create your media folder structure
```
my-project/
├── media/
│   ├── background.mp4
│   └── logo.png
├── my_video.swml
└── engine.py
```

**Step 2**: Create `my_video.swml`
```json
{
  "composition": {
    "width": 1920,
    "height": 1080,
    "fps": 30,
    "duration": 10
  },
  "sources": [
    { "id": "bg", "path": "media/background.mp4" },
    { "id": "logo", "path": "media/logo.png" }
  ],
  "tracks": [
    {
      "id": 1,
      "clips": [
        {
          "id": "background",
          "source_id": "bg",
          "start_time": 0,
          "end_time": 10
        }
      ]
    },
    {
      "id": 2,
      "clips": [
        {
          "id": "logo_clip",
          "source_id": "logo",
          "start_time": 2,
          "end_time": 8,
          "transform": {
            "position": { "cartesian": [0, 0] },
            "anchor": { "cartesian": [0, 0] }
          }
        }
      ],
      "transitions": [
        { "to_clip": "logo_clip", "duration": 1.0 },
        { "from_clip": "logo_clip", "duration": 1.0 }
      ]
    }
  ]
}
```

**Step 3**: Render your video

#### Using the installed package:

```bash
# After installing with pip
swimlane my_video.swml output.mp4
```

#### Using the script directly:

```bash
python engine.py my_video.swml output.mp4
```

That's it! You've just created a video with a background and a logo that fades in and out.

### Using the Package in Your Python Code

You can also use the Swimlane engine directly in your Python code:

```python
from swimlane import SwimlanesEngine

# Initialize the engine
engine = SwimlanesEngine()

# Load and validate a SWML file
engine.load_swml('my_video.swml')

# Render the video
engine.render('output.mp4')
```

## Creating Test Media

The package includes a utility to create test images for development:

```bash
# Generate a test image
swimlane-test-image

# Specify a custom output path
swimlane-test-image my_custom_path/test_image.png
```

## Tutorials

### Tutorial 1: Understanding the Basics

Every SWML file has three main sections:

#### 1. Composition (The Canvas)
Think of this as your video's "canvas settings":
```json
"composition": {
  "width": 1920,     // Video width in pixels
  "height": 1080,    // Video height in pixels
  "fps": 30,         // Frames per second
  "duration": 60,    // Total video length in seconds
  "output_format": "mp4"  // File format (mp4, mov, webm)
}
```

#### 2. Sources (Your Media Library)
List all media files you'll use:
```json
"sources": [
  { "id": "my_video", "path": "footage/main.mp4" },
  { "id": "my_logo", "path": "images/logo.png" },
  { "id": "music", "path": "audio/background.mp3" }
]
```

#### 3. Tracks (The Layers)
Think of tracks like layers in Photoshop. Lower IDs appear behind higher IDs:
```json
"tracks": [
  { "id": 1, "clips": [...] },  // Background layer
  { "id": 2, "clips": [...] },  // Middle layer
  { "id": 3, "clips": [...] }   // Top layer
]
```

### Tutorial 2: Working with Clips

A clip is a piece of media placed on your timeline:

```json
{
  "id": "my_clip",           // Unique name for this clip
  "source_id": "my_video",   // Which source file to use
  "start_time": 5,           // When to start (seconds)
  "end_time": 15,            // When to end (seconds)
  "source_start": 10         // Start from 10s into the source file
}
```

**Pro Tip**: Use `source_start` to trim the beginning of your source file.

### Tutorial 3: Positioning and Scaling

The transform system gives you precise control over clip placement:

#### Simple Centering
```json
"transform": {
  "position": { "cartesian": [0, 0] },    // Center of screen
  "anchor": { "cartesian": [0, 0] }       // Center of clip
}
```

#### Picture-in-Picture (Bottom Right)
```json
"transform": {
  "size": { "scale": [0.3, 0.3] },              // 30% of original size
  "position": { "pixels": [1820, 980] },        // 100px from edges
  "anchor": { "cartesian": [1, -1] }            // Bottom-right of clip
}
```

#### Understanding Coordinates

**Cartesian System** (Recommended for most use cases):
- `[-1, 1]` = Top-left corner
- `[0, 0]` = Center
- `[1, -1]` = Bottom-right corner

**Pixel System** (For precise positioning):
- `[0, 0]` = Top-left corner
- `[960, 540]` = Center of 1920x1080 video

### Tutorial 4: Creating Smooth Transitions

Transitions make your videos feel professional:

#### Fade In/Out
```json
"transitions": [
  { "to_clip": "my_clip", "duration": 2.0 },      // 2s fade-in
  { "from_clip": "my_clip", "duration": 1.5 }     // 1.5s fade-out
]
```

#### Cross-Transitions Between Clips
```json
"transitions": [
  {
    "from_clip": "clip_a",
    "to_clip": "clip_b",
    "duration": 1.0,
    "effect": "wipe",
    "direction": "left_to_right"
  }
]
```

Available effects:
- `fade` - Simple opacity transition
- `dissolve` - Cross-fade between clips
- `wipe` - Directional wipe (requires `direction`)

Wipe directions:
- `left_to_right`
- `right_to_left`
- `top_to_bottom`
- `bottom_to_top`

### Tutorial 5: Audio Integration

Add music and sound effects to your videos:

```json
{
  "id": 3,
  "type": "audio",
  "clips": [
    {
      "id": "background_music",
      "source_id": "my_music",
      "start_time": 0,
      "end_time": 30,
      "volume": 0.7,          // 70% volume
      "fade_in": 2.0,         // 2s fade-in
      "fade_out": 3.0         // 3s fade-out
    }
  ]
}
```

## Real-World Examples

### Example 1: Social Media Post
Create a 15-second Instagram-style video:

```json
{
  "composition": {
    "width": 1080,
    "height": 1080,
    "fps": 30,
    "duration": 15
  },
  "sources": [
    { "id": "product", "path": "media/product_video.mp4" },
    { "id": "logo", "path": "media/brand_logo.png" },
    { "id": "cta", "path": "media/call_to_action.png" }
  ],
  "tracks": [
    {
      "id": 1,
      "clips": [
        {
          "id": "main_content",
          "source_id": "product",
          "start_time": 0,
          "end_time": 12
        }
      ]
    },
    {
      "id": 2,
      "clips": [
        {
          "id": "brand_logo",
          "source_id": "logo",
          "start_time": 1,
          "end_time": 14,
          "transform": {
            "size": { "scale": [0.2, 0.2] },
            "position": { "cartesian": [0.8, 0.8] },
            "anchor": { "cartesian": [1, 1] }
          }
        },
        {
          "id": "final_cta",
          "source_id": "cta",
          "start_time": 12,
          "end_time": 15,
          "transform": {
            "position": { "cartesian": [0, 0] },
            "anchor": { "cartesian": [0, 0] }
          }
        }
      ],
      "transitions": [
        { "to_clip": "brand_logo", "duration": 0.5 },
        {
          "from_clip": "brand_logo",
          "to_clip": "final_cta",
          "duration": 0.8,
          "effect": "fade"
        }
      ]
    }
  ]
}
```

### Example 2: Educational Content
Create a tutorial video with multiple segments:

```json
{
  "composition": {
    "width": 1920,
    "height": 1080,
    "fps": 24,
    "duration": 120
  },
  "sources": [
    { "id": "intro", "path": "segments/intro.mp4" },
    { "id": "demo1", "path": "segments/demo_part1.mp4" },
    { "id": "demo2", "path": "segments/demo_part2.mp4" },
    { "id": "outro", "path": "segments/outro.mp4" },
    { "id": "music", "path": "audio/background.mp3" }
  ],
  "tracks": [
    {
      "id": 1,
      "clips": [
        { "id": "intro_clip", "source_id": "intro", "start_time": 0, "end_time": 10 },
        { "id": "demo1_clip", "source_id": "demo1", "start_time": 10, "end_time": 50 },
        { "id": "demo2_clip", "source_id": "demo2", "start_time": 50, "end_time": 100 },
        { "id": "outro_clip", "source_id": "outro", "start_time": 100, "end_time": 120 }
      ],
      "transitions": [
        { "to_clip": "intro_clip", "duration": 1.0 },
        { "from_clip": "intro_clip", "to_clip": "demo1_clip", "duration": 1.0, "effect": "dissolve" },
        { "from_clip": "demo1_clip", "to_clip": "demo2_clip", "duration": 1.0, "effect": "dissolve" },
        { "from_clip": "demo2_clip", "to_clip": "outro_clip", "duration": 1.0, "effect": "dissolve" },
        { "from_clip": "outro_clip", "duration": 1.0 }
      ]
    },
    {
      "id": 2,
      "type": "audio",
      "clips": [
        {
          "id": "bg_music",
          "source_id": "music",
          "start_time": 0,
          "end_time": 120,
          "volume": 0.3,
          "fade_in": 2.0,
          "fade_out": 3.0
        }
      ]
    }
  ]
}
```

## Advanced Features

### Custom Sizing and Scaling

```json
"transform": {
  "size": {
    "pixels": [800, 600],        // Set exact dimensions first
    "scale": [1.5, 1.5]          // Then scale by 150%
  }
}
```

### Precise Timing with Source Trimming

```json
{
  "id": "highlight_moment",
  "source_id": "long_video",
  "start_time": 30,              // Appears at 30s in final video
  "end_time": 35,                // Disappears at 35s in final video
  "source_start": 120            // Starts from 2 minutes into source
}
```

### Complex Multi-Layer Compositions

```json
"tracks": [
  { "id": 1, "clips": [...] },   // Background video
  { "id": 2, "clips": [...] },   // Picture-in-picture
  { "id": 3, "clips": [...] },   // Overlay graphics
  { "id": 4, "clips": [...] },   // Text/titles
  { "id": 10, "type": "audio", "clips": [...] }  // Audio track
]
```

## Audio Best Practices

1. **Background Music**: Use volume around 0.3-0.5 to avoid overpowering
2. **Fade Transitions**: Always use fade-in/out for smooth audio
3. **Multiple Audio Tracks**: Use different track IDs for different audio elements
4. **Volume Levels**: Test different volumes - 1.0 is often too loud

## Troubleshooting

### Common Issues

**"Command not found: blender"**
- Solution: Add Blender to your PATH or use the full path parameter

**"Source file not found"**
- Solution: Check file paths are correct relative to your SWML file

**"Clips overlapping strangely"**
- Solution: Ensure your timeline makes sense - clips on the same track shouldn't overlap without transitions

**"Audio not syncing"**
- Solution: Check your composition FPS matches your source video FPS

### Getting Help

1. Check file paths are correct
2. Verify Blender installation
3. Test with simple examples first
4. Check the console output for detailed error messages

## SWML Reference

### Composition Object
```json
{
  "width": 1920,           // Required: Video width
  "height": 1080,          // Required: Video height  
  "fps": 30,               // Required: Frames per second
  "duration": 60,          // Required: Total duration in seconds
  "output_format": "mp4"   // Optional: mp4, mov, webm
}
```

### Source Object
```json
{
  "id": "unique_id",       // Required: Unique identifier
  "path": "media/file.mp4" // Required: Path to media file
}
```

### Track Object
```json
{
  "id": 1,                 // Required: Unique numeric ID (lower = background)
  "type": "video",         // Optional: "video" or "audio"
  "clips": [...],          // Optional: Array of clips
  "transitions": [...]     // Optional: Array of transitions
}
```

### Clip Object
```json
{
  "id": "clip_name",       // Required: Unique within track
  "source_id": "source",   // Required: References source ID
  "start_time": 0,         // Required: Start time in seconds
  "end_time": 10,          // Required: End time in seconds
  "source_start": 0,       // Optional: Trim start of source
  "transform": {...},      // Optional: Size/position (video only)
  "volume": 1.0,           // Optional: Volume 0.0-1.0 (audio only)
  "fade_in": 0,            // Optional: Audio fade-in duration
  "fade_out": 0            // Optional: Audio fade-out duration
}
```

### Transform Object
```json
{
  "size": {
    "pixels": [1920, 1080],    // Set exact dimensions
    "scale": [1.0, 1.0]        // Scale factor
  },
  "position": {
    "pixels": [960, 540],      // Absolute position
    "cartesian": [0, 0]        // Relative position (-1 to 1)
  },
  "anchor": {
    "pixels": [100, 100],      // Absolute anchor point
    "cartesian": [0, 0]        // Relative anchor point (-1 to 1)
  }
}
```

### Transition Object
```json
{
  "from_clip": "clip_a",     // Optional: Source clip (null for fade-in)
  "to_clip": "clip_b",       // Optional: Target clip (null for fade-out)
  "duration": 1.0,           // Required: Transition duration
  "effect": "fade",          // Optional: fade, dissolve, wipe
  "direction": "left_to_right" // Required for wipe effect
}
```

## Usage

```bash
# Basic usage
python engine.py project.swml output.mp4

# With custom Blender path
python engine.py project.swml output.mp4 /path/to/blender

# Different output formats
python engine.py project.swml output.mov
python engine.py project.swml output.webm
```

## License

This project is open source. Feel free to use it in your projects.