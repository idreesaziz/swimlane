# Swimlane Engine

A declarative video rendering engine powered by Blender's Video Sequence Editor (VSE). Define your video composition, sources, tracks, and clips using the **Swimlane Markup Language (SWML)**, and let Swimlane Engine handle the complex video editing operations in Blender.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
  - [Command Line Interface](#command-line-interface)
  - [Basic Example](#basic-example)
- [SWML Reference](#swml-reference)
  - [General Structure](#general-structure)
  - [Composition](#composition)
  - [Sources](#sources)
  - [Tracks](#tracks)
  - [Clips](#clips)
  - [Transforms](#transforms)
  - [Transitions](#transitions)
- [Examples](#examples)
- [Development](#development)
- [License](#license)
- [Roadmap](#roadmap)

## Introduction

Creating videos with dynamic content often involves cumbersome processes with complex timelines, manual adjustments, and scripting in traditional video editors. Swimlane Engine simplifies this by providing a declarative, text-based approach to video composition.

By defining your video structure, assets, and desired effects in a human-readable JSON format (SWML), Swimlane Engine automates the rendering process using Blender's powerful Video Sequence Editor. This enables reproducible, version-controlled, and programmatic video generation.

## Features

- **Declarative Composition**: Define your entire video project in a single SWML file
- **Blender VSE Integration**: Leverages Blender's robust video editing capabilities
- **Automatic Media Handling**: Handles image and video sources with automatic framerate conversion
- **Flexible Transformations**: Define clip size, position, and anchor points using pixel or cartesian coordinates
- **Audio Control**: Adjust volume and apply fade effects on audio clips
- **Transition Effects**: Supports fade, wipe, and dissolve transitions between clips
- **Preview Mode**: Fast, low-quality rendering for quick iteration
- **Error Reporting**: Provides clear feedback on SWML parsing and validation issues
- **Cross-Platform**: Runs wherever Python and Blender are supported

## Prerequisites

Before installing Swimlane Engine, ensure you have the following software:

1. **Blender (v3.0 or later)**
   - Download from [blender.org](https://www.blender.org/download/)
   - Ensure the `blender` executable is in your system's PATH

2. **FFmpeg & FFprobe**
   - Used for media probing and preprocessing
   - Download from [ffmpeg.org](https://ffmpeg.org/download.html)
   - Ensure both are in your system's PATH

## Installation

Install Swimlane Engine via pip:

```bash
pip install swimlane
```

This installs the necessary Python dependencies and makes the `swimlane` command available in your terminal.

## Usage

### Command Line Interface

```bash
swimlane [--preview] <input.swml> <output.mp4> [path/to/blender]
```

**Arguments:**
- `<input.swml>`: Path to your SWML input file
- `<output.mp4>`: Path for the output video file (supports .mp4, .mov, .webm)
- `[path/to/blender]`: Optional path to Blender executable

**Options:**
- `--preview`: Use fast/low quality render settings for quick previews
- `--help, -h`: Show help message

**Examples:**

```bash
# Basic render
swimlane my_project.swml output_video.mp4

# Preview mode
swimlane --preview my_project.swml preview.mp4

# Custom Blender path
swimlane my_project.swml output.mp4 /Applications/Blender.app/Contents/MacOS/Blender
```

### Basic Example

Here's a simple SWML file (`example.swml`):

```json
{
  "composition": {
    "width": 1280,
    "height": 720,
    "fps": 30,
    "duration": 10
  },
  "sources": [
    {
      "id": "intro_video",
      "path": "media/intro.mp4"
    },
    {
      "id": "logo",
      "path": "media/logo.png"
    },
    {
      "id": "music",
      "path": "media/background.mp3"
    }
  ],
  "tracks": [
    {
      "id": "video_track",
      "type": "video",
      "clips": [
        {
          "id": "intro_clip",
          "source_id": "intro_video",
          "start_time": 0.0,
          "end_time": 8.0,
          "transform": {
            "size": {
              "scale": [0.8, 0.8]
            },
            "position": {
              "cartesian": [0, 0]
            }
          }
        },
        {
          "id": "logo_clip",
          "source_id": "logo",
          "start_time": 6.0,
          "end_time": 10.0,
          "transform": {
            "size": {
              "pixels": [200, 150]
            },
            "position": {
              "pixels": [100, 100]
            }
          }
        }
      ],
      "transitions": [
        {
          "from_clip": "intro_clip",
          "to_clip": "logo_clip",
          "duration": 2.0,
          "effect": "fade"
        }
      ]
    },
    {
      "id": "audio_track",
      "type": "audio",
      "clips": [
        {
          "id": "music_clip",
          "source_id": "music",
          "start_time": 0.0,
          "end_time": 10.0,
          "volume": 0.7,
          "fade_in": 1.0,
          "fade_out": 1.5
        }
      ]
    }
  ]
}
```

Render with:

```bash
swimlane example.swml output.mp4
```

## SWML Reference

SWML files are JSON documents with support for C-style comments (`//` and `/* */`).

### General Structure

A SWML document contains three top-level sections:

```json
{
  "composition": { /* Video properties */ },
  "sources": [ /* Media assets */ ],
  "tracks": [ /* Video and audio tracks */ ]
}
```

### Composition

Defines global properties of the output video:

| Property | Type | Required | Description | Default |
|----------|------|----------|-------------|---------|
| `width` | integer | Yes | Output width in pixels | - |
| `height` | integer | Yes | Output height in pixels | - |
| `fps` | number | Yes | Frames per second | - |
| `duration` | number | No | Total duration in seconds | Auto-calculated |
| `output_format` | string | No | Output format ("mp4", "mov", "webm") | "mp4" |

### Sources

Array of media files used in the project:

| Property | Type | Required | Description |
|----------|------|----------|-------------|
| `id` | string | Yes | Unique identifier |
| `path` | string | Yes | File path (absolute or relative to SWML file) |

### Tracks

Array of video or audio tracks:

| Property | Type | Required | Description | Default |
|----------|------|----------|-------------|---------|
| `id` | string | No | Unique identifier | Auto-generated |
| `type` | string | No | Track type ("video" or "audio") | "video" |
| `clips` | array | No | Array of clip objects | [] |
| `transitions` | array | No | Array of transition objects | [] |

### Clips

Media segments on a track:

| Property | Type | Required | Description | Default |
|----------|------|----------|-------------|---------|
| `id` | string | Yes | Unique identifier within track | - |
| `source_id` | string | Yes | References a source ID | - |
| `start_time` | number | No | Start time on timeline (seconds) | 0.0 |
| `end_time` | number | No | End time on timeline (seconds) | Auto-calculated |
| `source_start` | number | No | Start time in source file (seconds) | 0.0 |
| `volume` | number | No | Audio volume (0.0-1.0+) | 1.0 |
| `fade_in` | number | No | Audio fade-in duration (seconds) | 0.0 |
| `fade_out` | number | No | Audio fade-out duration (seconds) | 0.0 |
| `transform` | object | No | Visual transformations (video only) | None |

### Transforms

Visual transformations for video clips:

```json
{
  "transform": {
    "size": {
      "pixels": [width, height],    // Absolute size
      "scale": [scaleX, scaleY]     // Relative scale
    },
    "position": {
      "pixels": [x, y],             // Absolute position
      "cartesian": [x, y]           // Relative position (-1 to 1)
    },
    "anchor": {
      "pixels": [x, y],             // Absolute anchor point
      "cartesian": [x, y]           // Relative anchor point (-1 to 1)
    }
  }
}
```

**Coordinate Systems:**
- **Pixels**: Absolute coordinates from top-left corner
- **Cartesian**: Relative coordinates where (0,0) is center, (-1,-1) is top-left, (1,1) is bottom-right

### Transitions

Visual effects between clips (video tracks only):

| Property | Type | Required | Description | Default |
|----------|------|----------|-------------|---------|
| `from_clip` | string | * | Source clip ID | - |
| `to_clip` | string | * | Target clip ID | - |
| `duration` | number | No | Transition duration (seconds) | 1.0 |
| `effect` | string | No | Effect type ("fade", "wipe", "dissolve") | "fade" |
| `direction` | string | No | Wipe direction | "left_to_right" |

*At least one of `from_clip` or `to_clip` is required.

**Transition Types:**
- **Cross-transition**: Both clips specified (crossfade, wipe, dissolve)
- **Simple fade**: One clip specified (fade-in or fade-out)

## Examples

The package includes example SWML files in the `examples/` directory demonstrating various features with placeholder media files.

## Development

### Project Structure

- `cli.py`: Command-line interface and argument parsing
- `engine.py`: Core engine logic, SWML parsing, and validation
- `blender_template.py`: Python script template executed by Blender
- `.swimlane_cache/`: Directory for transcoded video sources (filename includes framerate for cache uniqueness)

### Contributing

1. Fork the repository
2. Create a feature branch
3. Implement changes with tests
4. Ensure existing tests pass
5. Submit a pull request

## License

This project is licensed under the MIT License.

## Roadmap

Future enhancements may include:

- Advanced transitions and effects
- Keyframe animation support  
- Text overlay capabilities
- Color grading features
- Enhanced error reporting
- Unit test coverage
