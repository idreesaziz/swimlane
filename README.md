# Swimlane Engine

A declarative video rendering framework designed for programmatic automation and AI-driven content generation. The Swimlane Engine enables complex video editing workflows through structured markup, making it ideal for systems that need to generate video content programmatically.

## Overview

The Swimlane Engine transforms video editing from a manual, GUI-based process into a code-driven workflow. By using SWML (Swimlane Markup Language), developers and AI systems can define complex video compositions through structured data rather than interactive editing tools.

**Key Design Principles:**
- **Programmatic Control**: Every aspect of video editing is controlled through structured data
- **AI-Friendly**: JSON-based format optimized for large language model generation and parsing  
- **Reproducible**: Version-controlled video projects with deterministic outputs
- **Scalable**: Designed for automated batch processing and template-based generation

## Architecture

The current implementation leverages Blender's Video Sequence Editor (VSE) as the rendering backend. Future versions will migrate to the MLT Multimedia Framework for improved performance and reduced dependencies.

**Current Stack:**
- SWML Parser (Python)
- Blender VSE Backend
- FFmpeg Preprocessing Pipeline

**Planned Migration:**
- MLT Framework Backend
- Enhanced Performance
- Reduced System Dependencies

## Installation

### Prerequisites
- Python 3.8+
- Blender 2.8+ (currently required for VSE backend)
- FFmpeg (media preprocessing and validation)

### Setup
```bash
git clone https://github.com/yourusername/swimlane-engine.git
cd swimlane-engine

# Verify dependencies
blender --version
ffmpeg -version

# Basic usage
python engine.py input.swml output.mp4
```

## SWML Language Specification

SWML is a JSON-based declarative language specifically designed for programmatic video generation. The language structure enables AI systems to generate complex video compositions without understanding underlying rendering mechanics.

### Core Structure

Every SWML document contains three primary sections:

```json
{
  "composition": {
    "width": 1920,
    "height": 1080, 
    "fps": 30,
    "duration": 120.0,
    "output_format": "mp4"
  },
  "sources": [
    { "id": "primary_footage", "path": "media/main.mp4" },
    { "id": "overlay_graphics", "path": "assets/logo.png" }
  ],
  "tracks": [
    {
      "id": "video_track_1",
      "type": "video",
      "clips": [...],
      "transitions": [...]
    }
  ]
}
```

### Composition Configuration

The composition object defines global rendering parameters:

| Parameter | Type | Description |
|-----------|------|-------------|
| `width` | Integer | Output video width in pixels |
| `height` | Integer | Output video height in pixels |
| `fps` | Integer | Target framerate for output |
| `duration` | Float | Total composition length in seconds |
| `background_color` | Array[3] | RGB values (0.0-1.0) for background |
| `output_format` | String | Container format: "mp4", "mov", "webm" |

### Source Management

Sources define all media assets referenced in the composition:

```json
"sources": [
  { "id": "main_video", "path": "footage/interview.mp4" },
  { "id": "b_roll", "path": "footage/cutaway.mp4" },
  { "id": "background_audio", "path": "audio/ambient.wav" },
  { "id": "title_graphic", "path": "graphics/title.png" }
]
```

### Track-Based Timeline

Tracks represent layers in the video composition, processed in sequential order:

```json
"tracks": [
  {
    "id": "audio_background",
    "type": "audio",
    "clips": [
      {
        "id": "ambient_sound",
        "source_id": "background_audio",
        "start_time": 0,
        "end_time": 60,
        "volume": 0.3,
        "fade_in": 2.0,
        "fade_out": 3.0
      }
    ]
  },
  {
    "id": "primary_video",
    "type": "video", 
    "clips": [
      {
        "id": "main_sequence",
        "source_id": "main_video",
        "start_time": 0,
        "end_time": 45,
        "source_start": 10.0,
        "transform": {
          "size": { "scale": [1.0, 1.0] },
          "position": { "cartesian": [0, 0] },
          "effects": {
            "color": {
              "brightness": 1.1,
              "contrast": 1.05,
              "saturation": 1.2
            }
          }
        }
      }
    ]
  }
]
```

## Advanced Features

### Transformation Pipeline

The transform system provides precise control over visual elements:

```json
"transform": {
  "size": {
    "pixels": [1280, 720],
    "scale": [0.8, 0.8]
  },
  "position": {
    "cartesian": [0.0, 0.0]  // Normalized coordinates
  },
  "anchor": {
    "cartesian": [0.0, 0.0]  // Pivot point
  },
  "effects": {
    "color": {
      "brightness": 1.2,
      "contrast": 1.1, 
      "saturation": 1.3,
      "gamma": 1.0,
      "hue": 15
    },
    "rotation": {
      "angle": 45
    },
    "lut": {
      "preset": "cinema",
      "strength": 0.7
    }
  }
}
```

### Transition System

Transitions enable smooth changes between clips:

```json
"transitions": [
  {
    "from_clip": "scene_1",
    "to_clip": "scene_2",
    "duration": 1.5,
    "effect": "fade"
  },
  {
    "from_clip": "scene_2", 
    "to_clip": "scene_3",
    "duration": 0.8,
    "effect": "wipe",
    "direction": "left_to_right"
  }
]
```

## Performance Optimization

### Intelligent Preprocessing
The engine automatically processes source media to ensure consistent timing and quality:

- **Framerate Normalization**: All sources converted to composition framerate
- **Format Standardization**: Intermediate format conversion for optimal performance
- **Smart Caching**: Processed media cached in `.swimlane_cache/` directory
- **Incremental Processing**: Only reprocesses modified sources

### Preview Mode
Accelerated preview generation for rapid iteration:

```bash
python engine.py --preview input.swml preview.mp4
```

Preview specifications:
- 10 FPS output
- 480p maximum resolution
- Optimized encoding settings
- ~10x faster processing

## Command Line Interface

```bash
# Standard rendering
python engine.py input.swml output.mp4

# Preview mode
python engine.py --preview input.swml preview.mp4

# Custom Blender path
python engine.py input.swml output.mp4 /usr/local/bin/blender

# Help
python engine.py --help
```

## Use Cases

### Automated Content Generation
- Template-based video creation
- Data-driven video generation
- Batch processing workflows
- A/B testing for video content

### AI Integration
- Large language model video generation
- Automated editing based on content analysis
- Dynamic video adaptation
- Programmatic story assembly

### Production Workflows
- Consistent branding application
- Multi-format output generation
- Version control for video projects
- Collaborative editing through code review

## Technical Implementation

### Current Backend: Blender VSE
The current implementation uses Blender's Video Sequence Editor for rendering:

**Advantages:**
- Mature rendering pipeline
- Professional-grade output quality
- Extensive format support
- Advanced effects processing

**Limitations:**
- Heavy system requirements
- GUI dependency
- Performance constraints for batch processing

### Future Backend: MLT Framework
Planned migration to MLT for improved automation capabilities:

**Benefits:**
- Lightweight architecture
- Server-optimized performance
- Headless operation
- Enhanced scalability

## Development Roadmap

### Phase 1: Current Implementation
- [x] SWML language specification
- [x] Blender VSE integration
- [x] Core rendering pipeline
- [x] Transformation system
- [x] Transition support

### Phase 2: MLT Migration
- [ ] MLT framework integration
- [ ] Performance benchmarking
- [ ] Feature parity validation
- [ ] Migration documentation

### Phase 3: Advanced Features
- [ ] Real-time preview
- [ ] Distributed rendering
- [ ] Plugin architecture
- [ ] Web-based editor

## Contributing

This project follows standard open-source contribution guidelines:

1. Fork the repository
2. Create a feature branch
3. Implement changes with tests
4. Submit a pull request

### Code Standards
- Python 3.8+ compatibility
- Type hints for public APIs
- Comprehensive error handling
- Performance-conscious implementation

## License

MIT License - see LICENSE file for details.