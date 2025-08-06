# Swimlane Engine

## Declarative Video Rendering with Blender's VSE

Swimlane Engine is a command-line tool that allows you to define video compositions declaratively using a JSON-based language called **SWML (Swimlane Markup Language)**. It leverages Blender's Video Sequence Editor (VSE) as its rendering backend, offering powerful composition capabilities without requiring direct Blender UI interaction.

This engine simplifies video creation by focusing on "what" you want to see rather than "how" to achieve it in a complex editor.

## Features

- **Declarative Composition:** Define your video structure, sources, clips, and effects in a human-readable JSON format.
- **Blender VSE Backend:** Utilizes Blender's robust video sequencing capabilities for high-quality rendering.
- **Flexible Source Handling:** Supports various video, audio, and image formats.
- **Automated Preprocessing:** Automatically converts video sources to the composition's target framerate using FFmpeg to ensure smooth playback and consistent timing.
- **Transformations:** Control clip size, position, and anchor points with flexible pixel or cartesian coordinate systems.
- **Transitions:** Implement various transitions like fades, wipes, and dissolves between clips.
- **Audio Control:** Adjust clip volume, apply fade-in/out effects, and mix multiple audio tracks.
- **Preview Mode:** Generate quick, lower-resolution, lower-framerate previews for rapid iteration (internal flag, not exposed via CLI currently).

## Installation

To use Swimlane Engine, you'll need the following:

1. **Python 3:** Ensure you have Python 3 installed.

2. **`ffmpeg-python` library:**
   ```bash
   pip install ffmpeg-python
   ```

3. **Blender:** Download and install Blender.
   - For command-line usage, ensure the Blender executable is either in your system's `PATH` environment variable or you provide its full path when running the engine.
   - **Windows Example:** `C:\Program Files\Blender Foundation\Blender\blender.exe`
   - **macOS Example:** `/Applications/Blender.app/Contents/MacOS/blender`
   - **Linux Example:** Often `/usr/bin/blender` if installed via package manager.

4. **FFmpeg:** Download and install FFmpeg. It is required for source probing and video preprocessing (framerate conversion).
   - Ensure the `ffmpeg` and `ffprobe` executables are in your system's `PATH`.

## Usage

To render a video, run the `engine.py` script from your terminal:

```bash
python engine.py <input.swml> <output_file> [path/to/blender_executable]
```

**Arguments:**

- `<input.swml>`: The path to your SWML composition file.
- `<output_file>`: The desired path for the rendered video output. The file extension (`.mp4`, `.mov`, `.webm`) will determine the output format and corresponding Blender codecs.
- `[path/to/blender_executable]`: (Optional) The full path to your Blender executable. If omitted, the engine will attempt to find `blender` in your system's `PATH`.

**Examples:**

```bash
# Basic usage (assuming Blender and FFmpeg are in PATH)
python engine.py my_composition.swml output.mp4

# Specifying Blender path on Windows
python engine.py my_composition.swml output.mp4 "C:\Program Files\Blender Foundation\Blender\blender.exe"

# Specifying Blender path on macOS/Linux
python engine.py my_composition.swml output.mp4 /Applications/Blender.app/Contents/MacOS/blender
```

## Swimlane Markup Language (SWML) Specification

SWML files are JSON documents that define your video project. They must contain three top-level keys: `composition`, `sources`, and `tracks`.

**Note on Comments:** SWML supports C-style comments (`//` for single-line and `/* ... */` for multi-line) within the JSON file. These comments are stripped *before* JSON parsing, allowing for human-friendly annotations.

### Root Structure

```json
{
  "composition": { /* ... defines global video properties */ },
  "sources": [ /* ... lists all media files used */ ],
  "tracks": [ /* ... defines layers of clips and transitions */ ]
}
```

### 1. `composition` Object

Defines the overall properties of the output video.

| Key | Type | Required | Description | Default | Validation / Coercion |
|:----|:-----|:---------|:------------|:--------|:---------------------|
| `width` | `number` | Yes | Output video width in pixels. | None | Must be a positive number. Coerced to `1` if non-positive. |
| `height` | `number` | Yes | Output video height in pixels. | None | Must be a positive number. Coerced to `1` if non-positive. |
| `fps` | `number` | Yes | Output video frames per second. | None | Must be a positive number. Coerced to `1` if non-positive. |
| `duration` | `number` | No | Total duration of the composition in seconds. If omitted, it's calculated as the `end_time` of the latest clip across all tracks. | Calculated max clip end time (min 0.001s) | Must be a positive number. If specified and non-positive, coerced to `10.0` seconds. |
| `output_format` | `string` | No | The desired output video container format. Supported values: `mp4`, `mov`, `webm`. Determines the FFmpeg codecs used in Blender. | `"mp4"` | If unsupported, defaults to `mp4`. |
| `background_color` | `array` | No | An array of three numbers `[R, G, B]` representing the background color of the composition. Each value should be between `0.0` (0%) and `1.0` (100%). This is rendered as a full-screen color strip in Blender's VSE on the lowest channel (channel 1). | `[0.0, 0.0, 0.0]` (black) | Must be an array of 3 numbers. Values outside `[0.0, 1.0]` are clamped. Invalid formats default to black. |

### 2. `sources` Array

Lists all media assets (videos, audios, images) used in the composition.

| Key | Type | Required | Description |
|:----|:-----|:---------|:------------|
| `id` | `string` | Yes | A unique identifier for this source, referenced by clips. |
| `path` | `string` | Yes | The file path to the media asset. Can be absolute or relative to the SWML file's location. The engine probes each source to determine its type, duration, and dimensions. |

### 3. `tracks` Array

Defines parallel timelines (layers) of clips. Clips on higher-indexed channels will appear "on top" visually. The engine assigns channels automatically based on track type and order.

| Key | Type | Required | Description | Default | Validation |
|:----|:-----|:---------|:------------|:--------|:----------|
| `id` | `string` | No | An optional identifier for the track, useful for debugging. | `track_N` | |
| `type` | `string` | No | The primary type of content this track handles. Supported values: `video`, `audio`, `audiovideo`. This helps the engine determine how to manage channels and effects. | `"video"` | If invalid, defaults to `"video"`. |
| `clips` | `array` | No | An array of `clip` objects, defining media segments placed on this track. | `[]` | All `clip.id`s within a single track must be unique. |
| `transitions` | `array` | No | An array of `transition` objects, defining how clips on this track transition between each other or fade in/out. | `[]` | References `clip.id`s. |

#### 3.1. `clip` Object (within `tracks.clips`)

Represents a segment of a source asset placed on the timeline.

| Key | Type | Required | Description | Default | Validation / Coercion |
|:----|:-----|:---------|:------------|:--------|:---------------------|
| `id` | `string` | Yes | A unique identifier for this clip within its track. Used for referencing in transitions. | None | Must be unique within its track. |
| `source_id` | `string` | Yes | The `id` of a source defined in the top-level `sources` array. | None | Must reference an existing `source.id`. Critical error if not found. |
| `start_time` | `number` | No | The time (in seconds) on the composition timeline where this clip begins. | `0.0` | Must be a non-negative number. Invalid values are coerced to `0.0`. |
| `end_time` | `number` | No | The time (in seconds) on the composition timeline where this clip ends. If omitted:<br>- For **images**: Defaults to `start_time + 5.0` seconds<br>- For **videos/audio**: Defaults to `start_time + (source_duration - source_start)` | Calculated | Must be a number. If `end_time <= start_time`, it's coerced to `start_time + (1/fps)` (minimum 1 frame duration). |
| `source_start` | `number` | No | For video/audio sources, the time (in seconds) from the beginning of the *source file* where this clip should start reading. | `0.0` | Must be a non-negative number. If beyond source duration, coerced to `0.0`. |
| `volume` | `number` | No | For `audio` or `audiovideo` tracks. The volume level of the clip, from `0.0` (silent) to `1.0` (full volume). | `1.0` | Must be non-negative. If negative, clamped to `0.0`. |
| `fade_in` | `number` | No | For `audio` or `audiovideo` tracks. The duration (in seconds) of an audio fade-in effect at the beginning of the clip. | `0.0` | Must be non-negative. Invalid values coerced to `0.0`. |
| `fade_out` | `number` | No | For `audio` or `audiovideo` tracks. The duration (in seconds) of an audio fade-out effect at the end of the clip. | `0.0` | Must be non-negative. Invalid values coerced to `0.0`. |
| `transform` | `object` | No | An object containing transformation properties (size, position, anchor) for video/image clips. | None (no transformation) | Invalid properties are warned and removed. |

#### `transform` Object (within `clip`)

Defines how a video or image clip is scaled and positioned on the screen. Transformations are applied in order: `size` (pixels then scale), then `position` (affected by `anchor`).

| Key | Type | Required | Description |
|:----|:-----|:---------|:------------|
| `size` | `object` | No | Defines the dimensions of the clip. |
| `position` | `object` | No | Defines the clip's placement relative to the composition. |
| `anchor` | `object` | No | Defines the point on the clip used for positioning. |

##### `transform.size` Object

Controls the width and height of the clip. The `pixels` property is applied first, then `scale` is applied as a multiplier.

| Key | Type | Required | Description | Validation |
|:----|:-----|:---------|:------------|:-----------|
| `pixels` | `array` | No | `[width, height]` in pixels. Sets exact pixel dimensions. | Must be array of two numbers. |
| `scale` | `array` | No | `[scale_x, scale_y]` as multipliers (e.g., `[0.5, 0.5]` for half size). | Must be array of two numbers. Values clamped to minimum `0.001`. |

**Examples:**
```json
"size": {"pixels": [640, 360]}  // Clip will be 640x360 pixels
"size": {"scale": [0.5, 0.5]}   // Clip will be half its original size
"size": {"pixels": [640, 360], "scale": [2.0, 2.0]}  // 640x360, then scaled to 1280x720
```

##### `transform.position` Object

Defines where the clip is placed on the composition canvas.

**Precedence:** If both `pixels` and `cartesian` are provided, `cartesian` takes precedence.

| Key | Type | Required | Description | Validation |
|:----|:-----|:---------|:------------|:-----------|
| `pixels` | `array` | No | `[x, y]` coordinates in pixels from top-left corner. | Must be array of two numbers. |
| `cartesian` | `array` | No | `[x, y]` coordinates where `[0,0]` is center, `[-1,-1]` is top-left, `[1,1]` is bottom-right. | Must be array of two numbers. |

**Examples:**
```json
"position": {"pixels": [100, 50]}        // Top-left at x=100, y=50
"position": {"cartesian": [0.0, 0.0]}    // Center of composition
"position": {"cartesian": [-1.0, 1.0]}   // Bottom-left of composition
```

##### `transform.anchor` Object

The anchor point defines which part of the clip is used for positioning.

**Precedence:** If both `pixels` and `cartesian` are provided, `cartesian` takes precedence.

| Key | Type | Required | Description | Validation |
|:----|:-----|:---------|:------------|:-----------|
| `pixels` | `array` | No | `[x, y]` pixel coordinates relative to clip's top-left corner. | Must be array of two numbers. |
| `cartesian` | `array` | No | `[x, y]` coordinates relative to clip's center. `[-1,-1]` = top-left, `[0,0]` = center, `[1,1]` = bottom-right. | Must be array of two numbers. |

**Examples:**
```json
"anchor": {"pixels": [0, 0]}         // Anchor at clip's top-left
"anchor": {"cartesian": [0.0, 0.0]}  // Anchor at clip's center
"anchor": {"cartesian": [-1.0, 1.0]} // Anchor at clip's bottom-left
```

#### 3.2. `transition` Object (within `tracks.transitions`)

Defines how clips fade in, fade out, or cross-fade between each other.

| Key | Type | Required | Description | Default | Validation |
|:----|:-----|:---------|:------------|:--------|:-----------|
| `from_clip` | `string` | No | The ID of the clip transitioning out. For fade-out only, set `to_clip` to `null`. | None | Must reference existing clip ID or be `null`. |
| `to_clip` | `string` | No | The ID of the clip transitioning in. For fade-in only, set `from_clip` to `null`. | None | Must reference existing clip ID or be `null`. |
| `duration` | `number` | No | Duration of the transition in seconds. | `1.0` | Must be positive number. |
| `effect` | `string` | No | Type of transition effect: `fade`, `wipe`, `dissolve`. | `"fade"` | Unsupported effects default to `fade`. |
| `direction` | `string` | No | For `wipe` transitions: `left_to_right`, `right_to_left`, `top_to_bottom`, `bottom_to_top`. | `"left_to_right"` | Only applies to wipe transitions. |

## Example SWML File

Save this as `my_composition.swml` and ensure you have corresponding media files:

```json
/*
  Example SWML Composition
  This JSON demonstrates various features of the Swimlane Markup Language.
*/
{
  "composition": {
    "width": 1280,
    "height": 720,
    "fps": 30,
    "duration": 60,
    "output_format": "mp4",
    "background_color": [0.1, 0.1, 0.2]
  },
  "sources": [
    {
      "id": "intro_video",
      "path": "./assets/intro.mp4"
    },
    {
      "id": "main_footage", 
      "path": "./assets/main_video.mov"
    },
    {
      "id": "logo_image",
      "path": "./assets/logo.png"
    },
    {
      "id": "background_music",
      "path": "./assets/music.mp3"
    },
    {
      "id": "outro_audio",
      "path": "./assets/outro_voiceover.wav"
    }
  ],
  "tracks": [
    {
      "id": "background_layer",
      "type": "video",
      "clips": [
        {
          "id": "clip_intro_bg",
          "source_id": "intro_video", 
          "start_time": 0.0,
          "end_time": 10.0,
          "transform": {
            "size": {"scale": [1.2, 1.2]},
            "position": {"cartesian": [0.0, 0.0]}
          }
        },
        {
          "id": "clip_main_bg",
          "source_id": "main_footage",
          "start_time": 8.0,
          "end_time": 58.0,
          "transform": {
            "size": {"scale": [1.0, 1.0]}
          }
        }
      ],
      "transitions": [
        {
          "from_clip": "clip_intro_bg",
          "to_clip": "clip_main_bg", 
          "duration": 2.0,
          "effect": "fade"
        }
      ]
    },
    {
      "id": "foreground_layer",
      "type": "video",
      "clips": [
        {
          "id": "logo_appearance",
          "source_id": "logo_image",
          "start_time": 5.0,
          "end_time": 15.0,
          "transform": {
            "size": {"pixels": [200, 200]},
            "position": {"pixels": [50, 50]},
            "anchor": {"pixels": [0, 0]}
          }
        },
        {
          "id": "logo_movement",
          "source_id": "logo_image",
          "start_time": 15.0,
          "end_time": 20.0,
          "transform": {
            "size": {"pixels": [200, 200]},
            "position": {"cartesian": [0.8, 0.8]},
            "anchor": {"cartesian": [1.0, 1.0]}
          }
        }
      ],
      "transitions": [
        {
          "from_clip": "logo_appearance",
          "to_clip": null,
          "duration": 1.0,
          "effect": "fade"
        },
        {
          "from_clip": null,
          "to_clip": "logo_movement",
          "duration": 1.0,
          "effect": "fade"
        },
        {
          "from_clip": "logo_movement",
          "to_clip": null,
          "duration": 0.5,
          "effect": "fade"
        }
      ]
    },
    {
      "id": "music_track",
      "type": "audio",
      "clips": [
        {
          "id": "main_music",
          "source_id": "background_music",
          "start_time": 0.0,
          "end_time": 55.0,
          "volume": 0.7,
          "fade_in": 2.0,
          "fade_out": 3.0
        }
      ]
    },
    {
      "id": "voiceover_track",
      "type": "audio", 
      "clips": [
        {
          "id": "outro_vo",
          "source_id": "outro_audio",
          "start_time": 50.0,
          "volume": 1.0,
          "fade_in": 1.0
        }
      ]
    }
  ]
}
```

## License

This project is provided as-is. Please ensure you have proper licenses for Blender, FFmpeg, and any media assets you use with this engine.
