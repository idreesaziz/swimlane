"""
Swimlane Engine - A declarative video rendering engine using Blender's VSE
Parses SWML (Swimlane Markup Language) files and generates videos
"""

import json
import sys
import os
import subprocess
import tempfile
import textwrap
import re # For comment stripping
import warnings # For custom warnings
import shutil # For cleanup
from typing import Dict, List, Any, Tuple, Optional

# ffprobe is still used for validation, so the dependency remains.
import ffmpeg


class SwmlError(Exception):
    """Custom exception for SWML parsing and processing errors"""
    pass

class SourceInfo(object):
    """Simple class to hold probed media information."""
    def __init__(self, path: str, duration: float = 0.0, width: int = 0, height: int = 0,
                 has_audio: bool = False, has_video: bool = False, is_image: bool = False):
        self.path = path
        self.duration = duration
        self.width = width
        self.height = height
        self.has_audio = has_audio
        self.has_video = has_video
        self.is_image = is_image

class SwimlaneEngine:
    DEFAULT_IMAGE_DURATION_SECONDS = 5.0

    def __init__(self, swml_path: str, output_path: str, blender_executable: str = 'blender'):
        self.swml_path = swml_path
        self.output_path = os.path.abspath(output_path)
        self.blender_executable = blender_executable
        self.swml_data = None
        self.source_info_cache: Dict[str, SourceInfo] = {} # Cache for ffprobe results
        self.converted_sources: Dict[str, str] = {} # Cache for framerate-converted sources

    def dry_run_preprocessing(self):
        """Test the preprocessing step without running Blender"""
        print("--- Swimlane Engine: Preprocessing Dry Run ---")
        print(f"1. Parsing SWML file: {self.swml_path}")
        self.parse_swml()
        
        print("2. Testing video preprocessing...")
        self._preprocess_video_sources()
        
        print("\n--- Preprocessing Results ---")
        for source_id, converted_path in self.converted_sources.items():
            print(f"  Source '{source_id}' converted to: {converted_path}")
            if os.path.exists(converted_path):
                print(f"    File exists: YES")
                # Check framerate of converted file
                try:
                    result = subprocess.run([
                        'ffprobe', '-v', 'quiet', '-select_streams', 'v:0', 
                        '-show_entries', 'stream=r_frame_rate', '-of', 'csv=p=0', 
                        converted_path
                    ], capture_output=True, text=True, check=True)
                    framerate = result.stdout.strip()
                    print(f"    Framerate: {framerate}")
                except:
                    print(f"    Framerate: Could not determine")
            else:
                print(f"    File exists: NO")
        
        if not self.converted_sources:
            print("  No video sources required conversion")

    def _warn(self, message: str):
        """Emits a non-blocking warning message."""
        warnings.warn(f"SWML Warning: {message}", UserWarning)
        # Also print to stderr for immediate visibility, especially in scripting environments
        print(f"SWML WARNING: {message}", file=sys.stderr)

    def _strip_comments(self, json_string: str) -> str:
        """Strips C-style comments (// and /* */) from a string."""
        # Remove /* ... */ comments
        json_string = re.sub(r'/\*.*?\*/', '', json_string, flags=re.DOTALL)
        # Remove // comments
        json_string = re.sub(r'//.*', '', json_string)
        return json_string

    def _probe_source(self, file_path: str) -> SourceInfo:
        """Probes a media file using ffprobe and caches the result."""
        abs_path = os.path.abspath(file_path)
        if abs_path in self.source_info_cache:
            return self.source_info_cache[abs_path]

        info = SourceInfo(path=abs_path)

        # Check if it's an image based on extension first
        image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.gif')
        if abs_path.lower().endswith(image_extensions):
            info.is_image = True
            try:
                probe_data = ffmpeg.probe(abs_path)
                # For images, look for ANY stream that has width/height.
                dimension_stream = next((s for s in probe_data['streams'] if 'width' in s and 'height' in s), None)

                if dimension_stream:
                    info.width = int(dimension_stream.get('width', 0))
                    info.height = int(dimension_stream.get('height', 0))
                    info.has_video = True # Mark as having video content (dimensions)
                else:
                    self._warn(f"Could not find dimensions (width/height) for image file: {file_path}")
            except Exception as e:
                self._warn(f"Failed to probe image file {file_path} for dimensions. Error: {e}")
            self.source_info_cache[abs_path] = info
            return info

        # For videos/audio files
        try:
            print(f"Probing: {abs_path}")
            probe_data = ffmpeg.probe(abs_path)

            video_stream = next((s for s in probe_data['streams'] if s['codec_type'] == 'video'), None)
            audio_stream = next((s for s in probe_data['streams'] if s['codec_type'] == 'audio'), None)

            if video_stream:
                info.has_video = True
                info.width = int(video_stream.get('width', 0))
                info.height = int(video_stream.get('height', 0))
            if audio_stream:
                info.has_audio = True
            
            # Get duration from format (container) or stream if available
            duration_str = probe_data['format'].get('duration')
            if duration_str:
                info.duration = float(duration_str)
            elif video_stream and video_stream.get('duration'):
                info.duration = float(video_stream['duration'])
            elif audio_stream and audio_stream.get('duration'):
                info.duration = float(audio_stream['duration'])

            self.source_info_cache[abs_path] = info
            return info
        except Exception as e:
            # Check if blender executable is found, if not, that's likely the cause of initial ffprobe issues
            try:
                subprocess.run([self.blender_executable, '--version'], check=True, capture_output=True)
            except (subprocess.CalledProcessError, FileNotFoundError):
                raise SwmlError(f"'{self.blender_executable}' not found. Please ensure Blender is installed and in your system's PATH, or specify the path.")
            # If Blender *is* found, then it's genuinely a media file issue.
            raise SwmlError(f"Failed to probe media file {file_path}. It may be corrupt or an unsupported format. Error: {e}")

    def _validate_transform(self, transform: Dict[str, Any], track_id: Any, clip_id: Any):
        """
        Validate transform object according to the explicit and sequential model.
        Coerces/removes invalid parts, raises error only for fundamental ambiguities.
        """
        if not transform:
            return # Empty transform is valid

        # 1. Remove unsupported keys
        allowed_transform_keys = {'size', 'position', 'anchor'}
        keys_to_remove = [key for key in transform.keys() if key not in allowed_transform_keys]
        for key in keys_to_remove:
            self._warn(f"Unsupported transform property '{key}' in clip '{clip_id}' in track {track_id}. Ignoring and removing.")
            transform.pop(key)

        # 2. Validate 'size' property
        if 'size' in transform:
            size = transform['size']
            if not isinstance(size, dict):
                raise SwmlError(f"Transform 'size' must be an object in clip '{clip_id}' in track {track_id}.") # Structural error

            # Validate 'pixels'
            if 'pixels' in size:
                pixels = size['pixels']
                if not (isinstance(pixels, list) and len(pixels) == 2 and all(isinstance(x, (int, float)) for x in pixels)):
                    self._warn(f"Transform 'size.pixels' must be a list of two numbers in clip '{clip_id}' in track {track_id}. Ignoring 'pixels' property.")
                    size.pop('pixels')

            # Validate 'scale'
            if 'scale' in size:
                scale = size['scale']
                if not (isinstance(scale, list) and len(scale) == 2 and all(isinstance(x, (int, float)) for x in scale)):
                    self._warn(f"Transform 'size.scale' must be a list of two numbers in clip '{clip_id}' in track {track_id}. Ignoring 'scale' property.")
                    size.pop('scale')
                else:
                    # Clamp scale values to a positive minimum to avoid issues like division by zero or invisibility
                    size['scale'] = [max(0.001, s) for s in scale] # Minimum scale of 0.001
                    if any(s < 0.001 for s in scale):
                        self._warn(f"Transform 'size.scale' in clip '{clip_id}' in track {track_id} contained non-positive values. Clamped to minimum 0.001.")
        
        # 3. Validate 'position' property
        if 'position' in transform:
            position = transform['position']
            if not isinstance(position, dict):
                raise SwmlError(f"Transform 'position' must be an object in clip '{clip_id}' in track {track_id}.")

            # NEW: cartesian dominates if both are present
            if 'pixels' in position and 'cartesian' in position:
                self._warn(f"Transform 'position' in clip '{clip_id}' in track {track_id} specifies both 'pixels' and 'cartesian'. 'cartesian' will take precedence and 'pixels' will be ignored.")
                position.pop('pixels') # Remove pixels, allowing cartesian to be used

            if 'pixels' not in position and 'cartesian' not in position:
                self._warn(f"Transform 'position' in clip '{clip_id}' in track {track_id} did not specify 'pixels' or 'cartesian'. Defaulting to center position.")
                # No need to remove/modify, Blender template defaults offset to 0,0 (center) if no transform.

            if 'pixels' in position:
                pixels = position['pixels']
                if not (isinstance(pixels, list) and len(pixels) == 2 and all(isinstance(x, (int, float)) for x in pixels)):
                    self._warn(f"Transform 'position.pixels' must be a list of two numbers in clip '{clip_id}' in track {track_id}. Ignoring 'pixels' property.")
                    position.pop('pixels')
            
            if 'cartesian' in position:
                cartesian = position['cartesian']
                if not (isinstance(cartesian, list) and len(cartesian) == 2 and all(isinstance(x, (int, float)) for x in cartesian)):
                    self._warn(f"Transform 'position.cartesian' must be a list of two numbers in clip '{clip_id}' in track {track_id}. Ignoring 'cartesian' property.")
                    position.pop('cartesian')

        # 4. Validate 'anchor' property (similar to 'position')
        if 'anchor' in transform:
            anchor = transform['anchor']
            if not isinstance(anchor, dict):
                raise SwmlError(f"Transform 'anchor' must be an object in clip '{clip_id}' in track {track_id}.")

            # NEW: cartesian dominates if both are present
            if 'pixels' in anchor and 'cartesian' in anchor:
                self._warn(f"Transform 'anchor' in clip '{clip_id}' in track {track_id} specifies both 'pixels' and 'cartesian'. 'cartesian' will take precedence and 'pixels' will be ignored.")
                anchor.pop('pixels') # Remove pixels, allowing cartesian to be used

            if 'pixels' not in anchor and 'cartesian' not in anchor:
                self._warn(f"Transform 'anchor' in clip '{clip_id}' in track {track_id} did not specify 'pixels' or 'cartesian'. Defaulting to center anchor.")

            if 'pixels' in anchor:
                pixels = anchor['pixels']
                if not (isinstance(pixels, list) and len(pixels) == 2 and all(isinstance(x, (int, float)) for x in pixels)):
                    self._warn(f"Transform 'anchor.pixels' must be a list of two numbers in clip '{clip_id}' in track {track_id}. Ignoring 'pixels' property.")
                    anchor.pop('pixels')
            
            if 'cartesian' in anchor:
                cartesian = anchor['cartesian']
                if not (isinstance(cartesian, list) and len(cartesian) == 2 and all(isinstance(x, (int, float)) for x in cartesian)):
                    self._warn(f"Transform 'anchor.cartesian' must be a list of two numbers in clip '{clip_id}' in track {track_id}. Ignoring 'cartesian' property.")
                    anchor.pop('cartesian')

    def _validate_tracks_and_clips(self, data: Dict[str, Any]):
        """
        Validates tracks and clips, applies defaults, and performs coercions.
        This includes handling missing end_time based on source type.
        """
        comp = data['composition']
        fps = comp['fps'] # Ensure fps is available for time conversions

        # Pre-build source info map for quick lookup
        source_info_map: Dict[str, SourceInfo] = {}
        for src in data['sources']:
            source_id = src.get('id')
            source_path = src.get('path')
            if source_id and source_path: # Already validated source existence earlier
                source_info_map[source_id] = self._probe_source(source_path)

        for track_idx, track in enumerate(data['tracks']):
            track_id = track.get('id', f"track_{track_idx}") # Use index as fallback ID for warnings
            track.setdefault('type', 'video')

            clips = track.get('clips', [])
            
            # Validate that all clip IDs are unique within the track, and all clips have IDs
            clip_ids_in_track = set()
            for clip_idx, clip in enumerate(clips):
                clip_id = clip.get('id')
                if not clip_id:
                    raise SwmlError(f"Clip at index {clip_idx} in track {track_id} missing required 'id' field. All clips must have a unique ID.")
                if clip_id in clip_ids_in_track:
                    raise SwmlError(f"Duplicate clip ID '{clip_id}' found in track {track_id}. Clip IDs must be unique within a track.")
                clip_ids_in_track.add(clip_id)

                # Validate source_id existence (should already be done by main parse_swml)
                source_id = clip.get('source_id')
                if source_id not in source_info_map:
                    raise SwmlError(f"Clip '{clip_id}' in track {track_id} references non-existent source_id '{source_id}'. This is a critical error.")

                source_info = source_info_map[source_id]

                # Coerce start_time
                start_time = clip.get('start_time', 0.0)
                if not isinstance(start_time, (int, float)) or start_time < 0:
                    self._warn(f"Clip '{clip_id}' in track {track_id} has invalid start_time '{start_time}'. Setting to 0.0.")
                    start_time = 0.0
                clip['start_time'] = start_time

                # Handle end_time: if missing, default based on source type
                end_time = clip.get('end_time')
                if end_time is None:
                    if source_info.is_image:
                        end_time = start_time + self.DEFAULT_IMAGE_DURATION_SECONDS
                        self._warn(f"Clip '{clip_id}' (image) in track {track_id} missing 'end_time'. Defaulting to {self.DEFAULT_IMAGE_DURATION_SECONDS}s duration (ends at {end_time:.2f}s).")
                    else: # Video/Audio
                        if source_info.duration > 0:
                            # If source_start is specified, relative duration from source_start
                            source_start = clip.get('source_start', 0.0)
                            if not isinstance(source_start, (int, float)) or source_start < 0:
                                self._warn(f"Clip '{clip_id}' in track {track_id} has invalid source_start '{source_start}'. Setting to 0.0.")
                                source_start = 0.0
                                clip['source_start'] = source_start # Update in data
                            
                            # Clamp source_start to not exceed source duration
                            if source_start >= source_info.duration:
                                self._warn(f"Clip '{clip_id}' in track {track_id} source_start '{source_start:.2f}s' is at or beyond source duration '{source_info.duration:.2f}s'. Setting source_start to 0.0.")
                                source_start = 0.0
                                clip['source_start'] = source_start

                            # Calculate effective duration from source
                            effective_source_duration = source_info.duration - source_start
                            if effective_source_duration < 0: # Should not happen with clamping, but defensive
                                effective_source_duration = 0.0
                            
                            end_time = start_time + effective_source_duration
                            self._warn(f"Clip '{clip_id}' (video/audio) in track {track_id} missing 'end_time'. Defaulting to source duration from source_start (ends at {end_time:.2f}s).")
                        else:
                            # Fallback if source duration is unknown (should be caught by probe earlier, but defensive)
                            end_time = start_time + self.DEFAULT_IMAGE_DURATION_SECONDS # Fallback to image default
                            self._warn(f"Clip '{clip_id}' (video/audio) in track {track_id} missing 'end_time' and source duration unknown. Defaulting to {self.DEFAULT_IMAGE_DURATION_SECONDS}s duration (ends at {end_time:.2f}s).")
                else:
                    # Validate provided end_time
                    if not isinstance(end_time, (int, float)):
                        self._warn(f"Clip '{clip_id}' in track {track_id} has non-numeric end_time '{end_time}'. Coercing to {start_time + self.DEFAULT_IMAGE_DURATION_SECONDS:.2f}s.")
                        end_time = start_time + self.DEFAULT_IMAGE_DURATION_SECONDS
                    elif end_time <= start_time:
                        original_end = end_time
                        min_duration = 1 / fps if fps > 0 else 0.1 # Minimum 1 frame or 0.1s
                        end_time = start_time + min_duration
                        self._warn(f"Clip '{clip_id}' in track {track_id} has end_time ({original_end:.2f}s) <= start_time ({start_time:.2f}s). Coercing end_time to {end_time:.2f}s.")
                clip['end_time'] = end_time

                # Validate and coerce audio clip properties
                if track.get('type') == 'audio':
                    # Volume
                    volume = clip.get('volume', 1.0)
                    if not isinstance(volume, (int, float)):
                        self._warn(f"Clip '{clip_id}' in track {track_id} has non-numeric volume '{volume}'. Defaulting to 1.0.")
                        volume = 1.0
                    elif volume < 0:
                        self._warn(f"Clip '{clip_id}' in track {track_id} has negative volume '{volume}'. Clamping to 0.0.")
                        volume = 0.0
                    elif volume > 1.0:
                         self._warn(f"Clip '{clip_id}' in track {track_id} has volume '{volume:.2f}' > 1.0. This may be clamped by Blender.")
                    clip['volume'] = volume

                    # Fade in/out
                    for fade_key in ['fade_in', 'fade_out']:
                        if fade_key in clip:
                            fade_val = clip[fade_key]
                            if not isinstance(fade_val, (int, float)) or fade_val < 0:
                                self._warn(f"Clip '{clip_id}' in track {track_id} has invalid {fade_key} '{fade_val}'. Setting to 0 (no fade).")
                                clip[fade_key] = 0.0
                                
                # Validate transform if present
                if 'transform' in clip:
                    self._validate_transform(clip['transform'], track_id, clip_id)

    def _validate_transitions(self, data: Dict[str, Any]):
        """
        Validate transition logic for all video tracks using the clip-reference model.
        Applies defaults, performs coercions, and raises errors for critical ambiguities.
        """
        # Collect all clips with their data for easy lookup
        all_clips_map: Dict[str, Dict[str, Any]] = {}
        for track in data['tracks']:
            for clip in track.get('clips', []):
                all_clips_map[clip['id']] = clip

        for track_idx, track in enumerate(data['tracks']):
            if track.get('type', 'video') != 'video': 
                continue
            
            track_id = track.get('id', f"track_{track_idx}")

            transitions_to_process = [] # Build a new list to avoid modifying during iteration
            for trans_idx, transition in enumerate(track.get('transitions', [])):
                from_clip_id = transition.get('from_clip')
                to_clip_id = transition.get('to_clip')
                trans_desc = f"Transition at index {trans_idx} in track {track_id}"

                # Must specify at least one of from_clip or to_clip
                if from_clip_id is None and to_clip_id is None:
                    self._warn(f"{trans_desc} must specify at least one of 'from_clip' or 'to_clip'. Skipping this transition.")
                    continue
                
                # from_clip and to_clip being the same is nonsensical for a cross-fade
                if from_clip_id is not None and to_clip_id is not None and from_clip_id == to_clip_id:
                    self._warn(f"{trans_desc} specifies 'from_clip' and 'to_clip' as the same clip ID '{from_clip_id}'. This is not supported for cross-transitions. Skipping this transition.")
                    continue

                # Validate that referenced clip IDs exist (critical for linking, as Blender would fail)
                if from_clip_id is not None and from_clip_id not in all_clips_map:
                    self._warn(f"{trans_desc} references unknown from_clip '{from_clip_id}'. Skipping this transition.")
                    continue
                if to_clip_id is not None and to_clip_id not in all_clips_map:
                    self._warn(f"{trans_desc} references unknown to_clip '{to_clip_id}'. Skipping this transition.")
                    continue

                # Coerce/Clamp Transition Duration
                duration = transition.get('duration', 1.0) # Default to 1.0s
                if not isinstance(duration, (int, float)):
                    self._warn(f"{trans_desc} has non-numeric duration '{duration}'. Setting to 1.0.")
                    duration = 1.0
                elif duration <= 0:
                    original_duration = duration
                    comp_fps = data['composition']['fps']
                    min_duration_val = 1 / comp_fps if comp_fps > 0 else 0.001 # Minimum duration of 1 frame or 0.001s
                    duration = min_duration_val
                    self._warn(f"{trans_desc} has duration ({original_duration:.2f}s) <= 0. Coercing to {duration:.2f}s.")
                transition['duration'] = duration # Update value in SWML data

                # Cross-transition specific logic (coercion/warnings, not errors)
                if from_clip_id is not None and to_clip_id is not None:
                    from_clip_data = all_clips_map[from_clip_id]
                    to_clip_data = all_clips_map[to_clip_id]
                    
                    from_clip_end = from_clip_data.get('end_time', 0.0)
                    to_clip_start = to_clip_data.get('start_time', 0.0)
                    
                    # Calculate actual overlap duration for clamping
                    actual_overlap_start = max(from_clip_data.get('start_time', 0.0), to_clip_start)
                    actual_overlap_end = min(from_clip_end, to_clip_data.get('end_time', 0.0))
                    
                    available_overlap_duration = max(0.0, actual_overlap_end - actual_overlap_start)

                    if duration > available_overlap_duration:
                        self._warn(f"{trans_desc} from '{from_clip_id}' to '{to_clip_id}': Requested duration ({duration:.2f}s) exceeds actual clip overlap duration ({available_overlap_duration:.2f}s). Clamping transition duration to {available_overlap_duration:.2f}s.")
                        transition['duration'] = available_overlap_duration # Clamp
                    
                    if transition['duration'] <= 0.001: # Check for effectively zero duration after clamping
                        self._warn(f"{trans_desc} from '{from_clip_id}' to '{to_clip_id}': Clamped duration became effectively zero ({transition['duration']:.2f}s). Skipping this cross-transition.")
                        continue # Skip this transition

                transitions_to_process.append(transition)
            
            # Replace the original transitions list with the processed one (which may have skipped entries)
            track['transitions'] = transitions_to_process


    def parse_swml(self) -> Dict[str, Any]:
        """Load and validate SWML file. Applies defaults and coercions extensively."""
        try:
            with open(self.swml_path, 'r') as f:
                raw_json_string = f.read()
            # Strip comments before parsing JSON
            clean_json_string = self._strip_comments(raw_json_string)
            data = json.loads(clean_json_string)
        except FileNotFoundError:
            raise SwmlError(f"SWML file not found: {self.swml_path}")
        except json.JSONDecodeError as e:
            raise SwmlError(f"Invalid JSON in SWML file (or syntax error after comment stripping): {e}")

        required_top_level_keys = ['composition', 'sources', 'tracks']
        for key in required_top_level_keys:
            if key not in data:
                raise SwmlError(f"Missing required top-level key: {key}")

        # --- Validate Composition ---
        comp = data['composition']
        comp_required_keys = ['width', 'height', 'fps']
        for key in comp_required_keys:
            if key not in comp:
                raise SwmlError(f"Missing required composition key: {key}")
            if not isinstance(comp[key], (int, float)) or comp[key] <= 0:
                self._warn(f"Composition '{key}' has invalid value '{comp[key]}'. Clamping to 1 for non-positive values.")
                comp[key] = max(1, int(comp[key])) # Coerce to minimum 1 pixel/fps

        # Handle composition duration: optional, calculate if missing, clamp if unreasonable
        # Temporarily set to None if missing or invalid, so _validate_tracks_and_clips can run first
        if 'duration' not in comp:
            self._warn("Composition duration not specified. Will be calculated from the maximum end time of all clips.")
            data['composition']['duration'] = None # Flag for internal calculation
        elif not isinstance(comp['duration'], (int, float)) or comp['duration'] <= 0:
            self._warn(f"Composition duration '{comp['duration']}' is invalid or non-positive. Setting to a default of 10.0 seconds.")
            data['composition']['duration'] = 10.0 # Coerce to a minimum reasonable value


        # --- Validate Sources ---
        abs_swml_dir = os.path.dirname(os.path.abspath(self.swml_path))
        for source in data['sources']:
            source_id = source.get('id')
            source_path = source.get('path')

            if not source_id:
                raise SwmlError(f"Source entry missing 'id' field: {source}. All sources must have a unique ID.")
            if not source_path:
                raise SwmlError(f"Source '{source_id}' missing 'path' field. This is a critical error.")

            if not os.path.isabs(source_path):
                source['path'] = os.path.join(abs_swml_dir, source_path)
            
            # Check file existence BEFORE probing to avoid long hangs on missing files
            if not os.path.exists(source['path']):
                raise SwmlError(f"Source file for ID '{source_id}' not found: {source['path']}. This is a critical error.")
            
            # Probe and cache source info
            self._probe_source(source['path'])

        # --- Validate Tracks and Clips (including end_time defaulting) ---
        # This must run BEFORE calculating overall duration, as it populates clip end_times
        self._validate_tracks_and_clips(data)

        # --- Validate Transitions (after all clips are processed and have final times) ---
        self._validate_transitions(data)
        
        # NEW: Calculate composition duration if it was not explicitly set (i.e., it's None)
        # This must happen AFTER _validate_tracks_and_clips ensures all clip end_times are numeric.
        if data['composition'].get('duration') is None:
            max_clip_end_time = 0.0
            for track in data['tracks']:
                # Only consider video/audio tracks for duration calculation
                if track.get('type') in ['video', 'audio']: # Audio tracks also contribute to overall duration
                    for clip in track.get('clips', []):
                        # clip['end_time'] is guaranteed to be a number by _validate_tracks_and_clips
                        max_clip_end_time = max(max_clip_end_time, clip['end_time'])
            
            # Ensure a minimal duration even if no clips, to avoid issues with 0 duration in Blender
            data['composition']['duration'] = max(max_clip_end_time, 0.001)


        self.swml_data = data
        return data

    def _get_blender_output_settings(self) -> Dict[str, str]:
        """Get Blender FFmpeg output settings based on SWML format"""
        output_format = self.swml_data['composition'].get('output_format', 'mp4').lower()
        if output_format == 'mp4':
            return {'format': 'MPEG4', 'codec': 'H264', 'audio_codec': 'AAC'}
        elif output_format == 'mov':
            return {'format': 'QUICKTIME', 'codec': 'PRORES', 'audio_codec': 'PCM'}
        elif output_format == 'webm':
            return {'format': 'WEBM', 'codec': 'VP9', 'audio_codec': 'OPUS'}
        else:
            self._warn(f"Unsupported output format '{output_format}'. Defaulting to 'mp4'.")
            return {'format': 'MPEG4', 'codec': 'H264', 'audio_codec': 'AAC'}

    def _generate_blender_script(self) -> str:
        """Generates the full Python script for Blender to execute."""
        # Proper JSON escaping, ensure backslashes are doubled for Python string literal
        swml_data_str = json.dumps(self.swml_data, indent=2).replace('\\', '\\\\').replace("'", "\\'")
        output_settings = self._get_blender_output_settings()
        
        script_dir = os.path.dirname(os.path.abspath(__file__))
        script_template_path = os.path.join(script_dir, "blender_template.py")
        
        try:
            with open(script_template_path, 'r') as template_file:
                script_template = template_file.read()
        except FileNotFoundError:
            raise SwmlError(f"Blender script template not found at: {script_template_path}")
        
        try:
            script = script_template.replace("{swml_data}", swml_data_str)
            script = script.replace("{output_path}", self.output_path)
            script = script.replace("{format}", output_settings['format'])
            script = script.replace("{codec}", output_settings['codec'])
            script = script.replace("{audio_codec}", output_settings['audio_codec'])
            return script
        except Exception as e:
            raise SwmlError(f"Error formatting script template: {e}")

    def _preprocess_video_sources(self):
        """Convert all video sources to composition framerate using ffmpeg."""
        if not self.swml_data:
            raise SwmlError("SWML data not loaded. Call parse_swml() first.")
        
        composition_fps = self.swml_data['composition']['fps']
        print(f"2. Preprocessing video sources for {composition_fps} FPS...")
        
        # Create cache directory in project root for converted videos
        cache_dir = os.path.join(os.path.dirname(self.swml_path), '.swimlane_cache')
        os.makedirs(cache_dir, exist_ok=True)
        
        for source in self.swml_data['sources']:
            source_id = source.get('id')
            source_path = source.get('path')
            abs_source_path = os.path.abspath(source_path)
            
            # Get source info from cache (already probed)
            source_info = self.source_info_cache.get(abs_source_path)
            
            # Skip if it's an image or doesn't have video
            if not source_info or source_info.is_image or not source_info.has_video:
                continue
            
            # Generate output filename
            base_name = os.path.splitext(os.path.basename(source_path))[0]
            converted_filename = f"{base_name}.mp4"
            converted_path = os.path.join(cache_dir, converted_filename)
            
            # Check if cached file already exists
            if os.path.exists(converted_path):
                print(f"   Using cached version: {converted_path}")
                # Update the source path in SWML data to point to cached file
                abs_converted_path = os.path.abspath(converted_path)
                source['path'] = abs_converted_path
                self.converted_sources[source_id] = abs_converted_path
                # Re-probe the cached file to update cache
                self._probe_source(abs_converted_path)
                continue
            
            print(f"   Converting video source '{source_id}': {source_path}")
            
            try:
                # Build ffmpeg command - preserve audio during video framerate conversion
                command = [
                    'ffmpeg',
                    '-i', abs_source_path,
                    '-r', str(composition_fps),          # Set video framerate
                    '-c:v', 'libx264',                   # Video codec
                    '-preset', 'ultrafast',              # Video encoding preset
                    '-crf', '15',                        # Video quality
                    '-c:a', 'aac',                       # Audio codec (preserve audio)
                    '-b:a', '128k',                      # Audio bitrate
                    '-y',                                # Overwrite output files without asking
                    converted_path
                ]
                
                print(f"     Running: {' '.join(command)}")
                
                # Run ffmpeg conversion
                result = subprocess.run(command, capture_output=True, text=True, check=True)
                
                # Update the source path in SWML data to point to converted file (use absolute path)
                abs_converted_path = os.path.abspath(converted_path)
                source['path'] = abs_converted_path
                self.converted_sources[source_id] = abs_converted_path
                
                # Re-probe the converted file to update cache
                self._probe_source(abs_converted_path)
                
                print(f"     ✓ Converted to: {converted_path}")
                
            except subprocess.CalledProcessError as e:
                # If ffmpeg fails, warn but continue with original file
                self._warn(f"Failed to convert video source '{source_id}' to {composition_fps} FPS. Using original file. Error: {e.stderr}")
                continue
            except FileNotFoundError:
                # ffmpeg not found
                self._warn("ffmpeg not found in PATH. Video sources will not be converted to composition framerate. This may cause timing issues.")
                break  # Don't try to convert other videos if ffmpeg is missing
            except Exception as e:
                # Other unexpected errors
                self._warn(f"Unexpected error converting video source '{source_id}': {e}")
                continue

    def render(self):
        """Main rendering function"""
        try:
            print("--- Swimlane Engine: Blender VSE Mode ---")
            print(f"1. Parsing SWML file: {self.swml_path}")
            self.parse_swml() # This will populate self.swml_data and apply defaults/coercions

            # Preprocess video sources to match composition framerate
            self._preprocess_video_sources()

            # Now, after parsing, do final critical checks based on fully processed data
            audio_source_issues, video_source_issues = [], []
            audio_source_ids = {c['source_id'] for t in self.swml_data.get('tracks', []) if t.get('type') == 'audio' for c in t.get('clips', [])}
            video_source_ids = {c['source_id'] for t in self.swml_data.get('tracks', []) if t.get('type', 'video') == 'video' for c in t.get('clips', [])}

            for source in self.swml_data['sources']:
                sid = source.get('id')
                path = source.get('path')
                
                # Retrieve from cache, already probed earlier
                source_info = self.source_info_cache.get(os.path.abspath(path))

                if sid in audio_source_ids and (not source_info or not source_info.has_audio):
                    audio_source_issues.append(f"'{sid}': {path} (has no audio stream but used in audio track)")
                if sid in video_source_ids and (not source_info or not source_info.has_video):
                    # An image used as a video source is fine if it has dimensions
                    if source_info and source_info.is_image and (source_info.width == 0 or source_info.height == 0):
                        video_source_issues.append(f"'{sid}': {path} (Image with no dimensions but used in video track)")
                    elif not source_info or not source_info.has_video:
                        video_source_issues.append(f"'{sid}': {path} (Video with no video stream but used in video track)")

            # These are critical errors: cannot proceed if sources are fundamentally wrong for their use
            if audio_source_issues: raise SwmlError("The following sources are used in audio tracks but lack an audio stream:\n" + "\n".join(audio_source_issues))
            if video_source_issues: raise SwmlError("The following sources are used in video tracks but lack a video stream (or dimensions for images):\n" + "\n".join(video_source_issues))


            print("3. Generating Blender script...")
            blender_script_content = self._generate_blender_script()
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as temp_script:
                temp_script.write(blender_script_content)
                script_path = temp_script.name

            print(f"4. Executing Blender...")
            print(f"   - Executable: {self.blender_executable}")
            print(f"   - Script: {script_path}")
            print(f"   - Output: {self.output_path}")
            print("   This may take a while. Blender output will follow:")
            print("--------------------------------------------------")

            command = [
                self.blender_executable,
                '--background',
                '--python', script_path
            ]
            
            result = subprocess.run(command, check=True, capture_output=True, text=True)
            
            print("--- Blender Output (STDOUT) ---")
            print(result.stdout)
            if result.stderr:
                print("--- Blender Errors/Warnings (STDERR) ---")
                print(result.stderr)
            print("--------------------------------------------------")
            
            print(f"✓ Video rendered successfully: {self.output_path}")

        except SwmlError as e:
            # Our custom errors are already formatted
            print(f"\nERROR: {e}", file=sys.stderr)
            sys.exit(1)
        except subprocess.CalledProcessError as e:
            print("!!! Blender execution failed !!!", file=sys.stderr)
            print(f"Return Code: {e.returncode}", file=sys.stderr)
            print("\n--- STDOUT ---\n", e.stdout, file=sys.stderr)
            print("\n--- STDERR ---\n", e.stderr, file=sys.stderr)
            sys.exit(1) # Exit with non-zero code to indicate failure
        except FileNotFoundError:
             print(f"\nERROR: Blender executable not found at '{self.blender_executable}'. Please ensure Blender is installed and in your PATH.", file=sys.stderr)
             sys.exit(1)
        except Exception as e:
            print(f"\nERROR: An unexpected rendering error occurred: {e}", file=sys.stderr)
            sys.exit(1)
        finally:
            if 'script_path' in locals() and os.path.exists(script_path):
                os.remove(script_path)


def main():
    script_name = os.path.basename(sys.argv[0])
    command_name = "swimlane" if "swimlane" in script_name else "python engine.py"
    
    if len(sys.argv) < 3 or len(sys.argv) > 4 or "--help" in sys.argv or "-h" in sys.argv:
        print("Swimlane Engine - SWML Video Renderer")
        print(f"Usage: {command_name} <input.swml> <output.mp4> [path/to/blender]")
        print("\nArguments:")
        print("  input.swml     Path to the SWML (Swimlane Markup Language) file")
        print("  output.mp4     Path for the output video file (can be .mp4, .mov, or .webm)")
        print("  path/to/blender  Optional path to the Blender executable (default: 'blender')")
        if "--help" in sys.argv or "-h" in sys.argv:
            sys.exit(0)
        else:
            sys.exit(1)
    
    swml_path = sys.argv[1]
    output_path = sys.argv[2]
    blender_exec = sys.argv[3] if len(sys.argv) == 4 else 'blender'
    
    try:
        engine = SwimlaneEngine(swml_path, output_path, blender_executable=blender_exec)
        engine.render()
    except KeyboardInterrupt:
        print("\nRendering cancelled by user", file=sys.stderr)
        sys.exit(1)
    # SwmlError and other exceptions are now handled within render() and exit with non-zero code


if __name__ == "__main__":
    main()