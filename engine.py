#!/usr/bin/env python3
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
from typing import Dict, List, Any, Tuple

# ffprobe is still used for validation, so the dependency remains.
import ffmpeg


class SwmlError(Exception):
    """Custom exception for SWML parsing and processing errors"""
    pass


class SwimlanesEngine:
    def __init__(self, swml_path: str, output_path: str, blender_executable: str = 'blender'):
        self.swml_path = swml_path
        self.output_path = os.path.abspath(output_path)
        self.blender_executable = blender_executable
        self.swml_data = None
        self.source_info_cache = {}  # Cache for ffprobe results

    def _probe_source(self, file_path: str) -> Dict[str, Any]:
        """Probes a media file using ffprobe and caches the result."""
        abs_path = os.path.abspath(file_path)
        if abs_path in self.source_info_cache:
            return self.source_info_cache[abs_path]
        try:
            print(f"Probing: {abs_path}")
            probe = ffmpeg.probe(abs_path)
            self.source_info_cache[abs_path] = probe
            return probe
        except Exception as e:
            try:
                subprocess.run([self.blender_executable, '--version'], check=True, capture_output=True)
            except (subprocess.CalledProcessError, FileNotFoundError):
                raise SwmlError(f"'{self.blender_executable}' not found. Please ensure Blender is installed and in your system's PATH, or specify the path.")
            raise SwmlError(f"Failed to probe media file {file_path}. It may be corrupt or an unsupported format. Error: {e}")

    def _validate_transform(self, transform, track_id, clip_id):
        """
        Validate that a transform object follows the new explicit and sequential model.
        """
        if not transform:
            return  # Empty transform is valid
            
        # Validate size property
        if 'size' in transform:
            size = transform['size']
            if not isinstance(size, dict):
                raise SwmlError(f"Transform 'size' must be an object in clip '{clip_id}' in track {track_id}")
                
            # Check pixels is valid if present
            if 'pixels' in size:
                pixels = size['pixels']
                if not isinstance(pixels, list) or len(pixels) != 2:
                    raise SwmlError(f"Transform 'size.pixels' must be a list of two numbers in clip '{clip_id}' in track {track_id}")
            
            # Check scale is valid if present
            if 'scale' in size:
                scale = size['scale']
                if not isinstance(scale, list) or len(scale) != 2:
                    raise SwmlError(f"Transform 'size.scale' must be a list of two numbers in clip '{clip_id}' in track {track_id}")
        
        # Validate position property
        if 'position' in transform:
            position = transform['position']
            if not isinstance(position, dict):
                raise SwmlError(f"Transform 'position' must be an object in clip '{clip_id}' in track {track_id}")
                
            # Must choose exactly one: pixels or cartesian
            if 'pixels' in position and 'cartesian' in position:
                raise SwmlError(f"Transform 'position' must use either 'pixels' or 'cartesian', not both in clip '{clip_id}' in track {track_id}")
                
            if 'pixels' not in position and 'cartesian' not in position:
                raise SwmlError(f"Transform 'position' must specify either 'pixels' or 'cartesian' in clip '{clip_id}' in track {track_id}")
                
            # Validate the format
            if 'pixels' in position:
                pixels = position['pixels']
                if not isinstance(pixels, list) or len(pixels) != 2:
                    raise SwmlError(f"Transform 'position.pixels' must be a list of two numbers in clip '{clip_id}' in track {track_id}")
                    
            if 'cartesian' in position:
                cartesian = position['cartesian']
                if not isinstance(cartesian, list) or len(cartesian) != 2:
                    raise SwmlError(f"Transform 'position.cartesian' must be a list of two numbers in clip '{clip_id}' in track {track_id}")
        
        # Validate anchor property
        if 'anchor' in transform:
            anchor = transform['anchor']
            if not isinstance(anchor, dict):
                raise SwmlError(f"Transform 'anchor' must be an object in clip '{clip_id}' in track {track_id}")
                
            # Must choose exactly one: pixels or cartesian
            if 'pixels' in anchor and 'cartesian' in anchor:
                raise SwmlError(f"Transform 'anchor' must use either 'pixels' or 'cartesian', not both in clip '{clip_id}' in track {track_id}")
                
            if 'pixels' not in anchor and 'cartesian' not in anchor:
                raise SwmlError(f"Transform 'anchor' must specify either 'pixels' or 'cartesian' in clip '{clip_id}' in track {track_id}")
                
            # Validate the format
            if 'pixels' in anchor:
                pixels = anchor['pixels']
                if not isinstance(pixels, list) or len(pixels) != 2:
                    raise SwmlError(f"Transform 'anchor.pixels' must be a list of two numbers in clip '{clip_id}' in track {track_id}")
                    
            if 'cartesian' in anchor:
                cartesian = anchor['cartesian']
                if not isinstance(cartesian, list) or len(cartesian) != 2:
                    raise SwmlError(f"Transform 'anchor.cartesian' must be a list of two numbers in clip '{clip_id}' in track {track_id}")
        
        # Note: Rotation property has been removed

    def _validate_transitions(self, tracks: List[Dict[str, Any]]) -> None:
        """
        Validate transition logic for all video tracks using the clip-reference model.
        """
        for track in tracks:
            if track.get('type', 'video') != 'video':
                continue

            clips = track.get('clips', [])
            transitions = track.get('transitions', [])
            
            # Build a map of clip IDs for validation
            clip_ids = {clip.get('id') for clip in clips if clip.get('id')}
            
            # Validate that all clip IDs are unique within the track
            clip_id_list = [clip.get('id') for clip in clips if clip.get('id')]
            if len(clip_id_list) != len(set(clip_id_list)):
                raise SwmlError(f"Duplicate clip IDs found in track {track.get('id', 'unknown')}")
            
            # Validate that all clips have IDs
            for clip in clips:
                if not clip.get('id'):
                    raise SwmlError(f"All clips must have an 'id' field in track {track.get('id', 'unknown')}")
                
                # Validate transform if present
                if 'transform' in clip:
                    self._validate_transform(clip['transform'], track.get('id', 'unknown'), clip.get('id'))
            
            # Validate transitions
            for transition in transitions:
                from_clip = transition.get('from_clip')
                to_clip = transition.get('to_clip')
                
                # At least one of from_clip or to_clip must be specified
                if from_clip is None and to_clip is None:
                    raise SwmlError(f"Transition must specify at least one of 'from_clip' or 'to_clip' in track {track.get('id', 'unknown')}")
                
                # Validate that referenced clip IDs exist
                if from_clip is not None and from_clip not in clip_ids:
                    raise SwmlError(f"Transition references unknown from_clip '{from_clip}' in track {track.get('id', 'unknown')}")
                    
                if to_clip is not None and to_clip not in clip_ids:
                    raise SwmlError(f"Transition references unknown to_clip '{to_clip}' in track {track.get('id', 'unknown')}")
                
                # Validate transition duration
                duration = transition.get('duration', 0)
                if duration <= 0:
                    raise SwmlError(f"Transition duration must be > 0 in track {track.get('id', 'unknown')}")
                
                # For cross-transitions, validate timing overlap
                if from_clip is not None and to_clip is not None:
                    from_clip_data = next(c for c in clips if c.get('id') == from_clip)
                    to_clip_data = next(c for c in clips if c.get('id') == to_clip)
                    
                    from_end = from_clip_data.get('end_time', 0)
                    to_start = to_clip_data.get('start_time', 0)
                    
                    # For cross-transitions, clips should overlap or be adjacent
                    if to_start > from_end:
                        raise SwmlError(f"Cross-transition from '{from_clip}' to '{to_clip}' requires overlapping or adjacent clips in track {track.get('id', 'unknown')}")
                    
                    # Check that transition duration doesn't exceed clip durations
                    from_duration = from_clip_data.get('end_time', 0) - from_clip_data.get('start_time', 0)
                    to_duration = to_clip_data.get('end_time', 0) - to_clip_data.get('start_time', 0)
                    
                    if duration > from_duration:
                        raise SwmlError(f"Transition duration ({duration}) exceeds from_clip duration ({from_duration}) for clip '{from_clip}' in track {track.get('id', 'unknown')}")
                    if duration > to_duration:
                        raise SwmlError(f"Transition duration ({duration}) exceeds to_clip duration ({to_duration}) for clip '{to_clip}' in track {track.get('id', 'unknown')}")

    def parse_swml(self) -> Dict[str, Any]:
        """Load and validate SWML file. (Largely unchanged, validation is still crucial)"""
        try:
            with open(self.swml_path, 'r') as f:
                data = json.load(f)
        except FileNotFoundError:
            raise SwmlError(f"SWML file not found: {self.swml_path}")
        except json.JSONDecodeError as e:
            raise SwmlError(f"Invalid JSON in SWML file: {e}")

        required_keys = ['composition', 'sources', 'tracks']
        for key in required_keys:
            if key not in data:
                raise SwmlError(f"Missing required key: {key}")

        # --- Validations (remain very important) ---
        comp = data['composition']
        comp_required = ['width', 'height', 'fps', 'duration']
        for key in comp_required:
            if key not in comp:
                raise SwmlError(f"Missing required composition key: {key}")
        
        # Make source paths absolute for the Blender script
        abs_swml_dir = os.path.dirname(os.path.abspath(self.swml_path))
        for source in data['sources']:
            source_path = source.get('path')
            if source_path and not os.path.isabs(source_path):
                source['path'] = os.path.join(abs_swml_dir, source_path)

        missing_files, audio_no_audio, video_no_video = [], [], []
        audio_source_ids = {c['source_id'] for t in data.get('tracks', []) if t.get('type') == 'audio' for c in t.get('clips', [])}
        video_source_ids = {c['source_id'] for t in data.get('tracks', []) if t.get('type', 'video') == 'video' for c in t.get('clips', [])}

        for source in data['sources']:
            sid = source.get('id')
            path = source.get('path')
            if not path or not os.path.exists(path):
                missing_files.append(f"'{sid}': {path}")
                continue
            if sid in audio_source_ids and not self.probe_has_audio(path):
                audio_no_audio.append(f"'{sid}': {path}")
            if sid in video_source_ids:
                try: self.probe_media_dimensions(path)
                except SwmlError: video_no_video.append(f"'{sid}': {path}")

        if missing_files: raise SwmlError("The following source files do not exist:\n" + "\n".join(missing_files))
        if audio_no_audio: raise SwmlError("The following sources are used in audio tracks but have no audio stream:\n" + "\n".join(audio_no_audio))
        if video_no_video: raise SwmlError("The following sources are used in video tracks but have no video stream:\n" + "\n".join(video_no_video))

        for track in data['tracks']:
            track.setdefault('type', 'video')
            for clip in track.get('clips', []):
                start, end = clip.get('start_time', 0), clip.get('end_time')
                if end is None: raise SwmlError(f"Clip in track {track.get('id', 'unknown')} missing 'end_time'")
                if end <= start: raise SwmlError(f"In track {track.get('id', 'unknown')}, end_time ({end}) must be > start_time ({start})")

        self._validate_transitions(data['tracks'])
        self.swml_data = data
        return data

    def probe_media_dimensions(self, file_path: str) -> Tuple[int, int]:
        probe = self._probe_source(file_path)
        video_stream = next((s for s in probe['streams'] if s['codec_type'] == 'video'), None)
        if not video_stream: raise SwmlError(f"No video stream found in {file_path}")
        return int(video_stream['width']), int(video_stream['height'])
    
    def probe_has_audio(self, file_path: str) -> bool:
        probe = self._probe_source(file_path)
        return any(s['codec_type'] == 'audio' for s in probe['streams'])

    def _get_blender_output_settings(self) -> Dict[str, str]:
        """Get Blender FFmpeg output settings based on SWML format"""
        output_format = self.swml_data['composition'].get('output_format', 'mp4').lower()
        if output_format == 'mp4':
            return {'format': 'MPEG4', 'codec': 'H264', 'audio_codec': 'AAC'}
        elif output_format == 'mov':
            # Using ProRes for MOV as it's a common, high-quality choice. qtrle is less common.
            return {'format': 'QUICKTIME', 'codec': 'PRORES', 'audio_codec': 'PCM'}
        elif output_format == 'webm':
            return {'format': 'WEBM', 'codec': 'VP9', 'audio_codec': 'OPUS'}
        else:
            raise SwmlError(f"Unsupported output format for Blender: {output_format}")

    def _generate_blender_script(self) -> str:
        """Generates the full Python script for Blender to execute."""
        # Properly escape JSON data for embedding in Python script
        swml_data_str = json.dumps(self.swml_data, indent=2).replace('\\', '\\\\').replace("'", "\\'")
        output_settings = self._get_blender_output_settings()
        
        # Get the path to the blender template file
        script_dir = os.path.dirname(os.path.abspath(__file__))
        script_template_path = os.path.join(script_dir, "blender_template.py")
        
        # Read the blender script template from the external file
        try:
            with open(script_template_path, 'r') as template_file:
                script_template = template_file.read()
        except FileNotFoundError:
            raise SwmlError(f"Blender script template not found at: {script_template_path}")
        
        # Replace placeholders with actual values
        try:
            script = script_template.replace("{swml_data}", swml_data_str)
            script = script.replace("{output_path}", self.output_path)
            script = script.replace("{format}", output_settings['format'])
            script = script.replace("{codec}", output_settings['codec'])
            script = script.replace("{audio_codec}", output_settings['audio_codec'])
            return script
        except Exception as e:
            raise SwmlError(f"Error formatting script template: {e}")

    def render(self):
        """Main rendering function"""
        try:
            print("--- Swimlanes Engine: Blender VSE Mode ---")
            print(f"1. Parsing SWML file: {self.swml_path}")
            self.parse_swml()
            
            print("2. Generating Blender script...")
            blender_script_content = self._generate_blender_script()
            
            # Use a temporary file to store the script
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as temp_script:
                temp_script.write(blender_script_content)
                script_path = temp_script.name

            print(f"3. Executing Blender...")
            print(f"   - Executable: {self.blender_executable}")
            print(f"   - Script: {script_path}")
            print(f"   - Output: {self.output_path}")
            print("   This may take a while. Blender output will follow:")
            print("--------------------------------------------------")

            command = [
                self.blender_executable,
                '--background',      # Run without UI
                '--python', script_path
            ]
            
            # Run Blender process
            result = subprocess.run(command, check=True, capture_output=True, text=True)
            
            print("--- Blender Output ---")
            print(result.stdout)
            if result.stderr:
                print("--- Blender Errors/Warnings ---")
                print(result.stderr)
            print("--------------------------------------------------")
            
            print(f"✓ Video rendered successfully: {self.output_path}")

        except SwmlError as e:
            # Pass our own errors up directly
            raise e
        except subprocess.CalledProcessError as e:
            # Handle errors from the Blender process itself
            print("!!! Blender execution failed !!!", file=sys.stderr)
            print(f"Return Code: {e.returncode}", file=sys.stderr)
            print("\n--- STDOUT ---\n", e.stdout, file=sys.stderr)
            print("\n--- STDERR ---\n", e.stderr, file=sys.stderr)
            raise SwmlError("Blender rendering failed. See the output above for details.")
        except FileNotFoundError:
             raise SwmlError(f"Blender executable not found at '{self.blender_executable}'. Please ensure Blender is installed and in your PATH.")
        except Exception as e:
            raise SwmlError(f"An unexpected rendering error occurred: {e}")
        finally:
            # Clean up the temporary script file
            if 'script_path' in locals() and os.path.exists(script_path):
                os.remove(script_path)


def main():
    if len(sys.argv) < 3 or len(sys.argv) > 4:
        print("Usage: python engine.py <input.swml> <output.mp4> [path/to/blender]")
        sys.exit(1)
    
    swml_path = sys.argv[1]
    output_path = sys.argv[2]
    blender_exec = sys.argv[3] if len(sys.argv) == 4 else 'blender'
    
    try:
        engine = SwimlanesEngine(swml_path, output_path, blender_executable=blender_exec)
        engine.render()
    except SwmlError as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nRendering cancelled by user", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
