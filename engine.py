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

    def _validate_transitions(self, tracks: List[Dict[str, Any]]) -> None:
        """
        Validate transition logic for all video tracks.
        Allows for simple fades (on a single clip) and cross-fades (between two clips).
        """
        for track in tracks:
            if track.get('type', 'video') != 'video':
                continue

            clips = track.get('clips', [])
            if len(clips) < 2:
                continue

            sorted_clips = sorted(clips, key=lambda c: c.get('start_time', 0))

            for i in range(len(sorted_clips) - 1):
                current_clip = sorted_clips[i]
                next_clip = sorted_clips[i + 1]

                current_end = current_clip.get('end_time', 0)
                next_start = next_clip.get('start_time', 0)

                # Check for adjacent clips that will be cross-faded
                if abs(current_end - next_start) > 0.001:
                    continue

                current_out = current_clip.get('transition_out', {})
                next_in = next_clip.get('transition_in', {})

                is_current_cross = current_out.get('cross', False)
                is_next_cross = next_in.get('cross', False)

                if is_current_cross != is_next_cross:
                    raise SwmlError(
                        f"Cross-transition mismatch in track {track.get('id', 'unknown')} at t={current_end}: "
                        f"An outgoing cross-transition must be met by an incoming cross-transition on the adjacent clip."
                    )

                if is_current_cross and is_next_cross:
                    if (current_out.get('type') != next_in.get('type') or
                        current_out.get('duration') != next_in.get('duration')):
                        raise SwmlError(
                            f"Cross-transition error in track {track.get('id', 'unknown')}: Adjacent clips have mismatched transitions."
                        )
                    # Duration validation
                    transition_duration = current_out.get('duration', 0)
                    if transition_duration > (current_clip.get('end_time', 0) - current_clip.get('start_time', 0)):
                        raise SwmlError(f"Transition duration exceeds clip duration for clip starting at {current_clip.get('start_time')}.")
                    if transition_duration > (next_clip.get('end_time', 0) - next_clip.get('start_time', 0)):
                        raise SwmlError(f"Transition duration exceeds clip duration for clip starting at {next_clip.get('start_time')}.")


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
        # (removed stray code fragment)
        swml_data_str = json.dumps(self.swml_data, indent=2).replace('\\', '\\\\').replace("'", "\\'")
        output_settings = self._get_blender_output_settings()

        # Using textwrap.dedent for a clean, readable script template
        script = textwrap.dedent(f"""
import bpy
import json
import os

# Embedded SWML data
SWML_DATA = json.loads('''{swml_data_str}''')
OUTPUT_PATH = r"{self.output_path}"

# Convert sources list to a dictionary for easy lookup
SOURCES_DICT = {{s['id']: s['path'] for s in SWML_DATA['sources']}}

def time_to_frame(t, fps):
    return int(round(t * fps)) + 1

def setup_scene():
    scene = bpy.context.scene
    comp = SWML_DATA['composition']
    scene.render.resolution_x = comp['width']
    scene.render.resolution_y = comp['height']
    scene.render.fps = comp['fps']
    scene.frame_end = time_to_frame(comp['duration'], comp['fps'])
    
    # Output settings
    scene.render.filepath = OUTPUT_PATH
    scene.render.image_settings.file_format = 'FFMPEG'
    scene.render.ffmpeg.format = "{output_settings['format']}"
    scene.render.ffmpeg.codec = "{output_settings['codec']}"
    if "{output_settings['audio_codec']}" != "NONE":
        scene.render.ffmpeg.audio_codec = "{output_settings['audio_codec']}"

    # Ensure VSE is the context
    if not scene.sequence_editor:
        scene.sequence_editor_create()
    
    # Clear existing sequences
    sequences = scene.sequence_editor.sequences
    for seq in list(sequences):
        sequences.remove(seq)
    
    print("Blender scene setup complete.")
    return scene, scene.sequence_editor

def process_tracks(scene, vse):
    comp = SWML_DATA['composition']
    fps = comp['fps']
    
    # Process tracks sorted by ID (like z-index)
    sorted_tracks = sorted(SWML_DATA['tracks'], key=lambda t: t.get('id', 0))
    
    # A map to store created strips for linking transitions
    clip_strip_map = {{}}

    for i, track in enumerate(sorted_tracks):
        channel = i * 2 + 1 # Use 2 channels per track for effects
        
        if track.get('type') == 'audio':
            process_audio_track(vse, track, channel, fps)
        else: # Default is 'video'
            process_video_track(vse, track, channel, fps, clip_strip_map)

    # Post-process to create cross-fades
    create_cross_fades(vse, sorted_tracks, fps, clip_strip_map)

def process_video_track(vse, track, channel, fps, clip_strip_map):
    comp = SWML_DATA['composition']
    sources = SOURCES_DICT

    for clip_idx, clip in enumerate(track.get('clips', [])):
        source_id = clip['source_id']
        source_path = sources[source_id]

        start_frame = time_to_frame(clip.get('start_time', 0), fps)
        end_frame = time_to_frame(clip.get('end_time', 0), fps)

        is_image = source_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))

        if is_image:
            strip = vse.sequences.new_image(
                name="{{}}_{{}}".format(track.get('id'), clip.get('source_id')),
                filepath=source_path,
                channel=channel,
                frame_start=start_frame
            )
            strip.frame_final_end = end_frame
            strip.frame_final_duration = end_frame - start_frame  # Ensure full duration for image
        else: # Is a video
            strip = vse.sequences.new_movie(
                name="{{}}_{{}}".format(track.get('id'), clip.get('source_id')),
                filepath=source_path,
                channel=channel,
                frame_start=start_frame
            )
            strip.frame_final_duration = end_frame - start_frame
            if 'source_start' in clip:
                strip.animation_offset_start = time_to_frame(clip['source_start'], fps)

        # Store the strip for later reference (e.g., cross-fades)
        # Use a unique tuple key (track_id, clip_index)
        clip_strip_map[(track.get('id'), clip_idx)] = strip

        # Handle Transformations
        apply_transform(vse, strip, clip, channel + 1)

        # Handle Simple Fades (non-cross)
        apply_simple_fades(strip, clip, fps)

def process_audio_track(vse, track, channel, fps):
    sources = SOURCES_DICT
    for clip in track.get('clips', []):
        source_path = sources[clip['source_id']]
        start_frame = time_to_frame(clip.get('start_time', 0), fps)
        end_frame = time_to_frame(clip.get('end_time'), fps)

        # Add as a sound strip directly
        sound_strip = vse.sequences.new_sound(
            name=f"audio_{{clip['source_id']}}",
            filepath=source_path,
            channel=channel,
            frame_start=start_frame
        )
        sound_strip.frame_final_duration = end_frame - start_frame

        if 'source_start' in clip:
             sound_strip.animation_offset_start = time_to_frame(clip['source_start'], fps)

        sound_strip.volume = clip.get('volume', 1.0)

        # Audio fades using keyframes
        if 'fade_in' in clip and clip['fade_in'] > 0:
            fade_in_frames = time_to_frame(clip['fade_in'], fps) - 1
            if fade_in_frames > 0:
                sound_strip.volume = 0.0
                sound_strip.keyframe_insert(data_path='volume', frame=int(sound_strip.frame_start))
                sound_strip.volume = clip.get('volume', 1.0)
                sound_strip.keyframe_insert(data_path='volume', frame=int(sound_strip.frame_start) + fade_in_frames)
        
        if 'fade_out' in clip and clip['fade_out'] > 0:
            fade_out_frames = time_to_frame(clip['fade_out'], fps) - 1
            if fade_out_frames > 0:
                sound_strip.volume = clip.get('volume', 1.0)
                sound_strip.keyframe_insert(data_path='volume', frame=int(sound_strip.frame_final_end) - fade_out_frames)
                sound_strip.volume = 0.0
                sound_strip.keyframe_insert(data_path='volume', frame=int(sound_strip.frame_final_end))

def apply_transform(vse, strip, clip, channel):
    comp = SWML_DATA['composition']
    transform = clip.get('transform', {{}})
    if not transform: return
    
    # Calculate transform values
    comp_w, comp_h = comp['width'], comp['height']
    source_w, source_h = strip.elements[0].orig_width, strip.elements[0].orig_height

    # Get transform parameters
    scale = transform.get('scale', 1.0)
    position = transform.get('position', [0, 0])  # Cartesian: 0,0 = center
    anchor = transform.get('anchor', [0, 0])      # Cartesian: 0,0 = center of clip
    
    # --- Revised Sizing Logic ---
    if 'size' in transform:
        # If 'size' is specified, it dictates the final dimensions.
        # This will stretch the media if the aspect ratio differs.
        final_w = transform['size'][0] * scale
        final_h = transform['size'][1] * scale
    else:
        # If 'size' is not specified, scale the original source dimensions,
        # which preserves the aspect ratio.
        final_w = source_w * scale
        final_h = source_h * scale

    # Convert cartesian coordinates to pixel coordinates
    # Position: cartesian (-1,-1)=top-left, (0,0)=center, (1,1)=bottom-right
    pos_x_px = (position[0] + 1) / 2 * comp_w
    pos_y_px = (1 - position[1]) / 2 * comp_h  # Flip Y for cartesian
    
    # Anchor: cartesian (-1,-1)=top-left of clip, (0,0)=center of clip, (1,1)=bottom-right of clip
    anchor_x_offset = (anchor[0] + 1) / 2 * final_w
    anchor_y_offset = (1 - anchor[1]) / 2 * final_h  # Flip Y for cartesian
    
    # Calculate final position
    top_left_x = pos_x_px - anchor_x_offset
    top_left_y = pos_y_px - anchor_y_offset
    
    center_x = top_left_x + final_w / 2
    center_y = top_left_y + final_h / 2
    
    # Apply transform directly to the strip to avoid clipping before scaling
    strip.transform.scale_x = final_w / source_w
    strip.transform.scale_y = final_h / source_h
    strip.transform.offset_x = center_x - comp_w / 2
    strip.transform.offset_y = center_y - comp_h / 2
    strip.blend_type = 'ALPHA_OVER'
    
    # Debug output
    import sys
    debug_info = f"Transform debug for {{strip.name}}:\\n"
    debug_info += f"  Source: {{source_w}}x{{source_h}}\\n"
    debug_info += f"  Final: {{final_w}}x{{final_h}}\\n"
    debug_info += f"  Scale: {{strip.transform.scale_x:.3f}}, {{strip.transform.scale_y:.3f}}\\n"
    debug_info += f"  Offset: {{strip.transform.offset_x:.1f}}, {{strip.transform.offset_y:.1f}}\\n"
    print(debug_info)
    sys.stdout.flush()
    # Also write to a debug file
    with open("C:/Dev/swimlane/debug_transforms.txt", "a") as f:
        f.write(debug_info + "\\n")
    
def apply_simple_fades(strip, clip, fps):
    # Animate the strip's opacity for simple fades
    # Note: For cross-fades, Blender's effect strip handles this automatically.
    trans_in = clip.get('transition_in', {{}})
    if trans_in and not trans_in.get('cross'):
        duration_frames = time_to_frame(trans_in.get('duration', 0), fps) - 1
        if duration_frames > 0:
            strip.blend_alpha = 0.0
            strip.keyframe_insert(data_path='blend_alpha', frame=int(strip.frame_start))
            strip.blend_alpha = 1.0
            strip.keyframe_insert(data_path='blend_alpha', frame=int(strip.frame_start) + duration_frames)

    trans_out = clip.get('transition_out', {{}})
    if trans_out and not trans_out.get('cross'):
        duration_frames = time_to_frame(trans_out.get('duration', 0), fps) - 1
        if duration_frames > 0:
            strip.blend_alpha = 1.0
            strip.keyframe_insert(data_path='blend_alpha', frame=int(strip.frame_final_end) - duration_frames)
            strip.blend_alpha = 0.0
            strip.keyframe_insert(data_path='blend_alpha', frame=int(strip.frame_final_end))

def create_cross_fades(vse, sorted_tracks, fps, clip_strip_map):
    for track in sorted_tracks:
        if track.get('type', 'video') != 'video': 
            continue
        
        clips = track.get('clips', [])
        sorted_clips = sorted(clips, key=lambda c: c.get('start_time', 0))

        for i in range(len(sorted_clips) - 1):
            clip_a = sorted_clips[i]
            clip_b = sorted_clips[i+1]
            
            # Check for an adjacent, cross-fade pair
            trans_out = clip_a.get('transition_out', {{}})
            trans_in = clip_b.get('transition_in', {{}})
            
            if not (trans_out.get('cross') and trans_in.get('cross')):
                continue
            
            strip_a = clip_strip_map.get((track.get('id'), i))
            strip_b = clip_strip_map.get((track.get('id'), i+1))

            if not strip_a or not strip_b: 
                continue
                
            duration_frames = time_to_frame(trans_out.get('duration', 0), fps) - 1
            
            # The transition effect needs to be on a higher channel
            transition_channel = max(strip_a.channel, strip_b.channel) + 2
            
            # Create the cross-fade effect
            xfade = vse.sequences.new_effect(
                name=f"xfade_{{strip_a.name}}_{{strip_b.name}}",
                type='GAMMA_CROSS',
                channel=transition_channel,
                frame_start=int(strip_b.frame_start),
                frame_end=int(strip_b.frame_start) + duration_frames,
                seq1=strip_a,
                seq2=strip_b
            )
            xfade.input_count = 2

def main():
    print("--- Starting Blender VSE Rendering ---")
    scene, vse = setup_scene()
    process_tracks(scene, vse)
    print("Track processing complete. Starting final render...")
    bpy.ops.render.render(animation=True, write_still=True)
    print("--- Blender VSE Rendering Finished ---")

if __name__ == "__main__":
    main()
        """)
        return script

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