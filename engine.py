#!/usr/bin/env python3
"""
Swimlane Engine - A declarative video rendering engine using FFmpeg
Parses SWML (Swimlane Markup Language) files and generates videos
"""

import json
import sys
import os
from typing import Dict, List, Any, Tuple, Optional
import ffmpeg


class SwmlError(Exception):
    """Custom exception for SWML parsing and processing errors"""
    pass


class SwimlanesEngine:
    def __init__(self, swml_path: str, output_path: str):
        self.swml_path = swml_path
        self.output_path = output_path
        self.swml_data = None
        self.source_info_cache = {}  # Cache for ffprobe results

    def _probe_source(self, file_path: str) -> Dict[str, Any]:
        """Probes a media file and caches the result."""
        if file_path in self.source_info_cache:
            return self.source_info_cache[file_path]
        try:
            probe = ffmpeg.probe(file_path)
            self.source_info_cache[file_path] = probe
            return probe
        except Exception as e:
            raise SwmlError(f"Failed to probe media file {file_path}: {e}")

    def _validate_transitions(self, tracks: List[Dict[str, Any]]) -> None:
        """Validate transition logic for all video tracks."""
        for track in tracks:
            if track.get('type', 'video') != 'video':
                continue
                
            clips = track.get('clips', [])
            if len(clips) < 2:
                continue
                
            # Sort clips by start_time for adjacency checking
            sorted_clips = sorted(clips, key=lambda c: c.get('start_time', 0))
            
            for i in range(len(sorted_clips) - 1):
                current_clip = sorted_clips[i]
                next_clip = sorted_clips[i + 1]
                
                # Check if clips are adjacent
                current_start = current_clip.get('start_time', 0)
                current_end = current_clip.get('end_time', current_start)
                next_start = next_clip.get('start_time', 0)
                
                # If clips are adjacent (no gap)
                if abs(current_end - next_start) < 0.001:  # Small epsilon for floating point comparison
                    current_cross = current_clip.get('cross', False)
                    next_cross = next_clip.get('cross', False)
                    
                    # Check for explicit cross-transitions
                    if current_cross or next_cross:
                        # Both clips must have cross=True
                        if not (current_cross and next_cross):
                            raise SwmlError(
                                f"Cross-transition mismatch in track {track['id']}: "
                                f"Both adjacent clips must have 'cross': true for cross-transitions"
                            )
                        
                        # Get transitions
                        current_out = current_clip.get('transition_out')
                        next_in = next_clip.get('transition_in')
                        
                        # Both must have transitions
                        if not current_out or not next_in:
                            raise SwmlError(
                                f"Cross-transition error in track {track['id']}: "
                                f"Both clips must have matching transitions when cross=true"
                            )
                        
                        # Transitions must match
                        if (current_out.get('type') != next_in.get('type') or 
                            current_out.get('duration') != next_in.get('duration')):
                            raise SwmlError(
                                f"Cross-transition error in track {track['id']}: "
                                f"Adjacent clips have mismatched transitions:\n"
                                f"- Clip ending at t={current_end}: {current_out}\n"
                                f"- Clip starting at t={next_start}: {next_in}\n"
                                f"Cross-transitions require identical transition types and durations"
                            )
                        
                        # Validate transition duration doesn't exceed clip duration
                        transition_duration = current_out.get('duration', 0)
                        current_duration = current_end - current_start
                        if transition_duration > current_duration:
                            raise SwmlError(
                                f"Transition duration ({transition_duration}s) exceeds clip duration ({current_duration}s) in track {track['id']}"
                            )
                        next_end = next_clip.get('end_time', next_start)
                        next_duration = next_end - next_start
                        if transition_duration > next_duration:
                            raise SwmlError(
                                f"Transition duration ({transition_duration}s) exceeds clip duration ({next_duration}s) in track {track['id']}"
                            )

    def parse_swml(self) -> Dict[str, Any]:
        """Load and validate SWML file"""
        try:
            with open(self.swml_path, 'r') as f:
                data = json.load(f)
        except FileNotFoundError:
            raise SwmlError(f"SWML file not found: {self.swml_path}")
        except json.JSONDecodeError as e:
            raise SwmlError(f"Invalid JSON in SWML file: {e}")

        # --- Base validation ---
        required_keys = ['composition', 'sources', 'tracks']
        for key in required_keys:
            if key not in data:
                raise SwmlError(f"Missing required key: {key}")

        comp = data['composition']
        comp_required = ['width', 'height', 'fps', 'duration']
        for key in comp_required:
            if key not in comp:
                raise SwmlError(f"Missing required composition key: {key}")

        comp.setdefault('output_format', 'mp4')
        comp.setdefault('background_color', 'black')

        # --- Source file validation ---
        missing_files = []
        audio_sources_without_audio = []
        video_sources_without_video = []

        audio_source_ids = {
            clip['source_id']
            for track in data.get('tracks', []) if track.get('type') == 'audio'
            for clip in track.get('clips', [])
        }
        video_source_ids = {
            clip['source_id']
            for track in data.get('tracks', []) if track.get('type', 'video') == 'video'
            for clip in track.get('clips', [])
        }

        for source_id, source_path in data['sources'].items():
            if not os.path.exists(source_path):
                missing_files.append(f"'{source_id}': {source_path}")
                continue
            
            if source_id in audio_source_ids and not self.probe_has_audio(source_path):
                audio_sources_without_audio.append(f"'{source_id}': {source_path}")
            if source_id in video_source_ids:
                try:
                    self.probe_media_dimensions(source_path)
                except SwmlError:
                     video_sources_without_video.append(f"'{source_id}': {source_path}")

        if missing_files:
            raise SwmlError("The following source files do not exist:\n" + "\n".join(missing_files))
        if audio_sources_without_audio:
            raise SwmlError("The following sources are used in audio tracks but have no audio stream:\n" + "\n".join(audio_sources_without_audio))
        if video_sources_without_video:
            raise SwmlError("The following sources are used in video tracks but have no video stream:\n" + "\n".join(video_sources_without_video))

        # Default track type to 'video' for backward compatibility
        for track in data['tracks']:
            track.setdefault('type', 'video')

        # Validate that all clips have required timing information
        for track in data['tracks']:
            for clip in track.get('clips', []):
                start_time = clip.get('start_time', 0)
                end_time = clip.get('end_time')
                
                if end_time is None:
                    raise SwmlError(f"Clip in track {track.get('id', 'unknown')} missing required 'end_time' field")
                
                if end_time <= start_time:
                    raise SwmlError(f"In track {track.get('id', 'unknown')}, clip end_time ({end_time}) must be greater than start_time ({start_time})")

        # Validate transitions
        self._validate_transitions(data['tracks'])

        self.swml_data = data
        return data

    def probe_media_dimensions(self, file_path: str) -> Tuple[int, int]:
        """Get native dimensions of a media file's video stream."""
        probe = self._probe_source(file_path)
        video_stream = next((s for s in probe['streams'] if s['codec_type'] == 'video'), None)
        if not video_stream:
            raise SwmlError(f"No video stream found in {file_path}")
        return int(video_stream['width']), int(video_stream['height'])

    def probe_has_audio(self, file_path: str) -> bool:
        """Check if a media file has an audio stream."""
        probe = self._probe_source(file_path)
        return any(s['codec_type'] == 'audio' for s in probe['streams'])

    def normalize_to_pixels(self, normalized_coord: float, canvas_dimension: int) -> int:
        """Convert normalized coordinate [-1, 1] to pixel coordinate"""
        return int((normalized_coord + 1) * canvas_dimension / 2)

    def calculate_clip_transform(self, clip: Dict[str, Any], source_path: str) -> Dict[str, Any]:
        """Calculate final pixel-based transform for a clip"""
        transform = clip.get('transform', {})
        source_width, source_height = self.probe_media_dimensions(source_path)
        comp = self.swml_data['composition']

        # Handle size - if specified, use it; otherwise use source dimensions
        if 'size' in transform:
            width, height = transform['size']
        else:
            width, height = source_width, source_height

        scale = transform.get('scale', 1.0)
        final_width = int(width * scale)
        final_height = int(height * scale)

        position = transform.get('position', [0, 0])
        anchor = transform.get('anchor', [-1, -1])

        pos_x = self.normalize_to_pixels(position[0], comp['width'])
        pos_y = self.normalize_to_pixels(position[1], comp['height'])

        anchor_x = self.normalize_to_pixels(anchor[0], final_width)
        anchor_y = self.normalize_to_pixels(anchor[1], final_height)

        final_x = pos_x - anchor_x
        final_y = pos_y - anchor_y

        return {'width': final_width, 'height': final_height, 'x': final_x, 'y': final_y, 
                'source_width': source_width, 'source_height': source_height}

    def _apply_clip_transitions(self, stream, clip, duration, is_cross=False):
        """Applies fade-in and fade-out transitions to a video clip stream using optimized fade filter."""
        transition_in = clip.get('transition_in')
        transition_out = clip.get('transition_out')
        
        # Apply fade-in transition (only if NOT part of a cross-transition)
        if transition_in and transition_in.get('type') == 'fade' and not is_cross:
            fade_duration = transition_in.get('duration', 0)
            if fade_duration > 0:
                stream = stream.filter('fade', type='in', start_time=0, duration=fade_duration, alpha=1)

        # Apply fade-out transition (only if NOT part of a cross-transition)
        if transition_out and transition_out.get('type') == 'fade' and not is_cross:
            fade_duration = transition_out.get('duration', 0)
            if fade_duration > 0:
                fade_out_start = duration - fade_duration
                stream = stream.filter('fade', type='out', start_time=max(0, fade_out_start), duration=fade_duration, alpha=1)
                
        return stream
    
    def _process_clip_stream(self, clip: Dict[str, Any], clip_stream: ffmpeg.Stream) -> Tuple[ffmpeg.Stream, float]:
        """Processes a raw clip stream by applying timing, scaling, and non-cross transitions."""
        source_id = clip['source_id']
        source_path = self.swml_data['sources'][source_id]
        
        start_time = clip.get('start_time', 0)
        end_time = clip.get('end_time', start_time)
        duration = end_time - start_time
        
        is_image = source_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
        if is_image:
            clip_stream = clip_stream.filter('loop', loop=-1, size=1, start=0)
            clip_stream = clip_stream.filter('trim', duration=duration).filter('setpts', 'PTS-STARTPTS')
        elif 'source_start' in clip or 'source_end' in clip:
            source_start = clip.get('source_start', 0)
            if 'source_end' in clip:
                source_duration = clip['source_end'] - source_start
                clip_stream = clip_stream.filter('trim', start=source_start, duration=source_duration)
            else:
                clip_stream = clip_stream.filter('trim', start=source_start)
            clip_stream = clip_stream.filter('setpts', 'PTS-STARTPTS')
        
        # All clips (image or video) get scaled
        transform = self.calculate_clip_transform(clip, source_path)
        clip_stream = clip_stream.filter('scale', transform['width'], transform['height'])
        
        # Apply regular (non-cross) transitions
        clip_stream = self._apply_clip_transitions(clip_stream, clip, duration, is_cross=False)
        
        return clip_stream, transform
        
    def _build_video_graph(self):
        """Builds the complex FFmpeg filter graph for video tracks."""
        comp = self.swml_data['composition']
        sources = self.swml_data['sources']
        video_tracks = [t for t in self.swml_data['tracks'] if t.get('type', 'video') == 'video']

        # Create background color source
        current_stream = (
            ffmpeg.input(f"color={comp['background_color']}:size={comp['width']}x{comp['height']}:duration={comp['duration']}:rate={comp['fps']}", f='lavfi')
            .filter('format', 'rgba')
        )

        source_clips = {}
        for track in video_tracks:
            for clip in track['clips']:
                source_id = clip['source_id']
                if source_id not in source_clips:
                    source_clips[source_id] = []
                source_clips[source_id].append(clip)
        
        source_streams = {}
        for source_id, clips in source_clips.items():
            source_path = sources[source_id]
            source_input = ffmpeg.input(source_path)
            
            if len(clips) > 1:
                split_stream = source_input['v'].filter_multi_output("split", len(clips))
                source_streams[source_id] = [split_stream.stream(i) for i in range(len(clips))]
            else:
                source_streams[source_id] = [source_input['v']]

        source_clip_counters = {source_id: 0 for source_id in source_clips.keys()}
        
        # Tracks are rendered from top down (higher ID on top)
        sorted_tracks = sorted(video_tracks, key=lambda t: t['id'], reverse=True)
        
        for track in sorted_tracks:
            clips = track['clips']
            sorted_clips = sorted(clips, key=lambda c: c.get('start_time', 0))
            
            i = 0
            while i < len(sorted_clips):
                clip = sorted_clips[i]
                
                # Check for a cross-transition pair
                next_clip = sorted_clips[i + 1] if i + 1 < len(sorted_clips) else None
                is_cross_pair = (next_clip and 
                               clip.get('cross', False) and 
                               next_clip.get('cross', False) and
                               abs(clip.get('end_time', 0) - next_clip.get('start_time', 0)) < 0.001)
                
                if is_cross_pair:
                    # --- XFADE PATH ---
                    # 1. Process both clips individually
                    clip_A_stream = source_streams[clip['source_id']][source_clip_counters[clip['source_id']]]
                    source_clip_counters[clip['source_id']] += 1
                    
                    clip_B_stream = source_streams[next_clip['source_id']][source_clip_counters[next_clip['source_id']]]
                    source_clip_counters[next_clip['source_id']] += 1

                    # Process each clip to apply transforms, but NOT non-cross transitions
                    # Note: We are not using the helper here to keep the logic explicit for xfade
                    def prepare_for_xfade(c, s):
                        start = c.get('start_time', 0)
                        end = c.get('end_time', start)
                        duration = end - start
                        
                        source_path = sources[c['source_id']]
                        is_image = source_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
                        if is_image:
                            s = s.filter('loop', loop=-1, size=1, start=0).filter('trim', duration=duration).filter('setpts', 'PTS-STARTPTS')
                        elif 'source_start' in c or 'source_end' in c:
                            src_start = c.get('source_start', 0)
                            if 'source_end' in c:
                                s = s.filter('trim', start=src_start, duration=c['source_end'] - src_start)
                            else:
                                s = s.filter('trim', start=src_start)
                            s = s.filter('setpts', 'PTS-STARTPTS')
                        
                        t = self.calculate_clip_transform(c, source_path)
                        s = s.filter('scale', t['width'], t['height']).filter('format', 'rgba')
                        return s, t, duration

                    processed_A, transform_A, duration_A = prepare_for_xfade(clip, clip_A_stream)
                    processed_B, transform_B, duration_B = prepare_for_xfade(next_clip, clip_B_stream)

                    # 2. Create full-size transparent canvases for each clip
                    canvas_A = ffmpeg.input(f"color=black@0.0:s={comp['width']}x{comp['height']}:d={duration_A}:r={comp['fps']}", f='lavfi', t=duration_A).filter('format', 'rgba')
                    canvas_B = ffmpeg.input(f"color=black@0.0:s={comp['width']}x{comp['height']}:d={duration_B}:r={comp['fps']}", f='lavfi', t=duration_B).filter('format', 'rgba')
                    
                    # 3. Overlay each processed clip onto its canvas at the correct position
                    positioned_A = ffmpeg.filter([canvas_A, processed_A], 'overlay', x=transform_A['x'], y=transform_A['y'])
                    positioned_B = ffmpeg.filter([canvas_B, processed_B], 'overlay', x=transform_B['x'], y=transform_B['y'])

                    # 4. Perform the xfade between the two positioned canvases
                    transition_props = clip.get('transition_out', {})
                    transition_duration = transition_props.get('duration', 0)
                    transition_type = transition_props.get('type', 'fade') # Default to fade
                    
                    xfade_offset = duration_A - transition_duration
                    
                    combined_stream = ffmpeg.filter(
                        [positioned_A, positioned_B], 'xfade',
                        transition=transition_type,
                        duration=transition_duration,
                        offset=xfade_offset
                    )
                    
                    # 5. Position the combined, transitioned stream on the main timeline
                    start_time = clip.get('start_time', 0)
                    combined_stream = combined_stream.filter('setpts', f'PTS-STARTPTS+{start_time}/TB')
                    
                    # 6. Overlay the final result onto the main composition stream
                    current_stream = ffmpeg.filter(
                        [current_stream, combined_stream], 'overlay',
                        x=0, y=0, eof_action='pass' # x/y are 0 because it's already a full canvas
                    )
                    
                    i += 2 # Skip the next clip
                else:
                    # --- REGULAR CLIP PATH ---
                    clip_stream_raw = source_streams[clip['source_id']][source_clip_counters[clip['source_id']]]
                    source_clip_counters[clip['source_id']] += 1
                    
                    # Process the clip normally
                    processed_stream, transform = self._process_clip_stream(clip, clip_stream_raw)
                    
                    # Position on timeline
                    start_time = clip.get('start_time', 0)
                    processed_stream = processed_stream.filter('setpts', f'PTS-STARTPTS+{start_time}/TB')
                    processed_stream = processed_stream.filter('format', 'rgba')
                    
                    # Overlay onto the main stream at its calculated position
                    current_stream = ffmpeg.filter(
                        [current_stream, processed_stream], 'overlay',
                        x=transform['x'], y=transform['y'], eof_action='pass'
                    )
                    
                    i += 1
        
        return current_stream
        
    def _build_audio_graph(self):
        """Builds the complex FFmpeg filter graph for audio tracks."""
        comp = self.swml_data['composition']
        sources = self.swml_data['sources']
        audio_tracks = [t for t in self.swml_data['tracks'] if t.get('type') == 'audio']

        if not audio_tracks:
            return None

        source_clips = {}
        for track in audio_tracks:
            for clip in track['clips']:
                source_id = clip['source_id']
                if source_id not in source_clips:
                    source_clips[source_id] = []
                source_clips[source_id].append(clip)
        
        source_streams = {}
        for source_id, clips in source_clips.items():
            source_path = sources[source_id]
            source_input = ffmpeg.input(source_path)
            
            if len(clips) > 1:
                split_stream = source_input['a'].filter_multi_output("asplit", len(clips))
                source_streams[source_id] = [split_stream.stream(i) for i in range(len(clips))]
            else:
                source_streams[source_id] = [source_input['a']]

        processed_audio_clips = []
        source_clip_counters = {source_id: 0 for source_id in source_clips.keys()}

        for track in audio_tracks:
            for clip in track['clips']:
                source_id = clip['source_id']
                clip_index = source_clip_counters[source_id]
                clip_stream = source_streams[source_id][clip_index]
                source_clip_counters[source_id] += 1
                
                start = clip.get('source_start', 0)
                
                start_time = clip.get('start_time', 0)
                end_time = clip.get('end_time')
                
                if end_time is not None:
                    clip_duration = end_time - start_time
                    if clip_duration <= 0:
                        raise SwmlError(f"In audio clip from source '{source_id}', end_time ({end_time}) must be greater than start_time ({start_time}).")
                    clip_stream = clip_stream.filter('atrim', start=start, duration=clip_duration)
                elif 'source_end' in clip:
                    source_end = clip['source_end']
                    if source_end <= start:
                        raise SwmlError(f"In audio clip from source '{source_id}', source_end ({source_end}) must be greater than source_start ({start}).")
                    calculated_duration = source_end - start
                    clip_stream = clip_stream.filter('atrim', start=start, duration=calculated_duration)
                elif 'source_start' in clip:
                    clip_stream = clip_stream.filter('atrim', start=start)
                else:
                    raise SwmlError(f"Audio clip from source '{source_id}' must specify either 'end_time' or 'source_end'")

                clip_stream = clip_stream.filter('asetpts', 'PTS-STARTPTS')

                if 'volume' in clip:
                    clip_stream = clip_stream.filter('volume', clip['volume'])

                if 'fade_in' in clip:
                    clip_stream = clip_stream.filter('afade', type='in', duration=clip['fade_in'])
                if 'fade_out' in clip:
                    clip_stream = clip_stream.filter('afade', type='out', duration=clip['fade_out'])

                start_time_ms = int(clip.get('start_time', 0) * 1000)
                clip_stream = clip_stream.filter('adelay', f"{start_time_ms}|{start_time_ms}")

                processed_audio_clips.append(clip_stream)

        if not processed_audio_clips:
            return None

        mixed_audio = ffmpeg.filter(processed_audio_clips, 'amix', inputs=len(processed_audio_clips), dropout_transition=0)
        return mixed_audio

    def get_output_codec_settings(self) -> Dict[str, str]:
        """Get codec settings based on output format"""
        output_format = self.swml_data['composition'].get('output_format', 'mp4')
        
        settings = {}
        if output_format == 'mp4':
            settings = {'vcodec': 'libx264', 'pix_fmt': 'yuv420p', 'acodec': 'aac'}
        elif output_format == 'mov':
            settings = {'vcodec': 'qtrle', 'pix_fmt': 'yuv420p', 'acodec': 'pcm_s16le'}
        elif output_format == 'webm':
            settings = {'vcodec': 'libvpx-vp9', 'pix_fmt': 'yuva420p', 'acodec': 'libopus'}
        else:
            raise SwmlError(f"Unsupported output format: {output_format}")
        
        return settings

    def render(self):
        """Main rendering function"""
        try:
            print(f"Parsing SWML file: {self.swml_path}")
            self.parse_swml()
            
            print("Building video filter graph...")
            final_video_stream = self._build_video_graph()
            
            print("Building audio filter graph...")
            final_audio_stream = self._build_audio_graph()
            
            print("Configuring output...")
            comp = self.swml_data['composition']
            codec_settings = self.get_output_codec_settings()
            
            output_streams = [final_video_stream]
            if final_audio_stream:
                output_streams.append(final_audio_stream)
            
            kwargs = {**codec_settings, 'shortest': None, 'r': comp['fps'], 't': comp['duration']}

            output = ffmpeg.output(*output_streams, self.output_path, **kwargs)
            
            print(f"Rendering video: {self.output_path}")
            print("This may take a while...")
            
            # For debugging, you can print the full ffmpeg command
            # print(ffmpeg.compile(output, overwrite_output=True))
            
            ffmpeg.run(output, overwrite_output=True, capture_stdout=True, capture_stderr=True)
            
            print(f"✓ Video rendered successfully: {self.output_path}")
            
        except ffmpeg.Error as e:
            print(f"FFmpeg error:")
            print(e.stderr.decode() if e.stderr else "Unknown FFmpeg error")
            raise SwmlError("FFmpeg rendering failed")
        except SwmlError as e:
            # Re-raise SwmlError to be caught by the main function
            raise e
        except Exception as e:
            raise SwmlError(f"An unexpected rendering error occurred: {e}")


def main():
    if len(sys.argv) != 3:
        print("Usage: python engine.py <input.swml> <output.mp4>")
        sys.exit(1)
    
    swml_path = sys.argv[1]
    output_path = sys.argv[2]
    
    try:
        engine = SwimlanesEngine(swml_path, output_path)
        engine.render()
    except SwmlError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nRendering cancelled by user", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()