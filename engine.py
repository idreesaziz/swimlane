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
                current_duration = current_clip.get('duration', 0)
                current_end = current_start + current_duration
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
                        if transition_duration > current_duration:
                            raise SwmlError(
                                f"Transition duration ({transition_duration}s) exceeds clip duration ({current_duration}s) in track {track['id']}"
                            )
                        if transition_duration > next_clip.get('duration', float('inf')):
                            next_duration = next_clip.get('duration', 'undefined')
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
        
        # Apply fade-in transition
        if transition_in and transition_in.get('type') == 'fade':
            fade_duration = transition_in.get('duration', 0)
            if fade_duration > 0:
                stream = stream.filter('fade', type='in', start_time=0, duration=fade_duration, alpha=1)

        # Apply fade-out transition
        if transition_out and transition_out.get('type') == 'fade':
            fade_duration = transition_out.get('duration', 0)
            if fade_duration > 0:
                fade_out_start = duration - fade_duration
                stream = stream.filter('fade', type='out', start_time=max(0, fade_out_start), duration=fade_duration, alpha=1)
                
        return stream

    def _calculate_clip_timing(self, clip: Dict[str, Any], next_clip: Optional[Dict[str, Any]] = None, 
                              prev_clip: Optional[Dict[str, Any]] = None) -> Tuple[float, float, bool, float]:
        """Calculate effective start time and duration for a clip, considering cross-transitions.
        
        Returns:
            (effective_start, base_duration, is_cross_transition, render_duration)
            - effective_start: When the clip starts rendering (may be earlier for cross-fade in)
            - base_duration: Original clip duration (unchanged)
            - is_cross_transition: Whether this clip participates in cross-transitions
            - render_duration: How long to render the clip (may be longer for overlaps)
        """
        start_time = clip.get('start_time', 0)
        
        # Get the base duration for this clip
        if 'duration' in clip:
            base_duration = clip['duration']
        elif 'source_end' in clip and 'source_start' in clip:
            base_duration = clip['source_end'] - clip['source_start']
        elif 'source_end' in clip:
            base_duration = clip['source_end']  # Assuming source_start defaults to 0
        else:
            # For images or when no duration specified, we'll set this later
            base_duration = 0
        
        effective_start = start_time
        render_duration = base_duration
        is_cross_transition = False
        
        # Check for cross-fade IN (with previous clip)
        if (prev_clip and 
            clip.get('cross', False) and 
            prev_clip.get('cross', False)):
            
            prev_start = prev_clip.get('start_time', 0)
            prev_duration = prev_clip.get('duration', 0)
            if 'source_end' in prev_clip and 'source_start' in prev_clip:
                prev_duration = prev_clip['source_end'] - prev_clip['source_start']
            elif 'source_end' in prev_clip:
                prev_duration = prev_clip['source_end']
                
            # Check if clips are adjacent
            if abs((prev_start + prev_duration) - start_time) < 0.001:
                transition_in = clip.get('transition_in')
                if transition_in and transition_in.get('type') == 'fade':
                    transition_duration = transition_in.get('duration', 0)
                    # Start rendering earlier to create overlap, but don't change logical timing
                    effective_start = start_time - transition_duration
                    render_duration = base_duration + transition_duration
                    is_cross_transition = True
        
        # Check for cross-fade OUT (with next clip)
        if (next_clip and 
            clip.get('cross', False) and 
            next_clip.get('cross', False) and
            abs((start_time + base_duration) - next_clip.get('start_time', 0)) < 0.001):  # Adjacent clips
            
            transition_out = clip.get('transition_out')
            if transition_out and transition_out.get('type') == 'fade':
                transition_duration = transition_out.get('duration', 0)
                # Extend rendering duration to create overlap, but don't change logical end time
                render_duration = base_duration + transition_duration
                is_cross_transition = True
        
        return effective_start, base_duration, is_cross_transition, render_duration
    
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
            
            for i, clip in enumerate(sorted_clips):
                source_id = clip['source_id']
                source_path = sources[source_id]
                
                clip_index = source_clip_counters[source_id]
                clip_stream = source_streams[source_id][clip_index]
                source_clip_counters[source_id] += 1
                
                # Calculate timing considering cross-transitions
                next_clip = sorted_clips[i + 1] if i + 1 < len(sorted_clips) else None
                prev_clip = sorted_clips[i - 1] if i > 0 else None
                effective_start, base_duration, is_cross_transition, render_duration = self._calculate_clip_timing(clip, next_clip, prev_clip)
                
                # For images without duration specified, use remaining composition time or default
                if base_duration == 0:
                    if 'duration' in clip:
                        base_duration = clip['duration']
                        render_duration = base_duration
                    else:
                        # Default duration for images is remaining time from start
                        base_duration = comp['duration'] - clip.get('start_time', 0)
                        render_duration = base_duration
                
                # Handle timing for images vs. videos
                is_image = source_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
                if is_image:
                    # For images, apply the same transform logic as videos
                    image_duration = render_duration
                    transform = self.calculate_clip_transform(clip, source_path)
                    
                    # Create the base image stream
                    clip_stream = clip_stream.filter('loop', loop=-1, size=1, start=0)
                    clip_stream = clip_stream.filter('trim', duration=image_duration).filter('setpts', 'PTS-STARTPTS')
                    
                    # Apply scaling to match the calculated transform dimensions
                    clip_stream = clip_stream.filter('scale', transform['width'], transform['height'])
                    
                elif 'source_start' in clip or 'source_end' in clip:
                    source_start = clip.get('source_start', 0)
                    if 'source_end' in clip:
                        source_duration = clip['source_end'] - source_start
                        clip_stream = clip_stream.filter('trim', start=source_start, duration=source_duration)
                    else:
                        clip_stream = clip_stream.filter('trim', start=source_start)
                    clip_stream = clip_stream.filter('setpts', 'PTS-STARTPTS')
                    
                else:
                    # No source timing specified, use the whole clip
                    pass
                
                # For non-image clips, apply scaling as before
                if not is_image:
                    transform = self.calculate_clip_transform(clip, source_path)
                    clip_stream = clip_stream.filter('scale', transform['width'], transform['height'])
                
                # Apply transitions using the BASE duration (not render duration)
                # This ensures fade timings are calculated based on the user's intended clip length
                clip_stream = self._apply_clip_transitions(clip_stream, clip, base_duration, is_cross_transition)

                # For cross-transitions, we need to handle the timeline positioning carefully
                if is_cross_transition:
                    # Use effective_start (which may be earlier) for positioning
                    clip_stream = clip_stream.filter('setpts', f'PTS-STARTPTS+{effective_start}/TB')
                else:
                    # Use the original start time for regular clips
                    clip_stream = clip_stream.filter('setpts', f'PTS-STARTPTS+{clip.get("start_time", 0)}/TB')

                clip_stream = clip_stream.filter('format', 'rgba')
                
                # Apply calculated transform for positioning (both images and videos)
                transform = self.calculate_clip_transform(clip, source_path)
                current_stream = ffmpeg.filter(
                    [current_stream, clip_stream], 'overlay',
                    x=transform['x'], y=transform['y'], eof_action='pass'
                )
        
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
                
                # --- UPDATED TIMING LOGIC ---
                start = clip.get('source_start', 0)
                
                if 'duration' in clip:
                    # Priority 1: Use explicit duration if provided
                    clip_stream = clip_stream.filter('atrim', start=start, duration=clip['duration'])
                elif 'source_end' in clip:
                    # Priority 2: Use source_end to calculate duration
                    end = clip['source_end']
                    if end <= start:
                        raise SwmlError(f"In clip from source '{source_id}', source_end ({end}) must be greater than source_start ({start}).")
                    calculated_duration = end - start
                    clip_stream = clip_stream.filter('atrim', start=start, duration=calculated_duration)
                elif 'source_start' in clip:
                    # Priority 3: No end point specified, so trim from start to end of source
                    clip_stream = clip_stream.filter('atrim', start=start)

                clip_stream = clip_stream.filter('asetpts', 'PTS-STARTPTS')
                # --- END OF UPDATED LOGIC ---

                # Apply volume
                if 'volume' in clip:
                    clip_stream = clip_stream.filter('volume', clip['volume'])

                # Apply fades
                if 'fade_in' in clip:
                    clip_stream = clip_stream.filter('afade', type='in', duration=clip['fade_in'])
                if 'fade_out' in clip:
                    clip_stream = clip_stream.filter('afade', type='out', duration=clip['fade_out'])

                # Apply delay to position the clip on the timeline for mixing
                start_time_ms = int(clip.get('start_time', 0) * 1000)
                clip_stream = clip_stream.filter('adelay', f"{start_time_ms}|{start_time_ms}")

                processed_audio_clips.append(clip_stream)

        if not processed_audio_clips:
            return None

        # Mix all processed audio clips together
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
            
            # Use shortest to ensure output duration matches composition if audio is longer
            kwargs = {**codec_settings, 'shortest': None, 'r': comp['fps'], 't': comp['duration']}

            output = ffmpeg.output(*output_streams, self.output_path, **kwargs)
            
            print(f"Rendering video: {self.output_path}")
            print("This may take a while...")
            
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