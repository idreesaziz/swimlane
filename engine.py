#!/usr/bin/env python3
"""
Swimlane Engine - A declarative video rendering engine using FFmpeg
Parses SWML (Swimlane Markup Language) files and generates videos
"""

import json
import sys
import os
from typing import Dict, List, Any, Tuple
import ffmpeg


class SwmlError(Exception):
    """Custom exception for SWML parsing and processing errors"""
    pass


class SwimlanesEngine:
    def __init__(self, swml_path: str, output_path: str):
        self.swml_path = swml_path
        self.output_path = output_path
        self.swml_data = None
        self.source_dimensions = {}  # Cache for source media dimensions
        
    def parse_swml(self) -> Dict[str, Any]:
        """Load and validate SWML file"""
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
        
        comp = data['composition']
        comp_required = ['width', 'height', 'fps', 'duration']
        for key in comp_required:
            if key not in comp:
                raise SwmlError(f"Missing required composition key: {key}")
        
        comp.setdefault('output_format', 'mp4')
        comp.setdefault('background_color', 'black')
        
        # Validate that all source files exist
        missing_files = []
        for source_id, source_path in data['sources'].items():
            if not os.path.exists(source_path):
                missing_files.append(f"'{source_id}': {source_path}")
        
        if missing_files:
            raise SwmlError(f"The following source files do not exist:\n" + "\n".join(missing_files))
        
        self.swml_data = data
        return data
    
    def probe_media_dimensions(self, file_path: str) -> Tuple[int, int]:
        """Get native dimensions of media file using ffprobe"""
        if file_path in self.source_dimensions:
            return self.source_dimensions[file_path]
        
        try:
            probe = ffmpeg.probe(file_path)
            video_stream = next((s for s in probe['streams'] if s['codec_type'] == 'video'), None)
            
            if not video_stream:
                raise SwmlError(f"No video stream found in {file_path}")

            width = int(video_stream['width'])
            height = int(video_stream['height'])
            
            self.source_dimensions[file_path] = (width, height)
            return width, height
            
        except Exception as e:
            raise SwmlError(f"Failed to probe media file {file_path}: {e}")
    
    def normalize_to_pixels(self, normalized_coord: float, canvas_dimension: int) -> int:
        """Convert normalized coordinate [-1, 1] to pixel coordinate"""
        return int((normalized_coord + 1) * canvas_dimension / 2)
    
    def calculate_clip_transform(self, clip: Dict[str, Any], source_path: str) -> Dict[str, Any]:
        """Calculate final pixel-based transform for a clip"""
        transform = clip.get('transform', {})
        source_width, source_height = self.probe_media_dimensions(source_path)
        
        width, height = transform.get('size', (source_width, source_height))
        scale = transform.get('scale', 1.0)
        final_width = int(width * scale)
        final_height = int(height * scale)
        
        position = transform.get('position', [0, 0])
        anchor = transform.get('anchor', [-1, -1])
        
        comp = self.swml_data['composition']
        
        pos_x = self.normalize_to_pixels(position[0], comp['width'])
        pos_y = self.normalize_to_pixels(position[1], comp['height'])
        
        anchor_x = self.normalize_to_pixels(anchor[0], final_width)
        anchor_y = self.normalize_to_pixels(anchor[1], final_height)
        
        final_x = pos_x - anchor_x
        final_y = pos_y - anchor_y
        
        return {'width': final_width, 'height': final_height, 'x': final_x, 'y': final_y}
    
    def build_filter_graph(self) -> Tuple[ffmpeg.nodes.FilterableStream, List[Any]]:
        """Build the complex FFmpeg filter graph"""
        comp = self.swml_data['composition']
        sources = self.swml_data['sources']
        tracks = self.swml_data['tracks']

        # Create background color source
        current_stream = (
            ffmpeg
            .input(f"color={comp['background_color']}:size={comp['width']}x{comp['height']}:duration={comp['duration']}:rate={comp['fps']}", f='lavfi')
            .filter('format', 'rgba')
        )

        all_inputs = []
        # Group clips by source to handle splitting properly
        source_clips = {}
        sorted_tracks = sorted(tracks, key=lambda t: t['id'], reverse=True)
        
        # First, collect all clips grouped by source
        for track in sorted_tracks:
            for clip in track['clips']:
                source_id = clip['source_id']
                if source_id not in source_clips:
                    source_clips[source_id] = []
                source_clips[source_id].append(clip)
        
        # Create inputs and split streams for each source
        source_streams = {}
        for source_id, clips in source_clips.items():
            source_path = sources[source_id]
            source_input = ffmpeg.input(source_path)
            all_inputs.append(source_input)
            
            if len(clips) > 1:
                # Multiple clips use this source, so we need to split
                split_stream = source_input['v'].filter_multi_output("split", len(clips))
                source_streams[source_id] = [split_stream.stream(i) for i in range(len(clips))]
            else:
                # Only one clip uses this source
                source_streams[source_id] = [source_input['v']]
        
        # Now process each clip with its dedicated stream
        source_clip_counters = {source_id: 0 for source_id in source_clips.keys()}
        
        for track in sorted_tracks:
            for clip in track['clips']:
                source_id = clip['source_id']
                source_path = sources[source_id]
                
                # Get the dedicated stream for this clip
                clip_index = source_clip_counters[source_id]
                clip_stream = source_streams[source_id][clip_index]
                source_clip_counters[source_id] += 1
                
                start_time = clip.get('start_time', 0)
                
                # Handle timing for images vs. videos
                if 'duration' in clip:
                    clip_stream = clip_stream.filter('loop', loop=-1, size=1, start=0)
                    clip_stream = clip_stream.filter('trim', duration=clip['duration']).filter('setpts', 'PTS-STARTPTS')
                elif 'source_start' in clip or 'source_end' in clip:
                    source_start = clip.get('source_start', 0)
                    if 'source_end' in clip:
                        duration = clip['source_end'] - source_start
                        clip_stream = clip_stream.filter('trim', start=source_start, duration=duration)
                    else:
                        clip_stream = clip_stream.filter('trim', start=source_start)
                    clip_stream = clip_stream.filter('setpts', 'PTS-STARTPTS')
                
                transform = self.calculate_clip_transform(clip, source_path)
                clip_stream = clip_stream.filter('scale', transform['width'], transform['height'])
                clip_stream = clip_stream.filter('format', 'rgba')
                
                # Calculate overlay enable timing
                clip_duration = comp['duration'] - start_time
                if 'duration' in clip:
                    clip_duration = clip['duration']
                elif 'source_end' in clip:
                    clip_duration = clip['source_end'] - clip.get('source_start', 0)
                end_time = start_time + clip_duration

                enable_expr = f"between(t,{start_time},{end_time})"
                current_stream = ffmpeg.filter(
                    [current_stream, clip_stream], 
                    'overlay',
                    x=transform['x'],
                    y=transform['y'],
                    enable=enable_expr
                )
        
        return current_stream, all_inputs

    def get_output_codec_settings(self) -> Dict[str, str]:
        """Get codec settings based on output format"""
        output_format = self.swml_data['composition'].get('output_format', 'mp4')
        
        if output_format == 'mp4':
            return {'vcodec': 'libx264', 'pix_fmt': 'yuv420p'}
        elif output_format == 'mov':
            return {'vcodec': 'qtrle', 'pix_fmt': 'yuv420p'}
        elif output_format == 'webm':
            return {'vcodec': 'libvpx-vp9', 'pix_fmt': 'yuva420p'}
        else:
            raise SwmlError(f"Unsupported output format: {output_format}")
    
    def render(self):
        """Main rendering function"""
        try:
            print(f"Parsing SWML file: {self.swml_path}")
            self.parse_swml()
            
            print("Building filter graph...")
            final_stream, _ = self.build_filter_graph()
            
            print("Configuring output...")
            comp = self.swml_data['composition']
            codec_settings = self.get_output_codec_settings()
            
            output = final_stream.output(
                self.output_path,
                **codec_settings,
                r=comp['fps'],
                t=comp['duration']
            )
            
            print(f"Rendering video: {self.output_path}")
            print("This may take a while...")
            
            ffmpeg.run(output, overwrite_output=True, capture_stdout=True, capture_stderr=True)
            
            print(f"✓ Video rendered successfully: {self.output_path}")
            
        except ffmpeg.Error as e:
            print(f"FFmpeg error:")
            print(e.stderr.decode() if e.stderr else "Unknown FFmpeg error")
            raise SwmlError("FFmpeg rendering failed")
        except Exception as e:
            raise SwmlError(f"Rendering failed: {e}")


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
        print(f"Error: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nRendering cancelled by user")
        sys.exit(1)


if __name__ == "__main__":
    main()