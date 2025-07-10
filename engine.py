# swml/engine.py

import json
import sys
import ffmpeg
import os

# ----------------- #
#  HELPER FUNCTIONS #
# ----------------- #

def get_media_dimensions(filepath):
    """Uses ffprobe to get the native width and height of a media file."""
    try:
        probe = ffmpeg.probe(filepath)
        video_stream = next((s for s in probe['streams'] if s['codec_type'] == 'video'), None)
        if video_stream:
            return video_stream['width'], video_stream['height']
    except ffmpeg.Error as e:
        print(f"Error probing {filepath}: {e.stderr}", file=sys.stderr)
    return None, None # Return None if not a video or error occurs

def calculate_pixel_coords(transform, comp_dims, source_dims):
    """Translates a SWML transform object into final pixel coordinates and size."""
    comp_w, comp_h = comp_dims['width'], comp_dims['height']

    base_size = transform.get('size')
    if base_size is None:
        base_w, base_h = source_dims
    else:
        base_w, base_h = base_size

    scale = transform.get('scale', 1.0)
    final_w = int(base_w * scale)
    final_h = int(base_h * scale)

    anchor_x, anchor_y = transform.get('anchor', [-1, -1])
    offset_x = (final_w * (anchor_x + 1)) / 2
    offset_y = (final_h * (anchor_y + 1)) / 2

    pos_x, pos_y = transform['position']
    target_x = (comp_w * (pos_x + 1)) / 2
    target_y = (comp_h * (pos_y + 1)) / 2
    
    final_x = int(target_x - offset_x)
    final_y = int(target_y - offset_y)

    return final_x, final_y, final_w, final_h

# ----------------- #
#    CORE ENGINE    #
# ----------------- #

def render_swml(swml_data, output_path):
    """The main rendering function."""
    comp = swml_data['composition']
    sources = swml_data['sources']
    tracks = sorted(swml_data['tracks'], key=lambda t: t['id'], reverse=True)

    # --- Pre-computation Step: Probe all sources ---
    source_dims = {
        source_id: get_media_dimensions(path) for source_id, path in sources.items()
    }

    # --- Setup for transparency ---
    # Determine pixel format based on if transparency is needed.
    output_format = comp.get('output_format', 'mp4').lower()
    if output_format == 'mov' or output_format == 'webm':
        PIXEL_FORMAT_ALPHA = 'yuva444p'
        vcodec = 'qtrle' if output_format == 'mov' else 'libvpx-vp9'
        print("Transparency mode enabled. Outputting to .mov/.webm with alpha channel.")
    else:
        PIXEL_FORMAT_ALPHA = 'yuv420p' # Standard format for MP4
        vcodec = 'libx264'
        print("Standard mode enabled. Outputting to .mp4 (no alpha).")

    # --- Create the base canvas ---
    canvas_color = comp.get('background_color', 'black')
    base_canvas = ffmpeg.input(
        f'color=c={canvas_color}:s={comp["width"]}x{comp["height"]}',
        f='lavfi',
        t=comp['duration']
    ).filter('format', pix_fmts=PIXEL_FORMAT_ALPHA)

    # --- Process Tracks and Composite Layers ---
    # For v1, we will simplify and not implement cross-transitions yet, focusing on layering.
    # We will overlay each clip individually.
    
    final_video = base_canvas
    
    for track in tracks:
        for clip in track['clips']:
            source_id = clip['source_id']
            filepath = sources[source_id]
            
            # Create the input stream and immediately set its format
            stream = ffmpeg.input(filepath).filter('format', pix_fmts=PIXEL_FORMAT_ALPHA)
            
            # Handle timing (trim/duration)
            if 'source_end' in clip:
                stream = stream.trim(start=clip.get('source_start', 0), end=clip['source_end']).setpts('PTS-STARTPTS')
            elif 'duration' in clip:
                 # Loop is a trick for still images
                stream = stream.loop(1, size=1).setpts('PTS-STARTPTS').trim(duration=clip['duration'])

            # Handle transform
            if 'transform' in clip:
                x, y, w, h = calculate_pixel_coords(
                    clip['transform'],
                    comp,
                    source_dims[source_id]
                )
                stream = stream.filter('scale', w, h)
            else: # Fullscreen if no transform
                x, y = 0, 0

            # Composite this clip onto the main video
            final_video = final_video.overlay(
                stream,
                x=x,
                y=y,
                enable=f"between(t,{clip['start_time']},{comp['duration']})" # Simple enable logic for v1
            )

    # --- Execute Render ---
    print(f"Rendering video to {output_path}...")
    try:
        final_video.output(
            output_path,
            vcodec=vcodec,
            pix_fmt=PIXEL_FORMAT_ALPHA, # This must match the stream format
            r=comp['fps'],
            t=comp['duration']
        ).overwrite_output().run(capture_stdout=True, capture_stderr=True)
        print("Render complete!")
    except ffmpeg.Error as e:
        print("FFmpeg Error:", file=sys.stderr)
        print(e.stderr.decode(), file=sys.stderr)
        sys.exit(1)


# ----------------- #
#  MAIN EXECUTION   #
# ----------------- #

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Usage: python {sys.argv[0]} <input.swml> <output_video>")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    if not os.path.exists(input_file):
        print(f"Error: Input file not found at '{input_file}'", file=sys.stderr)
        sys.exit(1)

    with open(input_file, 'r') as f:
        swml_spec = json.load(f)

    render_swml(swml_spec, output_file)