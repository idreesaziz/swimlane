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
            width = video_stream.get('width')
            height = video_stream.get('height')
            return width, height
    except ffmpeg.Error as e:
        print(f"Error probing {filepath}: {e.stderr.decode()}", file=sys.stderr)
    return None, None

def calculate_pixel_coords(transform, comp_dims, source_dims):
    """Translates a SWML transform object into final pixel coordinates and size."""
    comp_w, comp_h = comp_dims['width'], comp_dims['height']

    base_size = transform.get('size')
    if base_size is None:
        base_w, base_h = source_dims
        if base_w is None or base_h is None:
             base_w, base_h = comp_w, comp_h
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
    tracks = sorted(swml_data.get('tracks', []), key=lambda t: t['id'], reverse=True)

    source_dims = {
        source_id: get_media_dimensions(path) for source_id, path in sources.items()
    }

    output_format = comp.get('output_format', 'mp4').lower()
    if output_format in ['mov', 'webm']:
        PIXEL_FORMAT_ALPHA = 'yuva444p'
        vcodec = 'qtrle' if output_format == 'mov' else 'libvpx-vp9'
    else:
        PIXEL_FORMAT_ALPHA = 'yuv420p'
        vcodec = 'libx264'

    canvas_color = comp.get('background_color', 'black@0.0' if PIXEL_FORMAT_ALPHA == 'yuva444p' else 'black')
    base_canvas = ffmpeg.input(
        f'color=c={canvas_color}:s={comp["width"]}x{comp["height"]}',
        f='lavfi',
        t=comp['duration'],
        r=comp['fps']
    ).filter('format', pix_fmts=PIXEL_FORMAT_ALPHA)

    final_video = base_canvas
    
    for track in tracks:
        for clip in track.get('clips', []):
            source_id = clip['source_id']
            filepath = sources[source_id]
            is_image = 'duration' in clip

            if is_image:
                stream = ffmpeg.input(filepath, loop=1, r=comp['fps'])
            else:
                stream = ffmpeg.input(filepath)
            
            stream = stream.filter('format', pix_fmts=PIXEL_FORMAT_ALPHA)
            
            if is_image:
                stream = stream.setpts('PTS-STARTPTS').trim(duration=clip['duration'])
            else:
                if 'source_end' in clip:
                    stream = stream.trim(start=clip.get('source_start', 0), end=clip['source_end']).setpts('PTS-STARTPTS')

            if 'transform' in clip:
                x, y, w, h = calculate_pixel_coords(
                    clip['transform'],
                    comp,
                    source_dims[source_id]
                )
                stream = stream.filter('scale', w, h)
            else:
                x, y = 0, 0
            
            start = clip.get('start_time', 0)
            final_video = final_video.overlay(
                stream,
                x=x,
                y=y,
                enable=f"between(t,{start},{comp['duration']})"
            )

    # --- Execute Render ---
    print(f"Rendering video to {output_path}...")
    try:
        output_stream = final_video.output(
            output_path,
            vcodec=vcodec,
            pix_fmt=PIXEL_FORMAT_ALPHA,
            r=comp['fps'],
            t=comp['duration']
        ).overwrite_output()
        
        # To see the command, uncomment the next two lines
        # args = output_stream.get_args()
        # print("ffmpeg " + " ".join(args))

        output_stream.run(capture_stdout=True, capture_stderr=True)
        
        print("Render complete!")
    except ffmpeg.Error as e:
        print("\n[ERROR] An FFmpeg error occurred:", file=sys.stderr)
        print(e.stderr.decode(), file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] A Python error occurred: {e}", file=sys.stderr)
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

    print(f"Loading SWML spec from: {input_file}")
    with open(input_file, 'r') as f:
        swml_spec = json.load(f)

    render_swml(swml_spec, output_file)