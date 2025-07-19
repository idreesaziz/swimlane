import bpy
import json
import os
import math

# -------------------------------------------
# SWIMLANE ENGINE BLENDER TEMPLATE
# -------------------------------------------
#
# This template handles the conversion of SWML data to Blender VSE instructions.
# 
# IMPORTANT: All video sources are preprocessed to match the composition framerate
# before this template runs, so all frame calculations use composition FPS.
# 
# -------------------------------------------

# Embedded SWML data
SWML_DATA = json.loads('''{swml_data}''')
OUTPUT_PATH = r"{output_path}"

# Convert sources list to a dictionary for easy lookup
SOURCES_DICT = {s['id']: s['path'] for s in SWML_DATA['sources']}

def time_to_frame(t, fps):
    """Convert time in seconds to frame number (1-indexed for Blender)"""
    return int(round(t * fps))

def time_to_source_frame(t, fps):
    """Convert time to frame number for source media (0-indexed)"""
    return int(round(t * fps))


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
    scene.render.ffmpeg.format = "{format}"
    scene.render.ffmpeg.codec = "{codec}"
    if "{audio_codec}" != "NONE":
        scene.render.ffmpeg.audio_codec = "{audio_codec}"
    
    # Set audio bitrate for better quality
    scene.render.ffmpeg.audio_bitrate = 192

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
    clip_strip_map = {}

    for i, track in enumerate(sorted_tracks):
        base_channel = i * 3 + 1  # Use 3 channels per track (A, B, effects)
        
        if track.get('type') == 'audio':
            process_audio_track(vse, track, base_channel, fps)
        else: # Default is 'video'
            process_video_track(vse, track, base_channel, fps, clip_strip_map)

    # Post-process to create cross-transitions
    create_cross_transitions(vse, sorted_tracks, fps, clip_strip_map)

def process_video_track(vse, track, base_channel, fps, clip_strip_map):
    comp = SWML_DATA['composition']
    sources = SOURCES_DICT
    
    # Build a map of clip ID to clip data
    clips_by_id = {clip['id']: clip for clip in track.get('clips', [])}
    transitions = track.get('transitions', [])
    
    # Determine which clips need to be on alternate channels for cross-transitions
    clips_on_channel_b = set()
    for transition in transitions:
        from_clip = transition.get('from_clip')
        to_clip = transition.get('to_clip')
        
        # For cross-transitions (both clips specified), put to_clip on channel B
        if from_clip is not None and to_clip is not None:
            clips_on_channel_b.add(to_clip)
    
    # A/B roll channels for cross-fade transitions
    channel_a = base_channel
    channel_b = base_channel + 1

    for clip_idx, clip in enumerate(track.get('clips', [])):
        clip_id = clip['id']
        source_id = clip['source_id']
        source_path = sources[source_id]

        start_frame = time_to_frame(clip.get('start_time', 0), fps)
        end_frame = time_to_frame(clip.get('end_time', 0), fps)

        # Choose channel based on whether this clip is involved in cross-transitions
        current_channel = channel_b if clip_id in clips_on_channel_b else channel_a

        is_image = source_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))

        if is_image:
            strip = vse.sequences.new_image(
                name=clip_id,
                filepath=source_path,
                channel=current_channel,
                frame_start=start_frame
            )
            strip.frame_final_end = end_frame
            strip.frame_final_duration = end_frame - start_frame  # Ensure full duration for image
        else: # Is a video
            # Load the strip first to get its properties
            strip = vse.sequences.new_movie(
                name=clip_id,
                filepath=source_path,
                channel=current_channel,
                frame_start=start_frame
            )
            
            if 'source_start' in clip:
                # SIMPLIFIED VIDEO TRIMMING: Since all videos are now at composition FPS
                source_start_seconds = clip['source_start']
                
                # Calculate frame offset in the source video (now at composition fps)
                source_offset_frames = time_to_source_frame(source_start_seconds, fps)
                
                # Step 1: Trim the video source (skip frames at beginning)
                strip.frame_offset_start = source_offset_frames
                
                # Step 2: Set the final duration for the timeline
                strip.frame_final_duration = end_frame - start_frame
                
                # Step 3: Adjust the frame_start position to account for the offset
                # This ensures the trimmed content appears at the correct timeline position
                strip.frame_start = start_frame - source_offset_frames
            else:
                # Normal case - no source offset
                strip.frame_final_duration = end_frame - start_frame

        # Store the strip for later reference using clip ID
        clip_strip_map[clip_id] = strip

        # Handle Transformations (effects channel is base_channel + 2)
        apply_transform(vse, strip, clip, base_channel + 2)

        # Handle Simple Fades (transitions with only from_clip or to_clip)
        apply_simple_transitions(vse, strip, clip_id, transitions, fps)

def process_audio_track(vse, track, base_channel, fps):
    sources = SOURCES_DICT
    for clip in track.get('clips', []):
        source_path = sources[clip['source_id']]
        start_frame = time_to_frame(clip.get('start_time', 0), fps)
        end_frame = time_to_frame(clip.get('end_time'), fps)

        # Add as a sound strip directly
        sound_strip = vse.sequences.new_sound(
            name="audio_{}".format(clip['source_id']),
            filepath=source_path,
            channel=base_channel,
            frame_start=start_frame
        )
        sound_strip.frame_final_duration = end_frame - start_frame

        if 'source_start' in clip:
            # AUDIO TRIMMING: Use the same 3-step approach as video
            source_start_seconds = clip['source_start']
            source_offset_frames = time_to_source_frame(source_start_seconds, fps)
            
            # Step 1: Trim the audio source (skip frames at beginning)
            sound_strip.frame_offset_start = source_offset_frames
            
            # Step 2: Set the final duration for the timeline
            sound_strip.frame_final_duration = end_frame - start_frame
            
            # Step 3: Adjust the frame_start position to account for the offset
            # This ensures the trimmed content appears at the correct timeline position
            sound_strip.frame_start = start_frame - source_offset_frames

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
    transform = clip.get('transform', {})
    if not transform: return
    
    # Calculate transform values
    comp_w, comp_h = comp['width'], comp['height']
    source_w, source_h = strip.elements[0].orig_width, strip.elements[0].orig_height

    # --- Process size transformation (explicit & sequential model) ---
    size_transform = transform.get('size', {})
    
    # Default final dimensions to source dimensions
    final_w, final_h = source_w, source_h
    
    # Step 1: Check for pixels size (exact dimensions)
    if isinstance(size_transform, dict) and 'pixels' in size_transform:
        pixels = size_transform['pixels']
        if isinstance(pixels, list) and len(pixels) == 2:
            final_w, final_h = pixels[0], pixels[1]
    
    # Step 2: Apply scale (if present)
    scale = [1.0, 1.0]  # Default scale
    if isinstance(size_transform, dict) and 'scale' in size_transform:
        scale_value = size_transform['scale']
        if isinstance(scale_value, list) and len(scale_value) == 2:
            scale = scale_value
    
    # Apply scale to dimensions
    final_w *= scale[0]
    final_h *= scale[1]

    # --- Process position (explicit model) ---
    position_transform = transform.get('position', {})
    position_px = [comp_w / 2, comp_h / 2]  # Default: center of composition
    
    if isinstance(position_transform, dict):
        if 'pixels' in position_transform:
            # Direct pixel coordinates (from top-left)
            pixels = position_transform['pixels']
            if isinstance(pixels, list) and len(pixels) == 2:
                position_px = pixels
        elif 'cartesian' in position_transform:
            # Cartesian coordinates: [-1,-1] = top-left, [0,0] = center, [1,1] = bottom-right
            cartesian = position_transform['cartesian']
            if isinstance(cartesian, list) and len(cartesian) == 2:
                position_px[0] = (cartesian[0] + 1) / 2 * comp_w
                position_px[1] = (1 - cartesian[1]) / 2 * comp_h  # Flip Y for cartesian
    
    # --- Process anchor (explicit model) ---
    anchor_transform = transform.get('anchor', {})
    anchor_offset = [final_w / 2, final_h / 2]  # Default: center of the clip
    
    if isinstance(anchor_transform, dict):
        if 'pixels' in anchor_transform:
            # Direct pixel coordinates (from top-left of clip)
            pixels = anchor_transform['pixels']
            if isinstance(pixels, list) and len(pixels) == 2:
                anchor_offset = pixels
        elif 'cartesian' in anchor_transform:
            # Cartesian coordinates: [-1,-1] = top-left, [0,0] = center, [1,1] = bottom-right of clip
            cartesian = anchor_transform['cartesian']
            if isinstance(cartesian, list) and len(cartesian) == 2:
                anchor_offset[0] = (cartesian[0] + 1) / 2 * final_w
                anchor_offset[1] = (1 - cartesian[1]) / 2 * final_h  # Flip Y for cartesian
    
    # Calculate final position
    top_left_x = position_px[0] - anchor_offset[0]
    top_left_y = position_px[1] - anchor_offset[1]
    
    center_x = top_left_x + final_w / 2
    center_y = top_left_y + final_h / 2
    
    # For simple transforms, apply directly to the strip
    strip.transform.scale_x = final_w / source_w
    strip.transform.scale_y = final_h / source_h
    strip.transform.offset_x = center_x - comp_w / 2
    strip.transform.offset_y = center_y - comp_h / 2
    strip.blend_type = 'ALPHA_OVER'
    
def apply_simple_transitions(vse, strip, clip_id, transitions, fps):
    """Apply fade in/out transitions to a single clip."""
    for transition in transitions:
        # Only process transitions that involve this clip as a single clip (not cross-fade)
        from_clip = transition.get('from_clip')
        to_clip = transition.get('to_clip')
        
        # Simple fade out (clip has transition_out)
        if from_clip == clip_id and to_clip is None:
            effect_type = transition.get('effect', 'fade')
            duration = transition.get('duration', 1.0)
            duration_frames = time_to_frame(duration, fps)
            
            if duration_frames > 0:
                strip.blend_alpha = 1.0
                strip.keyframe_insert(data_path='blend_alpha', frame=int(strip.frame_final_end) - duration_frames)
                strip.blend_alpha = 0.0
                strip.keyframe_insert(data_path='blend_alpha', frame=int(strip.frame_final_end))
            
        # Simple fade in (clip has transition_in)
        elif to_clip == clip_id and from_clip is None:
            effect_type = transition.get('effect', 'fade')
            duration = transition.get('duration', 1.0)
            duration_frames = time_to_frame(duration, fps)
            
            if duration_frames > 0:
                strip.blend_alpha = 0.0
                strip.keyframe_insert(data_path='blend_alpha', frame=int(strip.frame_start))
                strip.blend_alpha = 1.0
                strip.keyframe_insert(data_path='blend_alpha', frame=int(strip.frame_start) + duration_frames)

def create_cross_transitions(vse, sorted_tracks, fps, clip_strip_map):
    for track in sorted_tracks:
        if track.get('type', 'video') != 'video': 
            continue
        
        transitions = track.get('transitions', [])
        
        # Process cross-transitions (those with both from_clip and to_clip)
        for transition in transitions:
            from_clip_id = transition.get('from_clip')
            to_clip_id = transition.get('to_clip')
            
            # Skip if not a cross-transition
            if from_clip_id is None or to_clip_id is None:
                continue
            
            strip_a = clip_strip_map.get(from_clip_id)
            strip_b = clip_strip_map.get(to_clip_id)

            if not strip_a or not strip_b: 
                continue
                
            duration_frames = time_to_frame(transition.get('duration', 1.0), fps)
            
            # The transition effect needs to be on the effects channel (highest for this track)
            # Calculate the base channel for this track and use the effects channel
            track_index = next(i for i, t in enumerate(sorted_tracks) if t.get('id') == track.get('id'))
            effects_channel = track_index * 3 + 3  # Third channel of the A/B/Effects trio
            
            # Get transition type and create appropriate effect
            transition_type = transition.get('effect', 'fade')
            effect_name = f"{transition_type}_{from_clip_id}_{to_clip_id}"
            
            if transition_type == 'fade':
                effect = vse.sequences.new_effect(
                    name=effect_name,
                    type='GAMMA_CROSS',
                    channel=effects_channel,
                    frame_start=int(strip_b.frame_start),
                    frame_end=int(strip_b.frame_start) + duration_frames,
                    seq1=strip_a,
                    seq2=strip_b
                )
            elif transition_type == 'wipe':
                effect = vse.sequences.new_effect(
                    name=effect_name,
                    type='WIPE',
                    channel=effects_channel,
                    frame_start=int(strip_b.frame_start),
                    frame_end=int(strip_b.frame_start) + duration_frames,
                    seq1=strip_a,
                    seq2=strip_b
                )
                # Configure wipe direction
                direction = transition.get('direction', 'left_to_right')
                if direction == 'left_to_right':
                    effect.angle = 0.0
                elif direction == 'right_to_left':
                    effect.angle = 3.14159  # 180 degrees
                elif direction == 'top_to_bottom':
                    effect.angle = 1.5708   # 90 degrees
                elif direction == 'bottom_to_top':
                    effect.angle = 4.71239  # 270 degrees
                    
            elif transition_type == 'dissolve':
                effect = vse.sequences.new_effect(
                    name=effect_name,
                    type='ALPHA_OVER',
                    channel=effects_channel,
                    frame_start=int(strip_b.frame_start),
                    frame_end=int(strip_b.frame_start) + duration_frames,
                    seq1=strip_a,
                    seq2=strip_b
                )
                # For dissolve, animate the blend factor
                effect.blend_alpha = 0.0
                effect.keyframe_insert(data_path='blend_alpha', frame=int(strip_b.frame_start))
                effect.blend_alpha = 1.0
                effect.keyframe_insert(data_path='blend_alpha', frame=int(strip_b.frame_start) + duration_frames)
                
            else:
                # Default to fade for unknown types
                effect = vse.sequences.new_effect(
                    name=effect_name,
                    type='GAMMA_CROSS',
                    channel=effects_channel,
                    frame_start=int(strip_b.frame_start),
                    frame_end=int(strip_b.frame_start) + duration_frames,
                    seq1=strip_a,
                    seq2=strip_b
                )

def main():
    print("--- Starting Blender VSE Rendering ---")
    scene, vse = setup_scene()
    process_tracks(scene, vse)
    print("Track processing complete. Starting final render...")
    bpy.ops.render.render(animation=True, write_still=True)
    print("--- Blender VSE Rendering Finished ---")

if __name__ == "__main__":
    main()
