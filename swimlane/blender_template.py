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
    return max(1, int(round(t * fps)) + 1)

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
    
    # Ensure VSE exists BEFORE setting use_sequencer
    if not scene.sequence_editor:
        scene.sequence_editor_create()
        print(f"Sequence Editor Created: {scene.sequence_editor is not None}")
    
    # --- IMPORTANT FIX ---
    # Explicitly tell Blender to render the VSE, not the 3D scene
    scene.render.use_sequencer = True
    
    # Additional VSE enforcement settings
    scene.render.resolution_percentage = 100
    
    # Clear 3D scene mesh objects as backup (color strip will be added later)
    bpy.ops.object.select_all(action='DESELECT')
    for obj in bpy.context.scene.objects:
        if obj.type == 'MESH':  # Only delete mesh objects (cube, etc.)
            obj.select_set(True)
    bpy.ops.object.delete(use_global=False)
    print("DEBUG: Cleared 3D scene mesh objects as backup")
    
    # Debug: Verify the setting
    print(f"VSE Render Setting: scene.render.use_sequencer = {scene.render.use_sequencer}")
    
    # Force scene update to apply VSE settings
    bpy.context.view_layer.update()
    print("DEBUG: Scene updated to apply VSE settings")
    
    # Output settings
    scene.render.filepath = OUTPUT_PATH
    scene.render.image_settings.file_format = 'FFMPEG'
    scene.render.ffmpeg.format = "{format}"
    scene.render.ffmpeg.codec = "{codec}"
    if "{audio_codec}" != "NONE":
        scene.render.ffmpeg.audio_codec = "{audio_codec}"
    
    # Set audio bitrate for better quality
    scene.render.ffmpeg.audio_bitrate = 192
    
    # Apply quality settings based on preview mode
    quality_mode = "{quality}"
    
    # For VSE-only rendering, use WORKBENCH engine to avoid 3D sampling overhead
    print("VSE Mode: Forcing WORKBENCH engine to eliminate 3D sampling")
    scene.render.engine = 'BLENDER_WORKBENCH'
    
    # Disable workbench anti-aliasing to minimize rendering overhead
    if hasattr(scene.display, 'render_aa'):
        scene.display.render_aa = 'OFF'
    
    if quality_mode == "preview":
        # Fast preview settings
        scene.render.ffmpeg.constant_rate_factor = "{blender_quality}"
        if hasattr(scene.render.ffmpeg, 'ffmpeg_preset'):
            scene.render.ffmpeg.ffmpeg_preset = "{blender_preset}"
        print(f"Preview mode: Using {'{blender_quality}'} quality with {'{blender_preset}'} preset for fast rendering")
    else:
        # High quality settings
        scene.render.ffmpeg.constant_rate_factor = "{blender_quality}"
        if hasattr(scene.render.ffmpeg, 'ffmpeg_preset'):
            scene.render.ffmpeg.ffmpeg_preset = "{blender_preset}"
        print(f"High quality mode: Using {'{blender_quality}'} quality with {'{blender_preset}'} preset")

    # Clear existing sequences if sequence editor exists
    if scene.sequence_editor:
        sequences = scene.sequence_editor.sequences
        for seq in list(sequences):
            sequences.remove(seq)
    
    # Add background color strip AFTER clearing sequences
    # This provides a background color and prevents Blender from falling back to 3D scene rendering
    bg_color = comp.get('background_color', [0.0, 0.0, 0.0])  # Default to black if not specified
    
    print(f"DEBUG: Adding background color strip with color RGB: {bg_color}")
    color_strip = scene.sequence_editor.sequences.new_effect(
        name="background_color",
        type='COLOR',
        channel=1,  # Use channel 1 as the background layer
        frame_start=1,
        frame_end=scene.frame_end
    )
    color_strip.color = tuple(bg_color)  # Convert list to tuple for Blender
    print(f"DEBUG: Added background color strip from frame 1 to {scene.frame_end}")
    
    print("Blender scene setup complete.")
    return scene, scene.sequence_editor

def process_tracks(scene, vse):
    comp = SWML_DATA['composition']
    fps = comp['fps']
    
    # Process tracks sorted by ID (like z-index)
    sorted_tracks = sorted(SWML_DATA['tracks'], key=lambda t: t.get('id', 0))
    
    # A map to store created strips for linking transitions
    clip_strip_map = {}

    current_channel = 2  # Start from channel 2 to avoid background on channel 1
    for i, track in enumerate(sorted_tracks):
        track_type = track.get('type', 'video')
        
        if track_type == 'audio':
            process_audio_track(vse, track, current_channel, fps)
            current_channel += 1  # Audio tracks use 1 channel
        elif track_type == 'audiovideo':
            # Process both video and audio for audiovideo tracks
            process_video_track(vse, track, current_channel, fps, clip_strip_map)
            process_audio_track(vse, track, current_channel + 3, fps)  # Audio goes after A, B, effects channels
            current_channel += 4  # Audiovideo tracks use 4 channels (A, B, effects, audio)
        else: # Default is 'video'
            process_video_track(vse, track, current_channel, fps, clip_strip_map)
            current_channel += 3  # Video tracks use 3 channels (A, B, effects)

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
    # Calculate scale factors
    scale_factor_x = final_w / source_w
    scale_factor_y = final_h / source_h
    
    # Detect flip requirements from negative scaling
    flip_horizontal = scale_factor_x < 0
    flip_vertical = scale_factor_y < 0
    
    # Apply absolute values for actual scaling
    strip.transform.scale_x = abs(scale_factor_x)
    strip.transform.scale_y = abs(scale_factor_y)
    strip.transform.offset_x = center_x - comp_w / 2
    strip.transform.offset_y = center_y - comp_h / 2
    strip.blend_type = 'ALPHA_OVER'
    
    # Apply flip effects if negative scaling was detected
    if flip_horizontal or flip_vertical:
        print(f"DEBUG: Detected flip requirement from negative scaling: horizontal={flip_horizontal}, vertical={flip_vertical}")
        if flip_horizontal:
            strip.use_flip_x = True
            print(f"DEBUG: Applied horizontal flip using use_flip_x: {strip.use_flip_x}")
        if flip_vertical:
            strip.use_flip_y = True
            print(f"DEBUG: Applied vertical flip using use_flip_y: {strip.use_flip_y}")
    
    # Apply effects if present
    effects = transform.get('effects', {})
    if effects:
        apply_effects(vse, strip, effects, channel)

def apply_effects(vse, strip, effects, channel):
    """Apply video effects to a strip."""
    
    # Apply color effects
    if 'color' in effects:
        apply_color_effects(strip, effects['color'])
    
    # Apply LUT effects
    if 'lut' in effects:
        apply_lut_effects(vse, strip, effects['lut'], channel)

def apply_color_effects(strip, color_effects):
    """Apply color adjustment effects using Blender modifiers."""
    
    # Brightness - adjust the strip's color multiplier
    if 'brightness' in color_effects:
        brightness = float(color_effects['brightness'])
        # Clamp brightness to reasonable range
        brightness = max(0.0, min(3.0, brightness))
        if hasattr(strip, 'color'):
            strip.color = (brightness, brightness, brightness)
    
    # Contrast - use Blender's color balance
    if 'contrast' in color_effects:
        contrast = float(color_effects['contrast'])
        contrast = max(0.0, min(2.0, contrast))
        if hasattr(strip, 'use_color_balance'):
            strip.use_color_balance = True
            if hasattr(strip, 'color_balance'):
                # Adjust gamma for contrast (1.0 = normal, <1.0 = more contrast, >1.0 = less contrast)
                gamma_val = 1.0 / contrast if contrast > 0 else 1.0
                strip.color_balance.gamma = (gamma_val, gamma_val, gamma_val)
    
    # Saturation - use color balance HSV adjustments
    if 'saturation' in color_effects:
        saturation = float(color_effects['saturation'])
        saturation = max(0.0, min(2.0, saturation))
        if hasattr(strip, 'color_saturation'):
            strip.color_saturation = saturation
        elif hasattr(strip, 'use_color_balance'):
            # Fallback method using color balance
            strip.use_color_balance = True
    
    # Hue shift
    if 'hue' in color_effects:
        hue = float(color_effects['hue'])
        # Hue is typically in range -180 to 180 degrees, normalize to 0-1
        hue_normalized = (hue % 360) / 360.0
        if hasattr(strip, 'color_hue'):
            strip.color_hue = hue_normalized
    
    # Gamma correction
    if 'gamma' in color_effects:
        gamma = float(color_effects['gamma'])
        gamma = max(0.1, min(3.0, gamma))
        if hasattr(strip, 'use_color_balance'):
            strip.use_color_balance = True
            if hasattr(strip, 'color_balance'):
                strip.color_balance.gamma = (gamma, gamma, gamma)
    
    # RGB channel adjustments
    if 'rgb' in color_effects:
        rgb = color_effects['rgb']
        if isinstance(rgb, list) and len(rgb) == 3:
            r, g, b = [max(0.0, min(2.0, float(c))) for c in rgb]
            if hasattr(strip, 'color'):
                strip.color = (r, g, b)
    
    print(f"DEBUG: Applied color effects: {color_effects}")

def apply_lut_effects(vse, strip, lut_effects, channel):
    """Apply LUT (Look-Up Table) effects."""
    
    if 'file' in lut_effects:
        lut_file = lut_effects['file']
        strength = lut_effects.get('strength', 1.0)
        
        try:
            # Try to apply LUT using Blender's built-in color grading
            # This requires the LUT file to be accessible
            if hasattr(strip, 'use_color_balance'):
                strip.use_color_balance = True
                # Note: Full LUT support would require loading the actual LUT file
                # and applying its color transformations. This is a placeholder.
                print(f"DEBUG: Basic LUT file setup applied with strength {strength}")
            
        except Exception as e:
            print(f"WARNING: Could not apply LUT effect: {e}")
            print("NOTE: Full LUT support may require additional Blender addons or compositor setup")
    
    # Preset LUT effects (built-in color grading presets)
    if 'preset' in lut_effects:
        preset = lut_effects['preset'].lower()
        strength = lut_effects.get('strength', 1.0)
        
        # Try direct color balance for movie/video strips
        if hasattr(strip, 'use_color_balance'):
            strip.use_color_balance = True
            
            if preset == 'warm':
                # Warm color grading - boost reds/yellows
                strip.color_balance.lift = (1.0 + 0.1 * strength, 1.0, 1.0 - 0.05 * strength)
                strip.color_balance.gamma = (1.0 + 0.05 * strength, 1.0, 1.0 - 0.1 * strength)
            elif preset == 'cool':
                # Cool color grading - boost blues
                strip.color_balance.lift = (1.0 - 0.05 * strength, 1.0, 1.0 + 0.1 * strength)
                strip.color_balance.gamma = (1.0 - 0.1 * strength, 1.0, 1.0 + 0.05 * strength)
            elif preset == 'vintage':
                # Vintage look - desaturated, warm shadows
                strip.color_balance.lift = (1.0 + 0.15 * strength, 1.0 + 0.1 * strength, 1.0 - 0.1 * strength)
                strip.color_balance.gamma = (1.0, 1.0 + 0.05 * strength, 1.0 - 0.05 * strength)
            elif preset == 'cinema':
                # Cinematic look - teal and orange
                strip.color_balance.lift = (1.0 + 0.1 * strength, 1.0 + 0.05 * strength, 1.0 - 0.15 * strength)
                strip.color_balance.gain = (1.0 - 0.05 * strength, 1.0, 1.0 + 0.1 * strength)
            
            print(f"DEBUG: Applied preset LUT '{preset}' with strength {strength}")
        
        # For image strips, use modifiers approach
        elif hasattr(strip, 'modifiers'):
            import bpy
            
            # Add color balance modifier for image strips
            modifier = strip.modifiers.new('ColorBalance', 'COLOR_BALANCE')
            
            if preset == 'warm':
                # Warm color grading - boost reds/yellows
                modifier.color_balance.lift = (1.0 + 0.1 * strength, 1.0, 1.0 - 0.05 * strength)
                modifier.color_balance.gamma = (1.0 + 0.05 * strength, 1.0, 1.0 - 0.1 * strength)
            elif preset == 'cool':
                # Cool color grading - boost blues
                modifier.color_balance.lift = (1.0 - 0.05 * strength, 1.0, 1.0 + 0.1 * strength)
                modifier.color_balance.gamma = (1.0 - 0.1 * strength, 1.0, 1.0 + 0.05 * strength)
            elif preset == 'vintage':
                # Vintage look - desaturated, warm shadows
                modifier.color_balance.lift = (1.0 + 0.15 * strength, 1.0 + 0.1 * strength, 1.0 - 0.1 * strength)
                modifier.color_balance.gamma = (1.0, 1.0 + 0.05 * strength, 1.0 - 0.05 * strength)
            elif preset == 'cinema':
                # Cinematic look - teal and orange
                modifier.color_balance.lift = (1.0 + 0.1 * strength, 1.0 + 0.05 * strength, 1.0 - 0.15 * strength)
                modifier.color_balance.gain = (1.0 - 0.05 * strength, 1.0, 1.0 + 0.1 * strength)
            
            print(f"DEBUG: Applied preset LUT '{preset}' with strength {strength}")
        
        else:
            print(f"WARNING: Strip type {type(strip)} does not support LUT effects")

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
        if track.get('type', 'video') not in ['video', 'audiovideo']: 
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
            effects_channel = track_index * 3 + 4  # Third channel of the A/B/Effects trio, adjusted for background on channel 1
            
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
                    input1=strip_a,
                    input2=strip_b
                )
            elif transition_type == 'wipe':
                effect = vse.sequences.new_effect(
                    name=effect_name,
                    type='WIPE',
                    channel=effects_channel,
                    frame_start=int(strip_b.frame_start),
                    frame_end=int(strip_b.frame_start) + duration_frames,
                    input1=strip_a,
                    input2=strip_b
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
                    input1=strip_a,
                    input2=strip_b
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
                    input1=strip_a,
                    input2=strip_b
                )

def main():
    print("--- Starting Blender VSE Rendering ---")
    scene, vse = setup_scene()
    process_tracks(scene, vse)
    
    # Debug: Final check before rendering
    if vse and vse.sequences:
        print(f"DEBUG: Final VSE check - Found {len(vse.sequences)} sequences before render")
        for seq in vse.sequences:
            print(f"DEBUG: Sequence '{seq.name}' type: {seq.type}, channel: {seq.channel}, frames: {seq.frame_start}-{seq.frame_final_end}")
    else:
        print("DEBUG: WARNING - No sequences found in VSE before render!")
    
    # Confirm VSE setting one more time
    print(f"DEBUG: Final VSE setting check: scene.render.use_sequencer = {scene.render.use_sequencer}")
    
    print("Track processing complete. Starting final render...")
    bpy.ops.render.render(animation=True, write_still=True)
    print("--- Blender VSE Rendering Finished ---")

if __name__ == "__main__":
    main()
