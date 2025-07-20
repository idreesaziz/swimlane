#!/usr/bin/env python3
"""
Test script for preview mode FPS override
"""

import sys
import os
import json

# Add the current directory to the path
sys.path.insert(0, os.path.dirname(__file__))

from swimlane.engine import SwimlaneEngine

def test_preview_fps_override():
    """Test that preview mode overrides FPS to 10"""
    
    print("=== Testing Preview Mode FPS Override ===\n")
    
    # Test with normal mode (should use original FPS)
    print("1. Testing normal mode...")
    try:
        engine_normal = SwimlaneEngine("examples/composition.swml", "output_normal.mp4", preview_mode=False)
        engine_normal.parse_swml()
        
        # Check original FPS
        original_fps = engine_normal.swml_data['composition']['fps']
        print(f"   Original composition FPS: {original_fps}")
        
        # Generate script (this is where FPS override would happen)
        script_content = engine_normal._generate_blender_script()
        
        # Extract SWML data from the generated script to verify FPS wasn't changed
        swml_start = script_content.find("SWML_DATA = json.loads('''") + len("SWML_DATA = json.loads('''")
        swml_end = script_content.find("''')", swml_start)
        swml_json_str = script_content[swml_start:swml_end]
        parsed_swml = json.loads(swml_json_str)
        
        normal_fps = parsed_swml['composition']['fps']
        print(f"   FPS in generated script: {normal_fps}")
        
        if normal_fps == original_fps:
            print("   ✓ Normal mode preserved original FPS")
        else:
            print(f"   ✗ Normal mode changed FPS from {original_fps} to {normal_fps}")
            return False
            
    except Exception as e:
        print(f"   ✗ Error with normal mode: {e}")
        return False
    
    # Test with preview mode (should override to 10 FPS)
    print("\n2. Testing preview mode...")
    try:
        engine_preview = SwimlaneEngine("examples/composition.swml", "output_preview.mp4", preview_mode=True)
        engine_preview.parse_swml()
        
        # Generate script (this is where FPS override should happen)
        script_content = engine_preview._generate_blender_script()
        
        # Extract SWML data from the generated script to verify FPS was overridden
        swml_start = script_content.find("SWML_DATA = json.loads('''") + len("SWML_DATA = json.loads('''")
        swml_end = script_content.find("''')", swml_start)
        swml_json_str = script_content[swml_start:swml_end]
        parsed_swml = json.loads(swml_json_str)
        
        preview_fps = parsed_swml['composition']['fps']
        print(f"   FPS in generated script: {preview_fps}")
        
        if preview_fps == 10:
            print("   ✓ Preview mode correctly overrode FPS to 10")
        else:
            print(f"   ✗ Preview mode FPS is {preview_fps}, expected 10")
            return False
            
    except Exception as e:
        print(f"   ✗ Error with preview mode: {e}")
        return False
    
    return True

def main():
    success = test_preview_fps_override()
    
    if success:
        print("\n✓ Preview FPS override test completed successfully!")
        print("Preview mode now correctly overrides composition FPS to 10.")
    else:
        print("\n✗ Preview FPS override test failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
