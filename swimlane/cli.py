#!/usr/bin/env python3
"""
Command-line interface for the Swimlane Engine
"""

import sys
import os

def main():
    """Main CLI entry point"""
    # Parse arguments and flags
    args = []
    preview_mode = False
    
    for arg in sys.argv[1:]:
        if arg == "--preview":
            preview_mode = True
        elif arg in ["--help", "-h"]:
            print("Swimlane Engine - SWML Video Renderer")
            print("Usage: swimlane [--preview] <input.swml> <output.mp4> [path/to/blender]")
            print("\nArguments:")
            print("  input.swml     Path to the SWML (Swimlane Markup Language) file")
            print("  output.mp4     Path for the output video file (can be .mp4, .mov, or .webm)")
            print("  path/to/blender  Optional path to the Blender executable (default: 'blender')")
            print("\nOptions:")
            print("  --preview      Use fast/low quality render settings for quick previews")
            sys.exit(0)
        else:
            args.append(arg)
    
    # Handle command-line arguments
    if len(args) < 2 or len(args) > 3:
        print("Swimlane Engine - SWML Video Renderer")
        print("Usage: swimlane [--preview] <input.swml> <output.mp4> [path/to/blender]")
        print("\nArguments:")
        print("  input.swml     Path to the SWML (Swimlane Markup Language) file")
        print("  output.mp4     Path for the output video file (can be .mp4, .mov, or .webm)")
        print("  path/to/blender  Optional path to the Blender executable (default: 'blender')")
        print("\nOptions:")
        print("  --preview      Use fast/low quality render settings for quick previews")
        sys.exit(1)
    
    # Import is done here to ensure fast startup for help message
    from swimlane.engine import SwimlaneEngine, SwmlError
    
    swml_path = args[0]
    output_path = args[1]
    blender_exec = args[2] if len(args) == 3 else 'blender'
    
    try:
        engine = SwimlaneEngine(swml_path, output_path, blender_executable=blender_exec, preview_mode=preview_mode)
        engine.render()
    except SwmlError as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nRendering cancelled by user", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
