#!/usr/bin/env python3
"""
Command-line interface for the Swimlane Engine
"""

import sys
import os

def main():
    """Main CLI entry point"""
    # Handle command-line arguments
    if len(sys.argv) < 3 or len(sys.argv) > 4 or "--help" in sys.argv or "-h" in sys.argv:
        print("Swimlane Engine - SWML Video Renderer")
        print("Usage: swimlane <input.swml> <output.mp4> [path/to/blender]")
        print("\nArguments:")
        print("  input.swml     Path to the SWML (Swimlane Markup Language) file")
        print("  output.mp4     Path for the output video file (can be .mp4, .mov, or .webm)")
        print("  path/to/blender  Optional path to the Blender executable (default: 'blender')")
        if "--help" in sys.argv or "-h" in sys.argv:
            sys.exit(0)
        else:
            sys.exit(1)
    
    # Import is done here to ensure fast startup for help message
    from swimlane.engine import SwimlanesEngine, SwmlError
    
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
