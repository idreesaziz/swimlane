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
    threads = None
    
    i = 1
    while i < len(sys.argv):
        arg = sys.argv[i]
        if arg == "--preview":
            preview_mode = True
        elif arg == "--threads":
            if i + 1 < len(sys.argv):
                threads_arg = sys.argv[i + 1]
                if threads_arg.lower() == "auto":
                    import multiprocessing
                    threads = multiprocessing.cpu_count()
                else:
                    try:
                        threads = int(threads_arg)
                        if threads <= 0:
                            print("ERROR: Thread count must be a positive integer or 'auto'", file=sys.stderr)
                            sys.exit(1)
                    except ValueError:
                        print("ERROR: Thread count must be a positive integer or 'auto'", file=sys.stderr)
                        sys.exit(1)
                i += 1  # Skip the next argument (thread count)
            else:
                print("ERROR: --threads requires a value (number or 'auto')", file=sys.stderr)
                sys.exit(1)
        elif arg in ["--help", "-h"]:
            print("Swimlane Engine - SWML Video Renderer")
            print("Usage: swimlane [options] <input.swml> <output.mp4> [path/to/blender]")
            print("\nArguments:")
            print("  input.swml       Path to the SWML (Swimlane Markup Language) file")
            print("  output.mp4       Path for the output video file (can be .mp4, .mov, or .webm)")
            print("  path/to/blender  Optional path to the Blender executable (default: 'blender')")
            print("\nOptions:")
            print("  --preview        Use fast/low quality render settings for quick previews")
            print("  --threads N      Number of CPU threads to use (default: 1)")
            print("                   Use 'auto' to automatically detect and use all CPU cores")
            print("\nExamples:")
            print("  swimlane input.swml output.mp4")
            print("  swimlane --preview input.swml output.mp4")
            print("  swimlane --threads auto input.swml output.mp4")
            print("  swimlane --threads 8 input.swml output.mp4")
            sys.exit(0)
        else:
            args.append(arg)
        i += 1
    
    # Handle command-line arguments
    if len(args) < 2 or len(args) > 3:
        print("Swimlane Engine - SWML Video Renderer")
        print("Usage: swimlane [options] <input.swml> <output.mp4> [path/to/blender]")
        print("\nArguments:")
        print("  input.swml       Path to the SWML (Swimlane Markup Language) file")
        print("  output.mp4       Path for the output video file (can be .mp4, .mov, or .webm)")
        print("  path/to/blender  Optional path to the Blender executable (default: 'blender')")
        print("\nOptions:")
        print("  --preview        Use fast/low quality render settings for quick previews")
        print("  --threads N      Number of CPU threads to use (default: 1)")
        print("                   Use 'auto' to automatically detect and use all CPU cores")
        print("\nExamples:")
        print("  swimlane input.swml output.mp4")
        print("  swimlane --preview input.swml output.mp4")
        print("  swimlane --threads auto input.swml output.mp4")
        print("  swimlane --threads 8 input.swml output.mp4")
        sys.exit(1)
    
    # Import is done here to ensure fast startup for help message
    from swimlane.engine import SwimlaneEngine, SwmlError
    
    swml_path = args[0]
    output_path = args[1]
    blender_exec = args[2] if len(args) == 3 else 'blender'
    
    try:
        engine = SwimlaneEngine(swml_path, output_path, blender_executable=blender_exec, preview_mode=preview_mode, threads=threads)
        engine.render()
    except SwmlError as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nRendering cancelled by user", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
