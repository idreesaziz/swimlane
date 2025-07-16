"""
Test script to verify that the swimlane package is correctly installed.
"""
import sys
from swimlane import SwimlanesEngine, __version__

def main():
    print(f"Swimlane Engine version: {__version__}")
    print(f"Python version: {sys.version}")
    print("SwimlanesEngine is available in the swimlane package.")
    print("Installation is successful!")
    
    # Check if commandline arguments are provided to run the engine
    if len(sys.argv) >= 3:
        swml_path = sys.argv[1]
        output_path = sys.argv[2]
        blender_exec = sys.argv[3] if len(sys.argv) >= 4 else 'blender'
        
        print(f"\nAttempting to render {swml_path} to {output_path}")
        try:
            engine = SwimlanesEngine(swml_path, output_path, blender_executable=blender_exec)
            engine.parse_swml()  # Only parse, don't render
            print("SWML file parsed successfully.")
        except Exception as e:
            print(f"Error: {e}")
    
if __name__ == "__main__":
    main()
