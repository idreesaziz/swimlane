#!/usr/bin/env python3
"""
Test script for the video preprocessing functionality
"""

import sys
import os

# Add the current directory to the path
sys.path.insert(0, os.path.dirname(__file__))

from swimlane.engine import SwimlaneEngine

def main():
    if len(sys.argv) < 2:
        print("Usage: python test_preprocessing.py <swml_file>")
        print("Example: python test_preprocessing.py examples/composition.swml")
        sys.exit(1)
    
    swml_path = sys.argv[1]
    output_path = "test_output.mp4"  # Dummy output path
    
    try:
        engine = SwimlaneEngine(swml_path, output_path)
        engine.dry_run_preprocessing()
        print("\n✓ Preprocessing test completed successfully!")
    except Exception as e:
        print(f"\n✗ Error during preprocessing test: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
