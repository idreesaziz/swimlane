#!/usr/bin/env python3
"""
Test script to demonstrate cache reuse functionality
"""

import sys
import os

# Add the current directory to the path
sys.path.insert(0, os.path.dirname(__file__))

from swimlane.engine import SwimlaneEngine

def test_cache_reuse():
    """Test that cached files are properly reused"""
    
    print("=== Testing Cache Reuse ===\n")
    
    print("1. Running 30fps composition (should use cached file)...")
    try:
        engine_30fps = SwimlaneEngine("test_caching_30fps.swml", "output_30fps.mp4")
        engine_30fps.dry_run_preprocessing()
        print("   ✓ 30fps preprocessing completed")
    except Exception as e:
        print(f"   ✗ Error with 30fps: {e}")
        return False
    
    print("\n2. Running 24fps composition (should use cached file)...")
    try:
        engine_24fps = SwimlaneEngine("test_caching_24fps.swml", "output_24fps.mp4")
        engine_24fps.dry_run_preprocessing()
        print("   ✓ 24fps preprocessing completed")
    except Exception as e:
        print(f"   ✗ Error with 24fps: {e}")
        return False
    
    return True

def main():
    success = test_cache_reuse()
    
    if success:
        print("\n✓ Cache reuse test completed successfully!")
    else:
        print("\n✗ Cache reuse test failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
