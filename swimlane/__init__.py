"""
Swimlane Engine - A declarative video rendering engine using Blender's VSE
Parses SWML (Swimlane Markup Language) files and generates videos
"""

from .engine import SwimlaneEngine, SwmlError, main

__version__ = "0.1.0"

# Command-line entry point
def cli_main():
    """Entry point for the console script."""
    import sys
    from .engine import main
    main()

if __name__ == "__main__":
    cli_main()
