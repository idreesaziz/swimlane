# SWML: The Swimlane Markup Language Engine

**swml** is a lightweight, JSON-based domain-specific language (DSL) and Python engine for describing and rendering layered, timeline-based video compositions.

## Features (v1.0)
- Declarative JSON format (`.swml`)
- Layer-based compositing with Z-ordering
- Resolution-independent positioning using a normalized coordinate system with anchors
- Flexible scaling and sizing of clips
- Support for video, image, and transparent sources (e.g., `.png`, `.webm`)
- Output to standard MP4 or transparency-enabled MOV/WebM formats

## Quick Start

**1. Prerequisites**
- Python 3.8+
- FFmpeg (must be installed and available in your system's PATH)

**2. Setup**
```bash
git clone https://github.com/idreesaziz/swimlane.git
cd swml
python -m venv venv
# Activate the virtual environment
# Windows: .\\venv\\Scripts\\activate
# macOS/Linux: source venv/bin/activate
pip install -r requirements.txt

**3. Run an Example**
Create your media files inside examples/media/. Then run the engine:

```bash
python engine.py examples/composition.swml output.mov