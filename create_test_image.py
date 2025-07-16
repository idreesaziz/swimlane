from PIL import Image, ImageDraw, ImageFont
import os
import sys

def create_test_image(output_path='examples/media/BeHappy.png'):
    """Create a test image with orientation indicators"""
    # Create a white image
    width, height = 600, 400
    img = Image.new('RGBA', (width, height), color=(255, 255, 255, 0))

    # Create a drawing context
    draw = ImageDraw.Draw(img)

    # Draw a rectangle with a border
    draw.rectangle([(20, 20), (width-20, height-20)], fill=(100, 200, 255, 200), outline=(0, 0, 0, 255), width=3)

    # Draw directional arrows
    # Top arrow
    draw.polygon([(width//2, 30), (width//2-20, 70), (width//2+20, 70)], fill=(255, 0, 0, 255))
    # Right arrow
    draw.polygon([(width-30, height//2), (width-70, height//2-20), (width-70, height//2+20)], fill=(0, 255, 0, 255))

    # Add text to show orientation
    try:
        # Try to load a font - fall back to default if not available
        font = ImageFont.truetype("arial.ttf", 36)
    except IOError:
        font = ImageFont.load_default()

    draw.text((width//2, height//2), "Be Happy", font=font, fill=(0, 0, 0, 255), anchor="mm")
    draw.text((width//2, height//2 + 50), "THIS WAY UP", font=font, fill=(255, 0, 0, 255), anchor="mm")

    # Add diagonal line to clearly show rotation
    draw.line([(30, 30), (width-30, height-30)], fill=(0, 0, 255, 255), width=5)

    # Create directory if it doesn't exist
    dirname = os.path.dirname(output_path)
    if dirname:  # Only create directories if there's a path
        os.makedirs(dirname, exist_ok=True)
    
    # Save the image
    img.save(output_path)
    print(f"Test image created at: {output_path}")
    return output_path

def main():
    """Command line entry point"""
    output_path = 'examples/media/BeHappy.png'
    
    # Check for command line argument
    if len(sys.argv) > 1:
        # Check if help flag is set
        if sys.argv[1] in ['-h', '--help']:
            print("Usage: swimlane-test-image [output_path]")
            print("\nCreates a test image with orientation indicators.")
            print("\nArguments:")
            print("  output_path    Path to save the generated image (default: examples/media/BeHappy.png)")
            sys.exit(0)
        else:
            output_path = sys.argv[1]
    
    create_test_image(output_path)

if __name__ == "__main__":
    main()
