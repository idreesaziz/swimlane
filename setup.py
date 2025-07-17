from setuptools import setup, find_packages
import os
import io

# Read the contents of your README file
try:
    with io.open('README.md', encoding='utf-8') as f:
        long_description = f.read()
except UnicodeDecodeError:
    # Fallback if the README.md has encoding issues
    long_description = "A declarative video rendering engine powered by Blender's Video Sequence Editor (VSE)"

# Read requirements from requirements.txt
try:
    with io.open('requirements.txt', 'r', encoding='utf-8') as f:
        requirements = f.read().splitlines()
except (FileNotFoundError, UnicodeDecodeError):
    # Fallback if requirements.txt is missing or has encoding issues
    requirements = ['ffmpeg-python>=0.2.0', 'pillow>=9.0.0']

setup(
    name="swimlane",
    version="0.1.1",
    author="idreesaziz",
    author_email="your.email@example.com",  # Replace with your actual email
    description="A declarative video rendering engine powered by Blender's Video Sequence Editor (VSE)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/idreesaziz/swimlane",
    packages=find_packages(),
    py_modules=["create_test_image"],  # Include root Python modules
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Multimedia :: Video",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.6",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "swimlane=swimlane.cli:main",  # Command line tool
            "swimlane-test-image=create_test_image:main",  # Test image generator
        ],
    },
    include_package_data=True,
    package_data={
        "swimlane": ["examples/*.swml", "examples/media/*"],
    },
)
