#!/usr/bin/env python3
"""
Setup script for rapidcadpy - A Python library for CAD sequence processing and manipulation.
"""

import pathlib

from setuptools import find_packages, setup

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (
    (HERE / "README.md").read_text(encoding="utf-8")
    if (HERE / "README.md").exists()
    else "A Python library for CAD sequence processing and manipulation."
)

setup(
    name="RapidCAD-Py",
    version="0.1.0",
    description="A Python library for CAD sequence processing and manipulation",
    long_description=README,
    long_description_content_type="text/markdown",
    author="Your Name",
    author_email="your.email@example.com",
    url="https://github.com/yourusername/rapidcadpy",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        # Core dependencies
        "numpy>=1.20.0",
        "torch>=1.10.0",
        "torch-geometric>=2.0.0",
        # Plotting and visualization
        "plotly>=5.0.0",
        "matplotlib>=3.5.0",
        # Image processing
        "Pillow>=8.0.0",
        # Progress bars
        "tqdm>=4.60.0",
        # Other utilities
        "dataclasses; python_version<'3.7'",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "pytest-cov>=2.10.0",
            "black>=21.0.0",
            "flake8>=3.8.0",
            "mypy>=0.800",
        ],
        "docs_old": [
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=1.0.0",
            "myst-parser>=0.18.0",
            "sphinx-autodoc-typehints>=1.12.0",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Computer Aided Design (CAD)",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    keywords="cad, geometry, 3d, design, sequence, modeling",
    include_package_data=True,
    zip_safe=False,
    entry_points={
        "console_scripts": [
            # Add command-line scripts if needed
            # "rapidcadpy=rapidcadpy.cli:main",
        ],
    },
)
