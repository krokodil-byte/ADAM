#!/usr/bin/env python3
"""
VectLLM - Continuous Self-Training Language Model
Setup script for installation
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README if exists
readme_path = Path(__file__).parent / "README.md"
long_description = ""
if readme_path.exists():
    long_description = readme_path.read_text(encoding="utf-8")

setup(
    name="vectllm",
    version="0.1.0",
    author="VectLLM Team",
    description="Continuous Self-Training Language Model with Dynamic Vocabulary",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/krokodil-byte/A.D.A.M-Adaptive-and-Dynamic-Agent-Module",

    # Package configuration
    packages=find_packages(where="A.D.A.M — Adaptive and Dynamic Agent Module"),
    package_dir={"": "A.D.A.M — Adaptive and Dynamic Agent Module"},

    # Dependencies
    install_requires=[
        "numpy>=1.20.0",
    ],

    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "pytest-mock>=3.10.0",
        ],
    },

    # Entry points
    entry_points={
        "console_scripts": [
            "vectllm=cli.vectllm:main",
        ],
    },

    # Metadata
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],

    # Include additional files
    include_package_data=True,
    package_data={
        "": ["*.cu", "*.cuh"],  # Include CUDA kernel files
    },
)
