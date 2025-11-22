#!/usr/bin/env python3
"""
VectLLM Runner Script
Simple script to run VectLLM without installation

Usage:
    python run.py init
    python run.py train input.txt -o model.ckpt
    python run.py stats -c model.ckpt
    python run.py chat -c model.ckpt
"""

import sys
from pathlib import Path

# Add package to path
package_dir = Path(__file__).parent / "A.D.A.M — Adaptive and Dynamic Agent Module"
sys.path.insert(0, str(package_dir))

# Now import and run
from cli.vectllm import main

if __name__ == "__main__":
    sys.exit(main())
