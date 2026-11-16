"""Regenerate the bundled scene PNGs into assets/."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from visionqa import make_scenes

if __name__ == "__main__":
    out = os.path.join(os.path.dirname(__file__), "..", "assets")
    for name, path in make_scenes(out).items():
        print(f"{name}: {path}")
