"""Bundled scenes: synthetic images with known ground truth.

Scenes are drawn with solid colours on a white background, objects
never touching, so the analyser's answers can be checked exactly.
"""
from __future__ import annotations

import os

from .image import Image, fill_circle, fill_polygon, fill_rect, save_png

# name -> (description, builder)
SCENES = {}


def scene_shapes() -> Image:
    """Red circle top-left, green triangle top-right, yellow circle
    centre, blue rectangle bottom-right."""
    img = Image.blank(120, 90)
    fill_circle(img, 30, 25, 14, (220, 30, 30))          # red circle
    fill_polygon(img, [(80, 8), (108, 8), (94, 32)], (30, 170, 60))  # green triangle
    fill_circle(img, 60, 62, 8, (240, 210, 30))          # yellow circle
    fill_rect(img, 82, 52, 30, 26, (30, 80, 220))        # blue rectangle
    return img


def scene_blocks() -> Image:
    """Orange block left, purple circle top-right, cyan block bottom-right."""
    img = Image.blank(100, 80)
    fill_rect(img, 8, 10, 30, 55, (240, 130, 20))        # orange rectangle
    fill_circle(img, 75, 20, 12, (140, 60, 190))         # purple circle
    fill_rect(img, 62, 48, 28, 20, (30, 200, 220))       # cyan rectangle
    return img


def scene_bars() -> Image:
    """Bar chart: short red bar, tall blue bar, medium green bar."""
    img = Image.blank(120, 90)
    fill_rect(img, 10, 80, 110, 2, (20, 20, 20))         # black baseline
    fill_rect(img, 18, 60, 22, 20, (220, 30, 30))        # red bar, height 20
    fill_rect(img, 50, 20, 22, 60, (30, 80, 220))        # blue bar, height 60
    fill_rect(img, 82, 40, 22, 40, (30, 170, 60))        # green bar, height 40
    return img


SCENES = {
    "shapes": ("Four coloured shapes on white: red circle, green triangle, "
               "yellow circle, blue rectangle.", scene_shapes),
    "blocks": ("Three coloured blocks: orange rectangle, purple circle, "
               "cyan rectangle.", scene_blocks),
    "bars": ("A three-bar chart: red (short), blue (tall), green (medium).",
             scene_bars),
}


def make_scenes(out_dir: str) -> dict:
    """Generate every bundled scene PNG; returns {name: path}."""
    os.makedirs(out_dir, exist_ok=True)
    paths = {}
    for name, (_desc, builder) in SCENES.items():
        path = os.path.join(out_dir, f"scene_{name}.png")
        save_png(builder(), path)
        paths[name] = path
    return paths
