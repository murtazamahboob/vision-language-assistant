"""Grounding: turn pixels into named colours, objects and regions.

Everything here is measured from the image itself — no model, no
guessing. An object is a connected region of one named colour (other
than the white background); its evidence is its bounding box, area,
centroid and 3x3 grid position, so every answer can point at it.
"""
from __future__ import annotations

import math
from collections import Counter, deque
from dataclasses import dataclass, field

from .image import Image

# Named palette: name -> prototype RGB. Nearest prototype wins.
NAMED_COLORS = {
    "red": (220, 30, 30),
    "blue": (30, 80, 220),
    "green": (30, 170, 60),
    "yellow": (240, 210, 30),
    "orange": (240, 130, 20),
    "purple": (140, 60, 190),
    "pink": (240, 130, 180),
    "cyan": (30, 200, 220),
    "brown": (140, 80, 30),
    "black": (20, 20, 20),
    "white": (250, 250, 250),
    "gray": (140, 140, 140),
}

GRID_ROWS = ("top", "middle", "bottom")
GRID_COLS = ("left", "center", "right")


def nearest_color(rgb: tuple) -> str:
    best, best_d = None, float("inf")
    for name, proto in NAMED_COLORS.items():
        d = sum((a - b) ** 2 for a, b in zip(rgb, proto))
        if d < best_d:
            best, best_d = name, d
    return best


def luminance(rgb: tuple) -> float:
    return 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]


def grid_position(cx: float, cy: float, width: int, height: int) -> str:
    row = GRID_ROWS[min(2, int(cy / height * 3))]
    col = GRID_COLS[min(2, int(cx / width * 3))]
    return f"{row}-{col}"


@dataclass
class VisualObject:
    color: str
    area: int
    bbox: tuple  # (x, y, w, h)
    centroid: tuple  # (cx, cy) floats
    shape: str
    fill_ratio: float
    position: str

    @property
    def x(self): return self.bbox[0]

    @property
    def y(self): return self.bbox[1]

    @property
    def w(self): return self.bbox[2]

    @property
    def h(self): return self.bbox[3]

    def evidence(self) -> str:
        return (f"{self.color} {self.shape} at {self.position}, "
                f"bbox x={self.x} y={self.y} w={self.w} h={self.h}, "
                f"area {self.area} px")


def _classify_shape(w: int, h: int, area: int) -> tuple:
    fill = area / max(1, w * h)
    aspect = w / max(1, h)
    if fill >= 0.97:
        return "rectangle", fill
    if 0.65 <= fill <= 0.90 and 0.8 <= aspect <= 1.25:
        return "circle", fill
    if 0.40 <= fill <= 0.62:
        return "triangle", fill
    if fill >= 0.90:
        return "rectangle", fill
    return "blob", fill


def find_objects(image: Image, min_area: int = 15) -> list:
    """Connected components of one named colour, white background excluded."""
    w, h = image.width, image.height
    labels = [[nearest_color(image.pixel(x, y)) for x in range(w)] for y in range(h)]
    seen = [[False] * w for _ in range(h)]
    objects = []
    for y in range(h):
        for x in range(w):
            if seen[y][x] or labels[y][x] == "white":
                continue
            color = labels[y][x]
            queue = deque([(x, y)])
            seen[y][x] = True
            pts = []
            while queue:
                cx, cy = queue.popleft()
                pts.append((cx, cy))
                for nx, ny in ((cx + 1, cy), (cx - 1, cy), (cx, cy + 1), (cx, cy - 1)):
                    if 0 <= nx < w and 0 <= ny < h and not seen[ny][nx] and labels[ny][nx] == color:
                        seen[ny][nx] = True
                        queue.append((nx, ny))
            if len(pts) < min_area:
                continue
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            bx, by = min(xs), min(ys)
            bw, bh = max(xs) - bx + 1, max(ys) - by + 1
            cx = sum(xs) / len(pts)
            cy = sum(ys) / len(pts)
            shape, fill = _classify_shape(bw, bh, len(pts))
            objects.append(VisualObject(
                color=color, area=len(pts), bbox=(bx, by, bw, bh),
                centroid=(cx, cy), shape=shape, fill_ratio=round(fill, 3),
                position=grid_position(cx, cy, w, h)))
    objects.sort(key=lambda o: -o.area)
    return objects


@dataclass
class Analysis:
    width: int
    height: int
    mean_rgb: tuple
    brightness: float
    color_counts: dict  # named colour -> pixel count
    objects: list = field(default_factory=list)
    grid: list = field(default_factory=list)  # 9 cells: (position, dominant colour, brightness)

    @property
    def dominant_color(self) -> str:
        counts = {c: n for c, n in self.color_counts.items() if c != "white"}
        if not counts:
            return "white"
        return max(counts, key=counts.get)

    @property
    def total_pixels(self) -> int:
        return self.width * self.height

    def objects_of_color(self, color: str) -> list:
        return [o for o in self.objects if o.color == color]

    def objects_of_shape(self, shape: str) -> list:
        return [o for o in self.objects if o.shape == shape]


def analyze(image: Image) -> Analysis:
    counts = Counter()
    total = [0, 0, 0]
    for rgb in image.pixels:
        counts[nearest_color(rgb)] += 1
        total[0] += rgb[0]
        total[1] += rgb[1]
        total[2] += rgb[2]
    n = max(1, len(image.pixels))
    mean_rgb = tuple(round(t / n, 1) for t in total)
    grid = []
    cw, ch = image.width // 3, image.height // 3
    for r in range(3):
        for c in range(3):
            cell = Counter()
            lum = []
            for y in range(r * ch, (r + 1) * ch if r < 2 else image.height):
                for x in range(c * cw, (c + 1) * cw if c < 2 else image.width):
                    rgb = image.pixel(x, y)
                    cell[nearest_color(rgb)] += 1
                    lum.append(luminance(rgb))
            pos = f"{GRID_ROWS[r]}-{GRID_COLS[c]}"
            grid.append((pos, cell.most_common(1)[0][0] if cell else "white",
                         round(sum(lum) / max(1, len(lum)), 1)))
    return Analysis(
        width=image.width, height=image.height, mean_rgb=mean_rgb,
        brightness=round(luminance(mean_rgb), 1),
        color_counts=dict(counts), objects=find_objects(image), grid=grid)


def describe(analysis: Analysis) -> str:
    parts = [f"A {analysis.width}x{analysis.height} image, "
             f"mean brightness {analysis.brightness}, "
             f"dominant colour {analysis.dominant_color}."]
    colors = sorted((c for c in analysis.color_counts if c != "white"),
                    key=lambda c: -analysis.color_counts[c])
    if colors:
        parts.append("Colours present: " + ", ".join(
            f"{c} ({analysis.color_counts[c]} px)" for c in colors) + ".")
    if not analysis.objects:
        parts.append("No distinct objects on the white background.")
    else:
        parts.append(f"{len(analysis.objects)} object(s): " + "; ".join(
            o.evidence() for o in analysis.objects) + ".")
    return " ".join(parts)
