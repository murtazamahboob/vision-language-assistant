"""Image loading and saving, stdlib only.

PNG is decoded and encoded from scratch (zlib + struct, no Pillow):
colour types 2 (RGB) and 6 (RGBA), bit depth 8, no interlacing.
PPM (P6) is supported too, for easy hand-made test images.
"""
from __future__ import annotations

import struct
import zlib
from dataclasses import dataclass, field


@dataclass
class Image:
    width: int
    height: int
    pixels: list  # row-major list of (r, g, b) tuples
    path: str = ""

    def pixel(self, x: int, y: int) -> tuple:
        return self.pixels[y * self.width + x]

    def set_pixel(self, x: int, y: int, rgb: tuple) -> None:
        if 0 <= x < self.width and 0 <= y < self.height:
            self.pixels[y * self.width + x] = (int(rgb[0]), int(rgb[1]), int(rgb[2]))

    @classmethod
    def blank(cls, width: int, height: int, rgb=(255, 255, 255)) -> "Image":
        return cls(width, height, [tuple(rgb)] * (width * height))


def _paeth(a: int, b: int, c: int) -> int:
    p = a + b - c
    pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
    if pa <= pb and pa <= pc:
        return a
    if pb <= pc:
        return b
    return c


def _unfilter(raw: bytes, width: int, height: int, bpp: int) -> bytes:
    stride = width * bpp
    out = bytearray()
    prev = bytearray(stride)
    pos = 0
    for _ in range(height):
        ftype = raw[pos]
        pos += 1
        line = bytearray(raw[pos:pos + stride])
        pos += stride
        if ftype == 0:
            pass
        elif ftype == 1:
            for i in range(bpp, stride):
                line[i] = (line[i] + line[i - bpp]) & 0xFF
        elif ftype == 2:
            for i in range(stride):
                line[i] = (line[i] + prev[i]) & 0xFF
        elif ftype == 3:
            for i in range(stride):
                left = line[i - bpp] if i >= bpp else 0
                line[i] = (line[i] + ((left + prev[i]) // 2)) & 0xFF
        elif ftype == 4:
            for i in range(stride):
                a = line[i - bpp] if i >= bpp else 0
                b = prev[i]
                c = prev[i - bpp] if i >= bpp else 0
                line[i] = (line[i] + _paeth(a, b, c)) & 0xFF
        else:
            raise ValueError(f"unsupported PNG filter type {ftype}")
        out.extend(line)
        prev = line
    return bytes(out)


def load_png(path: str) -> Image:
    data = open(path, "rb").read()
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"not a PNG file: {path}")
    pos = 8
    width = height = bit_depth = color_type = interlace = None
    idat = bytearray()
    while pos < len(data):
        (length,) = struct.unpack(">I", data[pos:pos + 4])
        ctype = data[pos + 4:pos + 8]
        chunk = data[pos + 8:pos + 8 + length]
        if ctype == b"IHDR":
            width, height, bit_depth, color_type, _comp, _filt, interlace = struct.unpack(
                ">IIBBBBB", chunk)
        elif ctype == b"IDAT":
            idat.extend(chunk)
        elif ctype == b"IEND":
            break
        pos += 12 + length
    if width is None:
        raise ValueError("PNG has no IHDR chunk")
    if bit_depth != 8:
        raise ValueError(f"only 8-bit PNG supported, got {bit_depth}")
    if color_type not in (2, 6):
        raise ValueError(f"only RGB/RGBA PNG supported, got colour type {color_type}")
    if interlace != 0:
        raise ValueError("interlaced PNG not supported")
    channels = 3 if color_type == 2 else 4
    raw = _unfilter(zlib.decompress(bytes(idat)), width, height, channels)
    pixels = []
    stride = width * channels
    for y in range(height):
        row = raw[y * stride:(y + 1) * stride]
        for x in range(width):
            o = x * channels
            pixels.append((row[o], row[o + 1], row[o + 2]))
    return Image(width, height, pixels, path=path)


def save_png(image: Image, path: str) -> None:
    raw = bytearray()
    for y in range(image.height):
        raw.append(0)  # filter type 0: none
        for x in range(image.width):
            raw.extend(image.pixel(x, y))
    comp = zlib.compress(bytes(raw), 9)

    def chunk(ctype: bytes, payload: bytes) -> bytes:
        return (struct.pack(">I", len(payload)) + ctype + payload +
                struct.pack(">I", zlib.crc32(ctype + payload) & 0xFFFFFFFF))

    ihdr = struct.pack(">IIBBBBB", image.width, image.height, 8, 2, 0, 0, 0)
    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n")
        f.write(chunk(b"IHDR", ihdr))
        f.write(chunk(b"IDAT", comp))
        f.write(chunk(b"IEND", b""))


def load_ppm(path: str) -> Image:
    with open(path, "rb") as f:
        magic = f.readline().strip()
        if magic != b"P6":
            raise ValueError(f"not a P6 PPM file: {path}")
        # Skip comments, read width/height/maxval tokens.
        tokens = []
        while len(tokens) < 3:
            line = f.readline()
            if not line:
                break
            line = line.split(b"#")[0]
            tokens.extend(line.split())
        width, height, maxval = int(tokens[0]), int(tokens[1]), int(tokens[2])
        if maxval != 255:
            raise ValueError(f"only maxval 255 PPM supported, got {maxval}")
        data = f.read(width * height * 3)
    pixels = [(data[i], data[i + 1], data[i + 2])
              for i in range(0, len(data), 3)]
    return Image(width, height, pixels, path=path)


def load_image(path: str) -> Image:
    if path.lower().endswith(".png"):
        return load_png(path)
    if path.lower().endswith((".ppm", ".pgm")):
        return load_ppm(path)
    # Sniff the magic bytes instead of trusting the extension.
    with open(path, "rb") as f:
        head = f.read(8)
    if head[:8] == b"\x89PNG\r\n\x1a\n":
        return load_png(path)
    if head[:2] == b"P6":
        return load_ppm(path)
    raise ValueError(f"unsupported image format: {path}")


def fill_rect(image: Image, x: int, y: int, w: int, h: int, rgb: tuple) -> None:
    for yy in range(y, y + h):
        for xx in range(x, x + w):
            image.set_pixel(xx, yy, rgb)


def fill_circle(image: Image, cx: int, cy: int, r: int, rgb: tuple) -> None:
    for yy in range(cy - r, cy + r + 1):
        for xx in range(cx - r, cx + r + 1):
            if (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r:
                image.set_pixel(xx, yy, rgb)


def fill_polygon(image: Image, points: list, rgb: tuple) -> None:
    """Scanline polygon fill (even-odd rule), deterministic."""
    if len(points) < 3:
        return
    ys = [p[1] for p in points]
    for y in range(min(ys), max(ys) + 1):
        xs = []
        n = len(points)
        for i in range(n):
            x1, y1 = points[i]
            x2, y2 = points[(i + 1) % n]
            if (y1 <= y < y2) or (y2 <= y < y1):
                x = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
                xs.append(x)
        xs.sort()
        for i in range(0, len(xs) - 1, 2):
            for x in range(int(xs[i]), int(xs[i + 1]) + 1):
                image.set_pixel(x, y, rgb)
