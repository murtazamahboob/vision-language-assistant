import os

import pytest

from visionqa.image import (
    Image, fill_circle, fill_polygon, fill_rect, load_image, load_png,
    load_ppm, save_png,
)
from visionqa.scenes import make_scenes

ASSETS = os.path.join(os.path.dirname(__file__), "..", "assets")


def test_png_roundtrip(tmp_path):
    img = Image.blank(20, 10, (255, 255, 255))
    fill_rect(img, 2, 2, 5, 4, (220, 30, 30))
    fill_circle(img, 15, 5, 3, (30, 80, 220))
    path = str(tmp_path / "t.png")
    save_png(img, path)
    back = load_png(path)
    assert (back.width, back.height) == (20, 10)
    assert back.pixel(3, 3) == (220, 30, 30)
    assert back.pixel(15, 5) == (30, 80, 220)
    assert back.pixel(0, 0) == (255, 255, 255)
    assert back.pixels == img.pixels


def test_png_decoder_handles_rgba(tmp_path):
    # Encode an RGBA PNG by hand (filter 0) and check the decoder drops alpha.
    import struct, zlib
    raw = b"".join(b"\x00" + bytes([255, 0, 0, 128, 0, 255, 0, 255])
                    for _ in range(2))

    def chunk(ctype, payload):
        return (struct.pack(">I", len(payload)) + ctype + payload +
                struct.pack(">I", zlib.crc32(ctype + payload) & 0xFFFFFFFF))

    path = str(tmp_path / "rgba.png")
    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n")
        f.write(chunk(b"IHDR", struct.pack(">IIBBBBB", 2, 2, 8, 6, 0, 0, 0)))
        f.write(chunk(b"IDAT", zlib.compress(raw)))
        f.write(chunk(b"IEND", b""))
    img = load_png(path)
    assert img.pixel(0, 0) == (255, 0, 0)
    assert img.pixel(1, 0) == (0, 255, 0)


def test_load_png_rejects_non_png(tmp_path):
    path = str(tmp_path / "x.png")
    open(path, "wb").write(b"not a png")
    with pytest.raises(ValueError):
        load_png(path)


def test_ppm_roundtrip(tmp_path):
    path = str(tmp_path / "t.ppm")
    with open(path, "wb") as f:
        f.write(b"P6\n2 1\n255\n" + bytes([255, 0, 0, 0, 0, 255]))
    img = load_ppm(path)
    assert (img.width, img.height) == (2, 1)
    assert img.pixel(0, 0) == (255, 0, 0)
    assert img.pixel(1, 0) == (0, 0, 255)


def test_load_image_sniffs_format(tmp_path):
    img = Image.blank(4, 4, (10, 20, 30))
    path = str(tmp_path / "noext")
    save_png(img, path)
    assert load_image(path).pixel(0, 0) == (10, 20, 30)
    bad = str(tmp_path / "bad.bin")
    open(bad, "wb").write(b"junkjunk")
    with pytest.raises(ValueError):
        load_image(bad)


def test_fill_polygon_triangle_area():
    img = Image.blank(20, 20)
    fill_polygon(img, [(5, 5), (15, 5), (10, 15)], (0, 0, 0))
    black = sum(1 for p in img.pixels if p == (0, 0, 0))
    # Triangle area ~50 px; allow rasterisation slack.
    assert 40 <= black <= 65


def test_scenes_generate_and_load(tmp_path):
    paths = make_scenes(str(tmp_path))
    assert set(paths) == {"shapes", "blocks", "bars"}
    for path in paths.values():
        img = load_png(path)
        assert img.width > 0 and img.height > 0
        assert len(img.pixels) == img.width * img.height
