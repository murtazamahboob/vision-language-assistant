import os

import pytest

from visionqa.analysis import analyze, describe, grid_position, nearest_color
from visionqa.image import Image, fill_circle, fill_rect, load_image
from visionqa.scenes import SCENES, make_scenes

ASSETS = os.path.join(os.path.dirname(__file__), "..", "assets")


@pytest.fixture(scope="module")
def analyses(tmp_path_factory):
    d = str(tmp_path_factory.mktemp("assets"))
    make_scenes(d)
    return {name: analyze(load_image(os.path.join(d, f"scene_{name}.png")))
            for name in SCENES}


def test_nearest_color():
    assert nearest_color((220, 30, 30)) == "red"
    assert nearest_color((30, 80, 220)) == "blue"
    assert nearest_color((250, 250, 250)) == "white"
    assert nearest_color((20, 20, 20)) == "black"


def test_grid_position():
    assert grid_position(5, 5, 120, 90) == "top-left"
    assert grid_position(110, 80, 120, 90) == "bottom-right"
    assert grid_position(60, 45, 120, 90) == "middle-center"


def test_shapes_objects(analyses):
    an = analyses["shapes"]
    assert (an.width, an.height) == (120, 90)
    assert len(an.objects) == 4
    by_color = {o.color: o for o in an.objects}
    assert by_color["red"].shape == "circle"
    assert by_color["red"].position == "top-left"
    assert by_color["green"].shape == "triangle"
    assert by_color["green"].position == "top-right"
    assert by_color["blue"].shape == "rectangle"
    assert by_color["blue"].position == "bottom-right"
    assert by_color["yellow"].shape == "circle"
    assert by_color["yellow"].position == "bottom-center"
    assert an.dominant_color == "blue"  # largest non-white colour by pixels (780 px)


def test_shapes_object_areas(analyses):
    an = analyses["shapes"]
    red = next(o for o in an.objects if o.color == "red")
    # Circle r=14: pi*r^2 ~= 616, rasterised.
    assert 550 <= red.area <= 700
    blue = next(o for o in an.objects if o.color == "blue")
    assert blue.area == 30 * 26
    assert blue.bbox == (82, 52, 30, 26)


def test_blocks_objects(analyses):
    an = analyses["blocks"]
    assert len(an.objects) == 3
    assert an.dominant_color == "orange"
    purple = next(o for o in an.objects if o.color == "purple")
    assert purple.shape == "circle" and purple.position == "top-right"


def test_bars_objects(analyses):
    an = analyses["bars"]
    # 3 bars + the black baseline.
    assert len(an.objects) == 4
    blue = next(o for o in an.objects if o.color == "blue")
    red = next(o for o in an.objects if o.color == "red")
    green = next(o for o in an.objects if o.color == "green")
    assert blue.h == 60 and red.h == 20 and green.h == 40
    assert blue.area > green.area > red.area


def test_blank_image_has_no_objects():
    an = analyze(Image.blank(10, 10))
    assert an.objects == []
    assert an.dominant_color == "white"


def test_describe_mentions_objects_and_colors(analyses):
    text = describe(analyses["shapes"])
    assert "120x90" in text
    assert "red circle" in text
    assert "blue rectangle" in text


def test_grid_cells_cover_image(analyses):
    an = analyses["shapes"]
    assert len(an.grid) == 9
    assert an.grid[0][0] == "top-left"
    assert an.grid[8][0] == "bottom-right"
    assert all(isinstance(lum, float) for _p, _c, lum in an.grid)
