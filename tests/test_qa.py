import os

import pytest

from visionqa.analysis import analyze
from visionqa.image import load_image
from visionqa.qa import ask
from visionqa.scenes import SCENES, make_scenes


@pytest.fixture(scope="module")
def analyses(tmp_path_factory):
    d = str(tmp_path_factory.mktemp("assets"))
    make_scenes(d)
    return {name: analyze(load_image(os.path.join(d, f"scene_{name}.png")))
            for name in SCENES}


def test_describe(analyses):
    ans = ask(analyses["shapes"], "Describe the image")
    assert "red circle" in ans.text
    assert ans.evidence  # grounded


def test_size(analyses):
    ans = ask(analyses["shapes"], "How big is the image?")
    assert "120 pixels wide" in ans.text and "90 pixels tall" in ans.text


def test_colours_and_dominant(analyses):
    ans = ask(analyses["blocks"], "What colours are in the image?")
    assert "orange" in ans.text and "purple" in ans.text and "cyan" in ans.text
    ans = ask(analyses["blocks"], "What is the dominant colour?")
    assert "orange" in ans.text


def test_count_objects(analyses):
    assert "4 distinct" in ask(analyses["shapes"], "How many objects are there?").text
    assert "3 distinct" in ask(analyses["blocks"], "How many objects are there?").text


def test_count_colour(analyses):
    ans = ask(analyses["shapes"], "How many red objects are there?")
    assert "1 red" in ans.text
    ans = ask(analyses["shapes"], "How many pink objects are there?")
    assert "no pink" in ans.text


def test_where_and_is_there(analyses):
    ans = ask(analyses["shapes"], "Where is the red circle?")
    assert "top-left" in ans.text and "bbox" in ans.text
    assert ask(analyses["shapes"], "Is there a green triangle?").text.startswith("Yes")
    assert ask(analyses["shapes"], "Is there a pink circle?").text.startswith("No")


def test_position_yes_no(analyses):
    assert ask(analyses["shapes"], "Is the blue rectangle on the right?").text.startswith("Yes")
    assert ask(analyses["shapes"], "Is the blue rectangle on the left?").text.startswith("No")
    assert ask(analyses["blocks"], "Is the purple circle on the top?").text.startswith("Yes")


def test_region_question(analyses):
    ans = ask(analyses["shapes"], "What is in the top-right region?")
    assert "green triangle" in ans.text
    ans = ask(analyses["blocks"], "What is in the middle-left region?")
    assert "orange" in ans.text
    ans = ask(analyses["blocks"], "What is in the bottom-left region?")
    assert "no object" in ans.text  # orange block's centroid is middle-left


def test_brightest_darkest(analyses):
    ans = ask(analyses["shapes"], "What is the brightest region?")
    assert "brightest region is" in ans.text
    ans = ask(analyses["bars"], "What is the darkest region?")
    assert "darkest region is" in ans.text


def test_comparison(analyses):
    ans = ask(analyses["bars"], "Which is taller, the red bar or the blue bar?")
    assert "blue" in ans.text and "60 px" in ans.text
    ans = ask(analyses["shapes"], "Which object is biggest?")
    assert "biggest object is the blue rectangle" in ans.text  # 780 px, largest area


def test_fallback_never_guesses(analyses):
    ans = ask(analyses["shapes"], "What is the cat's name?")
    assert "only answer from what I measured" in ans.text
    assert ans.grounded is False
    assert "red circle" in ans.text  # still gives the grounded description
