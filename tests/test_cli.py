import json
import os
import subprocess
import sys

import pytest

from visionqa.cli import main
from visionqa.scenes import make_scenes


@pytest.fixture(scope="module")
def assets(tmp_path_factory):
    d = str(tmp_path_factory.mktemp("assets"))
    make_scenes(d)
    return d


def test_cli_describe(assets, capsys):
    assert main(["describe", os.path.join(assets, "scene_shapes.png")]) == 0
    assert "red circle" in capsys.readouterr().out


def test_cli_ask_json(assets, capsys):
    assert main(["ask", os.path.join(assets, "scene_shapes.png"),
                 "Where is the red circle?", "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert "top-left" in out["answer"]
    assert out["evidence"]


def test_cli_objects_json(assets, capsys):
    assert main(["objects", os.path.join(assets, "scene_blocks.png"), "--json"]) == 0
    objs = json.loads(capsys.readouterr().out)
    assert len(objs) == 3
    assert {o["color"] for o in objs} == {"orange", "purple", "cyan"}


def test_cli_list_scenes(capsys):
    assert main(["list-scenes"]) == 0
    assert "shapes" in capsys.readouterr().out


def test_cli_benchmark(assets, tmp_path):
    assert main(["benchmark", "--assets", assets, "--out", str(tmp_path)]) == 0
    assert os.path.exists(os.path.join(str(tmp_path), "summary.json"))


def test_cli_benchmark_regenerates_missing_scenes(tmp_path):
    """A fresh clone has no assets/ yet: benchmark regenerates the scenes."""
    assets_dir = str(tmp_path / "assets")
    assert main(["benchmark", "--assets", assets_dir,
                 "--out", str(tmp_path / "out")]) == 0
    for name in ("shapes", "blocks", "bars"):
        assert os.path.exists(os.path.join(assets_dir, f"scene_{name}.png"))


def test_installed_entry_point_smoke(tmp_path):
    """The `visionqa` console script works after `pip install -e .`."""
    result = subprocess.run(
        [sys.executable, "-m", "visionqa.cli", "list-scenes"],
        capture_output=True, text=True, cwd=str(tmp_path))
    assert result.returncode == 0
    assert "shapes" in result.stdout
