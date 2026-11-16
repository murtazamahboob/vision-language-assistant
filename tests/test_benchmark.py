import os

import pytest

from visionqa.benchmark import QUESTIONS, run_benchmark, write_report
from visionqa.scenes import make_scenes


@pytest.fixture(scope="module")
def assets(tmp_path_factory):
    d = str(tmp_path_factory.mktemp("assets"))
    make_scenes(d)
    return d


def test_benchmark_all_pass(assets):
    result = run_benchmark(assets)
    assert result.total == len(QUESTIONS) == 16
    assert result.passed == result.total, [
        r for r in result.rows if not r["passed"]]
    assert result.accuracy == 1.0


def test_benchmark_report_written(assets, tmp_path):
    result = run_benchmark(assets)
    report, summary = write_report(result, str(tmp_path))
    assert "16/16" in report
    assert summary["accuracy"] == 1.0
    assert os.path.exists(os.path.join(str(tmp_path), "report.md"))
    assert os.path.exists(os.path.join(str(tmp_path), "summary.json"))
