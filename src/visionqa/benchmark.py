"""Benchmark: fixed questions with expected answers per scene.

A question passes when the expected phrase appears in the answer
(case-insensitive). Accuracy is reported per scene and overall.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass

from .analysis import analyze
from .image import load_image
from .qa import ask

# (scene, question, expected substring in the answer)
QUESTIONS = [
    ("shapes", "Describe the image", "red circle"),
    ("shapes", "What colours are in the image?", "blue"),
    ("shapes", "How many objects are there?", "4 distinct"),
    ("shapes", "How many red objects are there?", "1 red"),
    ("shapes", "Where is the red circle?", "top-left"),
    ("shapes", "Is there a green triangle?", "Yes"),
    ("shapes", "Is the blue rectangle on the right?", "Yes"),
    ("shapes", "What is in the top-right region?", "green triangle"),
    ("shapes", "How big is the image?", "120 pixels wide"),
    ("blocks", "How many objects are there?", "3 distinct"),
    ("blocks", "Where is the purple circle?", "top-right"),
    ("blocks", "Is there a cyan rectangle?", "Yes"),
    ("blocks", "What is the dominant colour?", "orange"),
    ("bars", "Which is taller, the red bar or the blue bar?", "blue"),
    ("bars", "How many objects are there?", "4 distinct"),
    ("bars", "What is in the bottom-left region?", "red"),
]


@dataclass
class BenchmarkResult:
    total: int
    passed: int
    rows: list  # dicts: scene, question, expected, answer, passed

    @property
    def accuracy(self) -> float:
        return round(self.passed / max(1, self.total), 3)


def run_benchmark(assets_dir: str) -> BenchmarkResult:
    rows = []
    for scene, question, expected in QUESTIONS:
        path = os.path.join(assets_dir, f"scene_{scene}.png")
        analysis = analyze(load_image(path))
        answer = ask(analysis, question)
        ok = expected.lower() in answer.text.lower()
        rows.append({"scene": scene, "question": question,
                     "expected": expected, "answer": answer.text,
                     "passed": ok, "evidence": answer.evidence})
    passed = sum(1 for r in rows if r["passed"])
    return BenchmarkResult(total=len(rows), passed=passed, rows=rows)


def write_report(result: BenchmarkResult, out_dir: str) -> tuple:
    os.makedirs(out_dir, exist_ok=True)
    lines = ["# Vision-Language Assistant — Benchmark\n",
             f"**{result.passed}/{result.total} questions correct "
             f"({result.accuracy * 100:.1f}% accuracy)** — every answer "
             "carries its grid/bounding-box evidence.\n",
             "| Scene | Question | Expected | Result |",
             "|---|---|---|---|"]
    for r in result.rows:
        lines.append(f"| {r['scene']} | {r['question']} | {r['expected']} | "
                     f"{'pass' if r['passed'] else 'FAIL'} |")
    lines.append("")
    for r in result.rows:
        lines.append(f"- **{r['scene']} — {r['question']}**\n  - Answer: {r['answer']}")
    report = "\n".join(lines) + "\n"
    with open(os.path.join(out_dir, "report.md"), "w") as f:
        f.write(report)
    summary = {"total": result.total, "passed": result.passed,
               "accuracy": result.accuracy,
               "rows": [{k: v for k, v in r.items() if k != "evidence"} for r in result.rows]}
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return report, summary
