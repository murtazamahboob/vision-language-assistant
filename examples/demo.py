"""Demo: regenerate scenes, describe each one, ask sample questions,
run the benchmark, and write outputs/report.md + outputs/summary.json.
Fully offline and deterministic.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from visionqa import analyze, ask, describe, load_image, make_scenes, run_benchmark
from visionqa.benchmark import write_report

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS = os.path.join(ROOT, "assets")
OUT = os.path.join(ROOT, "outputs")

SAMPLE_QUESTIONS = {
    "shapes": ["Describe the image",
               "Where is the red circle?",
               "Is there a green triangle?",
               "What is in the top-right region?"],
    "blocks": ["What colours are in the image?",
               "Where is the purple circle?"],
    "bars": ["Which is taller, the red bar or the blue bar?",
             "How many objects are there?"],
}


def main() -> None:
    make_scenes(ASSETS)
    for name in ("shapes", "blocks", "bars"):
        path = os.path.join(ASSETS, f"scene_{name}.png")
        analysis = analyze(load_image(path))
        print(f"\n== {name} ({path}) ==")
        print(describe(analysis))
        for q in SAMPLE_QUESTIONS[name]:
            ans = ask(analysis, q)
            print(f"  Q: {q}")
            print(f"  A: {ans.text}")
    result = run_benchmark(ASSETS)
    write_report(result, OUT)
    print(f"\nBenchmark: {result.passed}/{result.total} correct "
          f"({result.accuracy * 100:.1f}% accuracy)")
    print(f"Report: {os.path.join(OUT, 'report.md')}")


if __name__ == "__main__":
    main()
