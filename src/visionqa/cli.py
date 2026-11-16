"""CLI: visionqa describe|ask|objects|benchmark|list-scenes|make-scenes"""
from __future__ import annotations

import argparse
import json
import os
import sys

from .analysis import analyze, describe
from .benchmark import run_benchmark, write_report
from .image import load_image
from .qa import ask
from .scenes import SCENES, make_scenes


def _assets_dir() -> str:
    return os.path.join(os.path.dirname(__file__), "..", "..", "assets")


def _load(path: str):
    return analyze(load_image(path))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="visionqa",
                                description="Ask questions about an image — grounded in its pixels.")
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("describe", help="Describe an image")
    d.add_argument("image")
    d.add_argument("--json", action="store_true")

    a = sub.add_parser("ask", help="Ask a question about an image")
    a.add_argument("image")
    a.add_argument("question")
    a.add_argument("--json", action="store_true")

    o = sub.add_parser("objects", help="List detected objects in an image")
    o.add_argument("image")
    o.add_argument("--json", action="store_true")

    b = sub.add_parser("benchmark", help="Run the fixed Q&A benchmark")
    b.add_argument("--assets", default=None)
    b.add_argument("--out", default=None)

    sub.add_parser("list-scenes", help="List the bundled scenes")
    m = sub.add_parser("make-scenes", help="Regenerate the bundled scene PNGs")
    m.add_argument("--out", default=None)

    args = p.parse_args(argv)
    if args.cmd == "describe":
        an = _load(args.image)
        if args.json:
            print(json.dumps({"image": args.image, "width": an.width, "height": an.height,
                              "dominant_color": an.dominant_color,
                              "color_counts": an.color_counts,
                              "objects": [o.evidence() for o in an.objects]}, indent=2))
        else:
            print(describe(an))
        return 0
    if args.cmd == "ask":
        an = _load(args.image)
        ans = ask(an, args.question)
        if args.json:
            print(json.dumps({"question": args.question, "answer": ans.text,
                              "evidence": ans.evidence, "grounded": ans.grounded}, indent=2))
        else:
            print(ans.text)
            for e in ans.evidence:
                print(f"  evidence: {e}")
        return 0
    if args.cmd == "objects":
        an = _load(args.image)
        if args.json:
            print(json.dumps([{"color": o.color, "shape": o.shape, "area": o.area,
                               "bbox": o.bbox, "centroid": o.centroid,
                               "position": o.position} for o in an.objects], indent=2))
        else:
            for o in an.objects:
                print(o.evidence())
        return 0
    if args.cmd == "benchmark":
        assets = args.assets or os.path.abspath(_assets_dir())
        # The bundled scenes are deterministic build artefacts: regenerate
        # any that are missing (e.g. a fresh clone) instead of failing.
        missing = [n for n in SCENES
                   if not os.path.exists(os.path.join(assets, f"scene_{n}.png"))]
        if missing:
            make_scenes(assets)
        result = run_benchmark(assets)
        out = args.out or os.path.join(os.getcwd(), "outputs")
        write_report(result, out)
        print(f"{result.passed}/{result.total} correct "
              f"({result.accuracy * 100:.1f}% accuracy)")
        for r in result.rows:
            print(f"  [{'pass' if r['passed'] else 'FAIL'}] {r['scene']}: {r['question']}")
        return 0 if result.passed == result.total else 1
    if args.cmd == "list-scenes":
        for name, (desc, _b) in SCENES.items():
            print(f"{name}: {desc}")
        return 0
    if args.cmd == "make-scenes":
        out = args.out or os.path.abspath(_assets_dir())
        paths = make_scenes(out)
        for name, path in paths.items():
            print(f"{name}: {path}")
        return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
