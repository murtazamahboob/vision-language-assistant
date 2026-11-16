# Vision-Language Assistant — Ask Questions About an Image, Grounded in Pixels

**A multimodal vision-language assistant that answers questions about an image — colours, objects, positions, sizes — with bounding-box evidence for every answer. Fully offline, deterministic, stdlib-only, no model, no API key.**

Most "ask about an image" demos call a hosted vision model and return an answer you cannot check. This one does the opposite: it decodes the PNG itself, measures the pixels, finds the objects, and every answer points at the evidence — grid position, bounding box, area — so you can verify it against the image. If it cannot measure something, it says so instead of guessing.

## Results (measured, not claimed)

Benchmark: 16 fixed questions across 3 bundled scenes, each with a known expected answer:

| Scene | Question | Answer |
|---|---|---|
| shapes | Describe the image | 4 objects: blue rectangle, red circle, green triangle, yellow circle |
| shapes | Where is the red circle? | top-left, bbox x=16 y=11 w=29 h=29, area 613 px |
| shapes | Is there a green triangle? | Yes — top-right, area 374 px |
| shapes | What is in the top-right region? | a green triangle |
| shapes | How many red objects are there? | 1 |
| blocks | What is the dominant colour? | orange (1650 px) |
| blocks | Where is the purple circle? | top-right, area 441 px |
| bars | Which is taller, the red bar or the blue bar? | blue — height 60 px vs 20 px |
| bars | How many objects are there? | 4 (3 bars + black baseline) |

**Benchmark: 16/16 questions correct (100% accuracy)** — run it yourself with `visionqa benchmark`. Every answer carries its grid/bounding-box evidence; see `outputs/report.md`.

## Features

- **PNG codec built from scratch** — byte-level PNG decode and encode with `zlib` + `struct` only (RGB/RGBA, 8-bit, all five filter types incl. Paeth). No Pillow, no OpenCV. PPM (P6) supported too.
- **Grounded object detection** — connected-component regions of one named colour (12-colour palette, nearest-prototype matching), with bounding box, centroid, area, shape (circle / rectangle / triangle / blob from fill ratio) and 3×3 grid position.
- **Question answering** — describe, colours and dominant colour, object counts (total, per colour, per shape), where an object is, is-there-a checks, left/right/top/bottom and above/below, brightest/darkest region, what is in a region, bigger/taller comparisons. Unknown questions get a grounded description plus what *can* be asked — never a guessed answer.
- **Three bundled scenes with known ground truth** — `shapes` (4 coloured shapes), `blocks` (3 coloured blocks), `bars` (a 3-bar chart), generated deterministically by `scripts/make_scenes.py`. The scenes are build artefacts, not checked-in binaries: `visionqa benchmark` regenerates any missing scene automatically on first run, and `visionqa make-scenes` regenerates them any time.
- **CLI** — `describe`, `ask`, `objects`, `benchmark`, `list-scenes`, `make-scenes`, with `--json` output for scripting.

## Quickstart

```bash
pip install -r requirements.txt
pip install -e .

# Describe a bundled scene
visionqa describe assets/scene_shapes.png

# Ask a question (evidence printed under the answer)
visionqa ask assets/scene_shapes.png "Where is the red circle?"
visionqa ask assets/scene_bars.png "Which is taller, the red bar or the blue bar?" --json

# List the objects in an image
visionqa objects assets/scene_blocks.png

# Run the 16-question benchmark
visionqa benchmark

# Run the full demo (scenes + sample questions + benchmark)
python examples/demo.py
```

## Python API

```python
from visionqa import analyze, ask, load_image, run_benchmark

analysis = analyze(load_image("assets/scene_shapes.png"))
analysis.objects[0].evidence()  # 'blue rectangle at bottom-right, bbox x=82 y=52 w=30 h=26, area 780 px'

answer = ask(analysis, "Where is the red circle?")
answer.text       # 'The red circle is in the top-left of the image (...)'
answer.evidence   # ['red circle at top-left, bbox x=16 y=11 w=29 h=29, area 613 px']

result = run_benchmark("assets")
result.accuracy   # 1.0
```

## How it works

1. **Decode** — the PNG is decompressed and unfiltered scanline by scanline; each pixel is mapped to its nearest named colour.
2. **Detect** — flood-fill finds connected regions of one colour (white background excluded, min. area 15 px); shape comes from the region's fill ratio against its bounding box (circle ≈ 0.65–0.90, triangle ≈ 0.40–0.62, rectangle ≈ 1.0).
3. **Answer** — the question is matched to an intent (describe / count / where / is-there / compare / region / brightness); the answer is assembled from the measured objects only, with the bounding box and grid position as evidence.

## Project structure

```
src/visionqa/
  image.py      PNG/PPM decode + encode, drawing primitives (rect/circle/polygon)
  analysis.py   colour naming, object detection, 3x3 grid stats, describe()
  qa.py         question intents + grounded answers with evidence
  scenes.py     bundled scenes with known ground truth
  benchmark.py  16-question benchmark + Markdown/JSON report
  cli.py        visionqa command-line interface
assets/         scene PNGs (generated by scripts/make_scenes.py; regenerated
              automatically by `visionqa benchmark` if missing)
examples/demo.py  deterministic demo -> outputs/report.md + outputs/summary.json
tests/          36 pytest tests
```

## Verified

- **36/36 tests pass** (`pytest`), run twice in two fresh venvs created from `requirements.txt` + `pip install -e .` (identical results).
- **Demo verified twice, second time in a fresh venv** — benchmark 16/16 both times, identical `report.md` and `summary.json`.
- Three real bugs were caught and fixed during verification: a region question ("what is in the top-right?") was hijacked by the describe branch; a small yellow circle was misclassified as a blob (fill-ratio threshold too tight); and a "which object is biggest?" question with no colour named fell through to the fallback.
- No network, no API key, no secrets, no placeholders.

## License

MIT — see [LICENSE](LICENSE).
