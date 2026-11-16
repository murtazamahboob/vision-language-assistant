"""Question answering over a grounded analysis.

Every answer is built from the analysis only, and carries its
evidence (grid position, bounding box, area) so it can be checked
against the image. Unknown questions get a grounded description
plus what *can* be asked, never a guessed answer.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .analysis import NAMED_COLORS, Analysis, describe

COLORS = [c for c in NAMED_COLORS if c != "white"]
SHAPES = ("circle", "rectangle", "triangle", "blob")
REGIONS = ("top-left", "top-center", "top-right", "middle-left",
           "middle-center", "middle-right", "bottom-left",
           "bottom-center", "bottom-right")


@dataclass
class Answer:
    text: str
    evidence: list = field(default_factory=list)
    grounded: bool = True


def _color_in(q: str) -> str | None:
    for c in COLORS:
        if re.search(rf"\b{c}\b", q):
            return c
    return None


def _colors_in(q: str) -> list:
    return [c for c in COLORS if re.search(rf"\b{c}\b", q)]


def _shape_in(q: str) -> str | None:
    for s in SHAPES:
        if re.search(rf"\b{s}\b", q):
            return s
    return None


def _region_in(q: str) -> str | None:
    q2 = q.replace("top left", "top-left").replace("top right", "top-right")
    q2 = q2.replace("bottom left", "bottom-left").replace("bottom right", "bottom-right")
    q2 = q2.replace("middle left", "middle-left").replace("middle right", "middle-right")
    q2 = q2.replace("top centre", "top-center").replace("top middle", "top-center")
    q2 = q2.replace("bottom centre", "bottom-center").replace("bottom middle", "bottom-center")
    q2 = q2.replace("centre", "center")
    for r in REGIONS:
        if r in q2:
            return r
    return None


def _fmt_obj(o) -> str:
    return f"a {o.color} {o.shape} ({o.evidence()})"


def ask(analysis: Analysis, question: str) -> Answer:
    q = question.strip().lower()

    # --- size / dimensions ---
    if any(k in q for k in ("how big", "how large", "size", "dimensions", "width", "height", "resolution")):
        return Answer(
            f"The image is {analysis.width} pixels wide and {analysis.height} pixels tall "
            f"({analysis.width * analysis.height} pixels in total).",
            evidence=[f"size: {analysis.width}x{analysis.height}"])

    # --- what is in a region --- (before describe: "what is in the top-right"
    # would otherwise match the describe branch)
    _region = _region_in(q)
    if _region and ("what" in q or "in the" in q):
        objs = [o for o in analysis.objects if o.position == _region]
        if not objs:
            return Answer(f"There is no object in the {_region} region.",
                          evidence=[f"grid {_region}: no object centroid there"])
        return Answer(f"In the {_region} region: " +
                      ", ".join(_fmt_obj(o) for o in objs) + ".",
                      evidence=[o.evidence() for o in objs])

    # --- describe ---
    if any(k in q for k in ("describe", "what is in", "what's in", "what does", "summar", "what do you see", "what is this")):
        return Answer(describe(analysis),
                      evidence=[o.evidence() for o in analysis.objects])

    # --- how many objects ---
    if "how many object" in q or "how many thing" in q or "how many shape" in q:
        return Answer(
            f"There are {len(analysis.objects)} distinct object(s) on the white background: "
            + ", ".join(f"{o.color} {o.shape}" for o in analysis.objects) + ".",
            evidence=[o.evidence() for o in analysis.objects])

    # --- count of a colour ---
    color = _color_in(q)
    if "how many" in q and color:
        objs = analysis.objects_of_color(color)
        if not objs:
            return Answer(f"There are no {color} objects in the image.",
                          evidence=[f"colour counts: {analysis.color_counts}"])
        return Answer(
            f"There is {len(objs)} {color} object(s): " +
            ", ".join(_fmt_obj(o) for o in objs) + ".",
            evidence=[o.evidence() for o in objs])

    # --- colours present / dominant ---
    if "what color" in q or "which color" in q or "what colour" in q or "which colour" in q:
        colors = sorted((c for c in analysis.color_counts if c != "white"),
                        key=lambda c: -analysis.color_counts[c])
        return Answer(
            "The colours in the image are: " +
            ", ".join(f"{c} ({analysis.color_counts[c]} px)" for c in colors) +
            f". The dominant colour is {analysis.dominant_color}.",
            evidence=[f"dominant: {analysis.dominant_color}",
                      f"colour counts: {analysis.color_counts}"])
    if "dominant" in q or "main color" in q or "main colour" in q:
        return Answer(
            f"The dominant colour is {analysis.dominant_color} "
            f"({analysis.color_counts.get(analysis.dominant_color, 0)} px).",
            evidence=[f"dominant: {analysis.dominant_color}"])

    # --- brightest / darkest region ---
    if "brightest" in q or "darkest" in q:
        cells = sorted(analysis.grid, key=lambda g: g[2])
        if "brightest" in q:
            pos, col, lum = cells[-1]
            return Answer(f"The brightest region is {pos} (mean brightness {lum}, "
                          f"mostly {col}).",
                          evidence=[f"grid {pos}: brightness {lum}, colour {col}"])
        pos, col, lum = cells[0]
        return Answer(f"The darkest region is {pos} (mean brightness {lum}, "
                      f"mostly {col}).",
                      evidence=[f"grid {pos}: brightness {lum}, colour {col}"])

    # --- is there a ... ---
    if q.startswith("is there") or q.startswith("do you see") or "is there a" in q or "is there an" in q:
        shape = _shape_in(q)
        if color and shape:
            objs = [o for o in analysis.objects if o.color == color and o.shape == shape]
            label = f"{color} {shape}"
        elif color:
            objs = analysis.objects_of_color(color)
            label = f"{color} object"
        elif shape:
            objs = analysis.objects_of_shape(shape)
            label = f"{shape}"
        else:
            objs, label = [], "that"
        if objs:
            return Answer(f"Yes — there is {label}: " +
                          ", ".join(_fmt_obj(o) for o in objs) + ".",
                          evidence=[o.evidence() for o in objs])
        return Answer(f"No — there is no {label} in the image.",
                      evidence=["objects: " + ", ".join(
                          f"{o.color} {o.shape}" for o in analysis.objects)])

    # --- where is the ... ---
    if q.startswith("where"):
        shape = _shape_in(q)
        if color and shape:
            objs = [o for o in analysis.objects if o.color == color and o.shape == shape]
        elif color:
            objs = analysis.objects_of_color(color)
        elif shape:
            objs = analysis.objects_of_shape(shape)
        else:
            objs = []
        if objs:
            o = objs[0]
            return Answer(f"The {o.color} {o.shape} is in the {o.position} "
                          f"of the image ({o.evidence()}).",
                          evidence=[o.evidence()])
        return Answer("I cannot find that object in the image.",
                      evidence=["objects: " + ", ".join(
                          f"{o.color} {o.shape}" for o in analysis.objects)])

    # --- position: is X on the left / above Y ... ---
    if ("left" in q or "right" in q or "top" in q or "bottom" in q or "above" in q or "below" in q) and color:
        objs = analysis.objects_of_color(color)
        if not objs:
            return Answer(f"There is no {color} object in the image.",
                          evidence=["objects: " + ", ".join(
                              f"{o.color} {o.shape}" for o in analysis.objects)])
        o = objs[0]
        if "above" in q or "below" in q:
            others = _colors_in(q)
            others = [c for c in others if c != color]
            if others:
                o2 = analysis.objects_of_color(others[0])
                if o2:
                    rel = "above" if o.centroid[1] < o2[0].centroid[1] else "below"
                    return Answer(f"Yes — the {color} {o.shape} is {rel} the "
                                  f"{others[0]} {o2[0].shape}.",
                                  evidence=[o.evidence(), o2[0].evidence()])
            return Answer(f"The {color} {o.shape} is in the {o.position} region.",
                          evidence=[o.evidence()])
        side = "left" if "left" in q else "right" if "right" in q else "top" if "top" in q else "bottom"
        yes = side in o.position
        return Answer(f"{'Yes' if yes else 'No'} — the {color} {o.shape} is in the "
                      f"{o.position} of the image.",
                      evidence=[o.evidence()])

    # --- bigger / taller comparison ---
    if any(k in q for k in ("bigger", "larger", "taller", "biggest", "tallest", "largest")):
        colors = _colors_in(q)
        if len(colors) >= 2:
            a = analysis.objects_of_color(colors[0])
            b = analysis.objects_of_color(colors[1])
            if a and b:
                o1, o2 = a[0], b[0]
                if "tall" in q:
                    winner, loser = (o1, o2) if o1.h >= o2.h else (o2, o1)
                    metric = f"height {winner.h} px vs {loser.h} px"
                else:
                    winner, loser = (o1, o2) if o1.area >= o2.area else (o2, o1)
                    metric = f"area {winner.area} px vs {loser.area} px"
                return Answer(f"The {winner.color} {winner.shape} is bigger "
                              f"({metric}).",
                              evidence=[winner.evidence(), loser.evidence()])
        if analysis.objects:
            biggest = max(analysis.objects, key=lambda o: o.area)
            return Answer(f"The biggest object is the {biggest.color} {biggest.shape} "
                          f"({biggest.evidence()}).",
                          evidence=[biggest.evidence()])

    # --- count of a shape ---
    shape = _shape_in(q)
    if "how many" in q and shape:
        objs = analysis.objects_of_shape(shape)
        return Answer(f"There are {len(objs)} {shape}(s) in the image.",
                      evidence=[o.evidence() for o in objs])

    # --- fallback: grounded, never guessed ---
    return Answer(
        "I can only answer from what I measured in this image. " + describe(analysis) +
        " Try asking about its colours, how many objects there are, "
        "where an object is, or which object is biggest.",
        evidence=[o.evidence() for o in analysis.objects],
        grounded=False)
