"""Read a narrowly bounded, explicit three-dash marker beside a TK label.

This is punctuation detection, not inference from missing OCR text. Three
separate, aligned strokes must survive two dark-pixel thresholds. A blank crop,
single minus, digits, table rule, faint mark or merged blob is not accepted.
The detector has no learned model, reference answers or patient-specific rules.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from .models import Box

METHOD = "image_dash_triplet_v1"


@dataclass(frozen=True)
class DashEvidence:
    bbox: Box
    method: str = METHOD
    text: str = "---"


def _components(image: Image.Image, threshold: int) -> list[tuple[int, int, int, int, int]]:
    width, height = image.size
    pixels = image.load()
    ink = {(x, y) for y in range(height) for x in range(width) if pixels[x, y] < threshold}
    components = []
    while ink:
        point = ink.pop()
        pending = [point]
        x0 = x1 = point[0]
        y0 = y1 = point[1]
        count = 0
        while pending:
            x, y = pending.pop()
            count += 1
            x0, y0, x1, y1 = min(x0, x), min(y0, y), max(x1, x), max(y1, y)
            for neighbour in ((x-1, y), (x+1, y), (x, y-1), (x, y+1)):
                if neighbour in ink:
                    ink.remove(neighbour)
                    pending.append(neighbour)
        if count >= 3:  # Ignore isolated compression/scanner specks only.
            components.append((x0, y0, x1+1, y1+1, count))
    return sorted(components)


def _triplet(image: Image.Image, label_height: float, threshold: int) -> tuple[int, int, int, int] | None:
    components = _components(image, threshold)
    if len(components) != 3:
        return None
    widths = [c[2]-c[0] for c in components]
    heights = [c[3]-c[1] for c in components]
    centers = [(c[1]+c[3])/2 for c in components]
    if any(not max(3, label_height*.07) <= w <= label_height*.9 or h > max(3, label_height*.18)
           or w/max(1, h) < 2 for w, h in zip(widths, heights)):
        return None
    if max(widths) / min(widths) > 1.8 or max(centers)-min(centers) > max(1.5, label_height*.08):
        return None
    gaps = [components[i+1][0]-components[i][2] for i in (0, 1)]
    if any(not 1 <= gap <= label_height*.5 for gap in gaps):
        return None
    if max(gaps)/min(gaps) > 2.5 or components[-1][2]-components[0][0] > label_height*2.5:
        return None
    return components[0][0], min(c[1] for c in components), components[-1][2], max(c[3] for c in components)


def find_dash_triplet(image: Image.Image, label: Box) -> DashEvidence | None:
    """Return original-page coordinates only when an explicit marker is visible.

    The search starts just after an OCR TK label, covers eight label heights to
    its right, and stays within its text line. It never searches another row or
    a whole eye column for an unrelated dash.
    """
    width, height = image.size
    line_height = (label.y1-label.y0)*height
    if not 6 <= line_height <= height*.05:
        return None
    left = round(label.x1*width+line_height*.1)
    top = max(0, round(label.y0*height-line_height*.2))
    right = min(width, round(label.x1*width+line_height*8))
    bottom = min(height, round(label.y1*height+line_height*.2))
    if right <= left or bottom <= top or (right-left)*(bottom-top) > 250_000:
        return None
    with image.crop((left, top, right, bottom)) as crop:
        gray = crop.convert("L")
        try:
            bounds = [_triplet(gray, line_height, threshold) for threshold in (90, 120)]
        finally:
            gray.close()
    if any(b is None for b in bounds):
        return None
    first, second = bounds
    if any(abs(a-b) > max(2, line_height*.08) for a, b in zip(first, second)):
        return None
    return DashEvidence(Box(x0=(left+min(first[0], second[0]))/width,
                            y0=(top+min(first[1], second[1]))/height,
                            x1=(left+max(first[2], second[2]))/width,
                            y1=(top+max(first[3], second[3]))/height))


def open_source_image(path: str) -> Image.Image | None:
    """Missing images simply leave the OCR result unresolved."""
    if not path or not Path(path).is_file():
        return None
    try:
        return Image.open(path)
    except (OSError, ValueError):
        return None
