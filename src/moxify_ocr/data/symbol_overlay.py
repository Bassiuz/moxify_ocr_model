"""Synthesize foil training examples by replacing ``•`` with ``★`` in a crop.

Scryfall is overwhelmingly non-foil; this module detects the small dark ``•``
blob in the lower half of a non-foil crop, erases it with the local background
color, and draws a 5-point star polygon in its place using the dot's own
foreground color — producing a pseudo-foil crop for augmentation.

Pure-drawing approach (no licensed MTG font required). Detection is threshold-
based and may mis-fire on complex backgrounds; callers should catch
:class:`LookupError` and fall back to a fixed-position overlay.
"""

from __future__ import annotations

import math
import random

import numpy as np
from PIL import Image, ImageDraw

# Detection + drawing tuning for ~256x48 letterboxed crops. _STAR_RADIUS_SCALE
# is kept small so the rasterised star fits inside the erased dot bbox at tiny
# text sizes (aliased 5-point stars over-cover at radius >= half the dot).
_MIN_DOT_AREA = 3
_MAX_DOT_AREA = 30
_DARKNESS_MARGIN = 20
_RING_PAD = 5
_JITTER_PX = 1
_JITTER_DEG = 10.0
_STAR_RADIUS_SCALE = 0.25


def overlay_foil_star(crop: Image.Image, *, seed: int = 0) -> Image.Image:
    """Return a copy of ``crop`` with the non-foil dot replaced by a foil star.

    Raises :class:`ValueError` if ``crop`` is not RGB. Raises :class:`LookupError`
    if no plausible dot glyph is found in the lower half.
    """
    if crop.mode != "RGB":
        raise ValueError(f"overlay_foil_star requires RGB input, got mode={crop.mode!r}")

    rng = random.Random(seed)
    arr = np.array(crop, dtype=np.uint8)
    gray = np.dot(arr[..., :3], np.array([0.299, 0.587, 0.114])).astype(np.float32)
    x0, y0, x1, y1 = _find_dot_bbox(gray)
    blob = arr[y0:y1, x0:x1]
    dy, dx = divmod(int(np.argmin(gray[y0:y1, x0:x1])), blob.shape[1])
    star_color = tuple(int(c) for c in blob[dy, dx])
    bg_color = _surrounding_color(arr, (x0, y0, x1, y1))
    arr[y0:y1, x0:x1] = np.array(bg_color, dtype=np.uint8)
    cx = (x0 + x1) / 2.0 + rng.uniform(-_JITTER_PX, _JITTER_PX)
    cy = (y0 + y1) / 2.0 + rng.uniform(-_JITTER_PX, _JITTER_PX)
    rotation = rng.uniform(-_JITTER_DEG, _JITTER_DEG)
    radius = max(1.0, min(x1 - x0, y1 - y0) * _STAR_RADIUS_SCALE)
    polygon = _star_polygon(cx, cy, radius, rotation)
    base = Image.fromarray(arr, "RGB").convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    ImageDraw.Draw(overlay).polygon(polygon, fill=(*star_color, 255))
    base.alpha_composite(overlay)
    return base.convert("RGB")


def _find_dot_bbox(gray: np.ndarray) -> tuple[int, int, int, int]:
    """Return (x0, y0, x1, y1) of the detected dot in the lower half of ``gray``."""
    h, _w = gray.shape
    lower = gray[h // 2 :, :]
    threshold = float(lower.mean()) - _DARKNESS_MARGIN
    dark_mask = lower < threshold
    if not dark_mask.any():
        raise LookupError("no dark pixels in lower half of crop")

    labels = _label_components(dark_mask)
    candidates: list[tuple[int, int, int, int, int]] = []
    for label_id in range(1, int(labels.max()) + 1):
        ys, xs = np.where(labels == label_id)
        area = int(ys.size)
        if area < _MIN_DOT_AREA or area > _MAX_DOT_AREA:
            continue
        bx0, by0 = int(xs.min()), int(ys.min())
        bx1, by1 = int(xs.max()) + 1, int(ys.max()) + 1
        bw, bh = bx1 - bx0, by1 - by0
        if bw == 0 or bh == 0:
            continue
        aspect = bw / bh if bh >= bw else bh / bw
        if aspect < 0.5:
            continue
        candidates.append((area, bx0, by0, bx1, by1))

    if not candidates:
        raise LookupError("no dot-sized blob found in lower half of crop")

    candidates.sort(key=lambda c: c[0])
    _, bx0, by0, bx1, by1 = candidates[0]
    y_off = gray.shape[0] // 2
    return bx0, by0 + y_off, bx1, by1 + y_off


def _label_components(mask: np.ndarray) -> np.ndarray:
    """Label 4-connected components in a boolean mask with iterative BFS."""
    h, w = mask.shape
    labels = np.zeros((h, w), dtype=np.int32)
    next_label = 0
    for sy in range(h):
        for sx in range(w):
            if not mask[sy, sx] or labels[sy, sx] != 0:
                continue
            next_label += 1
            stack = [(sy, sx)]
            while stack:
                y, x = stack.pop()
                if y < 0 or y >= h or x < 0 or x >= w or not mask[y, x] or labels[y, x] != 0:
                    continue
                labels[y, x] = next_label
                stack.extend(((y + 1, x), (y - 1, x), (y, x + 1), (y, x - 1)))
    return labels


def _surrounding_color(
    rgb: np.ndarray, bbox: tuple[int, int, int, int]
) -> tuple[int, int, int]:
    """Mean RGB of a ring ``_RING_PAD`` pixels wide around ``bbox``."""
    h, w, _ = rgb.shape
    x0, y0, x1, y1 = bbox
    ox0, oy0 = max(0, x0 - _RING_PAD), max(0, y0 - _RING_PAD)
    ox1, oy1 = min(w, x1 + _RING_PAD), min(h, y1 + _RING_PAD)
    mask = np.ones((oy1 - oy0, ox1 - ox0), dtype=bool)
    mask[y0 - oy0 : y1 - oy0, x0 - ox0 : x1 - ox0] = False
    ring = rgb[oy0:oy1, ox0:ox1][mask]
    pool = ring if ring.size > 0 else rgb[oy0:oy1, ox0:ox1].reshape(-1, 3)
    mean = pool.mean(axis=0)
    return int(mean[0]), int(mean[1]), int(mean[2])


def _star_polygon(
    cx: float, cy: float, radius: float, rotation_deg: float
) -> list[tuple[float, float]]:
    """10-vertex 5-point star, alternating outer and inner radius."""
    inner = radius * 0.5
    rotation = math.radians(rotation_deg) - math.pi / 2
    return [
        (
            cx + (radius if i % 2 == 0 else inner) * math.cos(rotation + i * math.pi / 5),
            cy + (radius if i % 2 == 0 else inner) * math.sin(rotation + i * math.pi / 5),
        )
        for i in range(10)
    ]
