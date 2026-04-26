"""Line-level real-data compositor for OCR training augmentation.

Takes real bottom-region crops, splits each into a top half (line 1: collector
number + optional /total + rarity letter) and a bottom half (line 2: set code
+ foil glyph + language). At training time we mix-and-match: pair a top half
from card A with a bottom half from card B to produce new training samples
that use real fonts, real ink, and real print artifacts — but with novel
field combinations.

Why this beats the pure synthetic renderer (v3):
- Real font (MTG's actual Gotham/Relay-family typeface), not Montserrat
- Real print imperfections (ink bleed, slight color variations)
- Real anti-aliasing matching what the camera sees in production
- No domain gap between training and inference

What it doesn't do out of the box: the foil glyph ★ — Scryfall images are
almost always non-foil prints, so the • dot dominates real data. We patch
that by overlaying a drawn ★ over the • position on a fraction of the
sampled line-2 halves (controlled by ``foil_overlay_prob``).
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from moxify_ocr.data.crop import crop_bottom_region
from moxify_ocr.data.dataset import is_trainable
from moxify_ocr.data.labels import LANG_CODE, make_label
from moxify_ocr.data.manifest import ManifestEntry, read_manifest

# Half-image height. The full crop is 48px tall — cut exactly at the midpoint.
_HALF_H = 24
_FULL_H = 48
_FULL_W = 256

# Max pixel jitter when re-stacking the two halves (helps the model not learn
# the seam location as a feature).
_SEAM_JITTER = 2

# When overlaying ★ to fake a foil sample, this is the radius in pixels.
_STAR_RADIUS = 4


@dataclass(frozen=True, slots=True)
class LineSample:
    """One real top-half or bottom-half image + the text it represents."""

    image: np.ndarray  # (24, 256, 3) uint8
    text: str
    has_foil_finish: bool  # True if the source card has a 'foil' printing


@dataclass
class LineLibrary:
    """Per-line sample pool, indexed by language for line 2 to allow oversampling."""

    line1: list[LineSample]
    line2_by_lang: dict[str, list[LineSample]]

    @classmethod
    def build(
        cls,
        manifest_path: Path,
        images_root: Path,
        *,
        min_release: str = "2008-01-01",
        max_per_lang: int = 1000,
        max_line1: int = 10000,
    ) -> "LineLibrary":
        """Walk the manifest, split each crop into halves, build the library.

        We cap each language's line-2 pool at ``max_per_lang`` so a few common
        languages (mostly English) don't dominate. ``max_line1`` is the global
        cap on top halves — line 1 is mostly numeric so language isn't a
        factor; we just want a deep pool of digits/letters/PW-icon variations.
        """
        line1: list[LineSample] = []
        line2_by_lang: dict[str, list[LineSample]] = {}

        for entry in read_manifest(manifest_path):
            if not is_trainable(entry, min_release=min_release):
                continue
            card = _entry_to_card_dict(entry)
            try:
                full_label = make_label(card, is_foil=False)
            except (KeyError, ValueError):
                continue
            try:
                line1_text, line2_text = full_label.split("\n")
            except ValueError:
                continue

            try:
                pil = Image.open(images_root / entry.image_path).convert("RGB")
            except FileNotFoundError:
                continue
            cropped = crop_bottom_region(pil, target_size=(_FULL_W, _FULL_H))
            arr = np.asarray(cropped, dtype=np.uint8)

            top = arr[:_HALF_H].copy()
            bottom = arr[_HALF_H:].copy()
            has_foil = "foil" in (entry.finishes or [])

            if len(line1) < max_line1:
                line1.append(LineSample(top, line1_text, has_foil))

            lang_list = line2_by_lang.setdefault(entry.lang, [])
            if len(lang_list) < max_per_lang:
                lang_list.append(LineSample(bottom, line2_text, has_foil))

        return cls(line1=line1, line2_by_lang=line2_by_lang)

    def is_empty(self) -> bool:
        return len(self.line1) == 0 or not self.line2_by_lang


def composite_sample(
    library: LineLibrary,
    *,
    seed: int,
    lang_balance: bool = True,
    foil_overlay_prob: float = 0.30,
) -> tuple[np.ndarray, str]:
    """Sample one (image, label) pair by stitching real halves.

    Args:
        library: Pre-built :class:`LineLibrary`.
        seed: Per-call seed for determinism.
        lang_balance: When True, pick line-2's language uniformly across
            available languages instead of weighted by frequency. Equalizes
            non-EN representation.
        foil_overlay_prob: Probability of replacing the • with a drawn ★ on
            line 2. Only applied to samples whose source card has a foil
            printing available.

    Returns:
        ``(image, label)`` where image is ``(48, 256, 3)`` uint8 and label is
        the synthesized string with ``\\n`` separator.
    """
    rng = random.Random(seed)
    if library.is_empty():
        raise ValueError("line library is empty — nothing to sample from")

    line1 = rng.choice(library.line1)

    if lang_balance:
        non_empty_langs = [lang for lang, samples in library.line2_by_lang.items() if samples]
        lang = rng.choice(non_empty_langs)
        line2 = rng.choice(library.line2_by_lang[lang])
    else:
        all_l2: list[LineSample] = []
        for samples in library.line2_by_lang.values():
            all_l2.extend(samples)
        line2 = rng.choice(all_l2)

    bottom_arr = line2.image.copy()
    line2_text = line2.text

    # Foil overlay: only meaningful if the source card has a foil printing
    # available (otherwise the overlay creates an impossible state).
    if line2.has_foil_finish and rng.random() < foil_overlay_prob:
        overlaid = _try_overlay_star(bottom_arr, rng)
        if overlaid is not None:
            bottom_arr = overlaid
            line2_text = line2_text.replace("•", "★", 1)

    composite = _stack_with_jitter(line1.image, bottom_arr, rng)
    label = f"{line1.text}\n{line2_text}"
    return composite, label


# ---------- internal helpers ----------


def _stack_with_jitter(top: np.ndarray, bottom: np.ndarray, rng: random.Random) -> np.ndarray:
    """Stack two 24x256x3 halves with ±jitter to break the seam location."""
    canvas = np.zeros((_FULL_H, _FULL_W, 3), dtype=np.uint8)
    # Pick a seam y in [_HALF_H - jitter, _HALF_H + jitter].
    seam = _HALF_H + rng.randint(-_SEAM_JITTER, _SEAM_JITTER)
    seam = max(_HALF_H - _SEAM_JITTER, min(_HALF_H + _SEAM_JITTER, seam))

    # Place top so its bottom row sits at row `seam` (truncate or pad as needed).
    top_target_h = seam
    if top_target_h <= top.shape[0]:
        canvas[:top_target_h] = top[-top_target_h:]
    else:
        canvas[top_target_h - top.shape[0] : top_target_h] = top

    # Place bottom starting at row `seam`.
    bottom_target_h = _FULL_H - seam
    if bottom_target_h <= bottom.shape[0]:
        canvas[seam:] = bottom[:bottom_target_h]
    else:
        canvas[seam : seam + bottom.shape[0]] = bottom

    return canvas


def _try_overlay_star(bottom: np.ndarray, rng: random.Random) -> np.ndarray | None:
    """Detect the • in a line-2 image and overlay a drawn ★ on top.

    Returns the modified image, or None if the • couldn't be located reliably
    (in which case the caller should fall back to the unmodified non-foil sample).
    """
    pos = _find_dot_center(bottom)
    if pos is None:
        return None
    cx, cy = pos

    # Sample text foreground color from the brightest pixels nearby — keeps
    # color consistent with the rest of the line.
    fg = _sample_bright_color(bottom)

    pil = Image.fromarray(bottom)
    draw = ImageDraw.Draw(pil)

    # Erase a slightly-larger region than the original • so the dot doesn't
    # peek out from behind the new star.
    bg = _sample_dark_color(bottom)
    draw.ellipse(
        (cx - _STAR_RADIUS - 1, cy - _STAR_RADIUS - 1, cx + _STAR_RADIUS + 1, cy + _STAR_RADIUS + 1),
        fill=bg,
    )
    _draw_star(draw, (cx, cy), radius=_STAR_RADIUS, fill=fg, rotation_deg=rng.uniform(-15, 15))
    return np.asarray(pil, dtype=np.uint8)


def _find_dot_center(bottom: np.ndarray) -> tuple[int, int] | None:
    """Return (x, y) center of the • in a line-2 image, or None if not found.

    Heuristic: find the smallest connected light blob in the middle horizontal
    third of the image (where the • sits between SET and LANG). Tolerates
    some text bleed because the dot is small (~3-5 pixels) and roundish.
    """
    h, w, _ = bottom.shape
    luminance = bottom.astype(np.int32).mean(axis=2)
    # Search middle 50% horizontally — • is centered between SET and LANG.
    x_start = w // 4
    x_end = (3 * w) // 4
    region = luminance[:, x_start:x_end]
    bright_thresh = max(180, int(np.percentile(region, 95)))
    bright = region > bright_thresh

    components = _connected_components(bright)
    candidates: list[tuple[int, int, int]] = []  # (area, x, y)
    for ys, xs in components:
        if len(xs) < 2 or len(xs) > 30:
            continue
        bbox_w = xs.max() - xs.min() + 1
        bbox_h = ys.max() - ys.min() + 1
        if max(bbox_w, bbox_h) > 8:  # too big to be the dot
            continue
        aspect = max(bbox_w, bbox_h) / max(min(bbox_w, bbox_h), 1)
        if aspect > 2.0:  # not roundish
            continue
        # Center, in the original image's coordinate frame.
        cx = int(xs.mean()) + x_start
        cy = int(ys.mean())
        candidates.append((len(xs), cx, cy))

    if not candidates:
        return None
    candidates.sort()  # smallest first — • is the smallest blob
    _, cx, cy = candidates[0]
    return cx, cy


def _connected_components(mask: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    """Pure-numpy 4-connected component labeling. Returns list of (ys, xs) per blob."""
    visited = np.zeros_like(mask, dtype=bool)
    h, w = mask.shape
    components: list[tuple[np.ndarray, np.ndarray]] = []
    for y in range(h):
        for x in range(w):
            if not mask[y, x] or visited[y, x]:
                continue
            # BFS
            stack = [(y, x)]
            ys: list[int] = []
            xs: list[int] = []
            while stack:
                cy, cx = stack.pop()
                if cy < 0 or cy >= h or cx < 0 or cx >= w:
                    continue
                if visited[cy, cx] or not mask[cy, cx]:
                    continue
                visited[cy, cx] = True
                ys.append(cy)
                xs.append(cx)
                stack.extend([(cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)])
            components.append((np.asarray(ys), np.asarray(xs)))
    return components


def _draw_star(
    draw: ImageDraw.ImageDraw,
    center: tuple[int, int],
    *,
    radius: float,
    fill: tuple[int, int, int],
    rotation_deg: float = 0.0,
) -> None:
    """Draw a 5-pointed star at ``center`` (mirrors the helper in synthetic.py)."""
    cx, cy = center
    outer = radius
    inner = radius * 0.45
    pts: list[tuple[float, float]] = []
    base_angle = -math.pi / 2 + math.radians(rotation_deg)
    for i in range(10):
        angle = base_angle + i * math.pi / 5
        r = outer if i % 2 == 0 else inner
        pts.append((cx + r * math.cos(angle), cy + r * math.sin(angle)))
    draw.polygon(pts, fill=fill)


def _sample_bright_color(image: np.ndarray) -> tuple[int, int, int]:
    """Mean RGB of the brightest 5% of pixels — approximates text color."""
    lum = image.astype(np.int32).mean(axis=2)
    thresh = np.percentile(lum, 95)
    bright_pixels = image[lum >= thresh]
    if len(bright_pixels) == 0:
        return (240, 240, 240)
    mean = bright_pixels.mean(axis=0)
    return (int(mean[0]), int(mean[1]), int(mean[2]))


def _sample_dark_color(image: np.ndarray) -> tuple[int, int, int]:
    """Mean RGB of the darkest 30% of pixels — approximates background."""
    lum = image.astype(np.int32).mean(axis=2)
    thresh = np.percentile(lum, 30)
    dark_pixels = image[lum <= thresh]
    if len(dark_pixels) == 0:
        return (15, 13, 12)
    mean = dark_pixels.mean(axis=0)
    return (int(mean[0]), int(mean[1]), int(mean[2]))


def _entry_to_card_dict(entry: ManifestEntry) -> dict[str, Any]:
    """Reshape ManifestEntry into the dict shape ``make_label`` expects."""
    return {
        "collector_number": entry.collector_number,
        "set": entry.set_code,
        "lang": entry.lang,
        "rarity": entry.rarity,
        "type_line": entry.type_line,
        "released_at": entry.released_at,
        **({"printed_size": entry.printed_size} if entry.printed_size is not None else {}),
    }
