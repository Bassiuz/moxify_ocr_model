"""Synthetic bottom-region renderer — generates (image, label) pairs on the fly.

Real training data is dominated by English cards (Scryfall's natural
distribution). This module generates **synthetic** bottom-region crops with
uniformly-sampled fields so the model sees every language, rarity, and digit
combination at roughly equal rates.

Design decisions
----------------
- **Colors** match what we measured on 20 real crops:
    - Background: near-black RGB(~15, ~13, ~12) ± 10, slight warm tint
    - Text: near-white RGB(~244, ~242, ~240) ± 10, slight warm tint
- **Font** defaults to ``DejaVu Sans Bold`` (always available on Linux / Kaggle).
  Users can drop ``Montserrat-Medium.ttf`` or ``Inter-Medium.ttf`` into
  ``assets/fonts/`` for a closer visual match — picks one at random per sample.
- **Layout** mirrors the real bottom-left: two lines of text, left-aligned,
  small horizontal padding. Positions are jittered within ±2 pixels.
- **Output** is a ``uint8`` array matching the ``(94, 268, 3)`` crop dimensions
  real cards produce, so the downstream ``crop_bottom_region`` letterbox logic
  applies identically.
"""

from __future__ import annotations

import math
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from moxify_ocr.data.labels import LANG_CODE, PLANESWALKER_ICON

# The real crop dimensions that ``crop_bottom_region`` produces for a
# 672x936 Scryfall image with fractions (0, 0.90, 0.40, 1.00).
_SYNTH_HEIGHT = 94
_SYNTH_WIDTH = 268

# Real card distribution stats (measured empirically — see chat logs).
_BG_BASE = (16, 14, 12)  # mean real background
_TEXT_BASE = (244, 242, 240)  # mean real text

# Font path candidates, tried in order. First existing wins.
_FONT_CANDIDATES_ROOT = Path(__file__).resolve().parents[3] / "assets" / "fonts"
_FONT_CANDIDATES = [
    _FONT_CANDIDATES_ROOT / "Montserrat-Medium.ttf",
    _FONT_CANDIDATES_ROOT / "Inter-Medium.ttf",
    Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),  # Linux / Kaggle
    Path("/System/Library/Fonts/Supplemental/Arial Bold.ttf"),  # macOS fallback
]

_SAMPLEABLE_LANGS = list(LANG_CODE.keys())  # all the language codes we support
_RARITIES = ["C", "U", "R", "M", "L"]
_SET_CODE_CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def _draw_star(
    draw: ImageDraw.ImageDraw,
    center: tuple[float, float],
    radius: float,
    fill: tuple[int, int, int],
) -> None:
    """Draw a 5-pointed star. Used because most fonts don't include U+2605."""
    cx, cy = center
    outer = radius
    inner = radius * 0.45
    pts: list[tuple[float, float]] = []
    # 10 vertices alternating outer/inner — starting straight up so the star
    # sits "pointy side up".
    for i in range(10):
        angle = -math.pi / 2 + i * math.pi / 5
        r = outer if i % 2 == 0 else inner
        pts.append((cx + r * math.cos(angle), cy + r * math.sin(angle)))
    draw.polygon(pts, fill=fill)


def _draw_planeswalker_icon(
    draw: ImageDraw.ImageDraw,
    center: tuple[float, float],
    radius: float,
    fill: tuple[int, int, int],
) -> None:
    """Stub planeswalker icon — a vertical rectangle with two small tapering dots.

    Real PLST icons are more ornate, but for CTC the class just needs to be
    visually distinct from ★ and •. A filled ellipse with two small dots above
    and below is distinct enough for a classifier.
    """
    cx, cy = center
    # Main body: vertical ellipse.
    draw.ellipse(
        (cx - radius * 0.5, cy - radius, cx + radius * 0.5, cy + radius),
        fill=fill,
    )
    # Two small circles top and bottom to disambiguate from a regular •.
    small = radius * 0.3
    draw.ellipse(
        (cx - small, cy - radius - small * 2, cx + small, cy - radius),
        fill=fill,
    )
    draw.ellipse(
        (cx - small, cy + radius, cx + small, cy + radius + small * 2),
        fill=fill,
    )


def _find_font(size: int, rng: random.Random) -> ImageFont.FreeTypeFont:
    """Return a truetype font at the given pixel size. Picks from available candidates."""
    available = [p for p in _FONT_CANDIDATES if p.exists()]
    if not available:
        raise FileNotFoundError(
            f"No usable font found among {_FONT_CANDIDATES}. "
            "On Linux/Kaggle install fonts-dejavu, on macOS Arial is usually present."
        )
    path = rng.choice(available)
    return ImageFont.truetype(str(path), size=size)


def _bg_color(rng: random.Random) -> tuple[int, int, int]:
    """Near-black background with slight warm tint — matches real card distribution."""
    g = rng.randint(5, 25)
    return (
        max(0, min(255, g + rng.randint(-1, 3))),
        max(0, min(255, g + rng.randint(-2, 1))),
        max(0, min(255, g + rng.randint(-2, 0))),
    )


def _text_color(rng: random.Random) -> tuple[int, int, int]:
    """Near-white text with slight warm tint."""
    g = rng.randint(220, 255)
    return (
        g,
        max(0, g - rng.randint(0, 3)),
        max(0, g - rng.randint(0, 5)),
    )


def _random_set_code(rng: random.Random) -> str:
    """Random 3-5 char uppercase, occasionally with a trailing digit (like MB1, M13)."""
    length = rng.choices([3, 3, 3, 4, 5], k=1)[0]
    code = "".join(rng.choices(_SET_CODE_CHARS, k=length))
    if rng.random() < 0.1:
        code = code[:-1] + str(rng.randint(0, 9))
    return code


def _random_collector_number(rng: random.Random) -> str:
    """Mimic Scryfall's collector number styles: 1-4 digit, usually zero-padded to 3."""
    n = rng.randint(1, 999)
    if rng.random() < 0.8:
        return f"{n:03d}"
    return str(n)


def _random_sample_fields(
    rng: random.Random,
) -> tuple[str, int | None, str | None, str, str, bool, bool]:
    """Randomly sample a tuple of (collector, total, rarity, set_code, lang, foil, is_plst).

    Distribution: rarities and languages sampled uniformly (not weighted by
    real-world frequency) so the synthetic set compensates for Scryfall's
    EN/common skew.
    """
    collector = _random_collector_number(rng)
    total = rng.randint(200, 400) if rng.random() < 0.7 else None
    rarity: str | None = rng.choice(_RARITIES) if rng.random() < 0.9 else None
    set_code = _random_set_code(rng)
    lang = rng.choice(_SAMPLEABLE_LANGS)
    foil = rng.random() < 0.35
    is_plst = rng.random() < 0.08  # small PLST share
    return collector, total, rarity, set_code, lang, foil, is_plst


def _build_label(
    collector: str,
    total: int | None,
    rarity: str | None,
    set_code: str,
    lang: str,
    foil: bool,
    is_plst: bool,
) -> str:
    """Label string matching the format ``make_label`` produces."""
    line1_parts: list[str] = []
    if is_plst:
        line1_parts.append(PLANESWALKER_ICON)
    line1_parts.append(f"{collector}/{total}" if total is not None else collector)
    if rarity is not None:
        line1_parts.append(rarity)
    line1 = " ".join(line1_parts)
    foil_glyph = "★" if foil else "•"
    line2 = f"{set_code.upper()} {foil_glyph} {LANG_CODE[lang]}"
    return f"{line1}\n{line2}"


def generate_synthetic_crop(seed: int) -> tuple[np.ndarray, str]:
    """Render one ``(image, label)`` pair with the given seed.

    The image is a ``uint8`` ``(_SYNTH_HEIGHT, _SYNTH_WIDTH, 3)`` array matching
    the real crop dimensions, so the downstream ``crop_bottom_region`` letterbox
    pipeline applies identically.
    """
    rng = random.Random(seed)

    collector, total, rarity, set_code, lang, foil, is_plst = _random_sample_fields(rng)
    label = _build_label(collector, total, rarity, set_code, lang, foil, is_plst)

    # Render two lines separately (makes positioning + size predictable).
    bg = _bg_color(rng)
    text = _text_color(rng)
    img = Image.new("RGB", (_SYNTH_WIDTH, _SYNTH_HEIGHT), bg)
    draw = ImageDraw.Draw(img)

    font_size = rng.randint(22, 28)
    font = _find_font(font_size, rng)

    line1, line2 = label.split("\n")
    pad_x = rng.randint(4, 10)
    y1 = rng.randint(2, 10)
    y2 = rng.randint(_SYNTH_HEIGHT // 2 - 6, _SYNTH_HEIGHT // 2 + 4)
    _draw_line_with_custom_glyphs(draw, (pad_x, y1), line1, font, text)
    _draw_line_with_custom_glyphs(draw, (pad_x, y2), line2, font, text)

    return np.asarray(img, dtype=np.uint8), label


def _draw_line_with_custom_glyphs(
    draw: ImageDraw.ImageDraw,
    origin: tuple[int, int],
    line: str,
    font: ImageFont.FreeTypeFont,
    fill: tuple[int, int, int],
) -> None:
    """Draw a line of text, replacing ★ and the planeswalker icon with shapes.

    Most fonts don't include U+2605 (★) or our PUA U+E100 (planeswalker icon),
    so rendering them directly produces empty-glyph □ boxes. We draw those two
    symbols as polygons/ellipses instead, and let the font handle everything
    else.
    """
    x, y = origin
    # Space between characters — approximate from the font's "0" width.
    glyph_h = font.size
    for ch in line:
        if ch == "★":
            cx = x + glyph_h * 0.5
            cy = y + glyph_h * 0.5
            _draw_star(draw, (cx, cy), radius=glyph_h * 0.45, fill=fill)
            x += int(glyph_h * 1.1)
        elif ch == PLANESWALKER_ICON:
            cx = x + glyph_h * 0.35
            cy = y + glyph_h * 0.5
            _draw_planeswalker_icon(draw, (cx, cy), radius=glyph_h * 0.35, fill=fill)
            x += int(glyph_h * 0.9)
        else:
            draw.text((x, y), ch, font=font, fill=fill)
            # Advance by the glyph's real width (font.getlength handles kerning).
            x += int(font.getlength(ch))
