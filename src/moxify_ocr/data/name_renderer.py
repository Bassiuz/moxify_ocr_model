"""PIL compositor for synthetic card-name region crops.

Mirrors the role of :mod:`moxify_ocr.data.cardconjurer_dataset` from the
bottom-region pipeline: turns :class:`NameSpec` objects into ``(image, label)``
pairs the dataset reader can feed the CRNN.

Unlike the bottom-region pipeline (which drives a real CardConjurer browser
via Playwright) this module renders entirely in pure Python — it composites
CardConjurer's *static* frame asset PNGs onto a colored background, then
draws the card name with PIL.ImageDraw using fonts from the same asset
bundle. No Chromium, no JavaScript, no network.

The asset bundle lives at the path passed to :class:`NameRenderer` (default
``/tmp/cardconjurer-master``). The expected layout is::

    <root>/img/frames/<pack>/<color>.png   # 1500x2100 RGBA frame layers
    <root>/fonts/<font>.ttf                # the bundled MTG-style fonts

If a style's pack is missing the requested color, the renderer falls back to
the closest available color (color matters for visual variety, not for
correctness — the OCR target is the name text, not the frame color).
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from moxify_ocr.data.name_specs import NameSpec

CARD_W: int = 1500
CARD_H: int = 2100
TARGET_W: int = 512
TARGET_H: int = 48

# Diameter of mana symbols rendered into the right edge of the name region,
# in canvas (1500x2100) pixels. Real cards print these at ~5-6% of card
# width; we use 80px (~5.3%) as a deliberate slight under-size so a
# 5-symbol cost doesn't crowd the name.
_MANA_SYMBOL_PX: int = 80
# Horizontal gap between adjacent symbols.
_MANA_SYMBOL_GAP: int = 6
# Inset from the right edge of the bbox to the rightmost symbol's right edge.
_MANA_RIGHT_INSET: int = 18

# Mana symbol palette. Generic (numeric / X) costs print as a grey circle
# with a black numeral; colored mana is a tinted circle with a white inner
# letter. These match the visual "shape vocabulary" the OCR needs to learn
# to ignore — exact glyph fidelity to real m21 symbols isn't required.
_MANA_FILL: dict[str, tuple[int, int, int]] = {
    "w": (252, 246, 217),  # cream
    "u": (170, 207, 232),  # light blue
    "b": (170, 162, 158),  # mauve-grey (real B background)
    "r": (244, 178, 158),  # light salmon
    "g": (164, 207, 178),  # light green
    # Generic costs all share the same grey.
    "_generic": (192, 188, 180),
}
_MANA_INK: dict[str, tuple[int, int, int]] = {
    "w": (40, 35, 25),
    "u": (40, 35, 25),
    "b": (40, 35, 25),
    "r": (40, 35, 25),
    "g": (40, 35, 25),
    "_generic": (40, 35, 25),
}
_MANA_LABEL: dict[str, str] = {
    "w": "W", "u": "U", "b": "B", "r": "R", "g": "G", "x": "X",
    "0": "0", "1": "1", "2": "2", "3": "3", "4": "4",
    "5": "5", "6": "6", "7": "7", "8": "8", "9": "9",
}


@dataclass(frozen=True)
class StyleConfig:
    """How to render one :class:`NameSpec` style.

    Attributes:
        frame_pack: Subpath under ``<root>/img/frames`` containing the color
            PNGs for this style.
        available_colors: Color-letter codes the pack actually has files for.
            If the spec asks for a color outside this set we pick a fallback.
        font: Font filename under ``<root>/fonts`` to render the name with.
        font_size: Base size in pixels at the 1500x2100 canvas resolution.
            ``NameSpec.font_size_jitter`` multiplies this per-sample.
        text_color: RGB color for the name glyphs.
        bbox: ``(x1, y1, x2, y2)`` of the name region inside the 1500x2100
            canvas. The renderer left-aligns the text inside this box and
            crops to the box for output.
        rotate_cw_after_crop: For battle / split / aftermath / flip styles
            whose name reads sideways on the printed card. After cropping
            the bbox we rotate 90° clockwise so the model only sees
            horizontal text — matching the upstream-cropper contract.
        bg_palette: Per-color base background tones. The renderer picks the
            entry matching ``spec.frame_color`` (or "M" as fallback).
    """

    frame_pack: str
    available_colors: tuple[str, ...]
    font: str
    font_size: int
    # Static text color override. When ``None`` (preferred), the renderer
    # auto-picks black or white based on the mean brightness of the frame
    # region inside ``bbox`` — that's what real card printers do, and it
    # avoids "black text on dark frame" failures on showcase / old / DFC-back
    # styles whose name slot is dark.
    text_color: tuple[int, int, int] | None
    bbox: tuple[int, int, int, int]
    rotate_cw_after_crop: bool
    # Filename format for the color PNG inside the pack. ``{c}`` is the
    # lowercase color letter, ``{C}`` the uppercase. Default ``"{c}.png"``
    # works for most packs; transform / DFC packs need ``"front{C}.png"``
    # / ``"back{C}.png"`` etc.
    filename_format: str = "{c}.png"
    # Pixels of left-inset between ``bbox[0]`` and where the name text
    # starts. The bbox itself can be wider than the name slot to capture
    # surrounding context (e.g., the DFC transform-indicator on the top-
    # left of double-faced cards). Setting this larger pushes the text past
    # that context so it doesn't render *over* the indicator. Default 8 px
    # matches the historical behavior for normal layouts.
    text_left_inset: int = 8


# Background palette per frame color. Roughly matches the dominant tone of
# the corresponding real-card frame so the synthetic compositor doesn't look
# wildly off when the frame PNG has transparency in or near the name region.
_BG_PALETTE: dict[str, tuple[int, int, int]] = {
    "W": (235, 222, 188),  # cream
    "U": (60, 100, 160),   # blue
    "B": (45, 38, 38),     # dark gray-brown
    "R": (170, 50, 40),    # red
    "G": (40, 110, 60),    # green
    "M": (190, 165, 100),  # gold
    "A": (140, 145, 150),  # silver/slate
    "L": (140, 105, 75),   # brown
}

# Fallback when a pack lacks the requested color.
_COLOR_FALLBACK_ORDER: tuple[str, ...] = ("M", "A", "C", "W", "U", "B", "R", "G", "L")


# Per-style asset + geometry table. Coordinates are tuned to land the name
# box on the visible name strip for each frame era; smoke-test expected to
# refine these.
STYLE_TABLE: dict[str, StyleConfig] = {
    # Modern (2015 / M15) — the dominant prior.
    "modern_regular": StyleConfig(
        frame_pack="m15/regular",
        available_colors=("w", "u", "b", "r", "g", "m", "a", "l"),
        font="beleren-b.ttf",
        font_size=70,
        text_color=None,
        bbox=(110, 130, 1380, 220),
        rotate_cw_after_crop=False,
    ),
    "modern_legendary": StyleConfig(
        # CC's m15/crowns uses non-standard filenames; for v1 we fall back
        # to the regular m15 frame and accept the loss of legendary
        # filigree (the bbox is what matters for OCR coverage; smoke test
        # will tell us if real-card legendary names regress without it).
        frame_pack="m15/regular",
        available_colors=("w", "u", "b", "r", "g", "m", "a", "l"),
        font="beleren-b.ttf",
        font_size=68,
        text_color=None,
        bbox=(135, 138, 1380, 230),
        rotate_cw_after_crop=False,
    ),
    "modern_extended": StyleConfig(
        frame_pack="m15/regular",
        available_colors=("w", "u", "b", "r", "g", "m", "a", "l"),
        font="beleren-b.ttf",
        font_size=70,
        text_color=None,
        bbox=(110, 130, 1380, 220),
        rotate_cw_after_crop=False,
    ),
    "modern_borderless": StyleConfig(
        frame_pack="m15/regular",
        available_colors=("w", "u", "b", "r", "g", "m", "a", "l"),
        font="beleren-b.ttf",
        font_size=70,
        text_color=None,
        bbox=(110, 130, 1380, 220),
        rotate_cw_after_crop=False,
    ),
    "modern_showcase_storybook": StyleConfig(
        frame_pack="storybook",
        available_colors=("w", "u", "b", "r", "g", "m", "c"),
        font="goudy-medieval.ttf",
        font_size=72,
        text_color=(40, 30, 20),
        bbox=(180, 110, 1320, 220),
        rotate_cw_after_crop=False,
    ),
    "modern_showcase_kaldheim": StyleConfig(
        frame_pack="kaldheim",
        available_colors=("w", "u", "b", "r", "g", "m"),
        font="beleren-b.ttf",
        font_size=70,
        text_color=None,
        bbox=(110, 130, 1380, 220),
        rotate_cw_after_crop=False,
    ),
    "modern_showcase_tarkir": StyleConfig(
        frame_pack="tarkir",
        available_colors=("w", "u", "b", "r", "g", "m"),
        font="beleren-b.ttf",
        font_size=70,
        text_color=None,
        bbox=(110, 130, 1380, 220),
        rotate_cw_after_crop=False,
    ),
    "modern_showcase_neo": StyleConfig(
        frame_pack="neo/neon",
        available_colors=("w", "u", "b", "r", "g", "m", "a"),
        font="beleren-b.ttf",
        font_size=70,
        text_color=None,
        bbox=(110, 130, 1380, 220),
        rotate_cw_after_crop=False,
    ),
    # Pre-modern.
    "premodern_2003": StyleConfig(
        frame_pack="8th",
        available_colors=("w", "u", "b", "r", "g", "m", "a", "l"),
        font="matrix.ttf",
        font_size=78,
        text_color=None,
        bbox=(110, 105, 1380, 200),
        rotate_cw_after_crop=False,
    ),
    "premodern_1997": StyleConfig(
        frame_pack="seventh/regular",
        available_colors=("w", "u", "b", "r", "g", "m", "a", "l"),
        font="matrix.ttf",
        font_size=78,
        text_color=None,
        bbox=(110, 105, 1380, 200),
        rotate_cw_after_crop=False,
    ),
    "old_1993": StyleConfig(
        frame_pack="old/abu",
        available_colors=("w", "u", "b", "r", "g", "m", "a", "l", "bl"),
        font="matrix.ttf",
        font_size=82,
        text_color=None,
        bbox=(110, 95, 1380, 200),
        rotate_cw_after_crop=False,
    ),
    "old_8th": StyleConfig(
        frame_pack="8th",
        available_colors=("w", "u", "b", "r", "g", "m", "a", "l"),
        font="matrix.ttf",
        font_size=78,
        text_color=None,
        bbox=(110, 105, 1380, 200),
        rotate_cw_after_crop=False,
    ),
    "future_sight": StyleConfig(
        frame_pack="future",
        available_colors=("white", "gray"),  # the FUT pack uses prefix names
        font="beleren-b.ttf",
        font_size=66,
        text_color=None,
        bbox=(110, 120, 1380, 210),
        rotate_cw_after_crop=False,
    ),
    # Special layouts.
    "saga": StyleConfig(
        frame_pack="saga",
        available_colors=("w", "u", "b", "r", "g", "m", "a"),
        font="beleren-b.ttf",
        font_size=68,
        text_color=None,
        bbox=(140, 130, 1380, 220),
        rotate_cw_after_crop=False,
    ),
    # Battle / split frames have their name slots running sideways on the
    # actual card. The OCR contract is that the upstream cropper pre-rotates
    # those crops to horizontal — so what the *model* sees is a small,
    # cramped horizontal name strip. We mimic that here by rendering on the
    # m15 frame with a smaller font; faithful battle/split frame chrome is
    # deferred to v2 (it requires drawing rotated text in a slot that the
    # m15/{battle,split} frame PNGs only suggest indirectly).
    "battle": StyleConfig(
        frame_pack="m15/regular",
        available_colors=("w", "u", "b", "r", "g", "m", "a", "l"),
        font="beleren-b.ttf",
        font_size=58,
        text_color=None,
        bbox=(110, 140, 1380, 215),
        rotate_cw_after_crop=False,
    ),
    "split_left": StyleConfig(
        frame_pack="m15/regular",
        available_colors=("w", "u", "b", "r", "g", "m", "a"),
        font="beleren-b.ttf",
        font_size=54,
        text_color=None,
        bbox=(110, 140, 1380, 215),
        rotate_cw_after_crop=False,
    ),
    "split_right": StyleConfig(
        frame_pack="m15/regular",
        available_colors=("w", "u", "b", "r", "g", "m", "a"),
        font="beleren-b.ttf",
        font_size=54,
        text_color=None,
        bbox=(110, 140, 1380, 215),
        rotate_cw_after_crop=False,
    ),
    "adventure": StyleConfig(
        frame_pack="adventure/regular",
        available_colors=("w", "u", "b", "r", "g", "m", "a", "l"),
        font="beleren-b.ttf",
        font_size=70,
        text_color=None,
        bbox=(110, 130, 1380, 220),
        rotate_cw_after_crop=False,
    ),
    # DFC styles: the bbox captures the FULL top of the card (x=0..1500) so
    # the transform-indicator stays in the OCR crop, but the text starts at
    # x=170 (text_left_inset) to clear that indicator instead of rendering
    # over it. The 170-px inset matches the real-card layout where the
    # indicator occupies roughly the leftmost 10% of card width.
    "transform_front": StyleConfig(
        frame_pack="m15/transform/regular",
        available_colors=("w", "u", "b", "r", "g", "m", "a", "l"),
        font="beleren-b.ttf",
        font_size=66,
        text_color=None,
        bbox=(0, 80, 1500, 240),
        rotate_cw_after_crop=False,
        filename_format="front{C}.png",
        text_left_inset=230,
    ),
    "transform_back": StyleConfig(
        frame_pack="m15/transform/regular",
        available_colors=("w", "u", "b", "r", "g", "m", "a", "l"),
        font="beleren-b.ttf",
        font_size=66,
        text_color=None,
        bbox=(0, 80, 1500, 240),
        rotate_cw_after_crop=False,
        filename_format="back{C}.png",
        text_left_inset=230,
    ),
    "modal_dfc_front": StyleConfig(
        frame_pack="m15/transform/regular",
        available_colors=("w", "u", "b", "r", "g", "m", "a", "l"),
        font="beleren-b.ttf",
        font_size=66,
        text_color=None,
        bbox=(0, 80, 1500, 240),
        rotate_cw_after_crop=False,
        filename_format="front{C}.png",
        text_left_inset=230,
    ),
    "modal_dfc_back": StyleConfig(
        frame_pack="m15/transform/regular",
        available_colors=("w", "u", "b", "r", "g", "m", "a", "l"),
        font="beleren-b.ttf",
        font_size=66,
        text_color=None,
        bbox=(0, 80, 1500, 240),
        rotate_cw_after_crop=False,
        filename_format="back{C}.png",
        text_left_inset=230,
    ),
    "planeswalker": StyleConfig(
        frame_pack="planeswalker",
        available_colors=("w", "u", "b", "r", "g", "m"),
        font="beleren-b.ttf",
        font_size=70,
        text_color=None,
        bbox=(110, 130, 1380, 220),
        rotate_cw_after_crop=False,
    ),
    "aftermath": StyleConfig(
        frame_pack="m15/aftermath",
        available_colors=("w", "u", "b", "r", "g", "m", "a", "l"),
        font="beleren-b.ttf",
        font_size=64,
        text_color=None,
        bbox=(110, 130, 1380, 220),
        rotate_cw_after_crop=False,
    ),
    "flip": StyleConfig(
        frame_pack="m15/flip",
        available_colors=("w", "u", "b", "r", "g", "m", "a", "l", "c"),
        font="beleren-b.ttf",
        font_size=66,
        text_color=None,
        bbox=(110, 130, 1380, 220),
        rotate_cw_after_crop=False,
    ),
}


class NameRenderer:
    """Stateful renderer; caches frame PNGs and font objects across calls."""

    DEFAULT_ROOT: ClassVar[Path] = Path("/tmp/cardconjurer-master")

    def __init__(
        self,
        *,
        cardconjurer_root: Path | None = None,
    ) -> None:
        root = cardconjurer_root or self.DEFAULT_ROOT
        if not root.exists():
            raise FileNotFoundError(
                f"cardconjurer root not found at {root} — see "
                "infra/cardconjurer/README.md for one-time setup"
            )
        self._frames_root = root / "img" / "frames"
        self._fonts_root = root / "fonts"
        self._font_cache: dict[tuple[str, int], ImageFont.FreeTypeFont] = {}
        self._frame_cache: dict[Path, Image.Image] = {}
        self._mana_cache: dict[str, Image.Image] = {}

    def render(self, spec: NameSpec) -> tuple[np.ndarray, str]:
        """Render one ``NameSpec`` to a ``(48, 512, 3) uint8`` ndarray + label."""
        style = STYLE_TABLE[spec.style]
        frame = self._load_frame(style, spec.frame_color, seed=hash(spec) & 0xFFFFFFFF)
        bg_color = _BG_PALETTE.get(spec.frame_color, _BG_PALETTE["M"])
        canvas = Image.new("RGB", (CARD_W, CARD_H), bg_color)
        canvas.paste(frame, (0, 0), frame)
        # Mana cost is painted FIRST so the name's auto-shrink logic knows
        # how much horizontal room is left after the symbols claim the right
        # edge. Width consumed by the cost is subtracted from the bbox's
        # right side before computing max_w in :meth:`_draw_name`.
        mana_width_used = self._paste_mana_cost(canvas, spec, style)
        text_color = self._resolve_text_color(canvas, style)
        self._draw_name(
            canvas, spec, style, text_color=text_color, mana_width=mana_width_used
        )
        # Crop the name region.
        crop = canvas.crop(style.bbox)
        if style.rotate_cw_after_crop:
            crop = crop.rotate(-90, expand=True)
        # Resize to the model's input shape.
        crop = crop.resize((TARGET_W, TARGET_H), Image.LANCZOS)
        arr = np.asarray(crop.convert("RGB"), dtype=np.uint8)
        return arr, spec.name

    @staticmethod
    def _resolve_text_color(
        canvas: Image.Image, style: StyleConfig
    ) -> tuple[int, int, int]:
        """Pick black or white text based on bbox-region brightness.

        ``style.text_color`` overrides this when set. Otherwise the renderer
        samples the canvas at the bbox region (post-frame compositing) and
        picks the contrast color: white text on backgrounds with mean
        luminance < 128, black text otherwise. Matches what real card
        printers do — old/showcase/DFC-back frames with dark name slots get
        white name text automatically.
        """
        if style.text_color is not None:
            return style.text_color
        x1, y1, x2, y2 = style.bbox
        region = np.asarray(canvas.crop((x1, y1, x2, y2)).convert("RGB"))
        # Rec. 709 luma weights — close enough for a contrast decision.
        luma = float(
            (0.2126 * region[..., 0] + 0.7152 * region[..., 1] + 0.0722 * region[..., 2])
            .mean()
        )
        return (255, 255, 255) if luma < 128 else (0, 0, 0)

    # ---- internals ----

    def _load_font(self, name: str, size: int) -> ImageFont.FreeTypeFont:
        key = (name, size)
        font = self._font_cache.get(key)
        if font is None:
            path = self._fonts_root / name
            font = ImageFont.truetype(str(path), size)
            self._font_cache[key] = font
        return font

    def _load_frame(
        self, style: StyleConfig, frame_color: str, *, seed: int
    ) -> Image.Image:
        """Resolve the frame PNG path and return a cached RGBA image."""
        color_letter = self._resolve_color(style, frame_color, seed=seed)
        filename = style.filename_format.format(
            c=color_letter.lower(), C=color_letter.upper()
        )
        path = self._frames_root / style.frame_pack / filename
        cached = self._frame_cache.get(path)
        if cached is not None:
            return cached
        if not path.exists():
            # Fall back: pick any color-letter PNG in the pack. Excludes
            # thumbs, masks, and named helpers (stamp / holo / border /
            # frame.svg-like assets) which would render as garbage if loaded
            # as the main frame.
            pack_dir = self._frames_root / style.frame_pack
            disallow = ("thumb", "mask", "stamp", "holo", "border", "frame")
            candidates = sorted(
                p
                for p in pack_dir.glob("*.png")
                if not any(bad in p.stem.lower() for bad in disallow)
            )
            if not candidates:
                raise FileNotFoundError(
                    f"no usable frame PNGs in {pack_dir} for style "
                    f"{style.frame_pack!r} (filename {filename!r} not found, "
                    "and no color-letter PNG fallbacks)"
                )
            path = candidates[seed % len(candidates)]
        img = Image.open(path).convert("RGBA")
        if img.size != (CARD_W, CARD_H):
            img = img.resize((CARD_W, CARD_H), Image.LANCZOS)
        self._frame_cache[path] = img
        return img

    @staticmethod
    def _resolve_color(
        style: StyleConfig, frame_color: str, *, seed: int
    ) -> str:
        """Map a spec's frame_color (uppercase letter) to a pack file stem."""
        target = frame_color.lower()
        if target in style.available_colors:
            return target
        # Fallback chain: try the canonical preference order, then any.
        for cand in _COLOR_FALLBACK_ORDER:
            cl = cand.lower()
            if cl in style.available_colors:
                return cl
        # Last resort: pick the first available color deterministically.
        rng = random.Random(seed)
        return rng.choice(style.available_colors)

    def _paste_mana_cost(
        self, canvas: Image.Image, spec: NameSpec, style: StyleConfig
    ) -> int:
        """Paste the spec's mana symbols on the right edge of the bbox.

        Returns the horizontal pixel range claimed by the symbols (0 if
        empty), so the name auto-shrink can subtract it from the bbox width
        before computing the available text room.

        Symbols are painted right-to-left so the last symbol in the cost
        ends up rightmost — matching how mana cost is read. Centered
        vertically in the bbox.
        """
        if not spec.mana_cost:
            return 0
        x1, y1, x2, y2 = style.bbox
        symbols = list(spec.mana_cost)
        n = len(symbols)
        total_w = n * _MANA_SYMBOL_PX + (n - 1) * _MANA_SYMBOL_GAP
        # Right edge of the rightmost symbol (canvas px).
        right_edge = x2 - _MANA_RIGHT_INSET
        # Top of each symbol — vertical-center inside the bbox.
        bbox_h = y2 - y1
        sym_top = y1 + max(0, (bbox_h - _MANA_SYMBOL_PX) // 2)
        # Place each symbol from right to left.
        x_cursor = right_edge - _MANA_SYMBOL_PX
        for sym in reversed(symbols):
            img = self._load_mana(sym)
            canvas.paste(img, (x_cursor, sym_top), img)
            x_cursor -= _MANA_SYMBOL_PX + _MANA_SYMBOL_GAP
        return total_w + _MANA_RIGHT_INSET

    def _load_mana(self, sym: str) -> Image.Image:
        """Render (and cache) a mana symbol as a tinted circle with an inner glyph.

        We don't use CardConjurer's ``m21*.png`` assets directly because
        they're white silhouettes meant to be CSS-tinted at runtime in the
        web UI — pasting them as-is produces white blobs. Drawing here
        instead gives us full RGBA control: a grey/coloured circle plus an
        ink-coloured inner letter or digit drawn with a bundled font.
        """
        cached = self._mana_cache.get(sym)
        if cached is not None:
            return cached
        is_colored = sym in _MANA_FILL and sym != "_generic"
        fill = _MANA_FILL[sym] if is_colored else _MANA_FILL["_generic"]
        ink = _MANA_INK[sym] if is_colored else _MANA_INK["_generic"]
        label = _MANA_LABEL.get(sym, "?")

        size = _MANA_SYMBOL_PX
        img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        # Filled circle with a thin darker ring to mimic the printed bevel.
        draw.ellipse([(2, 2), (size - 3, size - 3)], fill=fill)
        ring = tuple(max(0, c - 60) for c in fill)  # subtle ring shadow
        draw.ellipse([(2, 2), (size - 3, size - 3)], outline=ring, width=2)
        # Inner glyph centered using a fresh ImageFont (PIL's FreeTypeFont
        # cache misbehaves across multiple Draw contexts — "invalid
        # reference" — so we deliberately skip the shared _font_cache
        # for mana glyphs).
        # plantin-semibold is close to the real-card mana glyph font; we
        # avoid mplantin.ttf because the file ships with a malformed table
        # that makes PIL 12 raise OSError("invalid reference") on getbbox.
        glyph_font = ImageFont.truetype(
            str(self._fonts_root / "plantin-semibold.otf"), int(size * 0.55)
        )
        bb = glyph_font.getbbox(label)
        text_w = bb[2] - bb[0]
        text_h = bb[3] - bb[1]
        gx = (size - text_w) // 2 - bb[0]
        gy = (size - text_h) // 2 - bb[1]
        draw.text((gx, gy), label, fill=ink, font=glyph_font)
        self._mana_cache[sym] = img
        return img

    def _draw_name(
        self,
        canvas: Image.Image,
        spec: NameSpec,
        style: StyleConfig,
        *,
        text_color: tuple[int, int, int],
        mana_width: int = 0,
    ) -> None:
        size = max(20, int(style.font_size * spec.font_size_jitter))
        font = self._load_font(style.font, size)
        draw = ImageDraw.Draw(canvas)
        x1, y1, x2, y2 = style.bbox
        text_x = x1 + style.text_left_inset
        # Auto-shrink the font if the name doesn't fit the bbox width minus
        # whatever the mana cost on the right edge already claimed.
        max_w = x2 - text_x - 12 - mana_width
        text = spec.name
        while size > 24:
            bbox = draw.textbbox((0, 0), text, font=font)
            text_w = bbox[2] - bbox[0]
            if text_w <= max_w:
                break
            size = int(size * 0.92)
            font = self._load_font(style.font, size)
        # Vertical center inside the bbox.
        bbox = draw.textbbox((0, 0), text, font=font)
        text_h = bbox[3] - bbox[1]
        y_off = y1 + max(0, (y2 - y1 - text_h) // 2) - bbox[1]
        draw.text((text_x, y_off), text, fill=text_color, font=font)
