"""Crop the bottom-left region of a card image and letterbox into the OCR input size.

The bottom-left region — collector number, rarity letter, set code, foil glyph,
language code — is the only part of the card the OCR model consumes. We extract
it with fixed normalized fractions (mirroring Moxify's ``mtg_scan_regions.dart``)
and letterbox-resize it into the model's fixed ``48x256`` input while preserving
aspect ratio.

See ``docs/plans/2026-04-23-bottom-region-extractor-design.md`` section 4.3 for
the full rationale.
"""

from __future__ import annotations

from PIL import Image

# Normalized fractions of the card image for the bottom-left region.
# Same as Moxify's mtg_scan_regions.dart: (x1=0, y1=0.90, x2=0.50, y2=1.00).
BOTTOM_REGION_FRACTIONS: tuple[float, float, float, float] = (0.0, 0.90, 0.50, 1.00)

# Target model input (H, W) — matches design doc §5.
TARGET_HEIGHT: int = 48
TARGET_WIDTH: int = 256

# Letterbox fill color (R, G, B), matches design §4.3.
LETTERBOX_FILL_RGB: tuple[int, int, int] = (114, 114, 114)


def crop_bottom_region(
    image: Image.Image,
    *,
    target_size: tuple[int, int] = (TARGET_WIDTH, TARGET_HEIGHT),
    fractions: tuple[float, float, float, float] = BOTTOM_REGION_FRACTIONS,
    fill: tuple[int, int, int] = LETTERBOX_FILL_RGB,
) -> Image.Image:
    """Crop the bottom-left region and letterbox-resize it to ``target_size``.

    - ``image`` must be an RGB PIL Image (mode ``"RGB"``). Callers are responsible
      for converting — we do NOT silently convert here (fail fast on wrong mode).
    - ``target_size`` is ``(width, height)`` — PIL convention.
    - The crop uses the given ``fractions = (x1, y1, x2, y2)`` with values in ``[0, 1]``.
    - Letterboxing preserves the crop's aspect ratio; unused pixels are filled
      with ``fill``.
    - Output is always exactly ``target_size``, mode ``"RGB"``.
    """
    if image.mode != "RGB":
        raise ValueError(
            f"crop_bottom_region requires RGB input, got mode={image.mode!r}"
        )

    src_w, src_h = image.size
    x1, y1, x2, y2 = fractions
    box = (round(src_w * x1), round(src_h * y1), round(src_w * x2), round(src_h * y2))
    crop = image.crop(box)

    crop_w, crop_h = crop.size
    target_w, target_h = target_size
    scale = min(target_w / crop_w, target_h / crop_h)
    resized_w, resized_h = round(crop_w * scale), round(crop_h * scale)
    resized = crop.resize((resized_w, resized_h), Image.Resampling.LANCZOS)

    canvas = Image.new("RGB", target_size, fill)
    offset = ((target_w - resized_w) // 2, (target_h - resized_h) // 2)
    canvas.paste(resized, offset)
    return canvas
