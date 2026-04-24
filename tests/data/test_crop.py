"""Tests for :mod:`moxify_ocr.data.crop`."""

from __future__ import annotations

import pytest
from PIL import Image, ImageChops

from moxify_ocr.data.crop import (
    BOTTOM_REGION_FRACTIONS,
    LETTERBOX_FILL_RGB,
    TARGET_HEIGHT,
    TARGET_WIDTH,
    crop_bottom_region,
)


def _solid(size: tuple[int, int], color: tuple[int, int, int]) -> Image.Image:
    """Create a solid-color RGB image of the given (W, H)."""
    return Image.new("RGB", size, color)


def test_output_size() -> None:
    """Output must be exactly 256x48 for the default Scryfall ~672x936 input."""
    image = _solid((672, 936), (255, 255, 255))
    out = crop_bottom_region(image)
    assert out.size == (TARGET_WIDTH, TARGET_HEIGHT)
    assert out.size == (256, 48)


def test_output_mode_rgb() -> None:
    """Output mode must be RGB."""
    image = _solid((672, 936), (255, 255, 255))
    out = crop_bottom_region(image)
    assert out.mode == "RGB"


def test_non_rgb_input_raises() -> None:
    """Grayscale (mode 'L') input must raise ValueError — no silent convert."""
    image = Image.new("L", (672, 936), 255)
    with pytest.raises(ValueError, match="RGB"):
        crop_bottom_region(image)  # type: ignore[arg-type]


def test_crop_region_pixel_bounds() -> None:
    """The bottom-left region must be sampled, not the rest."""
    w, h = 672, 936
    image = _solid((w, h), (0, 255, 0))  # green background
    # Paint the bottom-left region (x in [0, 0.4*w], y in [0.9*h, h]) pure red.
    # 0.40 matches BOTTOM_REGION_FRACTIONS (v2 narrowed from 0.50).
    x1 = 0
    y1 = round(0.90 * h)
    x2 = round(0.40 * w)
    y2 = h
    red_patch = _solid((x2 - x1, y2 - y1), (255, 0, 0))
    image.paste(red_patch, (x1, y1))

    out = crop_bottom_region(image)
    # Sample the center 30% of the output; it should be predominantly red.
    ow, oh = out.size
    cx1 = int(ow * 0.35)
    cx2 = int(ow * 0.65)
    cy1 = int(oh * 0.35)
    cy2 = int(oh * 0.65)
    pixels = [out.getpixel((x, y)) for x in range(cx1, cx2) for y in range(cy1, cy2)]
    red_pixels = sum(1 for p in pixels if isinstance(p, tuple) and p[0] > 200 and p[1] < 60)
    assert red_pixels / len(pixels) > 0.9


def test_letterbox_fill_on_square_input() -> None:
    """A square crop into a 256x48 target must produce letterbox bars."""
    image = _solid((500, 500), (255, 255, 255))
    # Use full-image fractions so the crop is 500x500 (square).
    out = crop_bottom_region(image, fractions=(0.0, 0.0, 1.0, 1.0))
    # With target (256, 48), scale = min(256/500, 48/500) = 48/500 = 0.096.
    # Resized crop: 48x48. Bars are on left/right: (256-48)/2 = 104 px each side.
    # Corner pixel (0, 0) is firmly in the left bar.
    assert out.getpixel((0, 0)) == LETTERBOX_FILL_RGB
    assert out.getpixel((255, 47)) == LETTERBOX_FILL_RGB


def test_aspect_ratio_preserved() -> None:
    """Aspect ratio of the non-letterbox content must match the crop's aspect."""
    # Create an input whose full-image crop (fractions 0,0,1,1) has a known aspect.
    # 400x100 crop → aspect 4:1. Target 256x48 → scale = min(256/400, 48/100) = 0.48.
    # Resized content: 192x48. So the content fills vertically (48 px tall), and
    # the letterbox bars are horizontal (left/right), each (256-192)/2 = 32 px wide.
    image = _solid((400, 100), (10, 20, 30))
    out = crop_bottom_region(image, fractions=(0.0, 0.0, 1.0, 1.0))

    # Content occupies columns [32, 32+192) and all rows.
    # Check that columns [0, 32) are fill color and [32, ...) are content.
    assert out.getpixel((0, 24)) == LETTERBOX_FILL_RGB
    assert out.getpixel((31, 24)) == LETTERBOX_FILL_RGB
    # First content column is 32; sample it (it's the content color).
    content_pixel = out.getpixel((32 + 96, 24))  # middle of content
    assert isinstance(content_pixel, tuple)
    assert content_pixel == (10, 20, 30)
    # Far right bar: column 32+192 = 224 is the first bar pixel on the right.
    assert out.getpixel((224, 24)) == LETTERBOX_FILL_RGB

    # Aspect: content is 192 wide / 48 tall = 4.0; crop was 400/100 = 4.0.
    content_w, content_h = 192, 48
    assert content_w / content_h == pytest.approx(400 / 100)


def test_custom_target_size_honored() -> None:
    """target_size must be honored."""
    image = _solid((672, 936), (255, 255, 255))
    out = crop_bottom_region(image, target_size=(100, 50))
    assert out.size == (100, 50)


def test_custom_fractions_honored() -> None:
    """Custom fractions must select a different region of the input."""
    w, h = 400, 400
    image = _solid((w, h), (0, 255, 0))  # green default
    # Put a magenta patch in the top-left quarter.
    image.paste(_solid((200, 200), (255, 0, 255)), (0, 0))

    # Crop top-left quarter with the same aspect as the target so no letterbox.
    out = crop_bottom_region(
        image,
        fractions=(0.0, 0.0, 0.5, 0.5),
        target_size=(200, 200),
    )
    # Sample center pixel — should be magenta.
    assert out.getpixel((100, 100)) == (255, 0, 255)


def test_custom_fill_color_honored() -> None:
    """Custom fill color must be used in letterbox bars."""
    image = _solid((500, 500), (255, 255, 255))
    out = crop_bottom_region(
        image,
        fractions=(0.0, 0.0, 1.0, 1.0),
        fill=(255, 0, 255),
    )
    assert out.getpixel((0, 0)) == (255, 0, 255)


def test_full_coverage_no_letterbox_needed() -> None:
    """When crop aspect matches target aspect, output must contain no fill pixels."""
    # 512x96 → aspect 512/96 ≈ 5.333; target 256/48 ≈ 5.333. Same aspect.
    image = _solid((512, 96), (10, 20, 30))
    out = crop_bottom_region(
        image,
        fractions=(0.0, 0.0, 1.0, 1.0),
        target_size=(256, 48),
    )
    assert out.size == (256, 48)
    # No pixel should equal the letterbox fill color.
    for x in (0, 128, 255):
        for y in (0, 24, 47):
            pix = out.getpixel((x, y))
            assert pix != LETTERBOX_FILL_RGB


def test_idempotent_on_already_target_sized_input() -> None:
    """Input already 256x48, full fractions → output ≈ input (LANCZOS rounding OK)."""
    # Use a gradient to actually exercise resampling.
    image = Image.new("RGB", (256, 48))
    for x in range(256):
        for y in range(48):
            image.putpixel((x, y), (x % 256, y * 5 % 256, (x + y) % 256))
    out = crop_bottom_region(image, fractions=(0.0, 0.0, 1.0, 1.0))
    assert out.size == (256, 48)
    diff = ImageChops.difference(image, out)
    # Should be pixel-identical or extremely close (scale=1.0 → LANCZOS is no-op).
    bbox = diff.getbbox()
    assert bbox is None


def test_constants_match_design() -> None:
    """Sanity-check the module-level constants.

    v2 narrowed the crop from x2=0.50 → x2=0.40 to exclude artist-name bleed.
    """
    assert BOTTOM_REGION_FRACTIONS == (0.0, 0.90, 0.40, 1.00)
    assert TARGET_HEIGHT == 48
    assert TARGET_WIDTH == 256
    assert LETTERBOX_FILL_RGB == (114, 114, 114)
