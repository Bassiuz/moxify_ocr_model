"""Tests for :mod:`moxify_ocr.data.symbol_overlay`."""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image, ImageChops, ImageDraw

from moxify_ocr.data.symbol_overlay import overlay_foil_star


def _make_dot_image(
    *,
    size: tuple[int, int] = (256, 48),
    bg: tuple[int, int, int] = (255, 255, 255),
    dot_center: tuple[int, int] = (140, 40),
    dot_size: int = 4,
    dot_color: tuple[int, int, int] = (0, 0, 0),
) -> Image.Image:
    """Synthetic crop with a small dark square dot in the lower half."""
    img = Image.new("RGB", size, bg)
    draw = ImageDraw.Draw(img)
    cx, cy = dot_center
    half = dot_size // 2
    draw.rectangle(
        (cx - half, cy - half, cx - half + dot_size - 1, cy - half + dot_size - 1),
        fill=dot_color,
    )
    return img


def test_overlay_changes_image() -> None:
    """Overlay must change at least one pixel of the input."""
    img = _make_dot_image()
    out = overlay_foil_star(img, seed=0)
    diff_bbox = ImageChops.difference(img, out).getbbox()
    assert diff_bbox is not None


def test_overlay_deterministic_with_seed() -> None:
    """Same input and seed → identical output bytes."""
    img = _make_dot_image()
    a = overlay_foil_star(img, seed=42)
    b = overlay_foil_star(img, seed=42)
    assert list(a.getdata()) == list(b.getdata())


def test_overlay_different_seeds_produce_different_outputs() -> None:
    """Different seeds → different outputs (due to jitter/rotation)."""
    img = _make_dot_image()
    a = overlay_foil_star(img, seed=42)
    b = overlay_foil_star(img, seed=43)
    assert list(a.getdata()) != list(b.getdata())


def test_overlay_raises_when_no_dot_found() -> None:
    """Pure white image with no dark blobs → LookupError."""
    img = Image.new("RGB", (256, 48), (255, 255, 255))
    with pytest.raises(LookupError):
        overlay_foil_star(img, seed=0)


def test_overlay_preserves_mode() -> None:
    """RGB input → RGB output."""
    img = _make_dot_image()
    out = overlay_foil_star(img, seed=0)
    assert out.mode == "RGB"


def test_overlay_preserves_size() -> None:
    """256x48 input → 256x48 output."""
    img = _make_dot_image()
    out = overlay_foil_star(img, seed=0)
    assert out.size == (256, 48)


def test_overlay_non_rgb_input_raises() -> None:
    """Grayscale ('L') input raises ValueError — no silent convert."""
    img = Image.new("L", (256, 48), 255)
    with pytest.raises(ValueError, match="RGB"):
        overlay_foil_star(img, seed=0)  # type: ignore[arg-type]


def test_overlay_dot_area_is_replaced() -> None:
    """The dot's bounding box should no longer be a solid black blob.

    After overlay, the star occupies part — but not all — of the dot bbox, so
    the majority of pixels in that small region should NOT be pure black.
    """
    dot_center = (140, 40)
    dot_size = 4
    img = _make_dot_image(dot_center=dot_center, dot_size=dot_size)
    out = overlay_foil_star(img, seed=0)
    cx, cy = dot_center
    half = dot_size // 2
    total = 0
    pure_black = 0
    for x in range(cx - half, cx - half + dot_size):
        for y in range(cy - half, cy - half + dot_size):
            pix = out.getpixel((x, y))
            assert isinstance(pix, tuple)
            total += 1
            if pix == (0, 0, 0):
                pure_black += 1
    # Majority of dot-bbox pixels are no longer pure black.
    assert pure_black / total < 0.5


def test_overlay_star_is_visible_in_vicinity() -> None:
    """After overlay, dark pixels must remain in the broader vicinity of the dot."""
    dot_center = (140, 40)
    img = _make_dot_image(dot_center=dot_center, dot_size=4)
    out = overlay_foil_star(img, seed=0)
    cx, cy = dot_center
    # Sample a 20x20 region centered on the dot.
    dark_pixels = 0
    for x in range(max(0, cx - 10), min(256, cx + 10)):
        for y in range(max(0, cy - 10), min(48, cy + 10)):
            pix = out.getpixel((x, y))
            assert isinstance(pix, tuple)
            if pix[0] < 100 and pix[1] < 100 and pix[2] < 100:
                dark_pixels += 1
    assert dark_pixels > 0


def test_overlay_with_realistic_crop() -> None:
    """On a real Scryfall-derived crop, overlay should not raise and preserve shape."""
    crops_dir = Path("data/debug_crops")
    if not crops_dir.exists():
        pytest.skip("data/debug_crops not available")
    candidates = sorted(crops_dir.glob("*.jpg"))[:10]
    if not candidates:
        pytest.skip("no debug crops available")
    for path in candidates:
        with Image.open(path) as im:
            crop = im.convert("RGB")
        try:
            out = overlay_foil_star(crop, seed=7)
        except LookupError:
            continue
        assert out.size == crop.size
        assert out.mode == "RGB"
        return
    pytest.skip("dot not detectable in any sample crop")
