"""Tests for :mod:`moxify_ocr.data.augment`."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from moxify_ocr.data.augment import apply_augmentation, build_augmentation_pipeline


def _synthetic_image(seed: int = 0) -> np.ndarray:
    """Random 48x256 uint8 RGB image."""
    rng = np.random.RandomState(seed)
    return rng.randint(0, 256, size=(48, 256, 3), dtype=np.uint8)


def test_pipeline_returns_same_shape() -> None:
    """Input 48x256x3 must produce output of the same shape."""
    image = _synthetic_image()
    pipeline = build_augmentation_pipeline(seed=0)
    out = apply_augmentation(image, pipeline, seed=1)
    assert out.shape == image.shape


def test_pipeline_returns_uint8() -> None:
    """Output dtype must be uint8."""
    image = _synthetic_image()
    pipeline = build_augmentation_pipeline(seed=0)
    out = apply_augmentation(image, pipeline, seed=1)
    assert out.dtype == np.uint8


def test_deterministic_with_seed() -> None:
    """Same input, same seed → identical arrays (bit-for-bit)."""
    image = _synthetic_image()
    pipeline = build_augmentation_pipeline(seed=0)
    out_a = apply_augmentation(image, pipeline, seed=42)
    out_b = apply_augmentation(image, pipeline, seed=42)
    assert np.array_equal(out_a, out_b)


def test_different_seeds_produce_different_outputs() -> None:
    """Same input, different seeds → arrays must differ."""
    image = _synthetic_image()
    pipeline = build_augmentation_pipeline(seed=0)
    out_a = apply_augmentation(image, pipeline, seed=1)
    out_b = apply_augmentation(image, pipeline, seed=2)
    assert not np.array_equal(out_a, out_b)


def test_augmentation_actually_modifies_image() -> None:
    """Output should not be bit-identical to input (probabilistic but overwhelming)."""
    image = _synthetic_image()
    pipeline = build_augmentation_pipeline(seed=0)
    out = apply_augmentation(image, pipeline, seed=1)
    assert not np.array_equal(out, image)


_DEBUG_CROPS = Path("data/debug_crops")


@pytest.mark.skipif(
    not _DEBUG_CROPS.exists() or not any(_DEBUG_CROPS.glob("*.jpg")),
    reason="Requires data/debug_crops/*.jpg from scripts/dump_sample_crops.py",
)
def test_handles_realistic_crop() -> None:
    """Real cropped image from disk must round-trip through augmentation."""
    sample = next(_DEBUG_CROPS.glob("*.jpg"))
    with Image.open(sample) as img:
        rgb = img.convert("RGB")
        arr = np.asarray(rgb, dtype=np.uint8)
    pipeline = build_augmentation_pipeline(seed=0)
    out = apply_augmentation(arr, pipeline, seed=0)
    assert out.shape == arr.shape
    assert out.dtype == np.uint8
