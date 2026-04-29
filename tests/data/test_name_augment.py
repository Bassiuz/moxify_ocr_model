"""Tests for the name-region augmentation pipeline."""

from __future__ import annotations

import numpy as np

from moxify_ocr.data.name_augment import (
    apply_name_augmentation,
    build_name_augmentation_pipeline,
)


def _fake_crop(seed: int = 0) -> np.ndarray:
    """A 48x512x3 uint8 strip with structure (so augmentations have something to do)."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(48, 512, 3), dtype=np.uint8)


def test_pipeline_returns_uint8_with_same_shape() -> None:
    pipeline = build_name_augmentation_pipeline(seed=0)
    img = _fake_crop()
    out = apply_name_augmentation(img, pipeline, seed=42)
    assert out.shape == (48, 512, 3)
    assert out.dtype == np.uint8


def test_pipeline_is_deterministic_per_seed() -> None:
    pipeline = build_name_augmentation_pipeline(seed=0)
    img = _fake_crop()
    a = apply_name_augmentation(img, pipeline, seed=42)
    b = apply_name_augmentation(img, pipeline, seed=42)
    assert np.array_equal(a, b)


def test_pipeline_actually_transforms() -> None:
    """Different seeds should produce different outputs from the same input."""
    pipeline = build_name_augmentation_pipeline(seed=0)
    img = _fake_crop()
    a = apply_name_augmentation(img, pipeline, seed=1)
    b = apply_name_augmentation(img, pipeline, seed=2)
    assert not np.array_equal(a, b)
