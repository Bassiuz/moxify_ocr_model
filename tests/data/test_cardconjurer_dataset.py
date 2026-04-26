"""Tests for the pre-rendered CardConjurer pool reader."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from moxify_ocr.data.cardconjurer_dataset import CardConjurerPool, sample_from_pool


def _write_fake_pool(root: Path, n: int = 5) -> None:
    """Write n fake (256x48 black) PNGs + labels.jsonl matching the renderer output."""
    (root / "images").mkdir(parents=True)
    rows = []
    for i in range(n):
        img_path = f"images/{i:08d}.png"
        Image.new("RGB", (256, 48), (i * 10, 0, 0)).save(root / img_path)
        rows.append({"image_path": img_path, "label": f"00{i}/100 R\nMID • EN"})
    with (root / "labels.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def test_load_pool_finds_all_entries(tmp_path: Path) -> None:
    _write_fake_pool(tmp_path, n=5)
    pool = CardConjurerPool.load(tmp_path)
    assert len(pool) == 5


def test_sample_returns_correct_shape_and_label(tmp_path: Path) -> None:
    _write_fake_pool(tmp_path, n=5)
    pool = CardConjurerPool.load(tmp_path)
    img, label = sample_from_pool(pool, seed=0)
    assert img.shape == (48, 256, 3)
    assert img.dtype == np.uint8
    assert "\n" in label


def test_sample_is_deterministic(tmp_path: Path) -> None:
    _write_fake_pool(tmp_path, n=10)
    pool = CardConjurerPool.load(tmp_path)
    a = sample_from_pool(pool, seed=42)
    b = sample_from_pool(pool, seed=42)
    assert np.array_equal(a[0], b[0])
    assert a[1] == b[1]


def test_empty_pool_raises(tmp_path: Path) -> None:
    (tmp_path / "labels.jsonl").touch()
    pool = CardConjurerPool.load(tmp_path)
    with pytest.raises(ValueError):
        sample_from_pool(pool, seed=0)
