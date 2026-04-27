"""Tests for the pre-rendered name-region pool reader."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from moxify_ocr.data.name_dataset import NamePool, sample_from_pool


def _write_fake_pool(root: Path, n: int = 5) -> None:
    """Write n fake (512x48 colored) PNGs + labels.jsonl matching the renderer output."""
    (root / "images").mkdir(parents=True)
    rows = []
    for i in range(n):
        img_path = f"images/{i:08d}.png"
        Image.new("RGB", (512, 48), (i * 10, 0, 0)).save(root / img_path)
        rows.append({"image_path": img_path, "label": f"Card Name {i}"})
    with (root / "labels.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def test_load_pool_finds_all_entries(tmp_path: Path) -> None:
    _write_fake_pool(tmp_path, n=5)
    pool = NamePool.load(tmp_path)
    assert len(pool) == 5


def test_sample_returns_correct_shape_and_label(tmp_path: Path) -> None:
    _write_fake_pool(tmp_path, n=5)
    pool = NamePool.load(tmp_path)
    img, label = sample_from_pool(pool, seed=0)
    assert img.shape == (48, 512, 3)
    assert img.dtype == np.uint8
    assert label.startswith("Card Name")


def test_sample_is_deterministic(tmp_path: Path) -> None:
    _write_fake_pool(tmp_path, n=10)
    pool = NamePool.load(tmp_path)
    a = sample_from_pool(pool, seed=42)
    b = sample_from_pool(pool, seed=42)
    assert np.array_equal(a[0], b[0])
    assert a[1] == b[1]


def test_empty_pool_raises(tmp_path: Path) -> None:
    (tmp_path / "labels.jsonl").touch()
    pool = NamePool.load(tmp_path)
    with pytest.raises(ValueError, match="NamePool is empty"):
        sample_from_pool(pool, seed=0)


def test_missing_manifest_returns_empty_pool(tmp_path: Path) -> None:
    pool = NamePool.load(tmp_path)
    assert len(pool) == 0


def test_split_pool_holds_out_by_label(tmp_path: Path) -> None:
    """Same label must never end up in both train and val."""
    from moxify_ocr.data.name_dataset import split_pool

    # Build a pool where each label appears 3 times.
    (tmp_path / "images").mkdir()
    rows = []
    for i in range(60):
        img_path = f"images/{i:08d}.png"
        Image.new("RGB", (512, 48), (i, 0, 0)).save(tmp_path / img_path)
        # 20 distinct labels, each 3 times.
        rows.append({"image_path": img_path, "label": f"Card {i % 20}"})
    with (tmp_path / "labels.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    pool = NamePool.load(tmp_path)
    train, val = split_pool(pool, val_fraction=0.2, seed=0)
    train_labels = {e.label for e in train.entries}
    val_labels = {e.label for e in val.entries}
    assert not (train_labels & val_labels), "labels leaked across split"
    # ~20% of the label set should land in val. Allow generous slack on a tiny pool.
    assert 1 <= len(val_labels) <= 10


def test_build_tf_dataset_yields_correct_shapes(tmp_path: Path) -> None:
    """End-to-end: pool → tf.data → batch shapes match what the model expects."""
    from moxify_ocr.data.name_dataset import build_tf_dataset

    _write_fake_pool(tmp_path, n=8)
    pool = NamePool.load(tmp_path)

    def fake_encode(label: str) -> list[int]:
        return [ord(c) % 50 + 1 for c in label]

    ds = build_tf_dataset(
        pool, encode_fn=fake_encode, batch_size=4, shuffle_buffer=0
    )
    images, labels = next(iter(ds))
    assert images.shape == (4, 48, 512, 3)
    assert images.dtype.name == "uint8"
    assert labels.shape[0] == 4
    assert labels.shape[1] >= 1  # padded to max label length
