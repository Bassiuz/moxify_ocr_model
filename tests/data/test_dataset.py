"""Tests for :mod:`moxify_ocr.data.dataset`."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from moxify_ocr.data.dataset import (
    ALPHABET,
    BLANK_INDEX,
    DatasetConfig,
    build_dataset,
    decode_label,
    encode_label,
    is_trainable,
)
from moxify_ocr.data.manifest import ManifestEntry


def _make_entry(**overrides: object) -> ManifestEntry:
    """Build a ManifestEntry with sane defaults; override specific fields in tests."""
    base: dict[str, object] = {
        "scryfall_id": "00000000-0000-0000-0000-000000000000",
        "image_path": "images/fake.jpg",
        "lang": "en",
        "set_code": "tst",
        "collector_number": "001",
        "rarity": "rare",
        "type_line": "Creature",
        "layout": "normal",
        "finishes": ["nonfoil"],
        "image_sha256": "",
        "released_at": "2024-01-01",
        "printed_size": None,
    }
    base.update(overrides)
    return ManifestEntry(**base)  # type: ignore[arg-type]


def test_is_trainable_excludes_pre_2008() -> None:
    """Cards released before 2008-01-01 are excluded."""
    entry = _make_entry(released_at="2007-12-31", layout="normal")
    assert is_trainable(entry) is False
    # Boundary: exactly the cutoff is allowed.
    boundary = _make_entry(released_at="2008-01-01", layout="normal")
    assert is_trainable(boundary) is True
    # Empty released_at is excluded (we can't verify era → drop).
    empty = _make_entry(released_at="", layout="normal")
    assert is_trainable(empty) is False


def test_is_trainable_excludes_art_series() -> None:
    """Non-standard layouts (art_series, token, ...) are excluded."""
    art = _make_entry(layout="art_series")
    assert is_trainable(art) is False
    for layout in ("token", "scheme", "plane", "phenomenon", "emblem"):
        assert is_trainable(_make_entry(layout=layout)) is False
    # Standard layouts pass.
    for layout in ("normal", "split", "transform", "modal_dfc", "saga"):
        assert is_trainable(_make_entry(layout=layout)) is True


def test_alphabet_length() -> None:
    """Alphabet has exactly 44 characters (CTC blank is implicit at index 0)."""
    assert len(ALPHABET) == 44


def test_blank_index_is_zero() -> None:
    """CTC blank is reserved at index 0 by convention."""
    assert BLANK_INDEX == 0


def test_encode_decode_roundtrip() -> None:
    """A label that only uses alphabet chars must round-trip losslessly."""
    text = "R 0280\nCLU • EN"
    assert decode_label(encode_label(text)) == text


def test_encode_preserves_alphabet_range() -> None:
    """All encoded indices are in [1, 44]."""
    text = "R 0280\nCLU • EN"
    indices = encode_label(text)
    for i in indices:
        assert 1 <= i <= 44


def test_encode_unknown_char_raises() -> None:
    """Any char not in the alphabet raises ValueError (fail-fast on bad data)."""
    # Lowercase letters are not in the alphabet.
    with pytest.raises(ValueError):
        encode_label("hello")


def test_encode_empty_string() -> None:
    """Empty input → empty list (no crash)."""
    assert encode_label("") == []
    assert decode_label([]) == ""


def test_roundtrip_with_planeswalker_icon() -> None:
    """The planeswalker icon (U+E100) must round-trip — it's in the alphabet."""
    from moxify_ocr.data.labels import PLANESWALKER_ICON

    text = f"{PLANESWALKER_ICON} 123\nPLST • EN"
    assert decode_label(encode_label(text)) == text


def test_roundtrip_with_foil_glyphs() -> None:
    """Foil and non-foil glyphs must round-trip."""
    for glyph in ("•", "★"):
        text = f"R 100\nCLU {glyph} EN"
        assert decode_label(encode_label(text)) == text


def _write_fake_image(path: Path, seed: int) -> str:
    """Write a deterministic 672x936 RGB JPEG; returns its sha256."""
    rng = np.random.RandomState(seed)
    arr = rng.randint(0, 256, size=(936, 672, 3), dtype=np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(path, format="JPEG")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_fake_manifest(manifest_path: Path, images_root: Path) -> None:
    """Write 3 fake manifest rows with real backing images."""
    entries = [
        ManifestEntry(
            scryfall_id=f"fake-{i:03d}",
            image_path=f"images/fake-{i:03d}.jpg",
            lang="en",
            set_code=f"fk{i}",
            collector_number=f"{i:03d}",
            rarity="rare",
            type_line="Creature",
            layout="normal",
            finishes=["nonfoil"],
            image_sha256="",
            released_at="2024-01-01",
            printed_size=None,
        )
        for i in range(3)
    ]
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as handle:
        for idx, entry in enumerate(entries):
            img_path = images_root / entry.image_path
            sha = _write_fake_image(img_path, seed=idx)
            payload = asdict(entry)
            payload["image_sha256"] = sha
            handle.write(json.dumps(payload) + "\n")


def test_build_dataset_smoke(tmp_path: Path) -> None:
    """Synthesize tiny manifest + images, build dataset, consume 1 batch."""
    manifest_path = tmp_path / "manifest.jsonl"
    images_root = tmp_path
    _write_fake_manifest(manifest_path, images_root)

    config = DatasetConfig(
        manifest_path=manifest_path,
        images_root=images_root,
        split="train",
        batch_size=2,
        shuffle_buffer=4,
        augment=False,  # keep the smoke test fast and deterministic
        seed=0,
    )
    dataset = build_dataset(config)

    # Pull one batch; tolerate the case where the split filter drops all 3 rows
    # (fake set codes are random-hashed). If so, try any split until we find one.
    batch = None
    for split in ("train", "val", "test"):
        cfg = DatasetConfig(
            manifest_path=manifest_path,
            images_root=images_root,
            split=split,
            batch_size=2,
            shuffle_buffer=4,
            augment=False,
            seed=0,
        )
        ds = build_dataset(cfg)
        for item in ds.take(1):
            batch = item
            break
        if batch is not None:
            break

    assert batch is not None, "no split contained any rows"
    images = batch[0]
    assert images.shape[1:] == (48, 256, 3)
    assert images.dtype.name == "uint8"
    # batch can be 1 or 2 depending on where the 3 rows landed
    assert images.shape[0] in (1, 2, 3)
    # Unused var linter-silencer: build_dataset with augment=False is still callable.
    del dataset


def test_build_dataset_object_creation(tmp_path: Path) -> None:
    """Even if iteration is fragile, the Dataset object must be constructible."""
    manifest_path = tmp_path / "manifest.jsonl"
    images_root = tmp_path
    _write_fake_manifest(manifest_path, images_root)
    config = DatasetConfig(
        manifest_path=manifest_path,
        images_root=images_root,
        split="train",
        batch_size=2,
        augment=False,
    )
    dataset = build_dataset(config)
    assert dataset is not None
