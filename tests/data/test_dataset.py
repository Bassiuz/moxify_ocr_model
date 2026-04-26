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


def _write_fake_cardconjurer_pool(root: Path, n: int = 5) -> None:
    """Write a tiny CardConjurer pool. Sentinel pixel: pure red (255, 0, 0).

    Real Scryfall crops + the (114, 114, 114) gray letterbox never produce
    pure-red pixels, so every cc-pool sample is identifiable.
    """
    import json as _json

    (root / "images").mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(n):
        img_path = f"images/cc-{i:08d}.png"
        # Pure red — sentinel that the dataset path actually drew from the pool.
        Image.new("RGB", (256, 48), (255, 0, 0)).save(root / img_path)
        rows.append({"image_path": img_path, "label": f"00{i}/100 R\nMID • EN"})
    with (root / "labels.jsonl").open("w") as f:
        for r in rows:
            f.write(_json.dumps(r) + "\n")


def test_build_dataset_raises_when_cc_ratio_set_but_no_pool(tmp_path: Path) -> None:
    """A nonzero cardconjurer_ratio without a pool path is a config bug — fail loud.

    Silent skip would have made training fall through to real-only data with
    plausible-looking loss curves.
    """
    manifest_path = tmp_path / "manifest.jsonl"
    images_root = tmp_path
    _write_fake_manifest(manifest_path, images_root)

    config = DatasetConfig(
        manifest_path=manifest_path,
        images_root=images_root,
        split="train",
        batch_size=2,
        shuffle_buffer=0,
        augment=False,
        seed=0,
        cardconjurer_ratio=0.5,
        cardconjurer_pool=None,  # <- the bug we're guarding against
    )
    with pytest.raises(ValueError, match="cardconjurer_pool"):
        build_dataset(config)


def test_build_dataset_cardconjurer_ratio_one_yields_only_pool(tmp_path: Path) -> None:
    """With cardconjurer_ratio=1.0 on the train split, every yielded sample
    comes from the cc pool.

    Uses pure-red sentinel pixels in the fake pool — the production crop +
    (114, 114, 114) gray letterbox path can never produce those.
    """
    manifest_root = tmp_path / "manifest_root"
    manifest_root.mkdir()
    manifest_path = manifest_root / "manifest.jsonl"
    # Bigger manifest → at least one row reliably hashes into train.
    entries = [
        ManifestEntry(
            scryfall_id=f"fake-{i:03d}",
            image_path=f"images/fake-{i:03d}.jpg",
            lang="en",
            set_code=f"fk{i:02d}",
            collector_number=f"{i:03d}",
            rarity="rare",
            type_line="Creature",
            layout="normal",
            finishes=["nonfoil"],
            image_sha256="",
            released_at="2024-01-01",
            printed_size=None,
        )
        for i in range(20)
    ]
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as handle:
        for idx, entry in enumerate(entries):
            img_path = manifest_root / entry.image_path
            sha = _write_fake_image(img_path, seed=idx)
            payload = asdict(entry)
            payload["image_sha256"] = sha
            handle.write(json.dumps(payload) + "\n")

    pool_root = tmp_path / "ccpool"
    pool_root.mkdir()
    _write_fake_cardconjurer_pool(pool_root, n=4)

    cfg = DatasetConfig(
        manifest_path=manifest_path,
        images_root=manifest_root,
        split="train",
        batch_size=1,
        shuffle_buffer=0,
        augment=False,
        seed=0,
        cardconjurer_pool=pool_root,
        cardconjurer_ratio=1.0,
    )
    ds = build_dataset(cfg)
    sampled = 0
    for batch in ds.take(5):
        images = batch[0].numpy()
        for image in images:
            assert image.shape == (48, 256, 3)
            # Entire frame is (255, 0, 0) — mean R == 255, mean G == B == 0.
            assert image[..., 0].mean() == 255, (
                f"expected pure-red R channel, got mean={image[..., 0].mean()}"
            )
            assert image[..., 1].mean() == 0
            assert image[..., 2].mean() == 0
            sampled += 1
    assert sampled >= 1, "dataset yielded zero batches — train split was empty"


def test_v3_config_yields_only_synthetic_train_samples(tmp_path: Path) -> None:
    """v3 invariant: cc_ratio + lc_ratio == 1.0 → train sees zero real-data samples.

    Loads the actual v3 ratios from ``configs/bottom_region_v3.yaml`` and
    builds a train dataset. cc-pool images carry pure-red sentinels — the
    real-data leg's crop_bottom_region path applied to the fake manifest
    images here cannot produce all-red frames, and neither can lc's stitcher
    (which composites halves of those same fake-random images). Count cc
    sentinels vs non-cc samples, then assert cc + non-cc == total — i.e.,
    every yielded sample is accounted for by a synthetic leg.

    Real-data leakage would surface as samples that are byte-identical to
    crop_bottom_region(manifest_image) for some manifest row. We detect this
    by checking that no yielded image matches any real-data crop.
    """
    import yaml as _yaml

    from moxify_ocr.data.crop import crop_bottom_region

    repo_root = Path(__file__).resolve().parents[2]
    v3_yaml = repo_root / "configs" / "bottom_region_v3.yaml"
    v3_cfg = _yaml.safe_load(v3_yaml.read_text())["data"]
    v3_cc_ratio = v3_cfg["cardconjurer_ratio"]
    v3_lc_ratio = v3_cfg["line_compositor_ratio"]
    # Pre-flight: v3 promises cc+lc == 1.0. If someone bumps these out of sync,
    # this test (and the v3 training run) need re-thinking — fail loud.
    assert v3_cc_ratio + v3_lc_ratio == 1.0, (
        f"v3 config promises cc+lc=1.0; got {v3_cc_ratio} + {v3_lc_ratio}"
    )

    manifest_root = tmp_path / "manifest_root"
    manifest_root.mkdir()
    manifest_path = manifest_root / "manifest.jsonl"
    entries = [
        ManifestEntry(
            scryfall_id=f"fake-{i:03d}",
            image_path=f"images/fake-{i:03d}.jpg",
            lang="en",
            set_code=f"fk{i:02d}",
            collector_number=f"{i:03d}",
            rarity="rare",
            type_line="Creature",
            layout="normal",
            finishes=["nonfoil"],
            image_sha256="",
            released_at="2024-01-01",
            printed_size=None,
        )
        for i in range(20)
    ]
    real_crops: list[bytes] = []
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as handle:
        for idx, entry in enumerate(entries):
            img_path = manifest_root / entry.image_path
            sha = _write_fake_image(img_path, seed=idx)
            payload = asdict(entry)
            payload["image_sha256"] = sha
            handle.write(json.dumps(payload) + "\n")
            # Capture the deterministic real-data crop bytes — any sample
            # matching one of these would prove the real-data leg fired.
            with Image.open(img_path) as raw:
                cropped = crop_bottom_region(raw.convert("RGB"), target_size=(256, 48))
            real_crops.append(np.asarray(cropped, dtype=np.uint8).tobytes())
    real_crop_set = set(real_crops)

    pool_root = tmp_path / "ccpool"
    pool_root.mkdir()
    _write_fake_cardconjurer_pool(pool_root, n=4)

    cfg = DatasetConfig(
        manifest_path=manifest_path,
        images_root=manifest_root,
        split="train",
        batch_size=1,
        shuffle_buffer=0,
        augment=False,  # disable so real-data crops stay byte-identical to our captured set
        seed=0,
        cardconjurer_pool=pool_root,
        cardconjurer_ratio=v3_cc_ratio,
        line_compositor_ratio=v3_lc_ratio,
    )
    ds = build_dataset(cfg)

    # Sample 200 train items. .repeat() on the train split makes this safe
    # even for our 20-row fake manifest.
    target_n = 200
    cc_count = 0
    lc_count = 0
    real_count = 0
    seen = 0
    for batch in ds.take(target_n):
        for image in batch[0].numpy():
            assert image.shape == (48, 256, 3)
            is_cc_sentinel = (
                image[..., 0].mean() == 255
                and image[..., 1].mean() == 0
                and image[..., 2].mean() == 0
            )
            if is_cc_sentinel:
                cc_count += 1
            elif image.tobytes() in real_crop_set:
                real_count += 1
            else:
                lc_count += 1
            seen += 1
            if seen >= target_n:
                break
        if seen >= target_n:
            break

    assert seen == target_n, f"expected {target_n} samples, got {seen}"
    assert real_count == 0, (
        f"v3 config promises synthetic-only train samples but {real_count}/"
        f"{target_n} matched a real-data crop — real-data leg leaked through"
    )
    assert cc_count + lc_count == target_n
    # Sanity: with cc_ratio=0.7, lc_ratio=0.3 we expect both legs to fire.
    assert cc_count > 0, "cc leg never fired — routing or pool wiring broken"
    assert lc_count > 0, "lc leg never fired — routing or library wiring broken"


def test_build_datasets_forwards_v3_yaml_ratios(tmp_path: Path) -> None:
    """Trainer's ``_build_datasets`` must forward cc + lc fields from yaml.

    Regression test for the v3 wiring bug where ``configs/bottom_region_v3.yaml``
    used ``manifest_path:`` (mismatching the trainer's read key ``manifest``)
    and ``_build_datasets`` only forwarded ``synthetic_ratio`` — cc + lc
    ratios were silently dropped, so v3 training would fall back to v1
    real-only behavior with no error.

    We materialize a tiny v3-shape yaml on disk, call ``_build_datasets``
    on it, and inspect the underlying ``DatasetConfig`` produced for the
    train split to confirm the cc/lc ratios survived the round-trip.
    """
    import yaml as _yaml

    from moxify_ocr.train.train import _build_datasets

    manifest_root = tmp_path / "manifest_root"
    manifest_root.mkdir()
    manifest_path = manifest_root / "manifest.jsonl"
    _write_fake_manifest(manifest_path, manifest_root)

    pool_root = tmp_path / "ccpool"
    pool_root.mkdir()
    _write_fake_cardconjurer_pool(pool_root, n=4)

    # v3-shape yaml: ``manifest:`` (not ``manifest_path:``), cc + lc ratios.
    cfg_path = tmp_path / "v3.yaml"
    cfg_path.write_text(
        _yaml.safe_dump(
            {
                "data": {
                    "manifest": str(manifest_path),
                    "images_root": str(manifest_root),
                    "batch_size": 2,
                    "shuffle_buffer": 0,
                    "min_release": "2008-01-01",
                    "cardconjurer_pool": str(pool_root),
                    "cardconjurer_ratio": 0.7,
                    "line_compositor_ratio": 0.3,
                },
                "model": {
                    "input_height": 48,
                    "input_width": 256,
                    "num_classes": 45,
                    "lstm_units": 256,
                },
                "train": {
                    "epochs": 1,
                    "lr": 5.0e-4,
                    "warmup_steps": 100,
                    "seed": 0,
                    "output_dir": str(tmp_path / "out"),
                },
            }
        )
    )

    # Capture the train DatasetConfig that _build_datasets constructs by
    # patching DatasetConfig in the train module to record its kwargs.
    captured: list[DatasetConfig] = []
    import moxify_ocr.train.train as train_mod

    original_dc = train_mod.DatasetConfig

    def _spy(*args: object, **kwargs: object) -> DatasetConfig:
        instance = original_dc(*args, **kwargs)  # type: ignore[arg-type]
        captured.append(instance)
        return instance

    train_mod.DatasetConfig = _spy  # type: ignore[assignment]
    try:
        cfg = _yaml.safe_load(cfg_path.read_text())
        _build_datasets(cfg)
    finally:
        train_mod.DatasetConfig = original_dc  # type: ignore[assignment]

    # Two DatasetConfig instances were built: train then val.
    assert len(captured) == 2, f"expected 2 DatasetConfigs, got {len(captured)}"
    train_dc, val_dc = captured

    # Train: cc/lc ratios must survive yaml -> _build_datasets -> DatasetConfig.
    assert train_dc.split == "train"
    assert train_dc.cardconjurer_ratio == 0.7
    assert train_dc.line_compositor_ratio == 0.3
    assert train_dc.cardconjurer_pool == Path(str(pool_root))
    # manifest:-key read worked (would KeyError otherwise).
    assert train_dc.manifest_path == Path(str(manifest_path))

    # Val: must NOT have cc/lc ratios — must stay 100% real Scryfall.
    assert val_dc.split == "val"
    assert val_dc.cardconjurer_ratio == 0.0
    assert val_dc.line_compositor_ratio == 0.0


def test_dataset_routes_to_line_compositor_when_ratio_is_one(tmp_path: Path) -> None:
    """With line_compositor_ratio=1.0 every sample must come from the stitcher.

    Counter-test against the cardconjurer leg: we wire up BOTH a cc pool (with
    pure-red sentinel images) AND set line_compositor_ratio=1.0. Because the lc
    branch is checked first and consumes the random draw, no cc-pool sentinel
    sample should leak through. Real Scryfall halves stitched by line_compositor
    cannot produce pure-red frames, so any (255, 0, 0) image proves the cc leg
    fired — which would be a routing bug.
    """
    manifest_root = tmp_path / "manifest_root"
    manifest_root.mkdir()
    manifest_path = manifest_root / "manifest.jsonl"
    # Bigger manifest → at least one row reliably hashes into train.
    entries = [
        ManifestEntry(
            scryfall_id=f"fake-{i:03d}",
            image_path=f"images/fake-{i:03d}.jpg",
            lang="en",
            set_code=f"fk{i:02d}",
            collector_number=f"{i:03d}",
            rarity="rare",
            type_line="Creature",
            layout="normal",
            finishes=["nonfoil"],
            image_sha256="",
            released_at="2024-01-01",
            printed_size=None,
        )
        for i in range(20)
    ]
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as handle:
        for idx, entry in enumerate(entries):
            img_path = manifest_root / entry.image_path
            sha = _write_fake_image(img_path, seed=idx)
            payload = asdict(entry)
            payload["image_sha256"] = sha
            handle.write(json.dumps(payload) + "\n")

    pool_root = tmp_path / "ccpool"
    pool_root.mkdir()
    _write_fake_cardconjurer_pool(pool_root, n=4)

    cfg = DatasetConfig(
        manifest_path=manifest_path,
        images_root=manifest_root,
        split="train",
        batch_size=1,
        shuffle_buffer=0,
        augment=False,
        seed=0,
        cardconjurer_pool=pool_root,
        cardconjurer_ratio=1.0,
        line_compositor_ratio=1.0,
    )
    ds = build_dataset(cfg)
    sampled = 0
    for batch in ds.take(5):
        images = batch[0].numpy()
        for image in images:
            assert image.shape == (48, 256, 3)
            # If lc fires first and consumes the rng draw, no cc red sentinel
            # should ever appear — any all-red frame proves wrong routing.
            is_red_sentinel = (
                image[..., 0].mean() == 255
                and image[..., 1].mean() == 0
                and image[..., 2].mean() == 0
            )
            assert not is_red_sentinel, (
                "cardconjurer-pool sentinel image leaked through with "
                "line_compositor_ratio=1.0 — lc branch did not take precedence"
            )
            sampled += 1
    assert sampled >= 1, "dataset yielded zero batches — train split was empty"
