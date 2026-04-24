"""Tests for :mod:`moxify_ocr.eval.eval`.

Each test builds a tiny synthetic manifest with fake 48×256 RGB JPEGs and an
untrained-but-initialized checkpoint in ``tmp_path`` — we never touch real
training artifacts because training may be running concurrently.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import string
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf
from PIL import Image

from moxify_ocr.data.manifest import ManifestEntry
from moxify_ocr.data.splits import assign_split
from moxify_ocr.eval import eval as eval_mod
from moxify_ocr.eval.eval import evaluate, main
from moxify_ocr.models.bottom_region import build_bottom_region_model


def _write_fake_jpeg(path: Path, seed: int) -> str:
    """Write a deterministic 672×936 RGB JPEG; return sha256."""
    rng = np.random.RandomState(seed)
    arr = rng.randint(0, 256, size=(936, 672, 3), dtype=np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(path, format="JPEG")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _build_fixture(tmp_path: Path, *, n_cards: int = 3) -> tuple[Path, Path, Path, str]:
    """Build manifest + images + an untrained-weights file.

    Returns ``(manifest_path, images_root, weights_path, split_name)``, where
    ``split_name`` is the single split where every fake row landed (all rows
    share a set_code so the split-hash is identical for each).
    """
    manifest_path = tmp_path / "manifest.jsonl"
    images_root = tmp_path
    weights_path = tmp_path / "weights.keras"

    # Pick a set_code that lands in "test" split with seed=0 so evaluate(split="test")
    # actually sees rows. "aap" is the lex-first 3-letter code that hashes to
    # "test" under assign_split(seed=0); keep the deterministic probe loop so
    # if the cutoffs ever change we still discover one rather than silently
    # running on an empty split.
    target_split = "test"
    candidate_code: str | None = None
    for a, b, c in itertools.product(string.ascii_lowercase, repeat=3):
        code = f"{a}{b}{c}"
        if assign_split(code, seed=0) == target_split:
            candidate_code = code
            break
    assert candidate_code is not None, "could not find a fake set_code in test split"

    entries = [
        ManifestEntry(
            scryfall_id=f"fake-{i:03d}",
            image_path=f"images/fake-{i:03d}.jpg",
            lang="en",
            set_code=candidate_code,
            collector_number=f"{i + 1:03d}",
            rarity="rare",
            type_line="Creature",
            layout="normal",
            finishes=["nonfoil"],
            image_sha256="",
            released_at="2024-01-01",
            printed_size=None,
        )
        for i in range(n_cards)
    ]
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as handle:
        for idx, entry in enumerate(entries):
            img = images_root / entry.image_path
            sha = _write_fake_jpeg(img, seed=idx)
            payload = asdict(entry)
            payload["image_sha256"] = sha
            handle.write(json.dumps(payload) + "\n")

    # Fresh untrained weights: build the model, run one forward pass to
    # materialize variables, then save.
    model = build_bottom_region_model()
    model(np.zeros((1, 48, 256, 3), dtype=np.uint8))
    model.save(str(weights_path))

    return manifest_path, images_root, weights_path, target_split


_EXPECTED_KEYS: dict[str, type] = {
    "total_samples": int,
    "cer": float,
    "exact_match_rate": float,
    "collector_number_accuracy": float,
    "set_code_accuracy": float,
    "set_total_accuracy": float,
    "rarity_accuracy": float,
    "language_accuracy": float,
    "foil_f1": float,
    "the_list_f1": float,
    "p50_latency_ms": float,
    "p95_latency_ms": float,
}


def test_evaluate_returns_all_required_keys(tmp_path: Path) -> None:
    """evaluate() returns every expected metric key with the correct type."""
    manifest, images_root, weights, split = _build_fixture(tmp_path, n_cards=3)
    metrics = evaluate(
        model_weights=weights,
        manifest=manifest,
        images_root=images_root,
        split=split,
        limit=3,
    )
    for key, expected_type in _EXPECTED_KEYS.items():
        assert key in metrics, f"missing metric: {key}"
        # bool is a subclass of int in Python; we don't have any bool keys here.
        assert isinstance(metrics[key], expected_type), (
            f"{key} has type {type(metrics[key]).__name__}, expected {expected_type.__name__}"
        )
    assert metrics["total_samples"] >= 1


def test_cer_is_nonzero_for_untrained_model(tmp_path: Path) -> None:
    """Untrained weights + random images cannot produce perfect strings."""
    manifest, images_root, weights, split = _build_fixture(tmp_path, n_cards=3)
    metrics = evaluate(
        model_weights=weights,
        manifest=manifest,
        images_root=images_root,
        split=split,
        limit=3,
    )
    assert metrics["cer"] > 0.1, f"untrained CER unexpectedly low: {metrics['cer']}"


def test_per_field_accuracy_in_range(tmp_path: Path) -> None:
    """All accuracy and F1 metrics are in [0.0, 1.0]."""
    manifest, images_root, weights, split = _build_fixture(tmp_path, n_cards=3)
    metrics = evaluate(
        model_weights=weights,
        manifest=manifest,
        images_root=images_root,
        split=split,
        limit=3,
    )
    for key in (
        "cer",
        "exact_match_rate",
        "collector_number_accuracy",
        "set_code_accuracy",
        "set_total_accuracy",
        "rarity_accuracy",
        "language_accuracy",
        "foil_f1",
        "the_list_f1",
    ):
        value = metrics[key]
        # CER can theoretically exceed 1 if predictions are longer than truth;
        # keep a looser bound for CER and a tight bound for accuracies/F1s.
        if key == "cer":
            assert value >= 0.0, f"{key} negative: {value}"
        else:
            assert 0.0 <= value <= 1.0, f"{key} out of range: {value}"


def test_latency_reported_positive(tmp_path: Path) -> None:
    """p50 latency is > 0 and p95 is at least p50."""
    manifest, images_root, weights, split = _build_fixture(tmp_path, n_cards=3)
    metrics = evaluate(
        model_weights=weights,
        manifest=manifest,
        images_root=images_root,
        split=split,
        limit=3,
    )
    assert metrics["p50_latency_ms"] > 0.0, metrics["p50_latency_ms"]
    assert metrics["p95_latency_ms"] >= metrics["p50_latency_ms"], (
        metrics["p95_latency_ms"],
        metrics["p50_latency_ms"],
    )


def test_cli_writes_metrics_json(tmp_path: Path) -> None:
    """CLI writes a pretty-printed JSON file with all required keys."""
    manifest, images_root, weights, split = _build_fixture(tmp_path, n_cards=3)
    output = tmp_path / "metrics.json"
    rc = main(
        [
            "--weights",
            str(weights),
            "--manifest",
            str(manifest),
            "--images-root",
            str(images_root),
            "--split",
            split,
            "--limit",
            "3",
            "--output",
            str(output),
        ]
    )
    assert rc == 0
    assert output.exists()
    payload = json.loads(output.read_text(encoding="utf-8"))
    for key in _EXPECTED_KEYS:
        assert key in payload, f"missing key in CLI output: {key}"


def test_perfect_predictions_give_zero_cer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Mock the greedy decode to echo ground-truth ids → CER=0 and exact match=1."""
    manifest, images_root, weights, split = _build_fixture(tmp_path, n_cards=3)

    # Pre-iterate the dataset once to snapshot the truth id sequence in split
    # order. evaluate() will then iterate the same split in the same order
    # (augment=False, shuffle_buffer=0), so we can replay these as predictions.
    from moxify_ocr.data.dataset import BLANK_INDEX, DatasetConfig, build_dataset

    ds = build_dataset(
        DatasetConfig(
            manifest_path=manifest,
            images_root=images_root,
            split=split,
            batch_size=1,
            shuffle_buffer=0,
            augment=False,
            seed=0,
        )
    )
    truth_ids_per_sample: list[list[int]] = []
    for batch in ds:
        _, labels, _ = batch
        truth_ids_per_sample.append(
            [int(v) for v in labels.numpy()[0] if int(v) != BLANK_INDEX]
        )

    call_idx = {"i": 0}

    def oracle_greedy_decode_single(logits: tf.Tensor) -> list[int]:
        i = call_idx["i"]
        call_idx["i"] += 1
        return list(truth_ids_per_sample[i])

    monkeypatch.setattr(eval_mod, "_greedy_decode_single", oracle_greedy_decode_single)

    metrics = evaluate(
        model_weights=weights,
        manifest=manifest,
        images_root=images_root,
        split=split,
        limit=len(truth_ids_per_sample),
    )
    assert metrics["total_samples"] == len(truth_ids_per_sample)
    assert metrics["cer"] == 0.0, metrics["cer"]
    assert metrics["exact_match_rate"] == 1.0, metrics["exact_match_rate"]
