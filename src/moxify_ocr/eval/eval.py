"""Evaluation harness: per-field metrics + CER + latency on a held-out split.

Runs a trained bottom-region CRNN over val/test samples, greedy-decodes the
CTC logits, parses both prediction and ground-truth into structured fields
via :func:`moxify_ocr.export.parse_bottom.parse_bottom`, and aggregates
micro-averaged CER, per-field accuracies, F1s for foil + "The List"
detection, and forward-pass latency percentiles.

Loads via :func:`build_bottom_region_model` + ``load_weights`` to sidestep
the custom-layer deserialization quirk with ``tf.keras.models.load_model``.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import editdistance
import numpy as np
import tensorflow as tf

from moxify_ocr.data.dataset import BLANK_INDEX, DatasetConfig, build_dataset, decode_label
from moxify_ocr.data.labels import LANG_CODE, _salvage_plst_fields
from moxify_ocr.data.manifest import read_manifest
from moxify_ocr.export.parse_bottom import BottomRegionResult, parse_bottom
from moxify_ocr.models.bottom_region import build_bottom_region_model


def _greedy_decode_single(logits: tf.Tensor) -> list[int]:
    """Greedy CTC decode on a single-sample ``[1, T, C]`` logits tensor."""
    logits_tm = tf.transpose(logits, perm=[1, 0, 2])
    time_steps = int(tf.shape(logits)[1].numpy())
    decoded_sparse, _ = tf.nn.ctc_greedy_decoder(
        inputs=logits_tm,
        sequence_length=tf.fill([1], time_steps),
        merge_repeated=True,
        blank_index=BLANK_INDEX,
    )
    dense = tf.sparse.to_dense(decoded_sparse[0], default_value=-1).numpy()
    return [int(v) for v in dense[0] if int(v) >= 0]


def _derive_known_set_codes(manifest_path: Path) -> set[str]:
    """Collect uppercase set codes printed on cards — salvages PLST originals."""
    codes: set[str] = set()
    for entry in read_manifest(manifest_path):
        card = {"set": entry.set_code, "collector_number": entry.collector_number}
        set_code, _num, _is_list = _salvage_plst_fields(card)
        codes.add(set_code.upper())
    return codes


def _f1(tp: int, fp: int, fn: int) -> float:
    """Binary F1 with 0.0 fallback when precision or recall is undefined."""
    if tp + fp == 0 or tp + fn == 0:
        return 0.0
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    if precision + recall == 0.0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def _acc(hits: int, totals: int) -> float:
    return hits / totals if totals > 0 else 0.0


def evaluate(
    *,
    model_weights: Path,
    manifest: Path,
    images_root: Path,
    split: str = "test",
    min_release: str = "2008-01-01",
    batch_size: int = 32,
    seed: int = 0,
    known_set_codes: set[str] | None = None,
    known_languages: set[str] | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    """Run the trained model on split samples, return a metrics dict."""
    del batch_size  # we iterate per-sample for clean latency timing
    if known_set_codes is None:
        known_set_codes = _derive_known_set_codes(manifest)
    if known_languages is None:
        known_languages = set(LANG_CODE.values())

    dataset = build_dataset(
        DatasetConfig(
            manifest_path=manifest,
            images_root=images_root,
            split=split,
            batch_size=1,
            shuffle_buffer=0,
            augment=False,
            seed=seed,
            min_release=min_release,
        )
    )

    model = build_bottom_region_model()
    model.load_weights(str(model_weights))

    total_distance = total_length = exact_matches = total_samples = 0
    # Field-level accuracy counters: (hits, totals) per field.
    acc: dict[str, list[int]] = {
        "collector_number": [0, 0],
        "set_code": [0, 0],
        "language": [0, 0],
        "set_total": [0, 0],
        "rarity": [0, 0],
    }
    # Binary F1 counters: (tp, fp, fn) for each positive class.
    foil = [0, 0, 0]
    the_list = [0, 0, 0]
    latencies_ms: list[float] = []

    def _update_f1(counters: list[int], truth: bool, pred: bool) -> None:
        if truth and pred:
            counters[0] += 1
        elif pred and not truth:
            counters[1] += 1
        elif truth and not pred:
            counters[2] += 1

    def _update_acc(key: str, truth: Any, pred: Any) -> None:
        if truth is None:
            return
        acc[key][1] += 1
        if truth == pred:
            acc[key][0] += 1

    def _parse(text: str) -> BottomRegionResult:
        return parse_bottom(
            text, known_set_codes=known_set_codes, known_languages=known_languages
        )

    for batch in dataset:
        images, labels, _label_length = batch
        start = time.perf_counter()
        logits = model(images, training=False)
        latencies_ms.append((time.perf_counter() - start) * 1000.0)

        pred_str = decode_label(_greedy_decode_single(logits))
        true_ids = [int(v) for v in labels.numpy()[0] if int(v) != BLANK_INDEX]
        true_str = decode_label(true_ids)

        total_distance += int(editdistance.eval(pred_str, true_str))
        total_length += max(len(true_str), 1)
        if pred_str == true_str:
            exact_matches += 1

        truth = _parse(true_str)
        pred = _parse(pred_str)
        for key in acc:
            _update_acc(key, getattr(truth, key), getattr(pred, key))
        _update_f1(foil, truth.foil_detected is True, pred.foil_detected is True)
        _update_f1(the_list, truth.on_the_list_detected, pred.on_the_list_detected)

        total_samples += 1
        if limit is not None and total_samples >= limit:
            break

    if latencies_ms:
        p50 = float(np.percentile(latencies_ms, 50))
        p95 = float(np.percentile(latencies_ms, 95))
    else:
        p50 = p95 = 0.0

    return {
        "total_samples": total_samples,
        "cer": total_distance / total_length if total_length > 0 else 0.0,
        "exact_match_rate": _acc(exact_matches, total_samples),
        "collector_number_accuracy": _acc(*acc["collector_number"]),
        "set_code_accuracy": _acc(*acc["set_code"]),
        "set_total_accuracy": _acc(*acc["set_total"]),
        "rarity_accuracy": _acc(*acc["rarity"]),
        "language_accuracy": _acc(*acc["language"]),
        "foil_f1": _f1(*foil),
        "the_list_f1": _f1(*the_list),
        "p50_latency_ms": p50,
        "p95_latency_ms": p95,
    }


def main(argv: list[str] | None = None) -> int:
    """CLI: ``python -m moxify_ocr.eval.eval --weights ... --manifest ...``."""
    parser = argparse.ArgumentParser(
        description="Evaluate a trained bottom-region OCR model on a held-out split."
    )
    parser.add_argument("--weights", type=Path, required=True, help="Path to .keras weights.")
    parser.add_argument("--manifest", type=Path, required=True, help="Path to manifest.jsonl.")
    parser.add_argument("--images-root", type=Path, required=True, help="Images root dir.")
    parser.add_argument("--split", type=str, default="test", help="train|val|test (default: test).")
    parser.add_argument("--min-release", type=str, default="2008-01-01")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None, help="Cap at N samples.")
    parser.add_argument(
        "--output", type=Path, default=None, help="Write metrics JSON here; defaults to stdout."
    )
    args = parser.parse_args(argv)

    metrics = evaluate(
        model_weights=args.weights,
        manifest=args.manifest,
        images_root=args.images_root,
        split=args.split,
        min_release=args.min_release,
        batch_size=args.batch_size,
        seed=args.seed,
        limit=args.limit,
    )

    pretty = json.dumps(metrics, indent=2, sort_keys=True)
    if args.output is None:
        print(pretty)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(pretty + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
