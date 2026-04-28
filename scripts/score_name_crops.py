"""Run name-OCR inference on a directory of pre-cropped 48x512 RGB PNGs.

This is the *offline* inference path — bypasses the on-device TFLite runtime
entirely and runs the trained Keras model on the host. The Moxify scan-loop
harness can save the 48x512 crops it would have fed to the on-device model,
ship them to the laptop, and pipe them through this script to recover the
real-world CER number that the runtime conflict between LiteRT 1.4.1 and
select-tf-ops:2.16.1 currently blocks.

Usage::

    .venv/bin/python scripts/score_name_crops.py \\
        --keras artifacts/name_v1/best.keras \\
        --input docs/perf-runs/<stamp>/predictions.jsonl \\
        --crops-root docs/perf-runs/<stamp>/crops/ \\
        --out docs/perf-runs/<stamp>/predictions_scored.jsonl

The ``--input`` JSONL must have one row per crop with at minimum
``{image_path, true_name}`` (the field name for ground truth can also be
``label`` or ``name``; we try those in turn). ``image_path`` is resolved
against ``--crops-root`` if relative, used as-is if absolute.

Output is one JSONL row per crop with the original fields plus
``pred_name`` (the greedy-decoded prediction) and a summary line printed
to stdout: overall CER, exact-match accuracy, per-layout breakdown if a
``layout`` field is present in the input.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import editdistance
import numpy as np
import tensorflow as tf
from PIL import Image

from moxify_ocr.data.dataset import BLANK_INDEX, decode_label
from moxify_ocr.data.name_alphabet import NAME_ALPHABET
from moxify_ocr.models.crnn import SqueezeHeight  # noqa: F401  used at load time

_TRUTH_FIELDS: tuple[str, ...] = ("true_name", "label", "name")


def _greedy_decode_batch(logits: tf.Tensor) -> list[str]:
    """Greedy CTC decode on ``[B, T, C]`` logits, returning predicted strings."""
    logits_tm = tf.transpose(logits, perm=[1, 0, 2])
    batch_size = int(tf.shape(logits)[0].numpy())
    time_steps = int(tf.shape(logits)[1].numpy())
    seq_len = tf.fill([batch_size], time_steps)
    decoded_sparse, _ = tf.nn.ctc_greedy_decoder(
        inputs=logits_tm,
        sequence_length=seq_len,
        merge_repeated=True,
        blank_index=BLANK_INDEX,
    )
    dense = tf.sparse.to_dense(decoded_sparse[0], default_value=-1).numpy()
    out: list[str] = []
    for row in dense:
        ids = [int(v) for v in row if int(v) >= 1]
        out.append(decode_label(ids, alphabet=NAME_ALPHABET))
    return out


def _truth_of(row: dict) -> str | None:
    for f in _TRUTH_FIELDS:
        v = row.get(f)
        if isinstance(v, str):
            return v
    return None


def _resolve_path(root: Path, image_path: str) -> Path:
    p = Path(image_path)
    return p if p.is_absolute() else root / p


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--keras", type=Path, required=True)
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="JSONL with image_path + true_name (or label / name) per row.",
    )
    parser.add_argument(
        "--crops-root",
        type=Path,
        default=Path("."),
        help="Resolves relative image_path values; defaults to cwd.",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    if not args.keras.exists():
        raise SystemExit(f"missing keras model: {args.keras}")
    if not args.input.exists():
        raise SystemExit(f"missing input JSONL: {args.input}")

    print(f"loading {args.keras} ...")
    model = tf.keras.models.load_model(
        args.keras, custom_objects={"SqueezeHeight": SqueezeHeight}, compile=False
    )

    rows = [
        json.loads(line)
        for line in args.input.read_text().splitlines()
        if line.strip()
    ]
    if not rows:
        raise SystemExit(f"no rows in {args.input}")
    print(f"  {len(rows):,} crops to score")

    scored: list[dict] = []
    for start in range(0, len(rows), args.batch_size):
        batch = rows[start : start + args.batch_size]
        imgs: list[np.ndarray] = []
        for r in batch:
            p = _resolve_path(args.crops_root, r["image_path"])
            arr = np.asarray(Image.open(p).convert("RGB"), dtype=np.uint8)
            if arr.shape != (48, 512, 3):
                raise SystemExit(
                    f"bad crop shape {arr.shape} at {p} — expected (48, 512, 3)"
                )
            imgs.append(arr)
        batch_imgs = np.stack(imgs)
        logits = model(batch_imgs, training=False)
        preds = _greedy_decode_batch(logits)
        for r, pred in zip(batch, preds, strict=True):
            scored.append({**r, "pred_name": pred})
        print(f"  scored {min(start + args.batch_size, len(rows))}/{len(rows)}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        for r in scored:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {args.out}")

    # Summary.
    total_dist = 0
    total_len = 0
    exact = 0
    n_with_truth = 0
    by_layout: dict[str, list[tuple[int, int, bool]]] = defaultdict(list)
    for r in scored:
        truth = _truth_of(r)
        if truth is None:
            continue
        n_with_truth += 1
        d = int(editdistance.eval(r["pred_name"], truth))
        L = max(len(truth), 1)
        total_dist += d
        total_len += L
        is_match = r["pred_name"] == truth
        if is_match:
            exact += 1
        layout = r.get("layout", "?")
        by_layout[layout].append((d, L, is_match))

    if n_with_truth == 0:
        print("(no truth fields found — predictions written without scoring)")
        return 0

    cer = total_dist / total_len
    em = exact / n_with_truth
    print()
    print(f"overall CER:      {cer:.4f}  ({total_dist}/{total_len} chars)")
    print(f"overall match:    {exact}/{n_with_truth}  ({em:.1%})")
    if len(by_layout) > 1:
        print()
        print("per-layout:")
        for layout in sorted(by_layout):
            entries = by_layout[layout]
            d = sum(e[0] for e in entries)
            L = sum(e[1] for e in entries)
            m = sum(1 for e in entries if e[2])
            print(
                f"  {layout:20s} n={len(entries):4d}  CER={d/max(L,1):.4f}  "
                f"exact={m}/{len(entries)} ({m/max(len(entries),1):.1%})"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
