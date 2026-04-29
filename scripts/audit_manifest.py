"""Manifest-vs-card audit tool: surface likely Scryfall metadata errors.

Hypothesis: a non-trivial fraction of v3 model "errors" are actually manifest
errors — the printed set code on the card disagrees with what Scryfall stores.
PLST cards already have a salvage path (see ``_salvage_plst_fields``); this
tool finds the others.

Strategy: run inference on the held-out test split. Where the model
**confidently** disagrees with the manifest, the card is more likely a
manifest error than an OCR error (a confident wrong prediction means the
model is reading characters cleanly; the disagreement is then on the
ground truth).

The script writes ``artifacts/manifest_audit/REVIEW.md`` plus per-card
bottom-region crops at human-readable resolution. The user reviews the
markdown, marks each card with a verdict (``M`` = manifest wrong, ``W`` =
model wrong, ``A`` = ambiguous), and ``ingest_audit.py`` (companion script)
computes the manifest-noise rate.

Buckets:
- 30 highest-confidence set-code disagreements ("likely manifest errors")
- 20 random set-code disagreements ("baseline of model errors")
- 10 random agreements ("control — should all be 'M' if anything")
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image

from moxify_ocr.data.crop import crop_bottom_region
from moxify_ocr.data.dataset import BLANK_INDEX, decode_label
from moxify_ocr.data.labels import LANG_CODE, _salvage_plst_fields
from moxify_ocr.data.manifest import read_manifest
from moxify_ocr.data.splits import assign_split
from moxify_ocr.eval.eval import _derive_known_set_codes
from moxify_ocr.export.parse_bottom import parse_bottom
from moxify_ocr.models.bottom_region import build_bottom_region_model

OUT_DIR = Path("artifacts/manifest_audit")


@dataclass
class Candidate:
    index: int
    scryfall_id: str
    image_path: str  # relative to images_root
    manifest_set: str
    manifest_number: str
    manifest_rarity: str
    manifest_lang: str
    predicted_set: str | None
    predicted_number: str | None
    predicted_rarity: str | None
    predicted_lang: str | None
    raw_pred_text: str
    confidence: float
    bucket: str = ""  # "confident-disagree" | "random-disagree" | "random-agree"
    crop_filename: str = ""


def _greedy_decode(logits: tf.Tensor) -> tuple[list[int], float]:
    """CTC greedy decode + per-card confidence (mean max-prob over emitted tokens).

    Confidence is the average per-timestep softmax-max ACROSS THE WHOLE SEQUENCE
    (including blanks). For a model that emits very confidently, this is close
    to 1.0; for a model that hedges, it drops. Cheap and good enough to rank
    candidates by "how sure is the model".
    """
    probs = tf.nn.softmax(logits, axis=-1).numpy()[0]
    max_probs = probs.max(axis=-1)
    confidence = float(max_probs.mean())
    ids = probs.argmax(axis=-1).tolist()
    out, prev = [], -1
    for i in ids:
        if i != prev and i != BLANK_INDEX:
            out.append(int(i))
        prev = i
    return out, confidence


def _normalize_set(s: str) -> str:
    return s.upper().strip()


def _manifest_truth_set(row: dict) -> str:
    """The 'set' as it should appear printed on the card.

    For PLST reprints, Scryfall stores set='plst' but the printed code is the
    parent set (extracted from collector_number). We use the parent for
    comparison since that's what the OCR reads.
    """
    if row.get("set_code", "").lower() == "plst":
        salvaged_set, _, _ = _salvage_plst_fields({
            "collector_number": row["collector_number"],
            "set": row["set_code"],
            "lang": row["lang"],
        })
        return _normalize_set(salvaged_set)
    return _normalize_set(row["set_code"])


def _save_high_res_crop(images_root: Path, image_path: str, out_path: Path) -> None:
    """Save the bottom-region crop at human-readable resolution."""
    src = Image.open(images_root / image_path).convert("RGB")
    cropped = crop_bottom_region(src, target_size=(960, 180))
    cropped.save(out_path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=Path("data/scryfall/manifest.jsonl"))
    parser.add_argument("--images-root", type=Path, default=Path("data/scryfall"))
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--n-confident", type=int, default=30)
    parser.add_argument("--n-random-disagree", type=int, default=20)
    parser.add_argument("--n-random-agree", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args(argv)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"loading model from {args.weights} ...", flush=True)
    model = build_bottom_region_model()
    model.load_weights(str(args.weights))

    print("scanning manifest ...", flush=True)
    rows: list[dict] = []
    for entry in read_manifest(args.manifest):
        if assign_split(entry.set_code) != args.split:
            continue
        rows.append({
            "scryfall_id": entry.scryfall_id,
            "image_path": entry.image_path,
            "set_code": entry.set_code,
            "collector_number": entry.collector_number,
            "rarity": entry.rarity,
            "lang": entry.lang,
        })
    print(f"  {len(rows)} cards in '{args.split}' split", flush=True)

    known_set_codes = _derive_known_set_codes(args.manifest)
    known_languages = set(LANG_CODE.values())

    print("running inference ...", flush=True)
    candidates: list[Candidate] = []
    for i, row in enumerate(rows):
        try:
            img = Image.open(args.images_root / row["image_path"]).convert("RGB")
        except FileNotFoundError:
            continue
        crop_arr = np.asarray(crop_bottom_region(img))[None, ...].astype("uint8")
        logits = model(crop_arr, training=False)
        ids, conf = _greedy_decode(logits)
        pred_text = decode_label(ids)
        parsed = parse_bottom(
            pred_text,
            known_set_codes=known_set_codes,
            known_languages=known_languages,
        )

        cand = Candidate(
            index=i,
            scryfall_id=row["scryfall_id"],
            image_path=row["image_path"],
            manifest_set=_manifest_truth_set(row),
            manifest_number=row["collector_number"],
            manifest_rarity=row["rarity"],
            manifest_lang=row["lang"],
            predicted_set=parsed.set_code,
            predicted_number=parsed.collector_number,
            predicted_rarity=parsed.rarity,
            predicted_lang=parsed.language,
            raw_pred_text=pred_text,
            confidence=conf,
        )
        candidates.append(cand)

        if (i + 1) % 100 == 0:
            print(f"  inference [{i + 1}/{len(rows)}]", flush=True)

    print(f"  ran inference on {len(candidates)} cards", flush=True)

    # Bucket selection.
    rng = random.Random(args.seed)
    disagree = [c for c in candidates if (c.predicted_set or "") != c.manifest_set]
    agree = [c for c in candidates if (c.predicted_set or "") == c.manifest_set]
    disagree.sort(key=lambda c: -c.confidence)
    confident = disagree[: args.n_confident]
    seen_ids = {c.scryfall_id for c in confident}
    random_disagree_pool = [c for c in disagree[args.n_confident :] if c.scryfall_id not in seen_ids]
    random_disagree = rng.sample(
        random_disagree_pool, k=min(args.n_random_disagree, len(random_disagree_pool))
    )
    seen_ids.update(c.scryfall_id for c in random_disagree)
    random_agree = rng.sample(agree, k=min(args.n_random_agree, len(agree)))

    for c in confident:
        c.bucket = "confident-disagree"
    for c in random_disagree:
        c.bucket = "random-disagree"
    for c in random_agree:
        c.bucket = "random-agree"

    selected = confident + random_disagree + random_agree
    print(f"  selected {len(selected)} cards for review:")
    print(f"    {len(confident)} confident-disagree")
    print(f"    {len(random_disagree)} random-disagree")
    print(f"    {len(random_agree)} random-agree")

    # Save crops + write REVIEW.md
    review_path = args.out_dir / "REVIEW.md"
    crops_subdir = args.out_dir / "crops"
    crops_subdir.mkdir(exist_ok=True)
    for k, c in enumerate(selected):
        c.crop_filename = f"crops/audit_{k:03d}.png"
        _save_high_res_crop(args.images_root, c.image_path, args.out_dir / c.crop_filename)

    lines: list[str] = [
        "# Manifest Audit — set-code disagreements\n",
        f"Model: `{args.weights}`  ·  Split: `{args.split}`  ·  "
        f"Total cards reviewed: {len(selected)}\n",
        "",
        "## How to review",
        "",
        "For each card, look at the cropped image and decide whether the *manifest* "
        "values or the *model's prediction* matches what's actually printed. "
        "Mark the **Verdict** field with one letter:",
        "",
        "| Letter | Meaning |",
        "|---|---|",
        "| `M` | **Manifest is wrong** — model prediction is correct (or closer) |",
        "| `W` | **We (the model) are wrong** — manifest is correct |",
        "| `A` | **Ambiguous** — can't tell from the crop, or both partially right |",
        "",
        "Replace each `_` in the Verdict column with one of M/W/A. When done, "
        "run:",
        "",
        "```bash",
        ".venv/bin/python scripts/ingest_audit.py --review artifacts/manifest_audit/REVIEW.md",
        "```",
        "",
        "to compute manifest-noise rate.",
        "",
        "---",
        "",
    ]

    for bucket_name, bucket_cards in (
        ("Confident disagreements (most likely manifest errors)", confident),
        ("Random disagreements (mixed model + manifest errors)", random_disagree),
        ("Random agreements (control — most should be 'manifest right')", random_agree),
    ):
        lines.append(f"## {bucket_name}")
        lines.append("")
        for c in bucket_cards:
            lines.extend([
                f"### {c.scryfall_id[:8]} · `{c.bucket}` · confidence={c.confidence:.3f}",
                "",
                f"![{c.scryfall_id[:8]}]({c.crop_filename})",
                "",
                "| Field | Manifest | Predicted |",
                "|---|---|---|",
                f"| set | `{c.manifest_set}` | `{c.predicted_set}` |",
                f"| number | `{c.manifest_number}` | `{c.predicted_number}` |",
                f"| rarity | `{c.manifest_rarity}` | `{c.predicted_rarity}` |",
                f"| lang | `{c.manifest_lang}` | `{c.predicted_lang}` |",
                "",
                f"Raw model output: `{c.raw_pred_text!r}`",
                "",
                "**Verdict**: `_`  *(replace with M / W / A)*",
                "",
                "---",
                "",
            ])

    review_path.write_text("\n".join(lines), encoding="utf-8")

    # Also write a JSON sidecar so ingest_audit can resolve indices reliably.
    sidecar = {
        "candidates": [
            {
                "scryfall_id": c.scryfall_id,
                "bucket": c.bucket,
                "confidence": c.confidence,
                "manifest_set": c.manifest_set,
                "predicted_set": c.predicted_set,
            }
            for c in selected
        ],
    }
    (args.out_dir / "candidates.json").write_text(json.dumps(sidecar, indent=2))

    print(f"\nwrote {review_path}")
    print(f"wrote {len(selected)} crops to {crops_subdir}")
    print(f"\nopen {review_path} in your editor (VS Code preview shows images), mark each Verdict, then run:")
    print(f"  .venv/bin/python scripts/ingest_audit.py --review {review_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
