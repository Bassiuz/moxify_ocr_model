"""Predict the bottom-region OCR fields on one or more arbitrary card images.

Use to spot-check the bottom-region model on real-world photos before
integrating it into the app — feed it a JPG/PNG of a card (full card,
not pre-cropped) and it'll print the parsed fields.

Usage::

    .venv/bin/python scripts/predict_bottom.py \\
        --weights "best_v3(1).keras" \\
        path/to/card1.jpg path/to/card2.png ...

The script does the same crop the dataset pipeline does
(``crop_bottom_region`` with the production fractions), runs inference,
greedy-decodes CTC, and parses fields via ``parse_bottom``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image

from moxify_ocr.data.crop import crop_bottom_region
from moxify_ocr.data.dataset import BLANK_INDEX, decode_label
from moxify_ocr.data.labels import LANG_CODE
from moxify_ocr.eval.eval import _derive_known_set_codes
from moxify_ocr.export.parse_bottom import parse_bottom
from moxify_ocr.models.bottom_region import build_bottom_region_model


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/scryfall/manifest.jsonl"),
        help="Used to build the known-set-codes vocabulary for parse_bottom.",
    )
    parser.add_argument(
        "--save-crops",
        type=Path,
        default=None,
        help="If set, save each card's bottom-region crop here (debug/QA).",
    )
    parser.add_argument("images", nargs="+", type=Path)
    args = parser.parse_args()

    print(f"loading {args.weights} ...")
    model = build_bottom_region_model()
    model.load_weights(str(args.weights))

    if args.manifest.exists():
        known_set_codes = _derive_known_set_codes(args.manifest)
        known_languages = set(LANG_CODE.values())
        print(f"  known set codes: {len(known_set_codes)}")
    else:
        # Fall back to no vocab — parse_bottom will return None for set/lang
        # but the raw decoded string is still printed.
        known_set_codes = set()
        known_languages = set(LANG_CODE.values())
        print("  no manifest — set/lang fields may parse as None")

    if args.save_crops:
        args.save_crops.mkdir(parents=True, exist_ok=True)

    for i, img_path in enumerate(args.images):
        if not img_path.exists():
            print(f"\n[{i}] {img_path}: missing")
            continue
        img = Image.open(img_path).convert("RGB")
        cropped = crop_bottom_region(img)
        if args.save_crops:
            crop_out = args.save_crops / f"{img_path.stem}_crop.png"
            cropped.save(crop_out)
        arr = np.asarray(cropped)[None, ...].astype("uint8")
        logits = model(arr, training=False)
        ids = logits.numpy().argmax(-1)[0]
        out, prev = [], -1
        for token in ids:
            if token != prev and token != BLANK_INDEX:
                out.append(int(token))
            prev = token
        pred_str = decode_label(out)
        parsed = parse_bottom(
            pred_str, known_set_codes=known_set_codes, known_languages=known_languages
        )
        print(f"\n[{i}] {img_path.name}")
        print(f"  raw: {pred_str!r}")
        print(
            f"  parsed: set={parsed.set_code}  num={parsed.collector_number}  "
            f"rarity={parsed.rarity}  lang={parsed.language}  foil={parsed.foil_detected}"
        )
        # Also print as JSON for piping
        print(
            "  json:",
            json.dumps(
                {
                    "image": str(img_path),
                    "raw": pred_str,
                    "set_code": parsed.set_code,
                    "collector_number": parsed.collector_number,
                    "rarity": parsed.rarity,
                    "language": parsed.language,
                    "foil_detected": parsed.foil_detected,
                    "set_total": parsed.set_total,
                }
            ),
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
