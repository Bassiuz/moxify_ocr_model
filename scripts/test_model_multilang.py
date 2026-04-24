"""Multi-language qualitative smoke test: one card per language."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from moxify_ocr.data.crop import crop_bottom_region
from moxify_ocr.data.dataset import decode_label
from moxify_ocr.models.bottom_region import build_bottom_region_model

#: Languages we want to probe, in order. Major languages first, then a few minors.
LANGS = ["en", "de", "fr", "it", "es", "pt", "ja", "ru", "ko", "zhs"]


def main() -> None:
    model = build_bottom_region_model()
    model.load_weights("artifacts/bottom_region_v1/best.keras")

    # Pick one non-PLST card per language (first match wins).
    picks: dict[str, dict[str, Any]] = {}
    manifest = Path("data/scryfall/manifest.jsonl")
    with manifest.open() as f:
        for line in f:
            row = json.loads(line)
            lang = row.get("lang")
            if (
                row.get("set_code") != "plst"
                and lang in LANGS
                and lang not in picks
            ):
                picks[lang] = row
            if len(picks) == len(LANGS):
                break

    for lang in LANGS:
        row = picks.get(lang)
        if row is None:
            print(f"[{lang}] no card found in manifest\n")
            continue

        img = Image.open(f"data/scryfall/{row['image_path']}").convert("RGB")
        arr = np.asarray(crop_bottom_region(img))[None, ...].astype("uint8")
        ids = model(arr).numpy().argmax(-1)[0]

        out: list[int] = []
        prev = -1
        for i in ids:
            if i != prev and i != 0:
                out.append(int(i))
            prev = i
        pred = decode_label(out)

        truth_summary = (
            f"num={row['collector_number']:>5}  "
            f"set={row['set_code'].upper():>4}  "
            f"rarity={row['rarity'][0].upper()}  "
            f"lang={lang.upper()}"
        )
        print(f"[{lang}] truth:     {truth_summary}")
        print(f"[{lang}] predicted: {pred!r}")
        print()


if __name__ == "__main__":
    main()
