"""Quick smoke test: predict bottom-region labels for the first 5 English non-PLST cards in the manifest."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from moxify_ocr.data.crop import crop_bottom_region
from moxify_ocr.data.dataset import decode_label
from moxify_ocr.models.bottom_region import build_bottom_region_model


def main() -> None:
    model = build_bottom_region_model()
    model.load_weights("artifacts/bottom_region_v1/best.keras")

    samples = []
    with Path("data/scryfall/manifest.jsonl").open() as f:
        for line in f:
            row = json.loads(line)
            if row["set_code"] != "plst" and row["lang"] == "en":
                samples.append(row)
            if len(samples) >= 10:
                break

    for row in samples:
        truth = (
            f"{row['collector_number']:>5}  "
            f"{row['rarity'][0].upper()}  /  "
            f"{row['set_code'].upper()}"
        )
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

        print(f"truth:     {truth}")
        print(f"predicted: {pred!r}")
        print()


if __name__ == "__main__":
    main()
