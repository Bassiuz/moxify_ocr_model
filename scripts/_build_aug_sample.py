"""Build a 4x4 contact sheet of augmented variants of one CardConjurer crop.

One-shot visual sanity check for the v3 augmentation pipeline. Loads a single
crop from the smoke pool, runs ``apply_augmentation`` 16 times with seeds
0-15, tiles the results into a 4x4 grid, and saves to artifacts/.

Usage::

    .venv/bin/python scripts/_build_aug_sample.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from moxify_ocr.data.augment import apply_augmentation, build_augmentation_pipeline

DEFAULT_CROP = Path(
    "/Users/bassiuz/Projects/moxify_ocr_model/artifacts/cc_smoke_pool/images/00000000.png"
)
DEFAULT_OUT = Path("artifacts/v3_aug_sample.png")
GRID = 4  # 4x4 = 16 variants
PAD = 4


def main() -> None:
    parser = argparse.ArgumentParser(description="Augmentation contact sheet")
    parser.add_argument("--crop", type=Path, default=DEFAULT_CROP)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    with Image.open(args.crop) as img:
        crop_rgb = np.asarray(img.convert("RGB"), dtype=np.uint8)

    h, w = crop_rgb.shape[:2]
    pipeline = build_augmentation_pipeline(seed=0)

    sheet_w = GRID * w + (GRID + 1) * PAD
    sheet_h = GRID * h + (GRID + 1) * PAD + h + 2 * PAD  # extra row for the original on top
    sheet = Image.new("RGB", (sheet_w, sheet_h), (24, 24, 24))

    # Top: the original crop, centered.
    orig_x = (sheet_w - w) // 2
    sheet.paste(Image.fromarray(crop_rgb), (orig_x, PAD))

    # Below: 4x4 grid of augmented variants.
    for i in range(GRID * GRID):
        row, col = divmod(i, GRID)
        aug = apply_augmentation(crop_rgb, pipeline, seed=i)
        x = PAD + col * (w + PAD)
        y = (h + 2 * PAD) + PAD + row * (h + PAD)
        sheet.paste(Image.fromarray(aug), (x, y))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.out)
    print(f"wrote {args.out}  ({sheet.size})")


if __name__ == "__main__":
    main()
