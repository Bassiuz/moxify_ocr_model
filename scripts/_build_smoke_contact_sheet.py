"""Build a 20-card contact sheet from a CardConjurer pool for visual review.

Adapted from _build_spike_contact_sheet.py — instead of rendering a fixed
list of variety specs, this picks 20 random entries from a generated pool
and stitches them into one PNG so the user can eyeball font fidelity, foil
star rendering, and overall consistency before committing to a 50K-100K run.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

# Row height roomy enough to contain the native preview AND the OCR-scale
# stripe with breathing room — prior 100px caused the bottom edge of the
# native preview to touch the next row visually.
ROW_HEIGHT = 140
LABEL_WIDTH = 480
NATIVE_WIDTH = 720
OCR_WIDTH = 256
OCR_HEIGHT = 48
PAD = 12


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a 20-card contact sheet from a pool.")
    parser.add_argument("--pool", type=Path, required=True, help="Pool root with images/ + labels.jsonl")
    parser.add_argument("--n", type=int, default=20, help="Number of samples to include")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for sample selection")
    parser.add_argument("--out", type=Path, required=True, help="Output PNG path")
    args = parser.parse_args()

    rows = [json.loads(line) for line in (args.pool / "labels.jsonl").read_text().splitlines() if line]
    if len(rows) < args.n:
        raise SystemExit(f"pool has only {len(rows)} entries; need at least {args.n}")
    rng = random.Random(args.seed)
    picks = rng.sample(rows, args.n)

    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 14)
    except OSError:
        font = ImageFont.load_default()

    sheet_w = NATIVE_WIDTH + PAD + OCR_WIDTH + PAD + LABEL_WIDTH + 2 * PAD
    sheet_h = ROW_HEIGHT * args.n + 2 * PAD + 30  # +30 for header
    sheet = Image.new("RGB", (sheet_w, sheet_h), (24, 24, 24))
    draw = ImageDraw.Draw(sheet)

    # Header
    draw.rectangle((0, 0, sheet_w, 30), fill=(40, 40, 40))
    draw.text(
        (PAD, 7),
        f"left: native crop | right: 256x48 OCR scale | label  ({args.n} random samples from {args.pool.name})",
        fill=(200, 200, 200),
        font=font,
    )

    for i, row in enumerate(picks):
        y = 30 + PAD + i * ROW_HEIGHT
        crop = Image.open(args.pool / row["image_path"]).convert("RGB")
        # Scale native to NATIVE_WIDTH preserving aspect
        native = crop.resize(
            (NATIVE_WIDTH, int(crop.height * NATIVE_WIDTH / crop.width)),
            Image.Resampling.LANCZOS,
        )
        sheet.paste(native, (PAD, y + (ROW_HEIGHT - native.height) // 2))

        # OCR scale (already 256x48 in pool — paste verbatim)
        ox = PAD + NATIVE_WIDTH + PAD
        oy = y + (ROW_HEIGHT - OCR_HEIGHT) // 2
        sheet.paste(crop, (ox, oy))
        draw.rectangle((ox - 1, oy - 1, ox + OCR_WIDTH, oy + OCR_HEIGHT), outline=(80, 80, 80))

        # Label
        label_lines = row["label"].split("\n")
        meta = (
            f"foil={row.get('foil', False)}  set={row.get('info_set', '?')}  "
            f"lang={row.get('info_language', '?')}  rar={row.get('info_rarity', '?')}  "
            f"#={row.get('info_number', '?')}"
        )
        lx = ox + OCR_WIDTH + PAD
        ly = y + ROW_HEIGHT // 2 - 24
        draw.text((lx, ly), label_lines[0], fill=(220, 220, 220), font=font)
        if len(label_lines) > 1:
            draw.text((lx, ly + 18), label_lines[1], fill=(220, 220, 220), font=font)
        draw.text((lx, ly + 38), meta, fill=(160, 160, 160), font=font)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.out)
    print(f"wrote {args.out}  ({sheet.size})")


if __name__ == "__main__":
    main()
