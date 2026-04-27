"""Build a contact sheet from a name-OCR pool for visual review.

Mirrors scripts/_build_smoke_contact_sheet.py for the name-region pool.
Picks one sample per style if possible, otherwise falls back to a random
draw, and stitches them into one PNG so the user can eyeball font fidelity,
frame variety, and label correctness before committing to a full render.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROW_HEIGHT = 80
LABEL_WIDTH = 380
NATIVE_WIDTH = 768  # the rendered PNG is 512 wide; scale up for legibility
PAD = 12


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--per-style",
        type=int,
        default=2,
        help="Number of samples to include per style (best-effort).",
    )
    args = parser.parse_args()

    rows = [
        json.loads(line)
        for line in (args.pool / "labels.jsonl").read_text().splitlines()
        if line
    ]
    by_style: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_style[r["style"]].append(r)
    picks: list[dict] = []
    for style in sorted(by_style):
        picks.extend(by_style[style][: args.per_style])

    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 13)
    except OSError:
        font = ImageFont.load_default()

    sheet_w = NATIVE_WIDTH + PAD + LABEL_WIDTH + 2 * PAD
    sheet_h = ROW_HEIGHT * len(picks) + 2 * PAD + 30
    sheet = Image.new("RGB", (sheet_w, sheet_h), (24, 24, 24))
    draw = ImageDraw.Draw(sheet)

    draw.rectangle((0, 0, sheet_w, 30), fill=(40, 40, 40))
    draw.text(
        (PAD, 7),
        f"name-OCR pool {args.pool.name} | {len(picks)} samples ({args.per_style}/style)",
        fill=(200, 200, 200),
        font=font,
    )

    for i, row in enumerate(picks):
        y = 30 + PAD + i * ROW_HEIGHT
        crop = Image.open(args.pool / row["image_path"]).convert("RGB")
        # Native: scale to NATIVE_WIDTH preserving aspect.
        native = crop.resize(
            (NATIVE_WIDTH, int(crop.height * NATIVE_WIDTH / crop.width)),
            Image.Resampling.LANCZOS,
        )
        sheet.paste(native, (PAD, y + (ROW_HEIGHT - native.height) // 2))

        # Label box with style + name + meta.
        lx = PAD + NATIVE_WIDTH + PAD
        ly = y + ROW_HEIGHT // 2 - 22
        draw.text(
            (lx, ly),
            f"{row['style']}",
            fill=(150, 220, 150),
            font=font,
        )
        draw.text((lx, ly + 18), row["label"], fill=(220, 220, 220), font=font)
        meta = (
            f"color={row.get('frame_color', '?')}  foil={row.get('foil', False)}"
        )
        draw.text((lx, ly + 36), meta, fill=(160, 160, 160), font=font)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.out)
    print(f"wrote {args.out}  ({sheet.size})")


if __name__ == "__main__":
    main()
