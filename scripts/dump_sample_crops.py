"""Dump sample bottom-left card crops for human visual inspection.

Reads the first ``--limit`` entries from a manifest, runs each card image
through :func:`moxify_ocr.data.crop.crop_bottom_region`, and writes the
resulting JPEGs to ``--out``. This is a smoke tool for the data pipeline — it
lets you eyeball crops before kicking off training. Missing source images are
reported to stderr and skipped (the script does not abort).

Usage::

    python scripts/dump_sample_crops.py \\
        --manifest data/scryfall/manifest.jsonl \\
        --images-root data/scryfall/ \\
        --out data/debug_crops/ \\
        [--limit N]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image

from moxify_ocr.data.crop import crop_bottom_region
from moxify_ocr.data.manifest import read_manifest

#: Default number of rows to process when ``--limit`` is not supplied.
_DEFAULT_LIMIT = 20


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns the process exit code."""
    parser = argparse.ArgumentParser(
        description="Dump sample bottom-left crops for visual inspection.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="Path to a manifest.jsonl file.",
    )
    parser.add_argument(
        "--images-root",
        type=Path,
        required=True,
        dest="images_root",
        help="Directory where manifest image_path entries resolve (e.g. data/scryfall/).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Directory to write cropped JPEGs to. Created if missing.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=_DEFAULT_LIMIT,
        help=f"Number of rows to process (default: {_DEFAULT_LIMIT}).",
    )
    args = parser.parse_args(argv)
    return _dump_crops(args.manifest, args.images_root, args.out, args.limit)


def _dump_crops(manifest: Path, images_root: Path, out: Path, limit: int) -> int:
    out.mkdir(parents=True, exist_ok=True)
    dumped = 0
    for entry in read_manifest(manifest):
        if dumped >= limit:
            break
        src = images_root / entry.image_path
        if not src.exists():
            print(f"warning: missing image {src}", file=sys.stderr)
            continue
        with Image.open(src) as img:
            crop = crop_bottom_region(img.convert("RGB"))
            crop.save(out / f"{entry.scryfall_id}.jpg", format="JPEG", quality=95)
        dumped += 1
    print(f"dumped {dumped} crops to {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
