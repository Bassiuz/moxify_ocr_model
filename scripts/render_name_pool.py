"""Render a synthetic card-name OCR pool to disk.

Drives :class:`moxify_ocr.data.name_renderer.NameRenderer` with random specs
(see :func:`moxify_ocr.data.name_specs.generate_specs`) and writes:

- ``{out_dir}/images/{seed:08d}.png`` — one (48x512x3) RGB PNG per sample.
- ``{out_dir}/labels.jsonl`` — one row per sample with
  ``{image_path, label, style, frame_color, foil}``.

Mirrors the on-disk format produced by
[scripts/render_cardconjurer_pool.py](scripts/render_cardconjurer_pool.py)
so the new dataset reader can be a small clone of
:class:`moxify_ocr.data.cardconjurer_dataset.CardConjurerPool`.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

from moxify_ocr.data.name_renderer import NameRenderer
from moxify_ocr.data.name_specs import generate_specs, load_card_names


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--n", type=int, required=True, help="Number of samples to render."
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--scryfall",
        type=Path,
        default=Path("data/scryfall/default-cards.json"),
        help="Path to the Scryfall default-cards.json dump.",
    )
    parser.add_argument(
        "--cardconjurer-root",
        type=Path,
        default=None,
        help="Override the CardConjurer asset root (default /tmp/cardconjurer-master).",
    )
    args = parser.parse_args()

    print(f"loading card names from {args.scryfall} ...")
    names = load_card_names(args.scryfall)
    print(f"  {len(names):,} usable name strings")

    images_dir = args.out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    manifest = args.out_dir / "labels.jsonl"

    renderer = NameRenderer(cardconjurer_root=args.cardconjurer_root)

    t0 = time.time()
    fails = 0
    with manifest.open("w") as f:
        for i, spec in enumerate(generate_specs(names=names, n=args.n, seed=args.seed)):
            try:
                arr, label = renderer.render(spec)
            except Exception as exc:  # noqa: BLE001 — log + skip; one bad spec
                fails += 1                # shouldn't kill a multi-hour render.
                if fails <= 5:
                    print(f"  [skip {i}] {type(exc).__name__}: {exc}  ({spec})")
                continue
            assert arr.shape == (48, 512, 3), f"unexpected shape {arr.shape}"
            assert arr.dtype == np.uint8

            relpath = f"images/{i:08d}.png"
            Image.fromarray(arr).save(args.out_dir / relpath)
            f.write(
                json.dumps(
                    {
                        "image_path": relpath,
                        "label": label,
                        "style": spec.style,
                        "frame_color": spec.frame_color,
                        "foil": spec.foil,
                    }
                )
                + "\n"
            )

            if (i + 1) % 100 == 0:
                elapsed = time.time() - t0
                rate = (i + 1) / max(elapsed, 0.001)
                eta = (args.n - i - 1) / max(rate, 0.001)
                print(
                    f"  [{i+1}/{args.n}]  {rate:.1f} samples/s  "
                    f"elapsed={elapsed:.0f}s  eta={eta:.0f}s  fails={fails}"
                )

    elapsed = time.time() - t0
    print(
        f"done: {args.n - fails} samples written, {fails} failures, "
        f"{elapsed:.0f}s ({(args.n - fails) / max(elapsed, 0.001):.1f} samples/s)"
    )
    print(f"  -> {args.out_dir}")


if __name__ == "__main__":
    main()
