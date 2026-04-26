"""Render a pre-computed pool of CardConjurer bottom-strip crops + labels.

Drives a local CardConjurer instance via Playwright. For each spec produced
by :func:`moxify_ocr.data.cardconjurer_specs.generate_specs`, renders the
``bottomInfoCanvas``, composites it on black, crops to a wide bottom strip,
resizes to ``(256, 48)`` and writes to ``{out_dir}/images/{i:08d}.png``.
Appends one JSONL row per spec to ``{out_dir}/labels.jsonl``.

The label is built in Python from the spec (matches ``labels.make_label``'s
two-line format), NOT read from the rendered pixels.

Assumes CardConjurer is reachable at ``--cardconjurer-url`` (default
``http://localhost:4242/creator/``). Bring it up via
``infra/cardconjurer/start.sh`` before running this script.

Usage::

    python scripts/render_cardconjurer_pool.py --n 100 --seed 0 \\
        --out-dir /tmp/cc_pool
"""

from __future__ import annotations

import argparse
import base64
import contextlib
import io
import json
import sys
import time
import urllib.error
import urllib.request
from dataclasses import asdict
from pathlib import Path

from PIL import Image
from playwright.sync_api import Browser, Page, sync_playwright

from moxify_ocr.data.cardconjurer_specs import CardSpec, generate_specs

#: Output crop size — must match the OCR model's expected input shape.
_CROP_SIZE = (256, 48)

#: Wide-crop fractions matching the spike (NOT the production crop). These
#: produced the contact sheet the user reviewed and approved.
_CROP_TOP_FRAC = 0.928
_CROP_BOTTOM_FRAC = 0.98
_CROP_LEFT_FRAC = 0.05
_CROP_RIGHT_FRAC = 0.95

#: Restart the Chromium tab every N specs to defend against memory leaks
#: in long-running Playwright contexts.
_TAB_RESTART_EVERY = 1000

#: Print a progress line every N specs.
_PROGRESS_EVERY = 100

#: Browser viewport — same as the spike (large enough that the canvas the
#: bottom-info layer is rasterized to is high-resolution).
_VIEWPORT = {"width": 1400, "height": 1100}

#: Fonts CardConjurer's bottom-info layer draws with. Force-loaded explicitly
#: so the @font-face rules from ``style-9.css`` actually fetch before any
#: ``drawImage`` happens, otherwise text falls back to serif.
_FONT_FAMILIES = (
    "gothammedium",
    "belerenb",
    "belerenbsc",
    "mplantin",
    "mplantin-i",
    "gothambold",
)


_FILL_INPUT_JS = """
([id, value]) => {
    const el = document.querySelector('#' + id);
    if (!el) return false;
    el.value = value;
    el.dispatchEvent(new Event('input', {bubbles: true}));
    el.dispatchEvent(new Event('change', {bubbles: true}));
    return true;
}
"""


def _build_label(spec: CardSpec) -> str:
    """Construct the canonical two-line OCR label for a spec.

    Mirrors :func:`moxify_ocr.data.labels.make_label` for the synthetic case:
    no PLST icon, no Basic-Land L override, all rarities map to a single
    letter, set total is included verbatim from the spec's collector number.
    """
    pip = "★" if spec.foil else "•"
    line1 = f"{spec.info_number} {spec.info_rarity}"
    line2 = f"{spec.info_set} {pip} {spec.info_language}"
    return f"{line1}\n{line2}"


def _check_cardconjurer_reachable(url: str) -> None:
    """Fail fast if CardConjurer isn't responding at ``url``."""
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            if resp.status >= 400:
                raise RuntimeError(
                    f"CardConjurer returned HTTP {resp.status} at {url}"
                )
    except (urllib.error.URLError, OSError, TimeoutError) as e:
        raise SystemExit(
            f"error: CardConjurer not reachable at {url}: {e}\n"
            f"hint: bring it up with infra/cardconjurer/start.sh"
        ) from e


def _prime_page(page: Page, url: str) -> None:
    """Load CardConjurer and prepare the page for repeated bottom-strip renders.

    Steps:
      1. Navigate to ``url``.
      2. Inject ``/css/reset.css`` + ``/css/style-9.css`` (the htmx fragment
         skips its own <head>, so @font-face rules are missing without this).
      3. Wait for ``drawCard`` / ``bottomInfoEdited`` / ``setBottomInfoStyle``.
      4. Force-load every font family the bottom layer draws with.
      5. Tick ``#enableCollectorInfo``, default ``card.bottomInfoColor`` to
         ``"white"``, run ``setBottomInfoStyle()`` — both are unset on a fresh
         session due to a CardConjurer init-order quirk.
    """
    page.goto(url, wait_until="networkidle", timeout=60000)
    page.evaluate(
        """() => {
            for (const href of ['/css/reset.css', '/css/style-9.css']) {
                const link = document.createElement('link');
                link.rel = 'stylesheet';
                link.href = href;
                document.head.appendChild(link);
            }
        }"""
    )
    page.wait_for_function("typeof drawCard === 'function'", timeout=30000)
    page.wait_for_function(
        "typeof bottomInfoEdited === 'function' "
        "&& typeof setBottomInfoStyle === 'function'",
        timeout=30000,
    )
    page.evaluate(
        """async (families) => {
            await Promise.all(families.map(f => document.fonts.load(`24px ${f}`)));
            await document.fonts.ready;
        }""",
        list(_FONT_FAMILIES),
    )
    page.evaluate(
        """async () => {
            document.querySelector('#enableCollectorInfo').checked = true;
            card.bottomInfoColor = 'white';
            await setBottomInfoStyle();
        }"""
    )
    page.wait_for_timeout(500)


def _open_fresh_page(browser: Browser, url: str) -> Page:
    """Open a fresh tab in ``browser`` and prime it for rendering."""
    page = browser.new_page(viewport=_VIEWPORT)
    _prime_page(page, url)
    return page


def _set_field(page: Page, dom_id: str, value: str) -> None:
    ok = page.evaluate(_FILL_INPUT_JS, [dom_id, value])
    if not ok:
        raise RuntimeError(f"input #{dom_id} not found in CardConjurer page")


def _apply_spec(page: Page, spec: CardSpec) -> None:
    """Apply ``spec`` to the loaded page and recompute the bottom-info layer.

    Patches:
      - ``card.bottomInfo.topLeft.text`` to ``"<num> <rarity>"`` (CardConjurer
        defaults to the reverse order — our labels.py emits "num rarity").
      - ``card.bottomInfo.midLeft.text`` to ``"<set> <pip> <lang>"`` where pip
        is ``★`` for foil, ``•`` otherwise.
    Then sets all six ``#info-*`` fields and calls ``bottomInfoEdited()``.
    """
    pip = "★" if spec.foil else "•"
    # Match labels.make_label exactly: dummy artist + year (rendered text uses
    # them only inside the midLeft template if a foil pip is appended via the
    # default suffix, which we override below).
    _set_field(page, "info-set", spec.info_set)
    _set_field(page, "info-language", spec.info_language)
    _set_field(page, "info-rarity", spec.info_rarity)
    _set_field(page, "info-number", spec.info_number)
    _set_field(page, "info-artist", "Test Artist")
    _set_field(page, "info-year", "2026")
    page.evaluate(
        """([pip]) => {
            if (card.bottomInfo && card.bottomInfo.midLeft) {
                card.bottomInfo.midLeft.text =
                    `{elemidinfo-set} ${pip} {elemidinfo-language}`;
            }
            if (card.bottomInfo && card.bottomInfo.topLeft) {
                card.bottomInfo.topLeft.text =
                    '{elemidinfo-number} {kerning3}{elemidinfo-rarity}{kerning0}';
            }
        }""",
        [pip],
    )
    page.evaluate("async () => { await bottomInfoEdited(); }")


def _capture_bottom_canvas_png(page: Page) -> bytes:
    """Read ``window.bottomInfoCanvas``, composite on black, return PNG bytes.

    We deliberately do NOT capture ``window.cardCanvas`` — that only composites
    when a frame image is loaded, which we skip for speed and to keep the strip
    free of frame artwork (training data is text-only on black).
    """
    result = page.evaluate(
        """() => {
            const c = window.bottomInfoCanvas;
            if (!c) return null;
            const out = document.createElement('canvas');
            out.width = c.width;
            out.height = c.height;
            const ctx = out.getContext('2d');
            ctx.fillStyle = 'black';
            ctx.fillRect(0, 0, out.width, out.height);
            ctx.drawImage(c, 0, 0);
            return {data_url: out.toDataURL('image/png')};
        }"""
    )
    if not result or not result.get("data_url", "").startswith(
        "data:image/png;base64,"
    ):
        raise RuntimeError("could not read window.bottomInfoCanvas")
    return base64.b64decode(result["data_url"].split(",", 1)[1])


def _crop_and_resize(png_bytes: bytes) -> Image.Image:
    """Crop a wide bottom strip and resize to ``_CROP_SIZE``.

    Crop fractions match the spike: ``(0.05, 0.928, 0.95, 0.98)``. Resize
    uses LANCZOS for sharp downsampling of the small text glyphs.
    """
    img = Image.open(io.BytesIO(png_bytes)).convert("RGB")
    w, h = img.size
    box = (
        int(w * _CROP_LEFT_FRAC),
        int(h * _CROP_TOP_FRAC),
        int(w * _CROP_RIGHT_FRAC),
        min(int(h * _CROP_BOTTOM_FRAC), h),
    )
    return img.crop(box).resize(_CROP_SIZE, Image.LANCZOS)


def _format_eta(seconds: float) -> str:
    """Format an ETA in ``HH:MM:SS``."""
    s = int(seconds)
    return f"{s // 3600:02d}:{(s % 3600) // 60:02d}:{s % 60:02d}"


def _render_one(page: Page, spec: CardSpec, image_path: Path) -> None:
    """Render ``spec`` to ``image_path`` (256x48 PNG)."""
    _apply_spec(page, spec)
    png_bytes = _capture_bottom_canvas_png(page)
    crop = _crop_and_resize(png_bytes)
    crop.save(image_path, format="PNG")


def _jsonl_row(spec: CardSpec, image_rel_path: str) -> dict[str, object]:
    """Build the JSONL row for one rendered spec."""
    row: dict[str, object] = {
        "image_path": image_rel_path,
        "label": _build_label(spec),
    }
    row.update(asdict(spec))
    return row


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Render a pool of CardConjurer bottom-strip crops + labels to disk."
        )
    )
    parser.add_argument(
        "--n", type=int, required=True, help="number of cards to render"
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="base seed for spec generation"
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="output directory; images/ and labels.jsonl written here",
    )
    parser.add_argument(
        "--cardconjurer-url",
        type=str,
        default="http://localhost:4242/creator/",
        help="URL of a running CardConjurer instance",
    )
    args = parser.parse_args(argv)

    if args.n <= 0:
        parser.error("--n must be positive")

    _check_cardconjurer_reachable(args.cardconjurer_url)

    images_dir = args.out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_path = args.out_dir / "labels.jsonl"

    print(
        f"rendering n={args.n} seed={args.seed} → {args.out_dir} "
        f"(cardconjurer={args.cardconjurer_url})",
        flush=True,
    )

    t_start = time.time()
    succeeded = 0
    failed = 0

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = _open_fresh_page(browser, args.cardconjurer_url)
        try:
            with labels_path.open("a", encoding="utf-8") as label_fp:
                for i, spec in enumerate(generate_specs(args.n, args.seed)):
                    if i > 0 and i % _TAB_RESTART_EVERY == 0:
                        # Defensive: close + reopen the tab to free any
                        # accumulated DOM/canvas memory. Browser stays up so
                        # we don't pay the per-launch overhead.
                        with contextlib.suppress(Exception):
                            page.close()
                        page = _open_fresh_page(browser, args.cardconjurer_url)

                    image_rel = f"images/{i:08d}.png"
                    image_path = args.out_dir / image_rel
                    try:
                        _render_one(page, spec, image_path)
                    except Exception as e:  # noqa: BLE001
                        # Single bad spec must NOT kill a 31-hour run.
                        print(
                            f"  ✗ spec {i}: {type(e).__name__}: {e}",
                            file=sys.stderr,
                            flush=True,
                        )
                        failed += 1
                        continue

                    label_fp.write(
                        json.dumps(_jsonl_row(spec, image_rel), ensure_ascii=False)
                        + "\n"
                    )
                    label_fp.flush()
                    succeeded += 1

                    if (i + 1) % _PROGRESS_EVERY == 0:
                        elapsed = time.time() - t_start
                        rate = elapsed / (i + 1)
                        eta = rate * (args.n - (i + 1))
                        print(
                            f"[{i + 1}/{args.n}] elapsed={elapsed:.1f}s "
                            f"rate={rate:.2f}s/card eta={_format_eta(eta)}",
                            flush=True,
                        )
        finally:
            with contextlib.suppress(Exception):
                page.close()
            browser.close()

    elapsed = time.time() - t_start
    print(
        f"done: {succeeded}/{args.n} ok, {failed} failed, "
        f"elapsed={_format_eta(elapsed)}",
        flush=True,
    )
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
