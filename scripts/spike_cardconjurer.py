"""Spike: drive a local CardConjurer instance via Playwright to render variety cards.

Renders ~10 cards covering the variety axes that matter for our OCR (set code,
language, rarity, foil treatment, collector-number format, layout). Saves each
full-card PNG and a tightly-cropped bottom region. A contact sheet of bottom
crops is the artifact the user inspects to decide if the data is workable.

Run after `docker run -p 4242:4242 cardconjurer-client` is up.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image
from playwright.sync_api import sync_playwright

OUT_DIR = Path(__file__).resolve().parent.parent / "artifacts" / "cardconjurer_spike"
CARDS_DIR = OUT_DIR / "cards"
BOTTOM_DIR = OUT_DIR / "bottom_crops"
CONTACT_SHEET = OUT_DIR / "contact_sheet.png"

CARDCONJURER_URL = "http://localhost:4242/creator/"


@dataclass
class CardSpec:
    """One card to render. Field names mirror the DOM input ids in CardConjurer."""

    name: str  # output filename stem
    title: str = "Spike Sample"
    info_set: str = "TST"
    info_language: str = "EN"
    info_rarity: str = "R"  # M / R / U / C / S
    info_number: str = "001/250"
    info_artist: str = "Test Artist"
    info_year: str = "2026"
    foil: bool = False  # affects mid-left line: ★ vs •
    notes: list[str] = field(default_factory=list)  # printed on contact sheet


SPECS: list[CardSpec] = [
    CardSpec(
        name="01_baseline_en",
        info_set="LEA", info_language="EN", info_rarity="C", info_number="161/302",
        notes=["EN baseline", "/total format"],
    ),
    CardSpec(
        name="02_german",
        info_set="LEA", info_language="DE", info_rarity="C", info_number="161/302",
        notes=["DE language code"],
    ),
    CardSpec(
        name="03_japanese",
        info_set="LEA", info_language="JA", info_rarity="C", info_number="161/302",
        notes=["JA — CJK font test"],
    ),
    CardSpec(
        name="04_foil_en",
        info_set="LEA", info_language="EN", info_rarity="R", info_number="265/302",
        foil=True,
        notes=["★ foil glyph"],
    ),
    CardSpec(
        name="05_x_collector",
        info_set="PMTG", info_language="EN", info_rarity="M", info_number="X12",
        notes=["unusual collector number"],
    ),
    CardSpec(
        name="06_no_total",
        info_set="MID", info_language="EN", info_rarity="R", info_number="042",
        notes=["bare collector number, no /total"],
    ),
    CardSpec(
        name="07_korean",
        info_set="MID", info_language="KO", info_rarity="C", info_number="042",
        notes=["KO — CJK"],
    ),
    CardSpec(
        name="08_french",
        info_set="MID", info_language="FR", info_rarity="C", info_number="042",
        notes=["FR diacritics"],
    ),
    CardSpec(
        name="09_high_rarity",
        info_set="LEA", info_language="EN", info_rarity="M", info_number="232/302",
        foil=True,
        notes=["mythic + foil ★"],
    ),
    CardSpec(
        name="10_mtga_long_set",
        info_set="VOW", info_language="EN", info_rarity="U", info_number="123/277",
        notes=["modern set code"],
    ),
]


# ---------- Playwright driving ----------


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


def _set_field(page, dom_id: str, value: str) -> None:
    ok = page.evaluate(_FILL_INPUT_JS, [dom_id, value])
    if not ok:
        raise RuntimeError(f"input #{dom_id} not found in CardConjurer page")


def _apply_spec(page, spec: CardSpec) -> None:
    """Apply a spec to the loaded CardConjurer page."""
    _set_field(page, "info-set", spec.info_set)
    _set_field(page, "info-language", spec.info_language)
    _set_field(page, "info-rarity", spec.info_rarity)
    _set_field(page, "info-number", spec.info_number)
    _set_field(page, "info-artist", spec.info_artist)
    _set_field(page, "info-year", spec.info_year)
    # Patch the midLeft template to drop the artist + foil-pip suffix and put
    # a literal ★/• between set and language. Matches our label format
    # "<set> <pip> <lang>" (see src/moxify_ocr/data/labels.py).
    pip = "★" if spec.foil else "•"
    page.evaluate(
        """([pip]) => {
            if (card.bottomInfo && card.bottomInfo.midLeft) {
                card.bottomInfo.midLeft.text =
                    `{elemidinfo-set} ${pip} {elemidinfo-language}`;
            }
            // Also reorder topLeft to {number} {rarity} (CardConjurer default
            // is {rarity} {number} but our labels.py outputs the other order).
            if (card.bottomInfo && card.bottomInfo.topLeft) {
                card.bottomInfo.topLeft.text =
                    '{elemidinfo-number} {kerning3}{elemidinfo-rarity}{kerning0}';
            }
        }""",
        [pip],
    )
    page.evaluate("async () => { await bottomInfoEdited(); }")
    page.wait_for_timeout(400)


def _capture_canvas(page) -> bytes:
    """Read the offscreen bottomInfoCanvas (just the bottom-strip text layer).

    CardConjurer maintains separate offscreen canvases for each layer (frame,
    text, watermark, bottomInfo). The full cardCanvas only composites if a
    frame is loaded — for spike-fidelity we directly grab bottomInfoCanvas,
    which is the layer holding {set} • {language} ★{artist} and {rarity} {number}.
    Composited on black to mimic real-card-bottom appearance.
    """
    result = page.evaluate(
        """() => {
            const c = window.bottomInfoCanvas;
            if (!c) return null;
            // Composite on black so transparent text becomes visible.
            const out = document.createElement('canvas');
            out.width = c.width;
            out.height = c.height;
            const ctx = out.getContext('2d');
            ctx.fillStyle = 'black';
            ctx.fillRect(0, 0, out.width, out.height);
            ctx.drawImage(c, 0, 0);
            return {data_url: out.toDataURL('image/png'), w: out.width, h: out.height};
        }"""
    )
    if not result or not result.get("data_url", "").startswith("data:image/png;base64,"):
        raise RuntimeError("could not read window.bottomInfoCanvas")
    import base64

    return base64.b64decode(result["data_url"].split(",", 1)[1])


def _crop_bottom(card_path: Path, out_path: Path) -> None:
    """Crop a wide bottom strip of the captured image for visual review."""
    img = Image.open(card_path)
    w, h = img.size
    top = int(h * 0.928)
    bottom = min(int(h * 0.98), h)
    left = int(w * 0.05)
    right = int(w * 0.95)
    cropped = img.crop((left, top, right, bottom))
    cropped.save(out_path)


def main() -> None:
    CARDS_DIR.mkdir(parents=True, exist_ok=True)
    BOTTOM_DIR.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1400, "height": 1100})
        print(f"loading {CARDCONJURER_URL} ...")
        page.goto(CARDCONJURER_URL, wait_until="networkidle", timeout=60000)
        # /creator/ is an htmx fragment without <head>, so its stylesheets
        # (which carry the @font-face rules) never load on direct navigation.
        # Inject them manually so canvas text renders in the real MTG fonts
        # rather than the serif fallback.
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
        # Give creator-23.js a chance to wire everything up.
        page.wait_for_function("typeof drawCard === 'function'", timeout=30000)
        page.wait_for_function(
            "typeof bottomInfoEdited === 'function' && typeof setBottomInfoStyle === 'function'",
            timeout=30000,
        )
        # Now that @font-face rules exist, force-load each font we draw with.
        # `document.fonts.load` is lazy on its own — call it explicitly so the
        # font files are fetched before any text rasterizes onto the canvas.
        page.evaluate(
            """async () => {
                const families = [
                    'gothammedium', 'belerenb', 'belerenbsc',
                    'mplantin', 'mplantin-i', 'gothambold',
                ];
                await Promise.all(families.map(f => document.fonts.load(`24px ${f}`)));
                await document.fonts.ready;
            }"""
        )
        # Tick the collector-info checkbox (off by default on a fresh session
        # because of a CardConjurer init-order quirk) and prime card.bottomInfo
        # with the standard M15 template.
        page.evaluate(
            """async () => {
                document.querySelector('#enableCollectorInfo').checked = true;
                // bottomInfoColor isn't set on a fresh page (only set in init paths
                // we don't traverse). Default it to white so text isn't drawn black.
                card.bottomInfoColor = 'white';
                await setBottomInfoStyle();
            }"""
        )
        page.wait_for_timeout(500)
        print("page ready, rendering specs ...")

        timings = []
        for spec in SPECS:
            t0 = time.time()
            try:
                _apply_spec(page, spec)
                png_bytes = _capture_canvas(page)
            except Exception as e:
                print(f"  ✗ {spec.name}: {e}")
                continue
            card_path = CARDS_DIR / f"{spec.name}.png"
            card_path.write_bytes(png_bytes)
            _crop_bottom(card_path, BOTTOM_DIR / f"{spec.name}_bottom.png")
            dt = time.time() - t0
            timings.append(dt)
            print(f"  ✓ {spec.name} ({dt:.2f}s)")

        browser.close()

    if timings:
        print(f"\nrendered {len(timings)} cards, avg {sum(timings) / len(timings):.2f}s/card")
    _build_contact_sheet()


def _build_contact_sheet() -> None:
    """Stack all bottom crops vertically with a label, save one PNG."""
    crops = sorted(BOTTOM_DIR.glob("*_bottom.png"))
    if not crops:
        print("no crops to stitch")
        return
    rows = [Image.open(p) for p in crops]
    target_w = max(r.width for r in rows)
    target_h_per_row = max(r.height for r in rows)
    sheet = Image.new("RGB", (target_w, target_h_per_row * len(rows)), (32, 32, 32))
    for i, row in enumerate(rows):
        sheet.paste(row, (0, i * target_h_per_row))
    sheet.save(CONTACT_SHEET)
    print(f"contact sheet → {CONTACT_SHEET}")


if __name__ == "__main__":
    main()
