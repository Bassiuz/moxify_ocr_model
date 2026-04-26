"""Build a labeled contact sheet for the CardConjurer spike output.

Layout per row: [bottom-strip at native res] [resized to 48x256 OCR scale] [label]
"""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

OUT_DIR = Path(__file__).resolve().parent.parent / "artifacts" / "cardconjurer_spike"
BOTTOM_DIR = OUT_DIR / "bottom_crops"
CONTACT_SHEET = OUT_DIR / "contact_sheet.png"

ROW_HEIGHT = 100
LABEL_WIDTH = 360
NATIVE_WIDTH = 720  # downscaled native crop preview
OCR_WIDTH = 256
OCR_HEIGHT = 48
PAD = 12

LABELS = {
    "01_baseline_en": "EN baseline — LEA 161/302 C",
    "02_german": "DE — LEA 161/302 C",
    "03_japanese": "JA (CJK) — LEA 161/302 C",
    "04_foil_en": "EN foil ★ — LEA 265/302 R",
    "05_x_collector": "EN — PMTG X12 M (unusual #)",
    "06_no_total": "EN — MID 042 R (no /total)",
    "07_korean": "KO (CJK) — MID 042 C",
    "08_french": "FR — MID 042 C",
    "09_high_rarity": "EN foil ★ — LEA 232/302 M",
    "10_mtga_long_set": "EN — VOW 123/277 U",
}


def main() -> None:
    crops = sorted(BOTTOM_DIR.glob("*_bottom.png"))
    if not crops:
        print("no crops found")
        return

    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 14)
    except Exception:
        font = ImageFont.load_default()

    sheet_w = NATIVE_WIDTH + PAD + OCR_WIDTH + PAD + LABEL_WIDTH + 2 * PAD
    sheet_h = ROW_HEIGHT * len(crops) + 2 * PAD
    sheet = Image.new("RGB", (sheet_w, sheet_h), (24, 24, 24))
    draw = ImageDraw.Draw(sheet)

    for i, crop_path in enumerate(crops):
        y = PAD + i * ROW_HEIGHT
        crop = Image.open(crop_path).convert("RGB")
        # Native preview: scale to NATIVE_WIDTH preserving aspect.
        native = crop.resize(
            (NATIVE_WIDTH, int(crop.height * NATIVE_WIDTH / crop.width)),
            Image.LANCZOS,
        )
        ny = y + (ROW_HEIGHT - native.height) // 2
        sheet.paste(native, (PAD, ny))

        # OCR-scale: 48x256 (what the model sees).
        ocr = crop.resize((OCR_WIDTH, OCR_HEIGHT), Image.LANCZOS)
        ox = PAD + NATIVE_WIDTH + PAD
        oy = y + (ROW_HEIGHT - OCR_HEIGHT) // 2
        sheet.paste(ocr, (ox, oy))

        # Border around OCR-scale to highlight it.
        draw.rectangle(
            (ox - 1, oy - 1, ox + OCR_WIDTH, oy + OCR_HEIGHT),
            outline=(80, 80, 80),
        )

        # Label
        stem = crop_path.stem.replace("_bottom", "")
        label = LABELS.get(stem, stem)
        lx = ox + OCR_WIDTH + PAD
        ly = y + ROW_HEIGHT // 2 - 10
        draw.text((lx, ly), label, fill=(220, 220, 220), font=font)

    # Header bar
    header = Image.new("RGB", (sheet_w, 30), (40, 40, 40))
    new_sheet = Image.new("RGB", (sheet_w, sheet_h + 30), (24, 24, 24))
    new_sheet.paste(header, (0, 0))
    draw = ImageDraw.Draw(new_sheet)
    draw.text(
        (PAD, 7),
        "left: native CardConjurer crop (720px wide) | right: 48x256 OCR scale | label",
        fill=(200, 200, 200),
        font=font,
    )
    new_sheet.paste(sheet, (0, 30))
    new_sheet.save(CONTACT_SHEET)
    print(f"sheet saved: {CONTACT_SHEET}  ({new_sheet.size})")


if __name__ == "__main__":
    main()
