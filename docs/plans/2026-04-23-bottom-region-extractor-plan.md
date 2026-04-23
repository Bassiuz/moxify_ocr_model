# Bottom-Region OCR Extractor Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build, train, and export a TFLite CRNN+CTC model that reads the bottom-left region of MTG cards into 7 structured fields (collector number, set total, rarity, set code, language, foil, The-List), ready to drop into the Moxify Flutter app.

**Architecture:** Python/Keras 3 training project. Data pipeline pulls from Scryfall bulk API, synthesizes labels from card metadata, crops bottom-left regions from full-card images, and augments with synthetic foil-star and planeswalker-icon overlays. CRNN stem (MobileNetV3-Small) → 1× BiLSTM-96 → CTC head over a 45-class alphabet. Post-training INT8 quantization. Outputs `.tflite` + `.json` metadata, consumed by a new `CustomOcrService` in Moxify.

**Tech Stack:** Python 3.11, uv (env & packaging), TensorFlow ≥ 2.16 with Keras 3, Albumentations (augmentation), Pillow + OpenCV (imaging), Scryfall REST API, pytest + ruff + mypy (dev).

**Reference docs:**
- Design: [2026-04-23-bottom-region-extractor-design.md](2026-04-23-bottom-region-extractor-design.md)
- Parent OCR plan: [../../ocr_training_doc.md](../../ocr_training_doc.md)
- Moxify integration points: `../moxify/lib/core/image_processing/ocr/` and `../moxify/lib/games/mtg/scanning/`

---

## Milestone layout

| # | Milestone | Tasks | Output |
|---|---|---|---|
| 1 | Bootstrap | 1–3 | Empty repo, tools configured, first green test |
| 2 | Scryfall ingestion | 4–7 | Cached bulk data + card images + manifest |
| 3 | Label synthesis + parser | 8–11 | `make_label` / `parse_bottom` with full test coverage |
| 4 | Region cropping | 12–13 | Bottom-left crops from Scryfall images |
| 5 | Symbol overlays | 14–15 | Foil-star and planeswalker-icon synthesis |
| 6 | Dataset pipeline | 16–18 | tf.data.Dataset with augmentation + set-aware splits |
| 7 | Model + training | 19–22 | Trained Keras checkpoint hitting CER target on eval |
| 8 | Evaluation harness | 23–24 | Per-field metrics + Moxify real-world fixture eval |
| 9 | TFLite export | 25–27 | `.tflite` + `.json` with parity check |
| 10 | Hand-off | 28–29 | README, version-compat note, Moxify integration hints |

---

## Milestone 1: Bootstrap

### Task 1: Initialize git + Python project scaffold

**Files:**
- Create: `.gitignore`
- Create: `pyproject.toml`
- Create: `README.md`
- Create: `.python-version`

**Step 1.** Initialize the repo and commit the existing design docs as the starting state.

```bash
cd /Users/bassiuz/Projects/moxify_ocr_model
git init
git add ocr_training_doc.md docs/
git commit -m "chore: initial commit of design docs"
```

**Step 2.** Write `.python-version` with `3.11`.

**Step 3.** Write `pyproject.toml`:

```toml
[project]
name = "moxify-ocr"
version = "0.1.0"
description = "Custom TFLite OCR models for Moxify MTG card scanning."
requires-python = ">=3.11,<3.13"
dependencies = [
    "tensorflow>=2.16.0",
    "numpy>=1.26",
    "pillow>=10.0",
    "opencv-python>=4.9",
    "albumentations>=1.4",
    "requests>=2.31",
    "tqdm>=4.66",
    "pyyaml>=6.0",
    "editdistance>=0.8",
    "tensorboard>=2.16",
]

[project.optional-dependencies]
dev = ["pytest>=8", "ruff>=0.4", "mypy>=1.10", "types-requests", "types-PyYAML"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/moxify_ocr"]

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.ruff.lint]
extend-select = ["I", "UP", "B", "C4", "SIM"]

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["src"]
```

**Step 4.** Write `.gitignore`:

```
__pycache__/
*.py[cod]
.venv/
.mypy_cache/
.ruff_cache/
.pytest_cache/
.DS_Store

# gitignored data — see README
data/
artifacts/
.tensorboard/
```

**Step 5.** Write a minimal `README.md` (3-4 lines pointing at the design doc + plan).

**Step 6.** Set up the environment:

```bash
uv venv
uv pip install -e ".[dev]"
```

**Step 7.** Commit.

```bash
git add .gitignore pyproject.toml .python-version README.md
git commit -m "chore: bootstrap pyproject + env"
```

---

### Task 2: Create module skeleton + first passing test

**Files:**
- Create: `src/moxify_ocr/__init__.py` (with `__version__ = "0.1.0"`)
- Create: `src/moxify_ocr/data/__init__.py`
- Create: `src/moxify_ocr/models/__init__.py`
- Create: `src/moxify_ocr/train/__init__.py`
- Create: `src/moxify_ocr/eval/__init__.py`
- Create: `src/moxify_ocr/export/__init__.py`
- Create: `tests/__init__.py`
- Create: `tests/test_package.py`

**Step 1.** Write the test:

```python
# tests/test_package.py
from moxify_ocr import __version__


def test_version_exposed() -> None:
    assert __version__ == "0.1.0"
```

**Step 2.** Run it:

```bash
uv run pytest tests/test_package.py -v
```

Expected: PASS.

**Step 3.** Run ruff and mypy to confirm tooling works:

```bash
uv run ruff check src tests
uv run mypy src
```

Expected: both clean.

**Step 4.** Commit.

```bash
git add src tests
git commit -m "feat: package skeleton + smoke test"
```

---

### Task 3: Add CI-friendly `make` targets

**Files:**
- Create: `Makefile`

**Step 1.** Write:

```makefile
.PHONY: test lint typecheck fmt check

test:
	uv run pytest -v

lint:
	uv run ruff check src tests

typecheck:
	uv run mypy src

fmt:
	uv run ruff format src tests
	uv run ruff check --fix src tests

check: lint typecheck test
```

**Step 2.** Run `make check` — all three targets must pass.

**Step 3.** Commit.

---

## Milestone 2: Scryfall ingestion

### Task 4: Scryfall bulk-data downloader with disk cache

**Files:**
- Create: `src/moxify_ocr/data/scryfall.py`
- Create: `tests/data/__init__.py`
- Create: `tests/data/test_scryfall_bulk.py`

**Step 1.** Write tests first (TDD). The downloader should:
- Fetch the bulk-data index from `https://api.scryfall.com/bulk-data`
- Find the `default_cards` entry
- Download its `download_uri` to a cache directory if missing or older than N days
- Return the local path

```python
# tests/data/test_scryfall_bulk.py
from unittest.mock import patch
from pathlib import Path
import json

from moxify_ocr.data.scryfall import fetch_default_cards_path


def test_fetch_uses_cache_when_fresh(tmp_path: Path) -> None:
    cached = tmp_path / "default-cards.json"
    cached.write_text(json.dumps([{"id": "x"}]))
    # pretend file is 1 hour old; cache max-age is 7 days
    # fetch should NOT hit the network
    with patch("moxify_ocr.data.scryfall.requests.get") as mock_get:
        result = fetch_default_cards_path(cache_dir=tmp_path, max_age_days=7)
    mock_get.assert_not_called()
    assert result == cached


def test_fetch_downloads_when_missing(tmp_path: Path) -> None:
    # First call: index. Second call: the bulk JSON.
    index_response = {"data": [{"type": "default_cards",
                                "download_uri": "https://example.com/default.json"}]}
    with patch("moxify_ocr.data.scryfall.requests.get") as mock_get:
        mock_get.side_effect = [
            _mock_response(json=index_response),
            _mock_response(content=b'[{"id": "abc"}]'),
        ]
        path = fetch_default_cards_path(cache_dir=tmp_path, max_age_days=7)
    assert path.exists()
    assert json.loads(path.read_text()) == [{"id": "abc"}]
```

(Add a small `_mock_response` helper in the same file.)

**Step 2.** Run: `uv run pytest tests/data/test_scryfall_bulk.py -v`. Expected: FAIL (module missing).

**Step 3.** Implement `src/moxify_ocr/data/scryfall.py` with `fetch_default_cards_path(cache_dir, max_age_days) -> Path`. Use `requests` with a polite User-Agent header ("moxify-ocr training/0.1"). Honor Scryfall's 50-100ms rate limit if you add more calls later — but for a single bulk-data fetch per week, one call is fine.

**Step 4.** Run tests — must pass.

**Step 5.** Commit.

---

### Task 5: Scryfall card-image downloader

**Files:**
- Modify: `src/moxify_ocr/data/scryfall.py`
- Create: `tests/data/test_scryfall_images.py`

**Step 1.** Tests. The image downloader should:
- Accept a Scryfall card dict (with `image_uris.large` or `image_uris.normal`)
- Skip layouts we can't handle (`art_series`, `token`, `double_faced_token`, `scheme`, `plane`, `phenomenon`)
- Skip cards missing `image_uris` (e.g., `card_faces`-only cards — in v1 pull `card_faces[0].image_uris` if present, else skip)
- Return a local path: `<cache>/<first-two-of-id>/<id>.jpg`
- Respect Scryfall's rate limit (sleep 100 ms between requests)
- Be idempotent (don't re-download if file exists)

Write tests covering: skip-by-layout, already-cached skip, successful download, card-faces fallback.

**Step 2–4.** Run → fail → implement `download_card_image(card, cache_dir) -> Path | None` → pass.

**Step 5.** Commit.

---

### Task 6: Manifest writer/reader

**Files:**
- Create: `src/moxify_ocr/data/manifest.py`
- Create: `tests/data/test_manifest.py`

Manifest is a JSONL file where each line = `{scryfall_id, image_path, lang, set_code, collector_number, rarity, type_line, layout, finishes, image_sha256}`. Used so training is reproducible and so a later run can re-use already-downloaded images.

**Step 1.** Tests for `append_manifest_entry`, `read_manifest` (returns iterator), and `manifest_has(scryfall_id)`.

**Step 2–4.** TDD cycle.

**Step 5.** Commit.

---

### Task 7: Bulk-ingest script `scripts/ingest_scryfall.py`

**Files:**
- Create: `scripts/ingest_scryfall.py`

Loads bulk data → iterates cards → downloads image → appends manifest entry. Has a `--limit` flag for quick smoke tests.

**Step 1.** Write the script using `click` or `argparse` (argparse is in stdlib; prefer it).

**Step 2.** Smoke test: `uv run python scripts/ingest_scryfall.py --limit 20 --out data/scryfall`. Must produce 20 image files + a manifest line per card.

**Step 3.** Commit.

---

## Milestone 3: Label synthesis + parser

### Task 8: Era detection + rarity helper

**Files:**
- Create: `src/moxify_ocr/data/labels.py`
- Create: `tests/data/test_labels_helpers.py`

**Step 1.** Tests:

```python
def test_rarity_letter_basic_land_returns_L() -> None:
    card = _card(rarity="common", type_line="Basic Land — Mountain")
    assert _rarity_letter(card) == "L"

def test_rarity_letter_common() -> None:
    card = _card(rarity="common", type_line="Creature — Elf")
    assert _rarity_letter(card) == "C"

def test_rarity_letter_special_returns_none() -> None:
    card = _card(rarity="special", type_line="Creature — Wizard")
    assert _rarity_letter(card) is None

def test_era_has_slash_total_modern_yes() -> None:
    card = _card(released_at="2024-02-09")
    assert _era_has_slash_total(card) is True

def test_era_has_slash_total_pre_2008_no() -> None:
    card = _card(released_at="2005-07-15")
    assert _era_has_slash_total(card) is False
```

Include edge cases: rarity='bonus', tribal lands (not basic), dual-faced Basic Land transform (unlikely but check).

**Step 2–4.** TDD cycle. The `_era_has_slash_total` rule is: cards released 2008-01-01 or later **AND** not in an era-styled set. Keep it simple; a date cutoff is ~99% correct.

**Step 5.** Commit.

---

### Task 9: `make_label` with full test matrix

**Files:**
- Modify: `src/moxify_ocr/data/labels.py`
- Create: `tests/data/test_make_label.py`

**Step 1.** Write tests covering each branch:

```python
def test_make_label_modern_rare_english_nonfoil() -> None:
    card = _card(collector_number="0280", printed_size=286,
                 rarity="rare", set_code="CLU", lang="en",
                 type_line="Land — Island Mountain", released_at="2024-02-09")
    assert make_label(card, is_foil=False) == "0280/286 R\nCLU • EN"

def test_make_label_modern_mythic_german_foil() -> None:
    card = _card(collector_number="0128", printed_size=249,
                 rarity="mythic", set_code="M13", lang="de",
                 type_line="Creature — Dragon", released_at="2012-07-13")
    assert make_label(card, is_foil=True) == "0128/249 M\nM13 ★ DE"

def test_make_label_basic_land() -> None:
    card = _card(collector_number="270", printed_size=286,
                 rarity="common", set_code="CLU", lang="en",
                 type_line="Basic Land — Mountain", released_at="2024-02-09")
    assert make_label(card, is_foil=False) == "270/286 L\nCLU • EN"

def test_make_label_the_list_planeswalker_icon() -> None:
    card = _card(collector_number="0041", printed_size=None,
                 rarity="rare", set_code="PLST", lang="en",
                 type_line="Creature — Elf", released_at="2022-03-18")
    #  is the PUA codepoint for the planeswalker-icon class
    assert make_label(card, is_foil=True) == " 0041 PLST ★ EN"

def test_make_label_pre_2008_no_slash_total() -> None:
    card = _card(collector_number="128", printed_size=None,
                 rarity="common", set_code="MMQ", lang="en",
                 type_line="Creature — Goblin", released_at="1999-10-04")
    assert make_label(card, is_foil=False) == "128 C\nMMQ • EN"
```

**Step 2–4.** TDD cycle. Use the Python spec from the design doc §7 verbatim.

**Step 5.** Commit.

---

### Task 10: `parse_bottom` — inverse of `make_label`

**Files:**
- Create: `src/moxify_ocr/export/parse_bottom.py`
- Create: `tests/export/__init__.py`
- Create: `tests/export/test_parse_bottom.py`

Parser takes a raw CTC-decoded string + a `known_set_codes` list (loaded from Scryfall) + a `known_language_codes` list, and returns a `BottomRegionResult`.

**Step 1.** Tests round-trip every example from Task 9:

```python
def test_parse_modern_rare_english_nonfoil() -> None:
    result = parse_bottom("0280/286 R\nCLU • EN",
                          known_set_codes={"CLU", "M13"},
                          known_languages={"EN", "DE"})
    assert result.collector_number == "0280"
    assert result.set_total == 286
    assert result.rarity == "R"
    assert result.set_code == "CLU"
    assert result.language == "EN"
    assert result.foil_detected is False
    assert result.on_the_list_detected is False
```

Also test: parser is line-agnostic (handles `0280/286 R CLU • EN` single-line); handles noise characters from OCR errors by tolerating extra whitespace; returns `None` for unresolvable fields (e.g., unknown set code) rather than guessing.

**Step 2–4.** TDD. Parser strategy: tokenize on whitespace + newline, scan tokens for vocab matches (set codes, lang codes, rarity letters, foil symbols, planeswalker class), use digit-run regex for collector number and `\d+/\d+` for set total.

**Step 5.** Commit.

---

### Task 11: Round-trip property test

**Files:**
- Create: `tests/data/test_label_roundtrip.py`

A cheap-but-effective integration check: for a sample of real Scryfall cards (first 500 from the manifest), `parse_bottom(make_label(card))` must recover all fields.

**Step 1.** Write the test. Skip if manifest is empty (so test doesn't fail in clean checkouts).

**Step 2.** Run: `uv run python scripts/ingest_scryfall.py --limit 500 --out data/scryfall`, then pytest — must pass.

**Step 3.** Commit the test.

---

## Milestone 4: Region cropping

### Task 12: Bottom-left region cropper

**Files:**
- Create: `src/moxify_ocr/data/crop.py`
- Create: `tests/data/test_crop.py`

**Step 1.** Tests: given a known Scryfall image size (672 × 936), the bottom-left crop should be exactly the right pixel rectangle, with letterbox padding to 48 × 256.

```python
def test_bottom_left_crop_dimensions() -> None:
    img = Image.new("RGB", (672, 936), "white")
    crop = crop_bottom_region(img)
    assert crop.size == (256, 48)

def test_bottom_left_crop_aspect_preserved_with_letterbox() -> None:
    # Input Scryfall is 672x936; bottom-left region = (0, 0.9, 0.5, 1.0)
    # → 336 x 93.6 → should be letterboxed into 256x48 preserving aspect
    ...
```

**Step 2–4.** TDD cycle. Use Pillow for crop + letterbox (fill color `RGB(114,114,114)` per design §4.3).

**Step 5.** Commit.

---

### Task 13: End-to-end crop sanity with real cards

**Files:**
- Create: `scripts/dump_sample_crops.py`

Loads 20 manifest entries, crops each, writes to `data/debug_crops/<id>.jpg`. **Visually inspect a few** to confirm the crop covers the full collector line and set/foil/language row without cutting anything off. Commit the script (NOT the crops — they're gitignored).

---

## Milestone 5: Symbol overlays

### Task 14: Foil-star overlay synthesizer

**Files:**
- Create: `src/moxify_ocr/data/symbol_overlay.py`
- Create: `tests/data/test_symbol_overlay.py`
- Create: `assets/fonts/mplantin.ttf` (download the MPlantin-style font used by MTG, or use a close approximation like Beleren — check licensing and document source in the README)

**Step 1.** Tests:

```python
def test_overlay_foil_star_replaces_center_region(loaded_crop: Image.Image) -> None:
    out = overlay_foil_star(loaded_crop, seed=42)
    # out should differ from input in the center of the lower row
    diff = ImageChops.difference(loaded_crop, out).getbbox()
    assert diff is not None  # something changed

def test_overlay_foil_star_deterministic_with_seed(loaded_crop: Image.Image) -> None:
    a = overlay_foil_star(loaded_crop, seed=42)
    b = overlay_foil_star(loaded_crop, seed=42)
    assert list(a.getdata()) == list(b.getdata())
```

**Step 2–4.** TDD. Implementation:
- Find the `•` glyph in the image (look for a dark, small, roundish blob in the lower half — a template-match against a rendered `•` works fine)
- Render `★` at the same position + size using PIL text rendering
- Return the modified image

If glyph detection is unreliable on Scryfall imagery, fall back to a fixed-position overlay (derived from the cropped region's known layout). Document the fallback.

**Step 5.** Commit.

---

### Task 15: Planeswalker-icon overlay + cross-set augmentation

**Files:**
- Modify: `src/moxify_ocr/data/symbol_overlay.py`
- Modify: `tests/data/test_symbol_overlay.py`
- Create: `assets/icons/planeswalker.png` — cropped from a high-res PLST card image

**Step 1.** Write tests for `overlay_planeswalker_icon(crop, seed)` — asserts the icon is placed at the correct left-edge position and size-matched to surrounding text.

**Step 2–4.** TDD. For training augmentation: extract 5–10 real planeswalker icons from different PLST cards and randomly sample one per call.

**Step 5.** Commit.

---

## Milestone 6: Dataset pipeline

### Task 16: Train / val / test split generator

**Files:**
- Create: `src/moxify_ocr/data/splits.py`
- Create: `tests/data/test_splits.py`

**Step 1.** Tests: the splitter groups by `set_code` (all printings of a card in the same set go to the same split), produces deterministic 85/10/5 splits keyed on a seed, and refuses to put a held-out set in train.

**Step 2–4.** TDD cycle. Write a deterministic hash-based split: `hash(seed + set_code) % 100 < 85 → train`, etc.

**Step 5.** Commit.

---

### Task 17: Augmentation pipeline

**Files:**
- Create: `src/moxify_ocr/data/augment.py`
- Create: `tests/data/test_augment.py`

**Step 1.** Tests: augmentation applied to a fixed input is deterministic given a seed; shapes are preserved; output dtype is uint8.

**Step 2–4.** Implement using Albumentations per design §5.4 (rotate ±5°, perspective warp ±5%, brightness/contrast/saturation ±25%, Gaussian noise σ∈[0,8], JPEG Q∈[60,95], motion blur 3-7px @ 20% prob, Gaussian blur σ∈[0,1.2], moiré thin horizontal wave, chromatic aberration ±1px, vignette).

**Step 5.** Commit.

---

### Task 18: `tf.data.Dataset` builder

**Files:**
- Create: `src/moxify_ocr/data/dataset.py`
- Create: `tests/data/test_dataset.py`

**Step 1.** Tests: dataset yields `(image: uint8[48,256,3], label_sparse, label_length, image_length)` tuples; batches correctly; shuffle buffer works; per-call the symbol overlay may kick in based on the card's `finishes`.

**Step 2–4.** Implement:
- `build_dataset(manifest_path, split, alphabet, is_training=True) -> tf.data.Dataset`
- Parallel map for image loading, crop, overlay, augment, resize
- Encode labels as sparse tensors with the alphabet

**Step 5.** Commit.

---

## Milestone 7: Model + training

### Task 19: Generic CRNN builder

**Files:**
- Create: `src/moxify_ocr/models/crnn.py`
- Create: `tests/models/__init__.py`
- Create: `tests/models/test_crnn.py`

**Step 1.** Tests:
- `build_crnn(input_h, input_w, alphabet_size, lstm_units)` returns a `keras.Model`
- Output shape is `(batch, time, alphabet_size + 1)` where `time = input_w // 4` (after two 2× downsamples in the stem)
- Param count is within an expected range (sanity check: ~200K–400K for the bottom-region config)

**Step 2–4.** Implement with MobileNetV3-Small stem (`include_top=False`, reduced filters), a 2D→1D reshape, a BiLSTM, and a Dense classifier.

**Step 5.** Commit.

---

### Task 20: Bottom-region-specific config + CTC loss

**Files:**
- Create: `src/moxify_ocr/models/bottom_region.py`
- Create: `configs/bottom_region_v1.yaml`
- Create: `tests/models/test_bottom_region.py`

**Step 1.** Tests: `build_bottom_region_model()` produces the expected input/output shapes (48×256 in, time=64 out); CTC loss computes finite gradients on a tiny synthetic batch.

**Step 2–4.** TDD. Config YAML holds alphabet, input shape, LR schedule, batch size, epochs, paths.

**Step 5.** Commit.

---

### Task 21: Training entry point + metric callbacks

**Files:**
- Create: `src/moxify_ocr/train/train.py`
- Create: `src/moxify_ocr/train/callbacks.py`
- Create: `tests/train/__init__.py`
- Create: `tests/train/test_callbacks.py`

**Step 1.** Tests for callbacks — CER callback computes correct CER on a synthetic `(pred, label)` pair; field-level callback tracks per-field exact match.

**Step 2–4.** Implement. CLI entry:

```bash
uv run python -m moxify_ocr.train.train --config configs/bottom_region_v1.yaml
```

Callbacks: TensorBoard, EarlyStopping (patience 10 epochs on val CER), ModelCheckpoint (best val CER), the two custom metric callbacks.

**Step 5.** Commit.

---

### Task 22: First end-to-end smoke training run

**Files:**
- None modified; this is an execution task.

**Step 1.** Ensure `data/scryfall` has at least 5,000 cards ingested (`scripts/ingest_scryfall.py --limit 5000`).

**Step 2.** Run 3 epochs on a 500-card subset with batch 32:

```bash
uv run python -m moxify_ocr.train.train \
    --config configs/bottom_region_v1.yaml \
    --override data.limit=500 epochs=3 batch_size=32
```

Expected: loss decreases epoch-over-epoch (sanity check only). CER might still be poor with only 500 cards.

**Step 3.** Capture baseline metrics in `artifacts/smoke_run/metrics.json`. Commit this file as evidence the pipeline is wired correctly.

---

## Milestone 8: Evaluation harness

### Task 23: Evaluation harness with per-field metrics

**Files:**
- Create: `src/moxify_ocr/eval/eval.py`
- Create: `tests/eval/__init__.py`
- Create: `tests/eval/test_eval.py`

**Step 1.** Tests: eval on a fixed 10-sample dataset produces the expected CER and per-field exact-match numbers.

**Step 2–4.** TDD. Emits a `metrics.json` with: CER, full-string exact match, per-field exact match (collector, set, language, rarity), foil F1, The-List F1, P50/P95 inference latency.

**Step 5.** Commit.

---

### Task 24: Moxify fixture loader (real-world eval)

**Files:**
- Create: `src/moxify_ocr/eval/moxify_fixtures.py`
- Create: `scripts/pull_moxify_fixtures.py`
- Create: `tests/eval/test_moxify_fixtures.py`

The Moxify repo has `test/scan_loop/fixtures/generated/`. Each fixture has a camera-captured bottom-region crop + a ground-truth JSON. Export these into this project's `data/moxify_fixtures/` as `(image_path, label_json)` pairs.

**Step 1.** Tests: the loader reads a sample fixture directory, yields the expected tuples.

**Step 2.** `pull_moxify_fixtures.py` rsyncs the relevant directory from `../moxify` (explicit path, no assumptions about remotes).

**Step 3–4.** Implement loader. Exercise it against a real fixture subdirectory.

**Step 5.** Commit.

---

## Milestone 9: TFLite export

### Task 25: Keras → TFLite with INT8 post-training quantization

**Files:**
- Create: `src/moxify_ocr/export/tflite.py`
- Create: `tests/export/test_tflite.py`

**Step 1.** Tests: `export_tflite(keras_model, calibration_dataset, out_path)` produces a valid `.tflite` file; loading it back via `tf.lite.Interpreter` works; input/output tensor shapes match the Keras model.

**Step 2–4.** TDD. Use full-integer quantization with a representative dataset of ~500 calibration samples (drawn from the val split). Fall back to dynamic-range quantization if full-integer degrades CER > 0.5pp.

**Step 5.** Commit.

---

### Task 26: Metadata JSON writer

**Files:**
- Create: `src/moxify_ocr/export/metadata.py`
- Create: `tests/export/test_metadata.py`

Schema from design §7 (in the parent doc) §7.1.

**Step 1.** Tests: `write_metadata(out_path, model_config)` produces the exact JSON shape the Moxify integration expects.

**Step 2–4.** TDD.

**Step 5.** Commit.

---

### Task 27: Parity test (Keras vs TFLite INT8)

**Files:**
- Create: `src/moxify_ocr/export/validate.py`
- Create: `tests/export/test_validate.py`

**Step 1.** Tests: `validate_parity(keras_model, tflite_path, dataset)` returns a dict with `cer_delta` and `exact_match_delta`. Fail the test if `cer_delta > 0.005` (i.e., >0.5pp regression).

**Step 2–4.** TDD.

**Step 5.** Commit.

---

## Milestone 10: Hand-off

### Task 28: `README.md` + version-compat note

**Files:**
- Modify: `README.md`

Document:
- What the project produces (`bottom_region_v1.tflite` + `.json` in `artifacts/models/`)
- How to reproduce a training run end-to-end (bootstrap → ingest → train → export)
- The model version + the corresponding Moxify app version range that can consume it
- Known limitations (pre-Mirage cards, oversized products)

Commit.

---

### Task 29: Moxify integration hints doc

**Files:**
- Create: `docs/moxify_integration.md`

Concrete, copy-paste-ready notes for the Moxify side:
- Where to place `bottom_region_v1.tflite` + `.json` (`assets/ml/ocr/`)
- New fields to add to `OCRResult` (reproduce the Dart snippet from design §10)
- `CustomOcrService` skeleton Dart code
- Fallback-to-MLKit conditions
- A checklist mapping each design §12 open question to a specific resolution step during integration

Commit.

---

## Verification gate (before claiming v1 complete)

Run all of these and capture outputs in `artifacts/v1_verification/`:

1. `make check` — lint + type + tests all green
2. `uv run python -m moxify_ocr.eval.eval --model artifacts/models/bottom_region_v1.tflite --dataset data/scryfall --split test` — per-field metrics meet design §9 targets
3. `uv run python -m moxify_ocr.eval.eval --model artifacts/models/bottom_region_v1.tflite --dataset data/moxify_fixtures` — real-world CER within 2pp of synthetic-test CER
4. `uv run python -m moxify_ocr.export.validate --keras artifacts/checkpoints/best.keras --tflite artifacts/models/bottom_region_v1.tflite` — post-quant regression < 0.5pp
5. TFLite file size ≤ 500 KB
6. Model version-compat note written (Task 28)

All six must pass before the model is considered ready for Moxify integration.

---

## Notes for the executing engineer

- **TDD discipline:** every task has a test-first step. Do not skip. Write the test, see it fail, implement, see it pass, commit. The entire pipeline is testable — don't cheat on the ML parts by skipping tests.
- **Commit frequently:** each task ends with a commit. Don't batch commits across tasks.
- **Ask before training overnight:** tasks 22 and the final training run consume GPU budget. Get approval before committing to a full 40–60 epoch run.
- **Scryfall politeness:** never parallelize image downloads > 8 concurrent; always sleep 100 ms between API calls to the JSON endpoints. Their docs say "be kind or we'll ban you."
- **Data licensing:** see `ocr_training_doc.md §12`. Card images are Wizards' IP; training use is fine but never redistribute. Keep `data/` out of git (it is, in `.gitignore`).
