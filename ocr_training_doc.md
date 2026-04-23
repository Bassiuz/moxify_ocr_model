> **SUPERSEDED** for Phase 1 by [`docs/plans/2026-04-23-bottom-region-extractor-design.md`](docs/plans/2026-04-23-bottom-region-extractor-design.md).
> This document remains the authority for Phase 2 (Latin name model), Phase 3 (Japanese name model), and the shared training infrastructure.

# Custom OCR Training Project — Design & Handoff

**Status:** Design — ready to start as a new standalone project
**Author:** Bas
**Date:** 2026-04-23

## 1. Why this exists

Moxify currently uses `google_mlkit_text_recognition` for both the card **name** region (top of card) and the **collector number** region (bottom-left of modern cards). This works but has three well-understood limitations:

1. **Speed on older Android** — MLKit runs 100-300ms per recognizer call on mid-range Android CPUs (e.g. OnePlus Nord class). Each scan calls it 2-5 times. On a busy camera pipeline this is the dominant cost.
2. **Accuracy on small stylized text** — the collector-number region is ~10-14px of tiny stylized text. One-character errors happen regularly ("1117" read as "1177", "SLD" as "SLO"), which break exact DB lookup in the scan pipeline.
3. **Can't be tuned** — MLKit is closed-source and generic. Moxify's domain is narrow: ~30K known card names, known fonts (Beleren + a handful of retro/showcase variants), known region layouts. A domain-specific model can be an order of magnitude smaller, faster, and more accurate on this narrow task.

This project builds **custom TFLite OCR models** for Moxify's region-specific OCR needs. The end deliverable is a set of TFLite model assets + a short integration contract that the Moxify app loads at scan-isolate startup.

## 2. Scope

Three models, shipped in phases:

| # | Phase | Model | Target | Shipped as |
|---|---|---|---|---|
| 1 | v1 | **Number-region model** | `A-Z 0-9 / ★ • -` alphabet, short strings, known layout | Bundled asset |
| 2 | v2 | **Latin name-region model** | Latin card names (English / German / French / Italian / Spanish / Portuguese / Russian-transliteration) | Bundled asset |
| 3 | v3 (optional) | **Japanese name-region model** | Full kana + common kanji | Download-on-demand |

Korean and Simplified/Traditional Chinese models follow the same template as Japanese; build only after phase 3 ships and if telemetry shows demand.

## 3. Out of scope

- Rectangle extraction (already done by Moxify's YOLO pipeline; training images here consume pre-cropped regions)
- Card-name matching / fuzzy search (handled post-OCR by Moxify's existing `MtgCardMatcher` + FTS + SymSpell)
- Serving infrastructure — models run on-device; this project produces files, not a server
- Flutter integration code — lives in the Moxify repo, not here

## 4. Architecture

### 4.1 Model family

**CRNN with CTC loss**, one variant per region. This is the same architecture that works for license-plate OCR, receipt OCR, and every constrained-alphabet on-device OCR shipped in the last 5 years. Rationale over alternatives:

- **vs. Transformer (e.g. TrOCR-small)** — transformer encoder-decoders are overkill for short fixed-structure strings and too big for older Android CPU.
- **vs. Object-detection-style char-by-char** — CTC is simpler to train, handles variable-length output natively, produces strictly sequential character output which matches our domain.
- **vs. Image-to-text (BLIP-style)** — way too big; offline inference burns seconds.

### 4.2 Per-region architecture choices

| Region | Stem | Sequence head | Decoder | Params | TFLite INT8 size |
|---|---|---|---|---|---|
| Number | MobileNetV3-Small (reduced filters) | 1× BiLSTM-64 | CTC greedy | ~200K | ~300KB |
| Latin name | MobileNetV3-Small (full) | 2× BiLSTM-128 | CTC greedy | ~1M | ~1.5MB |
| Japanese name | MobileNetV3-Small + ViT bottleneck | 2× BiLSTM-256 | CTC greedy | ~3M | ~3MB |

Latencies (estimates from comparable license-plate OCR deployments — **validate on real hardware early**):

| Region | Nord-class Android CPU | Modern iPhone |
|---|---|---|
| Number | 2-5ms | <2ms |
| Latin name | 10-30ms | 5-10ms |
| Japanese name | 20-40ms | 10-20ms |

### 4.3 Input normalization

All models take a fixed input aspect ratio. CRNN output sequence length is proportional to input width, so a consistent aspect matters.

| Region | Input size | Aspect | Notes |
|---|---|---|---|
| Number | 32 × 160 (H × W) | 1:5 | Short strings, small text |
| Latin name | 48 × 384 | 1:8 | Longer names like "Elesh Norn, Grand Cenobite" |
| Japanese name | 48 × 288 | 1:6 | Japanese names tend to be shorter character-count-wise |

Letterbox padding (not stretch) to preserve aspect ratio on crops that don't match.

### 4.4 Alphabet per model

```
Number model   : 0-9 A-Z / ★ • - space (blank = CTC blank)  → 41 classes
Latin name     : a-z A-Z 0-9 space apostrophe hyphen comma period colon
                 accented Latin (à-ÿ, Ä-Ÿ subset)             → ~90 classes
Japanese name  : Latin digits + common Japanese punctuation
                 hiragana (46) + katakana (46) + top-2000 kanji by Scryfall
                 frequency                                    → ~2,100 classes
```

The Japanese alphabet is clipped to the top 2,000 kanji by frequency in actual MTG printed names (we measure this from Scryfall data). Any card using kanji outside that set falls back to MLKit. Coverage target: 99%+ of Japanese cards.

## 5. Data pipeline

### 5.1 Source: Scryfall bulk data

Scryfall publishes daily bulk dumps:

- `default-cards.json` — one object per language printing (~400K entries)
- Each entry has:
  - `printed_name` — ground-truth label in the card's language
  - `image_uris.normal` / `image_uris.large` — card image URL
  - `lang` — ISO language code
  - `collector_number` + `set` + `rarity` — for the number model's label
  - `layout` — e.g. `normal`, `split`, `flip`, `transform`, `showcase`

**Download & caching:**
- Fetch bulk data weekly (not per run)
- Cache card images locally at highest available quality (`large` = 672 × 936 px)
- Keep a manifest: `scryfall_id → local_path` for reproducibility

### 5.2 Region cropping

Training data is *cropped regions*, not full cards. We leverage Moxify's existing knowledge of where these regions sit on a detected card:

```python
# Coordinates from moxify's mtg_scan_regions.dart — single source of truth.
NAME_REGION   = (0.04, 0.04, 0.90, 0.10)   # x1, y1, x2, y2  (fractions of card)
NUMBER_REGION = (0.00, 0.90, 0.50, 1.00)
# Vertical name region for tall-format cards:
VERT_NAME_REGION = (0.04, 0.06, 0.12, 0.50)  # rotated 90° at load
```

For training we **don't need a live YOLO detection** — Scryfall images are already pre-cropped card images (no background, straight angle, clean). Cropping uses the fractions above directly. This is much cleaner training signal than camera-derived crops.

### 5.3 Label generation

| Model | Label source |
|---|---|
| Number | Composed: `rarity + " " + collector_number + " " + set_code.upper()` — matches the printed bottom line format |
| Latin name | `printed_name` (verbatim; uppercase/lowercase preserved as Scryfall has it) |
| Japanese name | `printed_name` for cards where `lang == "ja"` |

Labels are Unicode-normalized (NFC) before training. All labels are character-segmented for CTC.

### 5.4 Augmentation

The gap between "Scryfall image" (clean, straight, perfect lighting) and "real camera capture through the Moxify pipeline" is huge. Augmentation closes it:

**Geometric**
- Rotation ±5°
- Perspective warp ±5% corners
- Translation ±3%
- Scale 0.85-1.15×

**Photometric**
- Brightness ±25%
- Contrast ±25%
- Saturation ±20%
- Gaussian noise σ=[0, 8]
- JPEG re-encoding Q=[60, 95]
- Motion blur kernel 3-7px (20% probability)
- Gaussian blur σ=[0, 1.2]

**Camera-specific (add on top)**
- Moiré simulation (thin horizontal wave) — simulates laptop-screen capture for scan-test rig
- Chromatic aberration ±1px
- Vignetting

Train/val/test split: use Scryfall's `set` groupings to prevent leakage — all printings of the same card in the same set go to the same split. Hold out e.g. 5% of sets for test.

### 5.5 Real-world evaluation set

Moxify already has labeled camera-frame fixtures in `test/scan_loop/fixtures/generated/`. Export a subset as an **external eval set** for this project — synthetic-trained models are scored against real captures to catch domain gap. This is the unique advantage over "generic" OCR training efforts; use it.

## 6. Training

### 6.1 Framework

**Keras 3 with TensorFlow backend.** Rationale:

- Direct TFLite export, no extra conversion step
- Integer-quantization tooling (post-training + QAT) is mature
- Reasonable PyTorch compatibility if a contributor prefers it (Keras 3 can swap backends)

Alternative if you prefer PyTorch: train in PyTorch → ONNX → ONNX→TFLite via `onnx2tf`. Works but adds a conversion step that sometimes loses accuracy. Stick with Keras unless there's a reason to diverge.

### 6.2 Loss & decoder

- **CTC loss** at train time (handles variable-length output without alignment)
- **CTC greedy decode** at inference (beam search only if greedy hurts measurably)
- Optionally a language-model beam search with Scryfall name prior for the Latin model — but ship greedy first, measure, add only if needed

### 6.3 Training config

| Hyperparam | Number | Latin name | Japanese name |
|---|---|---|---|
| Optimizer | AdamW | AdamW | AdamW |
| LR | 1e-3 (cosine decay) | 5e-4 (cosine decay) | 3e-4 (cosine decay) |
| Warmup steps | 500 | 1,000 | 2,000 |
| Batch size | 256 | 128 | 64 |
| Epochs | 30-50 | 50-100 | 100-200 |
| Mixed precision | fp16 | fp16 | fp16 |
| GPU needed | Single mid-range (L4, RTX 4070) | Single mid-range | Single high-end (A10, RTX 4090) — kanji embedding is memory-heavy |

All three models train comfortably on a single GPU. Cloud budget: probably $50-150 total on Lambda Labs / Vast.ai / Colab Pro+ for all experiments across all three phases. No distributed training.

### 6.4 Evaluation metrics

At each eval step:

| Metric | Definition | Target |
|---|---|---|
| **CER** (Character Error Rate) | Levenshtein(pred, label) / len(label) | ≤ 2% (number), ≤ 5% (Latin), ≤ 8% (Japanese) |
| **Exact match rate** | `pred == label` | ≥ 92% (number), ≥ 70% (Latin), ≥ 60% (Japanese) |
| **Top-N recoverable** | Expected label within Levenshtein-2 of any top-N beam | ≥ 98% |
| **P50 inference latency** | Measured on a reference Android device | Budget in 4.2 table |
| **P95 inference latency** | Same | ≤ 2× P50 |

Exact-match numbers look low on names because *the matching pipeline downstream doesn't need exact — it needs "close enough that fuzzy FTS recovers it"*. The Top-N recoverable metric is the one that predicts end-to-end scan accuracy; tune for that.

### 6.5 Post-training quantization (PTQ)

- Calibration dataset: ~500 real Scryfall image regions
- Mode: full-integer INT8 quantization (weights + activations)
- Validate: post-quant CER must not regress more than +0.5 absolute percentage points
- Fall back to dynamic-range quantization if full-integer hurts accuracy too much

### 6.6 Hard-case sampling

After a first training run completes, identify the cards where the model fails (CER > 10%). Common buckets:

- Showcase / borderless frames
- Retro / alpha-style cards
- Cards with foil treatment in the training image
- Promo stamps

Over-sample these by 5-10× in a fine-tuning pass. This is the single most effective accuracy improvement after the baseline is up.

## 7. Integration contract with Moxify

The training project ships **two files per model**:

| File | Purpose |
|---|---|
| `<model>.tflite` | Quantized TFLite graph |
| `<model>.json` | Metadata: alphabet, input shape, preprocessing, model version |

### 7.1 Metadata JSON schema

```json
{
  "version": "number-v1.0.0",
  "model_type": "number" | "name_latin" | "name_japanese",
  "input_shape": [1, 32, 160, 3],
  "input_dtype": "uint8",
  "input_preprocessing": {
    "letterbox_pad_rgb": [114, 114, 114],
    "normalize": null
  },
  "output_shape": [1, 40, 42],
  "alphabet": "0123456789ABC...★•- ",
  "ctc_blank_index": 0,
  "trained_at": "2026-05-15",
  "scryfall_data_hash": "sha256:..."
}
```

### 7.2 Moxify-side integration

Moxify adds a new `CustomOcrService` alongside `MtgOcrResultParser` with roughly this API:

```dart
class CustomOcrService {
  Future<void> initialize({
    required Uint8List numberModelBuffer,
    required Uint8List nameLatinModelBuffer,
    Uint8List? nameJapaneseModelBuffer, // optional
  });

  /// Returns null if confidence too low — caller falls back to MLKit.
  String? recognizeNumber(Uint8List jpegBytes);
  String? recognizeName(Uint8List jpegBytes, {String script = 'latin'});
}
```

Loaded from `assets/ml/ocr/` at scan-isolate startup. Japanese model downloaded lazily via Moxify's existing download-manager when `detectedLanguage == 'ja'` is observed on-device.

### 7.3 Shipping & rollout

- **v1 (number)**: bundled as asset, rolls out with the next Moxify release. No feature flag — ship directly since the number path has a clear MLKit fallback.
- **v2 (Latin name)**: bundled asset, gated behind a remote-config flag `use_custom_latin_ocr` so we can A/B and roll back fast.
- **v3 (Japanese name)**: downloaded lazily (~3MB), only after a user scans a card where `detectedLanguage == 'ja'`. Local cache invalidation on model version bump.

## 8. Project repo structure

Suggested layout for the standalone training project:

```
moxify-ocr-train/
├── README.md
├── pyproject.toml                # Python 3.11+, uv / pdm
├── .gitignore                    # ignore data/, checkpoints/, artifacts/
│
├── src/moxify_ocr/
│   ├── data/
│   │   ├── scryfall.py           # bulk-data fetch, manifest, image download
│   │   ├── crop.py               # region extraction from Scryfall images
│   │   ├── augment.py            # augmentation pipeline (Albumentations)
│   │   ├── dataset.py            # tf.data.Dataset builder
│   │   └── splits.py             # train/val/test split generator
│   │
│   ├── models/
│   │   ├── crnn.py               # generic CRNN builder (stem + seq + CTC)
│   │   ├── number_model.py       # phase-1 spec
│   │   ├── name_latin.py         # phase-2 spec
│   │   └── name_ja.py            # phase-3 spec
│   │
│   ├── train/
│   │   ├── train.py              # single entry point: `python -m moxify_ocr.train.train --config configs/number_v1.yaml`
│   │   ├── callbacks.py          # CER logger, TensorBoard, early stop
│   │   └── hard_cases.py         # hard-case mining for fine-tune passes
│   │
│   ├── eval/
│   │   ├── eval.py               # run model against a dataset, emit metrics
│   │   ├── moxify_fixtures.py    # load Moxify scan-test fixtures as eval set
│   │   └── benchmark_latency.py  # device-adb latency bench
│   │
│   └── export/
│       ├── tflite.py             # Keras → TFLite with PTQ
│       ├── metadata.py           # write <model>.json alongside
│       └── validate.py           # post-export parity test
│
├── configs/
│   ├── number_v1.yaml
│   ├── name_latin_v1.yaml
│   └── name_ja_v1.yaml
│
├── notebooks/                    # ad-hoc exploration
│
├── data/                         # gitignored
│   ├── scryfall/                 # raw bulk data + images
│   ├── crops/                    # generated training crops
│   └── moxify_fixtures/          # pulled from Moxify repo
│
├── artifacts/                    # gitignored
│   ├── checkpoints/
│   └── models/                   # final .tflite + .json pairs
│
└── scripts/
    ├── download_scryfall.sh
    ├── train_all.sh
    └── export_all.sh
```

### 8.1 Dependencies

```toml
# pyproject.toml
[project]
dependencies = [
    "tensorflow>=2.16.0",          # includes Keras 3
    "numpy",
    "pillow",
    "opencv-python",
    "albumentations",              # augmentation
    "requests",
    "tqdm",
    "pyyaml",
    "editdistance",                # CER computation
    "tensorboard",
]

[project.optional-dependencies]
dev = ["pytest", "ruff", "mypy"]
onnx = ["onnx", "onnxruntime", "onnx2tf"]  # only if PyTorch branch chosen
```

Python 3.11 minimum. uv or pdm for env management. Single `make train-number` / `make export-number` entry points.

## 9. Evaluation loop with Moxify

Integration happens before shipping. The training project stays standalone; the loop is:

1. Train model in this repo → produce `.tflite` + `.json`
2. Copy into Moxify repo: `assets/ml/ocr/<model>.tflite` + `assets/ml/ocr/<model>.json`
3. Flip a branch flag in Moxify: `USE_CUSTOM_OCR = true`
4. Run `scripts/run_scan_test_remote_both.sh total_100` (Moxify's scan-test harness)
5. Compare accuracy + p50/p95 latency against the last known baseline
6. If improved: merge. If regressed on some axis: iterate on training.

This gives us a real end-to-end eval loop — not just CER on synthetic test data.

## 10. Timeline

| Phase | Elapsed weeks | Wall-clock (part-time) | Milestones |
|---|---|---|---|
| 0: bootstrap | 0 → 1 | ~1 week | repo scaffolding, Scryfall fetcher, crop pipeline, Moxify fixture loader |
| 1: number v1 | 1 → 3 | ~2 weeks | first model trained, TFLite exported, >92% exact on eval set, shipped as asset |
| 2: Latin name v1 | 3 → 6 | ~3 weeks | trained, exported, ≥70% exact / ≥98% top-N recoverable on eval, flag-gated in Moxify |
| 3: Japanese name v1 (optional) | 6 → 10 | ~4 weeks | only if CJK telemetry justifies — trained, exported, lazy-downloaded |

**Realistic to v2 ship: 6 weeks part-time.** v3 is a separate decision after v1+v2 are in production.

## 11. Open questions for the project owner

Keep these in the project README, fill in as you learn:

1. **Training hardware**: single RTX 4090 (~$0.50/hr on Lambda), Vast.ai mid-tier, or local if someone has a decent GPU? Budget implications are small.
2. **Scryfall bulk-data cadence**: weekly re-download, or only when new MTG sets release? (We recommend weekly to pick up errata.)
3. **Model versioning**: semver or date-based? How do we manage asset version pinning in Moxify?
4. **Fallback policy**: is MLKit always the fallback, or do we retire it once a custom model passes eval?
5. **Telemetry**: do we ship a per-method detection counter to Firebase so we know which path (custom / MLKit) runs in production?
6. **Japanese kanji coverage**: does the v3 alphabet include OLD Japanese printings (pre-4E), or only modern? Affects training-data scope.

## 12. Licensing & credits

- **Scryfall data**: CC0 (bulk data license) — free to use for training
- **Card images**: bound by Wizards' fan-content / reference policy — training use is fine, redistribution is not. Don't upload training data publicly.
- **Our models**: choose a license at project-start. MIT is the usual for this kind of utility ML asset.

## 13. References

- fast-plate-ocr — <https://github.com/ankandrew/fast-plate-ocr> (CCT + CTC for license plates — closest architectural analog)
- PaddleOCR PP-OCRv5 — <https://paddlepaddle.github.io/PaddleOCR/main/en/version3.x/algorithm/PP-OCRv5/PP-OCRv5.html> (state of mobile OCR in 2026 — aspirational benchmark)
- Scryfall bulk data — <https://scryfall.com/docs/api/bulk-data>
- Moxify scan-test harness — `docs/plans/2026-04-17-scan-loop-test-harness-design.md`
- Moxify region coordinates — `lib/games/mtg/scanning/mtg_scan_regions.dart`
- Moxify OCR pipeline — `lib/core/image_processing/ocr/ocr_isolate.dart`
