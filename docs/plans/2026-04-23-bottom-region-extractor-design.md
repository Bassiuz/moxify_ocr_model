# Bottom-Region OCR Extractor — Design

**Status:** Design
**Author:** Bas
**Date:** 2026-04-23
**Supersedes:** Phase 1 "Number model" from `ocr_training_doc.md` §2. That parent doc remains the authority for the Latin/Japanese name models (Phases 2 & 3), the training infrastructure, data pipeline, and framework choice.

## 1. Why this exists

The parent design doc scopes Phase 1 as a "number model" whose label is a short string: `rarity + collector_number + set_code`. Two things changed since then:

1. **Expanded field set.** We want the bottom-left region to yield seven structured fields instead of three: collector number, set total, rarity, set code, language, foil indicator, and The-List indicator.
2. **Moxify handles foil and The-List outside OCR today.** Foil is a manual UI toggle (`isScanningFoil`), and The-List is penalized post-match via the `plst` set code. Surfacing both from OCR lets the UX layer show smarter defaults without forcing user taps.

This doc redesigns Phase 1 as a **bottom-region extractor**. Same CRNN+CTC architecture, broader alphabet, deterministic post-OCR parsing into structured fields.

## 2. Scope

**In scope**

- One CRNN+CTC model covering the full bottom-left region
- Structured output: seven fields (see §3)
- Label synthesis from Scryfall bulk data
- Symbol detection: foil indicator (★ vs •) and The-List planeswalker icon
- TFLite export + `.json` metadata + integration contract with Moxify

**Out of scope**

- Name region (Latin / Japanese) — covered by parent doc Phases 2 & 3
- Physical foil detection via reflectance/lighting — this model only reads the printed ★ glyph
- Pre-Mirage cards (1993–1996) that have no collector number printed — these fall back to name-region matching
- Oversized products (Planechase planes, Archenemy schemes) that don't match Moxify's portrait-aspect rectangle detector

**Covered in scope (clarification):** split cards, aftermath cards, and battles. Their artwork/name is rotated 90°, but the physical card is still portrait-oriented — the collector info sits in the standard bottom-left `numberRegion`. The rotated-name problem is handled separately by Moxify's existing `verticalNameRegion` + rotation pipeline.

## 3. Output contract

The CTC head emits a single string. A deterministic parser in `src/moxify_ocr/export/parse_bottom.py` reads it into:

```python
@dataclass
class BottomRegionResult:
    collector_number: str | None   # "0280"
    set_total: int | None          # 286 (only if "/NNN" was printed)
    rarity: str | None             # one of 'C','U','R','M','L' or None
    set_code: str | None           # "CLU"
    language: str | None           # "EN","DE","FR","IT","ES","PT","JA","RU","KO","ZH","PH","LA"
    foil_detected: bool | None     # True if ★ parsed; False if • parsed; None if neither found
    on_the_list_detected: bool     # True iff planeswalker icon class emitted
    raw: str                       # original CTC-decoded string (for debugging)
    confidence: float              # minimum per-timestep softmax along the chosen path
```

**Signal, not decision.** Per the user's direction, this project only emits signals. How Moxify applies them (prefill UI, replace the foil toggle, surface confidence) is a downstream UX decision tracked in the Moxify repo, not here.

## 4. Reading the printed bottom line

MTG bottom-line formats vary by era and layout. The model must handle:

| Era | Typical layout (left-to-right, possibly two lines) |
|---|---|
| Modern (2018–present) | `<rarity> <num>/<total>` then `<set> <foil> <lang>` |
| Mid-modern (2008–2017) | same structure, sometimes single-line |
| Pre-2008 | rarity letter may be absent; language may be absent |
| The List (2021–present) | planeswalker icon left of / replacing rarity; set code is usually `PLST` |
| Secret Lair / Showcase | same fields, different font/positioning — handled via augmentation, not label schema |

**Design decision:** CTC reads the entire region as one left-to-right top-to-bottom string, with `\n` synthesized into the label wherever the printed text wraps to a second line. The parser is line-agnostic — it searches the decoded string for vocabulary-anchored patterns (known set codes, language codes, digit runs, rarity letters, symbol glyphs) rather than relying on fixed positions.

Example labels:

```
0280/286 R\nCLU ★ EN
128/249 C\nM13 • EN
⟨PW⟩ 0041 PLST ★ EN
0127 SLD • JA
```

where `⟨PW⟩` denotes the single-class planeswalker-icon output, stored internally as Unicode PUA codepoint `U+E100` so it never collides with real text.

## 5. Model architecture

Same family as parent doc §4.2 "Number" row. Two deltas:

| Axis | Parent doc | This design | Why |
|---|---|---|---|
| Input (H × W) | 32 × 160 | **48 × 256** | Fit two text lines plus longer strings like `0280/286 M\nCLU ★ EN` |
| Alphabet size | 41 | **45** | Adds `\n`, `⟨PW⟩`, full uppercase A–Z coverage for language codes, `.` for decimal edge cases |
| Seq head | 1× BiLSTM-64 | **1× BiLSTM-96** | Slightly more capacity for the longer sequence |

- Stem, loss, decoder, framework, quantization, export pipeline: **inherited unchanged** from parent §4 & §6.
- Estimated TFLite INT8 size: **~450 KB** (parent estimate 300 KB; modest growth).
- Estimated Nord-class CPU latency: **3–6 ms** (parent: 2–5 ms). Still well inside the scan-loop budget; name models remain the dominant cost.

## 6. Alphabet

```
Digits       : 0 1 2 3 4 5 6 7 8 9                      (10)
Uppercase    : A B C D E F G H I J K L M N O P Q R S T
               U V W X Y Z                               (26)
Punctuation  : space  /  -  .                             (4)
Layout       : \n (line separator; never a visible glyph) (1)
Foil         : •  ★                                       (2)
The-List     : ⟨PW⟩  (U+E100, single class)               (1)
             ───────
Total                                                    44 + CTC blank = 45 outputs
```

Lowercase is excluded — the bottom region is uppercase-only. Non-ASCII letters are excluded — all set and language codes are ASCII.

## 7. Label synthesis from Scryfall

For each Scryfall printing we synthesize the expected bottom-line string from metadata fields:

```python
RARITY_LETTER = {"common": "C", "uncommon": "U", "rare": "R", "mythic": "M"}
# Basic lands print "L" even though Scryfall tags them 'common' — handled in _rarity_letter.

LANG_CODE = {
    "en": "EN", "de": "DE", "fr": "FR", "it": "IT", "es": "ES",
    "pt": "PT", "ja": "JA", "ru": "RU", "ko": "KO", "zhs": "ZH",
    "zht": "ZH", "ph": "PH", "la": "LA",
}

def make_label(card) -> str:
    line1 = []
    if card.set.code.lower() == "plst":
        line1.append("")  # planeswalker icon class

    num = card.collector_number
    if card.set.printed_size and _era_has_slash_total(card):
        line1.append(f"{num}/{card.set.printed_size}")
    else:
        line1.append(num)

    letter = _rarity_letter(card)  # 'L' if Basic Land, else RARITY_LETTER.get(card.rarity), else None
    if letter and card.set.code.lower() != "plst":
        line1.append(letter)

    foil_glyph = "★" if _this_sample_is_foil(card) else "•"
    line2 = f"{card.set.code.upper()} {foil_glyph} {LANG_CODE[card.lang]}"

    return " ".join(line1) + "\n" + line2


def _rarity_letter(card) -> str | None:
    if "Basic Land" in card.type_line:
        return "L"
    # Scryfall 'special' (e.g. Time Spiral Timeshifted) and 'bonus' (e.g. some
    # MB playtest cards) do not print a distinct letter on the card — skip them.
    return RARITY_LETTER.get(card.rarity)
```

The `_era_has_slash_total` helper is a small lookup table by set release year. `_this_sample_is_foil` is sampled per training example — see §8.

**Per-language examples.** Scryfall yields one entry per language printing, so the same card in `en` and `de` naturally produces two training samples with different language-code labels.

## 8. Symbol training-data strategy

This is the riskiest part of the design, because Scryfall images do not reliably show the physical foil star — most of the image dump is from non-foil prints.

| Symbol | Source | Strategy |
|---|---|---|
| `•` non-foil dot | Already baked into most Scryfall images | Use directly |
| `★` foil star | Rare in Scryfall images | Synthesize: overlay a rendered `★` glyph on top of the `•` pixels, with font-size jitter matched to the surrounding text. Produce foil labels for printings where `'foil' in finishes`. |
| `⟨PW⟩` The-List icon | Present in PLST Scryfall images | Use directly; also augment by cropping the icon and overlaying it onto non-PLST bottom-left crops to reach ≥ 5× class balance |

**Implementation:** a new module `src/moxify_ocr/data/symbol_overlay.py` that takes a cropped bottom-left region and returns a (possibly-augmented) image plus the matching label. Uses PIL text rendering with font-size auto-matched to detected surrounding text height.

**Reproducibility:** the overlay RNG seed is recorded in the per-example manifest alongside the Scryfall ID so training runs are replayable.

**Risk & mitigation:** synthetic `★` overlays may look different from real-camera foil stars in production. Mitigation — evaluate foil precision/recall against Moxify's real-world fixture set (parent §5.5) before shipping; if foil F1 < target, add real captured foil frames to training via the scan-test harness export.

## 9. Training & evaluation

Inherits parent §6 (AdamW, CTC loss, cosine LR decay, fp16 mixed precision, single mid-range GPU, Lambda/Vast budget). Adjustments:

| Hyperparam | Value |
|---|---|
| LR | 1e-3, cosine decay |
| Warmup steps | 500 |
| Batch size | **192** (parent 256; reduced for larger input) |
| Epochs | 40–60 |

Evaluation metrics build on parent §6.4 but add per-field accuracy:

| Metric | Target |
|---|---|
| CER on full bottom string | ≤ 3% |
| Exact match on full string | ≥ 88% |
| Collector-number exact match | ≥ 97% |
| Set-code exact match | ≥ 98% (vocab-constrained) |
| Language exact match | ≥ 99% (vocab-constrained) |
| Foil F1 (★ vs •) | ≥ 95% |
| The-List F1 (⟨PW⟩ present) | ≥ 97% |

Vocab-constrained fields score higher because the parser can validate outputs against the known set-code and language-code enumerations.

## 10. Integration contract with Moxify

**Additions to `OCRResult`** (`../moxify/lib/core/image_processing/ocr/ocr_result.dart`):

```dart
class OCRResult {
  // --- existing, unchanged ---
  final List<String> cardNameCandidates;
  final List<String> verticalCardNameCandidates;
  final String? collectorsNumber;
  final String? setCode;
  final Map<String, Uint8List> cutouts;
  final Map<String, String> additionalTexts;

  // --- new, emitted by CustomOcrService ---
  final int? setTotal;
  final String? rarity;             // 'C' | 'U' | 'R' | 'M' | 'S' | 'T' | 'B' | null
  final String? language;           // 'EN' | 'DE' | ... | null
  final bool? foilDetected;         // null = unknown
  final bool? onTheListDetected;    // null = unknown
  final double? bottomRegionConfidence;
}
```

**New service** `CustomOcrService` (parallel to the existing MLKit path):

```dart
class CustomOcrService {
  Future<void> initialize({required Uint8List bottomRegionModelBuffer});

  /// Null if confidence below threshold — caller falls back to MLKit.
  BottomRegionResult? recognizeBottomRegion(Uint8List jpegBytes);
}
```

**Fallback to MLKit** triggers when any of the following holds:

- Model inference throws
- Parser yields `collector_number == null AND set_code == null`
- `confidence < THRESHOLD` (calibrated on eval set, initial guess 0.55)

**Downstream UX is out of scope here** — Moxify decides how to consume the new signals (replace the foil toggle, prefill variants, surface confidence, etc.).

## 11. Project structure delta

No changes to the repo layout from parent §8. New files:

```
src/moxify_ocr/
  data/
    symbol_overlay.py           # NEW — §8
  export/
    parse_bottom.py             # NEW — §3 parser
configs/
  bottom_region_v1.yaml         # REPLACES number_v1.yaml
```

The parent's `number_v1.yaml` is retired; its content folds into `bottom_region_v1.yaml`.

## 12. Open questions

1. **Confidence calibration.** The 0.55 threshold in §10 is a starting guess. Calibrate against Moxify's real-world fixture eval to target ~5% false-fallback rate.
2. **Model version bumps.** The metadata JSON (`bottom_region_v1.json`) carries a `version` field; Moxify should refuse to load a model whose version is newer than the app's supported range. Define the version-compat matrix when integrating v1.

### Decided during brainstorming

- **Rarity alphabet:** `C U R M L` only. Scryfall `special` and `bonus` rarities don't print a distinct letter on cards, so they map to `rarity: None`.
- **Secret Lair / showcase frames:** no pre-planned fine-tune. SLD volume (~2,000+ cards in Scryfall) gives natural representation in training; include SLD in the held-out eval set and trigger hard-case mining (parent §6.6) only if SLD CER materially regresses vs. baseline.

## 13. Timeline delta vs parent doc

Parent §10 budgeted **~2 weeks** for the Number v1 model. This expanded scope adds:

- +3 days — symbol overlay synthesis & validation
- +2 days — per-field eval harness + confidence calibration
- +2 days — Moxify integration (new fields, `CustomOcrService` wiring)

**Revised Phase 1 estimate: ~3 weeks part-time.** Still inside the parent's 6-week-to-v2-ship target. Phases 2 and 3 remain unchanged.

## 14. References

- Parent design: [`ocr_training_doc.md`](../../ocr_training_doc.md)
- Moxify OCR isolate: `../moxify/lib/core/image_processing/ocr/ocr_isolate.dart`
- Moxify region config: `../moxify/lib/games/mtg/scanning/mtg_scan_regions.dart:32-37`
- Moxify result parser: `../moxify/lib/games/mtg/scanning/mtg_ocr_result_parser.dart`
- Moxify variant scoring (The List / foil): `../moxify/lib/core/image_processing/image_processing/image_processing_isolate.dart:1268-1288`
- Scryfall bulk data: https://scryfall.com/docs/api/bulk-data
