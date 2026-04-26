# v3 Synthetic-Only Training Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Retrain the bottom-region OCR using only synthetic data (CardConjurer pool + line_compositor real-half stitcher) and validate on real Scryfall scans the model has never seen during training. This mirrors the YOLO-style approach the user used successfully on the card-detection model: synthetic-only train → arbitrary scale + perfect labels; real val → measures real-world accuracy.

**Architecture:** No model-architecture changes — keep [v2](docs/plans/2026-04-24-v2-plan.md) (custom OCR stem + 2× BiLSTM + CTC head). All gains here come from data: a 100K-card synthetic pool plus runtime stitching that effectively multiplies variety, with stronger augmentation to bridge the synthetic→real domain gap.

**Tech stack:** Python 3.11 + uv, Playwright (rendering), TensorFlow / `tf.data` (training), Albumentations (augmentation), all already in the project.

**Depends on:** [docs/plans/2026-04-26-cardconjurer-synthetic-renderer.md](docs/plans/2026-04-26-cardconjurer-synthetic-renderer.md) Tasks 1-7 must be complete (renderer working, dataset wiring in place). This plan starts where that one leaves off — at the pre-render step — and pushes through to a trained v3 checkpoint.

---

## What changes vs v2

| | v2 baseline | v3 synthetic-only |
|---|---|---|
| Train data | Real Scryfall renders (set-aware split) | CardConjurer + line_compositor (no real Scryfall) |
| Train pool size | ~9K real cards × `.repeat()` | 100K synthetic + infinite stitch combinations |
| `cardconjurer_ratio` | 0.0 (disabled) | 0.70 |
| `line_compositor_ratio` | already wired | 0.30 |
| Real-Scryfall in train | 100% | 0% (val/test only) |
| Augmentation | conservative (rotate ±3°, blur σ≤1) | stronger to bridge synthetic→real gap |
| Val / test | Scryfall set-aware split | unchanged — same val/test sets |
| Epochs | 30 | 50 (more synthetic samples → can overfit slower) |

The val/test sets staying identical is the key invariant that lets us compare v3 vs v2 val_cer apples-to-apples.

## Why this should work

1. **Perfect labels.** Synthetic data is generated FROM a known field tuple, so labels can't be wrong. Scryfall data labels can be wrong (occasional manifest oddities — promo collector numbers, unusual `printed_size` fields, missing rarities).
2. **Coverage of rare classes.** Real Scryfall data is heavily skewed toward common rarities + EN. Synthetic generation can sample uniformly across rarities, languages, and number formats.
3. **Real val captures what matters.** The model has to read real cards in production. If val_cer drops vs v2 even though train has zero real data, that's strong evidence the synthetic pool generalizes.

## Risks

1. **Synthetic→real gap is real.** Synthetic crops have pure black backgrounds and pure white text; Scryfall scans have ink bleed, slight color casts, anti-aliasing artifacts. Stronger augmentation should bridge this, but if it doesn't we may need to mix in a small `real_ratio` (say 0.10) and call this "near-synthetic" instead of "pure synthetic." Task 7 has an explicit checkpoint for this.
2. **100K render is ~28 hours.** Run overnight + day. If the smoke test in Task 2 reveals issues, we eat that time twice.
3. **line_compositor leg may be redundant or harmful.** The whole point of CardConjurer was to replace synthetic.py; line_compositor is a third leg. If it doesn't help (or hurts) val_cer, drop it. Task 9 includes the ablation.

---

## Background reading for the implementer

1. [docs/plans/2026-04-24-v2-plan.md](docs/plans/2026-04-24-v2-plan.md) — what v2 changed and why it plateaued.
2. [docs/plans/2026-04-26-cardconjurer-synthetic-renderer.md](docs/plans/2026-04-26-cardconjurer-synthetic-renderer.md) — the renderer. This plan assumes Tasks 1-7 there are merged.
3. [src/moxify_ocr/data/dataset.py](src/moxify_ocr/data/dataset.py) — config + sampling logic. Look at how `synthetic_ratio` and `cardconjurer_ratio` route samples.
4. [src/moxify_ocr/data/line_compositor.py](src/moxify_ocr/data/line_compositor.py) — the existing real-half stitcher. Note it samples lazily at runtime (zero render cost).
5. [src/moxify_ocr/data/augment.py](src/moxify_ocr/data/augment.py) — current pipeline. The "design doc §5.4 called for more aggressive augmentation; we've scaled back" comment is exactly the constraint we're now relaxing.
6. [configs/bottom_region_v1.yaml](configs/bottom_region_v1.yaml) — the existing config. v3 forks from this.

---

## Task 1: Wire `line_compositor` into `dataset.py` (TDD)

**Why:** The real-half stitcher is implemented and tested in isolation but not yet a runtime data source. Add a `line_compositor_ratio` knob parallel to `cardconjurer_ratio`.

**Files:**
- Modify: `src/moxify_ocr/data/dataset.py`
- Modify: `tests/data/test_dataset.py`

**Step 1: Write failing test**

In `tests/data/test_dataset.py`, add:

```python
def test_dataset_routes_to_line_compositor_when_ratio_is_one(tmp_path):
    """With line_compositor_ratio=1.0 every sample must come from the stitcher."""
    # Build a tiny manifest + image fixture (mirror existing tests) and a
    # cardconjurer_pool with a sentinel image. Assert that with
    # line_compositor_ratio=1.0 and cardconjurer_ratio=0.0, no sample matches
    # the cardconjurer-pool sentinel.
    ...
```

**Step 2: Run test, verify it fails**

```bash
.venv/bin/pytest tests/data/test_dataset.py::test_dataset_routes_to_line_compositor_when_ratio_is_one -v
```

Expected: FAIL — option not wired.

**Step 3: Add config field**

In `BottomRegionDatasetConfig`:

```python
# Path to manifest used by line_compositor for its real-half pool.
# Defaults to the same manifest used for the real-data leg.
line_compositor_manifest: Path | None = None
line_compositor_ratio: float = 0.0
```

**Step 4: Wire the branch**

Mirror the cardconjurer branch in the per-sample loop:

```python
from moxify_ocr.data.line_compositor import LineLibrary, composite_sample

lc_lib: LineLibrary | None = None
if config.line_compositor_ratio > 0:
    lc_manifest = config.line_compositor_manifest or config.manifest
    lc_lib = LineLibrary.build(lc_manifest, config.images_root)
    if lc_lib.is_empty():
        raise ValueError(
            f"line_compositor manifest at {lc_manifest} produced an empty library"
        )

# In the loop, BEFORE the cardconjurer check:
if lc_lib is not None and rng.random() < config.line_compositor_ratio:
    lc_seed = seed * 1_000_007 + 20_000_000 + lc_counter
    lc_counter += 1
    lc_img, lc_label = composite_sample(lc_lib, seed=lc_seed)
    lc_ids = np.asarray(encode_label(lc_label), dtype=np.int32)
    yield lc_img, lc_ids, np.int32(len(lc_ids))
    continue
```

**Step 5: Run all dataset tests**

```bash
.venv/bin/pytest tests/data/ -v
```

Expected: all green.

**Step 6: Commit**

```bash
git add src/moxify_ocr/data/dataset.py tests/data/test_dataset.py
git commit -m "feat(data): wire line_compositor leg into dataset config"
```

---

## Task 2: Smoke render — 1K cards before committing to the 100K run

**Why:** A 100K render is ~28 hours. The CardConjurer-renderer plan's Task 5 already does a 100-card smoke; this task scales it up to 1K to stress-test for memory leaks, browser-tab restarts, and render-rate regressions before the overnight run.

**Files:**
- Modify: `data/synth_cardconjurer/smoke_v3/` (gitignored)
- Modify: `artifacts/cc_smoke_1k_contact_sheet.png` — committed sample for the user to review

**Step 1: Render 1000 cards**

```bash
infra/cardconjurer/start.sh
.venv/bin/python scripts/render_cardconjurer_pool.py --n 1000 --seed 0 \
  --out-dir data/synth_cardconjurer/smoke_v3
```

Expected: ~18-20 min wall time. 1000 PNGs + labels.jsonl. **No memory growth past ~500 MB** in the chromium process — if it climbs past 1 GB, the per-1000 page-restart in the renderer isn't kicking in and Task 4 of the renderer plan needs a fix.

**Step 2: Sample a contact sheet from the 1K pool**

Adapt `scripts/_build_spike_contact_sheet.py` to randomly pick 20 entries from the 1K pool and write `artifacts/cc_smoke_1k_contact_sheet.png`.

**Step 3: User checkpoint**

Pause and ask the user: "open `artifacts/cc_smoke_1k_contact_sheet.png` — these 20 random samples representative of what you want? Foil ★ visible on some, no holo/List artifacts, font fidelity matches what you approved earlier?"

If user says yes → proceed. If no → fix and re-smoke.

**Step 4: Commit the contact sheet** (only the contact sheet, not 1K PNGs)

```bash
git add artifacts/cc_smoke_1k_contact_sheet.png
git commit -m "test(data): cardconjurer 1k smoke pool sample"
```

---

## Task 3: Generate the 100K pool

**Why:** This is the actual training data. Run when you're confident in Task 2.

**Files:**
- Modify: `data/synth_cardconjurer/v3_100k/` (gitignored)

**Step 1: Render the pool**

```bash
infra/cardconjurer/start.sh
.venv/bin/python scripts/render_cardconjurer_pool.py --n 100000 --seed 0 \
  --out-dir data/synth_cardconjurer/v3_100k 2>&1 \
  | tee artifacts/render_v3_100k.log
```

Expected: ~28-30 hours. Run overnight + into next day. Tee'd log preserves the throughput line per 100 cards so we can post-mortem any slowdowns.

**Step 2: Verify pool integrity**

```bash
.venv/bin/python -c "
from pathlib import Path
from moxify_ocr.data.cardconjurer_dataset import CardConjurerPool
pool = CardConjurerPool.load(Path('data/synth_cardconjurer/v3_100k'))
print(f'pool size: {len(pool)}')
assert len(pool) >= 99500, f'too many failed renders: {len(pool)}'
"
```

Expected: pool size ≥ 99,500 (allow <0.5% failures).

**Step 3: Commit the render log** (no PNGs — already gitignored)

```bash
git add artifacts/render_v3_100k.log
git commit -m "data: render log for 100k v3 cardconjurer pool"
```

---

## Task 4: Strengthen the augmentation pipeline (TDD)

**Why:** The current [augment.py](src/moxify_ocr/data/augment.py) was tuned for real Scryfall-only training where the train→val gap is small. With pure synthetic train + real val, the gap widens and we need stronger augmentation to compensate. The `# we've scaled back` comment in that file is the explicit knob to relax.

**Files:**
- Modify: `src/moxify_ocr/data/augment.py`
- Modify: `tests/data/test_augment.py`

**Step 1: Update the pipeline**

Increase parameters that bridge clean-synthetic → ink-bleed-real:

```python
# src/moxify_ocr/data/augment.py

# Was 5/255; bump for the synthetic→real gap.
_GAUSS_NOISE_MAX_UINT8: float = 12.0
_GAUSS_NOISE_STD_RANGE: tuple[float, float] = (0.0, _GAUSS_NOISE_MAX_UINT8 / 255.0)


def build_augmentation_pipeline(*, seed: int = 0) -> A.Compose:
    pipeline = A.Compose(
        [
            A.Affine(rotate=(-5, 5), scale=(0.92, 1.08), p=1.0),  # was ±3°, 0.95-1.05
            A.Perspective(scale=(0.01, 0.05), p=1.0),  # was up to 0.03
            A.RandomBrightnessContrast(
                brightness_limit=0.30,  # was 0.20
                contrast_limit=0.30,
                p=1.0,
            ),
            A.HueSaturationValue(  # NEW — synthetic is pure white text, real has color cast
                hue_shift_limit=8,
                sat_shift_limit=15,
                val_shift_limit=15,
                p=0.5,
            ),
            A.GaussNoise(std_range=_GAUSS_NOISE_STD_RANGE, p=1.0),
            A.ImageCompression(quality_range=(60, 95), p=1.0),  # was 70-95
            A.GaussianBlur(sigma_limit=(0.0, 1.5), p=1.0),  # was 0-1.0
            A.MotionBlur(p=0.20),  # was 0.15
        ],
        seed=seed,
    )
    return pipeline
```

**Step 2: Update test expectations**

In `tests/data/test_augment.py`, the deterministic-output tests will break (different parameters → different pixel hashes). Update the expected hashes after a manual sanity check.

**Step 3: Run tests**

```bash
.venv/bin/pytest tests/data/test_augment.py -v
```

Expected: green after hash update.

**Step 4: Sanity-check augmented samples by eye**

Build a contact sheet of 16 augmented variants of one CardConjurer crop. Save to `artifacts/v3_aug_sample.png`. Eyeball that none of them are over-corrupted.

**Step 5: Commit**

```bash
git add src/moxify_ocr/data/augment.py tests/data/test_augment.py artifacts/v3_aug_sample.png
git commit -m "feat(augment): stronger pipeline to bridge synthetic-real gap (v3)"
```

---

## Task 5: Author `configs/bottom_region_v3.yaml`

**Why:** The training entrypoint reads YAML; v3 needs its own config so we can A/B against v1.

**Files:**
- Create: `configs/bottom_region_v3.yaml`

**Step 1: Author the config**

```yaml
# configs/bottom_region_v3.yaml
data:
  manifest: data/scryfall/manifest.jsonl  # used only for val/test splits + line_compositor pool
  images_root: data/scryfall
  batch_size: 192
  shuffle_buffer: 4096                    # bumped — larger pool deserves a larger window
  min_release: "2008-01-01"

  # YOLO-style: train sees zero real Scryfall samples.
  # Synthetic legs:
  cardconjurer_pool: data/synth_cardconjurer/v3_100k
  cardconjurer_ratio: 0.70
  line_compositor_ratio: 0.30
  # The remaining 0.0 means: never sample from real Scryfall during training.
  # Real Scryfall still drives val + test (set-aware split).

model:
  input_height: 48
  input_width: 256
  num_classes: 45
  lstm_units: 256                          # v2 setting

train:
  epochs: 50                               # was 30 — more synthetic samples = slower overfit
  lr: 5.0e-4
  warmup_steps: 2000                       # was 1000 — synthetic CTC may need more warmup
  seed: 0
  output_dir: artifacts/bottom_region_v3
```

**Step 2: Commit**

```bash
git add configs/bottom_region_v3.yaml
git commit -m "feat(train): v3 config — synthetic-only train, real val"
```

---

## Task 6: Verify the dataset config sums to 1.0 train

**Why:** `cardconjurer_ratio + line_compositor_ratio = 1.0` means every train sample comes from a synthetic leg. Worth a sanity test before kicking off a 50-epoch run.

**Files:**
- Modify: `tests/data/test_dataset.py`

**Step 1: Add a test**

```python
def test_v3_config_yields_only_synthetic_train_samples(tmp_path):
    """With ratios summing to 1.0 every train sample must be synthetic — no real Scryfall."""
    # Build a manifest of 5 cards with sentinel pixel value (e.g., RED=255).
    # Build a CardConjurer pool of 5 cards with GREEN=255.
    # Build line_compositor LineLibrary from the same manifest (its samples are stitched
    # from real images, so still RED-derived but uniquely shaped).
    # Sample 200 train samples; assert ZERO of them are exactly the original RED images
    # (line_compositor is allowed since it's "synthetic" in our definition).
    ...
```

**Step 2: Run**

```bash
.venv/bin/pytest tests/data/test_dataset.py::test_v3_config_yields_only_synthetic_train_samples -v
```

Expected: green.

**Step 3: Commit**

```bash
git add tests/data/test_dataset.py
git commit -m "test(data): assert v3 config produces synthetic-only train samples"
```

---

## Task 7: Smoke train — 2 epochs, watch val_cer

**Why:** Catch broken wiring before committing to a 50-epoch run. 2 epochs takes ~1-2 hours; full run is ~50 hours.

**Files:**
- Modify: `artifacts/bottom_region_v3_smoke/`

**Step 1: Run a 2-epoch training**

```bash
.venv/bin/python -m moxify_ocr.train --config configs/bottom_region_v3.yaml \
  --epochs 2 \
  --output-dir artifacts/bottom_region_v3_smoke
```

Expected:
- training loss should drop monotonically (not stuck or NaN)
- val_cer at epoch 2 should be **better than 1.0** (random) — even after 2 epochs we expect <0.7 on a working pipeline
- if val_cer is stuck near 1.0, the synthetic→real gap is too large; abort and consider mixing in a small `real_ratio` (e.g. 0.10) before the long run

**Step 2: User checkpoint**

If val_cer trends down: proceed. If stuck: stop, diagnose. Most likely culprit is augmentation too weak — open `artifacts/v3_aug_sample.png` and crank up the noise/blur/JPEG quality.

**Step 3: Commit smoke logs** (not weights — too big)

```bash
ls artifacts/bottom_region_v3_smoke/
# expect: events.* (TF logs), best.keras (gitignore), train.log (commit)
echo 'artifacts/bottom_region_v3_smoke/best.keras' >> .gitignore
echo 'artifacts/bottom_region_v3_smoke/events.*' >> .gitignore
git add .gitignore artifacts/bottom_region_v3_smoke/train.log
git commit -m "test(train): v3 smoke (2 epochs) baseline"
```

---

## Task 8: Full v3 training — 50 epochs

**Why:** The actual training run.

**Files:**
- Modify: `artifacts/bottom_region_v3/`

**Step 1: Kick off**

```bash
.venv/bin/python -m moxify_ocr.train --config configs/bottom_region_v3.yaml \
  2>&1 | tee artifacts/bottom_region_v3/train.log
```

Expected: ~40-60 hours wall time on a GPU box, depending on hardware. Run on Colab or a workstation with persistent storage.

**Step 2: Watch the val_cer curve**

```bash
tail -f artifacts/bottom_region_v3/train.log | grep --line-buffered -E "val_cer|Epoch"
```

Stop early (Ctrl-C) if val_cer plateaus for 5+ epochs. Save the `best.keras` checkpoint at the lowest val_cer.

**Step 3: Summarize results**

In `ocr_training_doc.md`, append a v3 section:
- Final val_cer
- Comparison to v2 baseline (val_cer ~0.43)
- Per-language val_cer breakdown (`scripts/test_model_multilang.py`)
- Surprising failures or wins

**Step 4: Commit log + the new doc section**

```bash
git add artifacts/bottom_region_v3/train.log ocr_training_doc.md
git commit -m "feat(train): v3 50-epoch training results — synthetic-only"
```

---

## Task 9: Ablations — drop line_compositor, drop CardConjurer

**Why:** v3 has two synthetic legs. Knowing which one carries the load matters for future work and explains v3 behavior.

**Files:**
- Create: `configs/bottom_region_v3_cc_only.yaml` (cardconjurer_ratio=1.0, line_compositor_ratio=0.0)
- Create: `configs/bottom_region_v3_lc_only.yaml` (cardconjurer_ratio=0.0, line_compositor_ratio=1.0)
- Modify: `ocr_training_doc.md`

**Step 1: Author both configs** (clones of v3 with the ratio flipped).

**Step 2: Run both for 20 epochs each** (shorter — these are diagnostic, not deployment candidates).

```bash
.venv/bin/python -m moxify_ocr.train --config configs/bottom_region_v3_cc_only.yaml --epochs 20
.venv/bin/python -m moxify_ocr.train --config configs/bottom_region_v3_lc_only.yaml --epochs 20
```

**Step 3: Compare val_cer at epoch 20**

If `cc_only` and `lc_only` both lose to mixed: line_compositor + CardConjurer compose nicely → keep both.
If `cc_only` ≈ mixed: line_compositor adds nothing → simplify, drop it.
If `lc_only` ≈ mixed: CardConjurer adds nothing → unwind the renderer, save 28h render time per regen.

**Step 4: Document results in ocr_training_doc.md and commit**

---

## Task 10: Promote v3 if it wins

**Files:**
- Modify: `configs/bottom_region_v1.yaml` → leave alone (kept as v1 reference)
- Modify: production model artifact path

**Step 1: Comparison summary**

Table in `ocr_training_doc.md`:

| Run | Train data | val_cer | per-lang val_cer |
|---|---|---|---|
| v1 | Real Scryfall | 0.47 |  |
| v2 | Real Scryfall | 0.43 |  |
| v3 | 100% synthetic | ??? |  |

**Step 2: If v3 wins** (val_cer < 0.43): the production model becomes `artifacts/bottom_region_v3/best.keras`. Update any downstream consumers.

**Step 3: If v3 loses or ties:** decide whether to mix some real Scryfall back in (the "near-synthetic" fallback in the Risks section), or accept v2 as production. Document the call.

---

## Acceptance criteria

1. ✅ `data/synth_cardconjurer/v3_100k/` contains ≥99,500 valid PNGs.
2. ✅ A v3 50-epoch training run completes; final val_cer is logged in `ocr_training_doc.md`.
3. ✅ Ablations for cc-only and lc-only documented.
4. ✅ Decision recorded: v3 promoted to production, or rolled back with rationale.

## Out of scope

- **Real phone-photo validation set.** No such data exists in the repo. The closest is Scryfall set-aware splits, which is what we're using. If we ever capture real photos, that becomes a separate evaluation task.
- **Architecture changes.** v2 stem + 2× BiLSTM + CTC stays. v4 (if needed) is a separate plan.
- **DFC / split-card support.** The CardConjurer renderer is bottom-region only. The card-name OCR is its own future plan.
