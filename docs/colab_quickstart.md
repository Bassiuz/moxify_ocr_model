# Training on Google Colab / Kaggle

Local training on a Mac CPU is ~1 s/step. On a free-tier GPU it drops to 50–150 ms/step — **8–20× faster**. This doc gives you copy-paste cells for both platforms.

## tl;dr

| Platform | GPU | Time for 30 epochs on ~26k cards | Cost |
|---|---|---|---|
| Local Mac (CPU) | none | ~3 h | $0 |
| **Google Colab free** | T4 | ~25 min | $0 |
| **Kaggle notebooks** | T4 (30 h/wk) | ~25 min | $0 |
| Colab Pro | V100 / A100 | ~10 min | $10 / mo |
| Lambda / Vast.ai | RTX 4090 | ~5 min | ~$0.50 / hr |

Kaggle is the easiest free option — no time limits during the session and Drive mounting is optional. Colab free has a 12 h session cap.

## Workflow (either platform)

1. Push repo + manifest to GitHub (the manifest is 20-ish MB, fits in a commit; images live in Drive / re-ingest).
2. In the notebook: clone repo, install deps, pull images from Drive or re-ingest from Scryfall, run `python -m moxify_ocr.train.train`, download the best checkpoint.

## Before you start

```bash
# From your local repo — publish the manifest so the notebook can grab it
git checkout -b training-data
git add -f data/scryfall/manifest.jsonl  # override .gitignore
git commit -m "chore: publish training manifest for Colab runs"
git push origin training-data
# (delete the branch after training — don't merge to main)
```

For the **image files** (3–4 GB), two options:
- **A**: Zip `data/scryfall/images/` and upload to Google Drive once, then mount Drive in the notebook.
- **B**: Re-ingest inside the notebook (~40 min on Colab; then training on top of that).

Option A is faster per run, worth it if you train more than once.

## Google Colab cells

```python
# --- Cell 1: GPU check ---
import tensorflow as tf
print("TF:", tf.__version__)
print("GPUs:", tf.config.list_physical_devices("GPU"))
# Expected: at least one T4, V100, or A100 device listed.
```

```python
# --- Cell 2: Clone + install ---
!git clone -b training-data https://github.com/Bassiuz/moxify_ocr_model.git
%cd moxify_ocr_model
!pip install -q -e ".[dev]"
```

```python
# --- Cell 3a: Option A — pull images from Drive ---
from google.colab import drive
drive.mount("/content/drive")
# Assumes you uploaded data/scryfall/images.zip to MyDrive/moxify_ocr/
!mkdir -p data/scryfall
!unzip -q /content/drive/MyDrive/moxify_ocr/images.zip -d data/scryfall/
!ls data/scryfall/images | head
```

```python
# --- Cell 3b: Option B — re-ingest from Scryfall ---
!python scripts/ingest_scryfall.py --out data/scryfall
# ~40 min — grabs ~26k paper cards released on the 2015 frame
```

```python
# --- Cell 4: Train ---
!python -m moxify_ocr.train.train \
    --config configs/bottom_region_v1.yaml \
    --override data.batch_size=192
# On T4: ~25 min for 30 epochs. Watch val_cer decrease each epoch.
```

```python
# --- Cell 5: Download best checkpoint back to your machine ---
from google.colab import files
files.download("artifacts/bottom_region_v1/best.keras")
# Also download the train.log for reference:
files.download("artifacts/bottom_region_v1/train.log")
```

```python
# --- Cell 6: Smoke test on one card ---
import json, numpy as np
from PIL import Image
from moxify_ocr.models.bottom_region import build_bottom_region_model
from moxify_ocr.data.crop import crop_bottom_region
from moxify_ocr.data.dataset import decode_label

with open("data/scryfall/manifest.jsonl") as f:
    for line in f:
        row = json.loads(line)
        if row["set_code"] != "plst" and row["lang"] == "en":
            break
print("truth:", row["set_code"].upper(), row["collector_number"], row["rarity"].upper(), "EN")

m = build_bottom_region_model()
m.load_weights("artifacts/bottom_region_v1/best.keras")
img = Image.open(f"data/scryfall/{row['image_path']}").convert("RGB")
arr = np.asarray(crop_bottom_region(img))[None, ...].astype("uint8")
ids = m(arr).numpy().argmax(-1)[0]
out, prev = [], -1
for i in ids:
    if i != prev and i != 0:
        out.append(int(i))
    prev = i
print("predicted:", repr(decode_label(out)))
```

## Kaggle cells

Kaggle is similar but uses a dataset-upload workflow instead of Drive mounting. Practical flow:

1. On Kaggle, create a new dataset from your local `data/scryfall/images/` zip. Note the dataset slug (e.g. `yourname/moxify-ocr-images`).
2. Create a new notebook, **Add Data** → your dataset.
3. Kaggle's GPU option: **Settings → Accelerator → GPU T4 x2**.
4. Use the same training cells but replace the Drive-mount block with:

```python
# Kaggle attaches your dataset under /kaggle/input/<slug>/
!mkdir -p data/scryfall
!unzip -q /kaggle/input/moxify-ocr-images/images.zip -d data/scryfall/

# Manifest — either upload as a second dataset or commit to the branch
!cp /kaggle/input/moxify-ocr-images/manifest.jsonl data/scryfall/manifest.jsonl
```

5. For the output: Kaggle saves notebook outputs under `/kaggle/working/`. Download via **Output** tab after the run.

## Tuning knobs

Once the baseline trains cleanly, these are the cheapest wins:

| Change | How | Why |
|---|---|---|
| Bigger batch (T4 can handle 256) | `--override data.batch_size=256` | Faster epochs, better gradient estimate |
| More data | Re-ingest without `--limit` (all paper 2015-frame cards = ~55k) | Reduces overfitting |
| Longer training | `--override train.epochs=60` | Useful once the loss is still dropping at epoch 30 |
| Augmentation off early | (patch `dataset.py` `build_dataset` → skip augment for first N epochs) | Stabilises early CTC convergence |

Don't bother with these without first running the baseline once — they're diminishing returns on top of an already-working model.

## Gotchas

- **Colab sessions expire after 12 h idle.** Long ingest + long training can blow this. Use Option A (Drive-pulled zip) to skip ingest.
- **Free-tier T4s are shared.** First few minutes of a run may be slow while GPU warms up / gets allocated.
- **Save `best.keras` during training** — already done by the `ModelCheckpoint` callback. If the session dies, you still have the best snapshot.
- **`safe_mode=False` for loading** is no longer needed since the `v0.5` fix — the model uses proper `Rescaling` + subclassed layers now. But if you hit a Lambda-deserialization error with an old checkpoint, that's why.
