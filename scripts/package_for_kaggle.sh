#!/usr/bin/env bash
# Package the source code + the rendered name pool into two zips, ready to
# upload as Kaggle datasets for notebooks/train_name_v1_kaggle.ipynb.
#
# Usage:
#     scripts/package_for_kaggle.sh                  # default pool dir
#     scripts/package_for_kaggle.sh data/synth_names/v2  # different pool
#
# Outputs:
#     /tmp/moxify-ocr-source.zip
#     /tmp/<pool-dir-basename>.zip
#
# Upload each as a separate Kaggle dataset (one zip per dataset). The slugs
# you pick on Kaggle will need to match SOURCE_DATASET / POOL_DATASET in the
# notebook.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
POOL_DIR="${1:-data/synth_names/v1}"

if [ ! -d "$REPO_ROOT/$POOL_DIR" ]; then
  echo "pool dir not found: $REPO_ROOT/$POOL_DIR" >&2
  echo "  run scripts/render_name_pool.py first, or pass a different path" >&2
  exit 1
fi

# 1. Source zip — repo root minus heavy/local-only dirs.
SRC_ZIP=/tmp/moxify-ocr-source.zip
rm -f "$SRC_ZIP"
cd "$REPO_ROOT"
zip -rq "$SRC_ZIP" . \
  -x '.git/*' '.git' \
  -x '.venv/*' '.venv' \
  -x 'data/*' 'data' \
  -x 'artifacts/*' 'artifacts' \
  -x '.pytest_cache/*' '.mypy_cache/*' '.ruff_cache/*' \
  -x '__pycache__/*' '*/__pycache__/*' \
  -x '.worktrees/*' '.worktrees' \
  -x 'logs/*' 'logs' \
  -x '.tensorboard/*' '.tensorboard' \
  -x '*.tflite' '*.h5' '*.keras' '*.onnx'
echo "wrote $SRC_ZIP ($(du -h "$SRC_ZIP" | cut -f1))"

# 2. Pool zip — images/ + labels.jsonl. Keep them at the zip root so the
# notebook's pool_root is the dataset mount directly (no nested dir).
POOL_BASENAME="$(basename "$POOL_DIR")"
POOL_ZIP="/tmp/moxify-ocr-name-pool-${POOL_BASENAME}.zip"
rm -f "$POOL_ZIP"
cd "$REPO_ROOT/$POOL_DIR"
zip -rq "$POOL_ZIP" images/ labels.jsonl
echo "wrote $POOL_ZIP ($(du -h "$POOL_ZIP" | cut -f1))"

cat <<EOF

Next:
  1. Upload $SRC_ZIP to Kaggle as a dataset (slug: 'moxify-ocr-source' by default).
  2. Upload $POOL_ZIP to Kaggle as a dataset (slug: 'moxify-ocr-name-pool-${POOL_BASENAME}').
  3. Open notebooks/train_name_v1_kaggle.ipynb on Kaggle, attach both datasets, enable GPU + Internet.
  4. Run all cells.

If you used different slugs, edit SOURCE_DATASET / POOL_DATASET in the notebook's first code cell.
EOF
