"""Tests for :mod:`moxify_ocr.models.bottom_region` + v1 config."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf
import yaml

from moxify_ocr.models.bottom_region import build_bottom_region_model, ctc_loss


def test_build_bottom_region_model_shape() -> None:
    """v1 bottom-region model accepts (B, 48, 256, 3) uint8 and outputs (B, T, 45)."""
    model = build_bottom_region_model()
    dummy = np.zeros((2, 48, 256, 3), dtype=np.uint8)
    output = model(dummy, training=False)
    batch, _time, classes = output.shape
    assert int(batch) == 2
    assert int(classes) == 45


def test_ctc_loss_finite_on_synthetic_batch() -> None:
    """CTC loss is finite (not NaN/inf) on reasonable random logits + sparse labels."""
    rng = np.random.default_rng(seed=0)
    batch, time_steps, classes = 4, 32, 45
    # Random logits centred at 0 — softmax will be roughly uniform so no NaNs.
    y_pred = tf.constant(
        rng.standard_normal(size=(batch, time_steps, classes)).astype(np.float32)
    )
    # Labels shorter than time_steps, zero-padded. Non-zero entries are real.
    y_true_np = np.zeros((batch, 8), dtype=np.int32)
    y_true_np[0, :5] = [1, 2, 3, 4, 5]
    y_true_np[1, :3] = [10, 11, 12]
    y_true_np[2, :7] = [1, 2, 1, 2, 1, 2, 1]
    y_true_np[3, :4] = [40, 41, 42, 43]
    y_true = tf.constant(y_true_np)

    loss = ctc_loss(y_true, y_pred)
    loss_val = float(loss.numpy())
    assert np.isfinite(loss_val), f"CTC loss is not finite: {loss_val}"
    assert loss_val > 0.0


def test_config_parses() -> None:
    """The v1 YAML loads and has all the keys the training code relies on."""
    cfg_path = Path(__file__).resolve().parents[2] / "configs" / "bottom_region_v1.yaml"
    assert cfg_path.exists(), cfg_path
    with cfg_path.open("r", encoding="utf-8") as handle:
        cfg: dict[str, Any] = yaml.safe_load(handle)

    for section in ("data", "model", "train"):
        assert section in cfg, f"missing section: {section}"

    for key in ("manifest", "images_root", "batch_size", "shuffle_buffer", "min_release"):
        assert key in cfg["data"], f"missing data.{key}"
    for key in ("input_height", "input_width", "num_classes", "lstm_units"):
        assert key in cfg["model"], f"missing model.{key}"
    for key in ("epochs", "lr", "warmup_steps", "seed", "output_dir"):
        assert key in cfg["train"], f"missing train.{key}"

    assert cfg["model"]["num_classes"] == 45
    assert cfg["model"]["input_height"] == 48
    assert cfg["model"]["input_width"] == 256
