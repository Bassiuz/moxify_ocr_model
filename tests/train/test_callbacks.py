"""Tests for :mod:`moxify_ocr.train.callbacks`."""

from __future__ import annotations

import numpy as np
import tensorflow as tf

from moxify_ocr.data.dataset import ALPHABET, encode_label
from moxify_ocr.train.callbacks import CERCallback


class _PerfectModel(tf.keras.Model):
    """Stub model whose forward pass echoes pre-built per-sample logits.

    The callback calls ``self.model(images, training=False)`` — we ignore the
    image and instead return logits indexed by the order they're asked for.
    """

    def __init__(self, logits_batches: list[tf.Tensor]) -> None:
        super().__init__()
        self._logits_batches = logits_batches
        self._i = 0

    def call(  # type: ignore[override]
        self,
        inputs: tf.Tensor,
        training: bool | None = None,
    ) -> tf.Tensor:
        out = self._logits_batches[self._i]
        self._i += 1
        return out


def _one_hot_logits(label_ids: list[int], time_steps: int, num_classes: int) -> np.ndarray:
    """Build [1, T, C] logits that greedy-decode exactly to ``label_ids``.

    Each label id occupies one timestep, then a blank timestep separates
    repeats; remaining timesteps are blanks.
    """
    logits = np.full((1, time_steps, num_classes), -10.0, dtype=np.float32)
    # Default every step to blank (index 0).
    logits[0, :, 0] = 10.0
    t = 0
    for i, label in enumerate(label_ids):
        if t >= time_steps:
            break
        if i > 0 and label_ids[i - 1] == label:
            # Force a blank timestep to prevent CTC merge on equal-consecutive.
            t += 1
            if t >= time_steps:
                break
        logits[0, t, 0] = -10.0
        logits[0, t, label] = 10.0
        t += 1
    return logits


def _dataset_from_pairs(
    images: list[np.ndarray],
    labels: list[np.ndarray],
    batch_size: int,
) -> tf.data.Dataset:
    """Stitch (image, label, length) lists into a batched tf.data.Dataset."""
    padded_len = max((len(label) for label in labels), default=1)
    padded = np.zeros((len(labels), padded_len), dtype=np.int32)
    lengths = np.zeros((len(labels),), dtype=np.int32)
    for i, label in enumerate(labels):
        padded[i, : len(label)] = label
        lengths[i] = len(label)
    image_stack = np.stack(images, axis=0)
    ds = tf.data.Dataset.from_tensor_slices((image_stack, padded, lengths))
    return ds.batch(batch_size)


def test_cer_callback_zero_cer_on_perfect_predictions() -> None:
    """Predicted logits that greedy-decode to the exact labels produce ~0 CER."""
    num_classes = len(ALPHABET) + 1  # 45
    time_steps = 32
    labels = ["R 001", "C 12"]
    label_ids = [encode_label(label) for label in labels]

    per_sample_logits = [
        _one_hot_logits(ids, time_steps=time_steps, num_classes=num_classes)
        for ids in label_ids
    ]
    batch_logits = tf.constant(np.concatenate(per_sample_logits, axis=0))
    model = _PerfectModel(logits_batches=[batch_logits])

    images = [np.zeros((48, 256, 3), dtype=np.uint8) for _ in labels]
    val_ds = _dataset_from_pairs(images, [np.asarray(ids, dtype=np.int32) for ids in label_ids], batch_size=2)

    callback = CERCallback(val_ds)
    callback.set_model(model)

    cer = callback.compute_cer()
    assert cer == 0.0, f"expected zero CER on perfect logits, got {cer}"


def test_cer_callback_high_cer_on_random_predictions() -> None:
    """Random logits mostly mis-decode, so CER should be well above 0.5."""
    num_classes = len(ALPHABET) + 1
    time_steps = 32
    rng = np.random.default_rng(seed=42)
    labels = ["R 001 C", "C 12 U", "M 345 R", "U 9 C"]
    label_ids = [encode_label(label) for label in labels]

    random_logits = rng.standard_normal(size=(len(labels), time_steps, num_classes)).astype(
        np.float32
    )
    model = _PerfectModel(logits_batches=[tf.constant(random_logits)])

    images = [np.zeros((48, 256, 3), dtype=np.uint8) for _ in labels]
    val_ds = _dataset_from_pairs(
        images, [np.asarray(ids, dtype=np.int32) for ids in label_ids], batch_size=len(labels)
    )

    callback = CERCallback(val_ds)
    callback.set_model(model)

    cer = callback.compute_cer()
    assert cer > 0.5, f"expected high CER on random logits, got {cer}"
