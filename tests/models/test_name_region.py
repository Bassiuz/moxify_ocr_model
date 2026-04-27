"""Smoke tests for the name-region CRNN entrypoint."""

from __future__ import annotations

import numpy as np
import tensorflow as tf

from moxify_ocr.data.name_alphabet import NAME_ALPHABET
from moxify_ocr.models.name_region import (
    NAME_INPUT_SHAPE,
    NAME_NUM_CLASSES,
    build_name_region_model,
    ctc_loss,
)


def test_input_shape_and_class_count_consistent() -> None:
    assert NAME_INPUT_SHAPE == (48, 512, 3)
    assert NAME_NUM_CLASSES == len(NAME_ALPHABET) + 1


def test_model_output_shape() -> None:
    model = build_name_region_model()
    batch = np.zeros((2, 48, 512, 3), dtype=np.uint8)
    out = model(batch, training=False)
    # T = W / 4 = 128 timesteps.
    assert out.shape == (2, 128, NAME_NUM_CLASSES)


def test_ctc_loss_runs_on_one_batch() -> None:
    """Verify the CTC loss works end-to-end with the wider name shape."""
    model = build_name_region_model()
    batch_imgs = np.zeros((2, 48, 512, 3), dtype=np.uint8)
    # Two short labels — second is shorter (zero-padded to match).
    batch_labels = tf.constant([[1, 2, 3, 4, 0], [5, 6, 0, 0, 0]], dtype=tf.int32)
    logits = model(batch_imgs, training=True)
    loss = ctc_loss(batch_labels, logits)
    assert loss.shape == ()
    assert tf.math.is_finite(loss).numpy()
