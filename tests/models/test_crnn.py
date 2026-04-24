"""Tests for :mod:`moxify_ocr.models.crnn`."""

from __future__ import annotations

import numpy as np
import tensorflow as tf

from moxify_ocr.models.crnn import build_crnn


def test_output_shape() -> None:
    """A single uint8 (48, 256, 3) input produces rank-3 logits of width num_classes."""
    model = build_crnn(input_shape=(48, 256, 3), num_classes=45, lstm_units=96)
    dummy = np.zeros((1, 48, 256, 3), dtype=np.uint8)
    output = model(dummy, training=False)
    assert output.shape.rank == 3
    batch, _time, classes = output.shape
    assert batch == 1
    assert classes == 45


def test_time_dim_proportional_to_width() -> None:
    """Output time dimension T is approximately W / 4.

    CTC needs T ≥ 2L - 1 where L is the longest label (~20 chars for our
    labels), so we need T ≥ ~40. v2's custom stem has exactly two width-halving
    stages ((2,2) then (2,2) then three (n,1) height-only collapses) so T = W/4.
    For the default W=256 that gives T=64, safely above the alignment minimum.
    """
    model = build_crnn(input_shape=(48, 256, 3), num_classes=45, lstm_units=96)
    dummy = np.zeros((1, 48, 256, 3), dtype=np.uint8)
    output = model(dummy, training=False)
    time_steps = int(output.shape[1])
    assert 56 <= time_steps <= 72, f"expected T≈W/4=64, got {time_steps}"


def test_param_count_reasonable() -> None:
    """Total trainable params sit in the [100k, 10M] range — small but capable CRNN.

    v2 at defaults (lstm_units=256, 2 BiLSTM layers) is ~3.4M. Earlier versions
    with smaller LSTM (96) are still inside the range.
    """
    model = build_crnn(input_shape=(48, 256, 3), num_classes=45, lstm_units=96)
    params = int(model.count_params())
    assert 100_000 <= params <= 10_000_000, f"got {params} params"


def test_accepts_uint8_input() -> None:
    """Feeding a uint8 tensor works end-to-end without a dtype error."""
    model = build_crnn(input_shape=(48, 256, 3), num_classes=45, lstm_units=96)
    dummy = tf.zeros((2, 48, 256, 3), dtype=tf.uint8)
    output = model(dummy, training=False)
    assert output.dtype == tf.float32
    assert int(output.shape[0]) == 2
