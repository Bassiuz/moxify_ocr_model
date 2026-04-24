"""Bottom-region model wiring: CRNN builder wrapper + CTC loss.

This is the specific instantiation of :func:`moxify_ocr.models.crnn.build_crnn`
we train for the bottom-of-card OCR head, plus the CTC loss used by
:func:`tf.keras.Model.compile`.

Label convention
----------------
``y_true`` is a dense ``int32[B, L]`` tensor of class indices, where:

- Class ``0`` is both the CTC blank and the padding value — labels are shorter
  than ``L`` in general, and we use ``0`` to pad to the batch's longest label.
- Classes ``1..num_classes-1`` are real characters.

To feed ``tf.nn.ctc_loss`` we convert the dense batch to a sparse tensor by
treating any non-zero position as a real label entry.
"""

from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import Model

from moxify_ocr.models.crnn import build_crnn


def build_bottom_region_model() -> Model:
    """Return the v1.1 bottom-region CRNN (48x256 RGB -> logits).

    v1.1 bumped ``lstm_units`` from 96 to 256 after v1 plateaued at CER 0.43.
    The v1 model learned label *structure* (length, separators, trailing
    "SET . LANG") but not per-glyph character reading — too few recurrent
    parameters to absorb the full alphabet × positional variation. ~3x more
    params should put the model into the 0.1-0.2 CER range.
    """
    return build_crnn(input_shape=(48, 256, 3), num_classes=45, lstm_units=256)


def ctc_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """CTC loss wrapper for :func:`tf.keras.Model.compile`.

    Args:
        y_true: ``int32[B, max_label_len]``; zero-padded (0 = CTC blank/pad).
        y_pred: ``float32[B, T, C]`` logits straight from the CRNN head.

    Returns:
        Scalar ``float32`` loss — the mean per-sample CTC negative log-likelihood.
    """
    y_true = tf.cast(y_true, tf.int32)
    batch_size = tf.shape(y_pred)[0]
    time_steps = tf.shape(y_pred)[1]

    # Build sparse labels from the dense padded tensor. Any position with a
    # non-zero value is a real character; zeros are padding/blank and dropped.
    mask = tf.not_equal(y_true, 0)
    sparse_labels = tf.sparse.from_dense(tf.where(mask, y_true, tf.zeros_like(y_true)))
    # from_dense produces a sparse tensor whose stored values include zeros at
    # every dense position; strip them so only the actual label entries remain.
    keep = tf.not_equal(sparse_labels.values, 0)
    sparse_labels = tf.sparse.SparseTensor(
        indices=tf.boolean_mask(sparse_labels.indices, keep),
        values=tf.boolean_mask(sparse_labels.values, keep),
        dense_shape=sparse_labels.dense_shape,
    )

    logit_length = tf.fill([batch_size], time_steps)

    per_sample_loss = tf.nn.ctc_loss(
        labels=sparse_labels,
        logits=y_pred,
        label_length=None,
        logit_length=logit_length,
        logits_time_major=False,
        blank_index=0,
    )
    return tf.reduce_mean(per_sample_loss)
