"""Custom training callbacks for the OCR CRNN.

At this stage there is exactly one: :class:`CERCallback`, which evaluates
character-error-rate on a validation dataset at the end of every epoch and
stores it on ``logs['val_cer']`` so standard Keras callbacks
(EarlyStopping, ModelCheckpoint, TensorBoard) can see it.
"""

from __future__ import annotations

from typing import Any

import editdistance
import numpy as np
import tensorflow as tf

from moxify_ocr.data.dataset import BLANK_INDEX, decode_label


def _greedy_decode(logits: tf.Tensor) -> list[list[int]]:
    """Run greedy CTC decoding on ``[B, T, C]`` logits, returning per-sample id lists."""
    logits_tm = tf.transpose(logits, perm=[1, 0, 2])
    batch_size = int(tf.shape(logits)[0].numpy())
    time_steps = int(tf.shape(logits)[1].numpy())
    seq_len = tf.fill([batch_size], time_steps)
    decoded_sparse, _ = tf.nn.ctc_greedy_decoder(
        inputs=logits_tm,
        sequence_length=seq_len,
        merge_repeated=True,
        blank_index=BLANK_INDEX,
    )
    sparse = decoded_sparse[0]
    dense = tf.sparse.to_dense(sparse, default_value=-1).numpy()
    result: list[list[int]] = []
    for row in dense:
        ids = [int(v) for v in row if int(v) >= 0]
        result.append(ids)
    return result


def _labels_from_dense(y_true: np.ndarray) -> list[list[int]]:
    """Strip trailing zero-padding from a dense ``[B, L]`` int label batch."""
    out: list[list[int]] = []
    for row in y_true:
        ids = [int(v) for v in row if int(v) != BLANK_INDEX]
        out.append(ids)
    return out


class CERCallback(tf.keras.callbacks.Callback):  # type: ignore[misc]
    """End-of-epoch Character Error Rate on a validation dataset.

    Iterates the validation dataset, runs greedy CTC decoding, maps both the
    prediction and ground-truth index sequences back to strings via
    :func:`decode_label`, and averages per-character edit distance over the
    total reference length. The resulting scalar is stored as ``val_cer`` on
    the epoch's logs dict so standard Keras monitors can see it.
    """

    def __init__(self, val_dataset: tf.data.Dataset) -> None:
        super().__init__()
        self.val_dataset = val_dataset

    def on_epoch_end(
        self,
        epoch: int,
        logs: dict[str, Any] | None = None,
    ) -> None:
        cer = self.compute_cer()
        if logs is not None:
            logs["val_cer"] = cer

    def compute_cer(self) -> float:
        """Run the validation dataset once, return mean CER as a Python float."""
        total_distance = 0
        total_length = 0
        for batch in self.val_dataset:
            images, labels, _label_length = batch
            logits = self.model(images, training=False)
            pred_ids_batch = _greedy_decode(logits)
            true_ids_batch = _labels_from_dense(labels.numpy())
            for pred_ids, true_ids in zip(pred_ids_batch, true_ids_batch, strict=True):
                pred_str = decode_label(pred_ids)
                true_str = decode_label(true_ids)
                total_distance += int(editdistance.eval(pred_str, true_str))
                total_length += max(len(true_str), 1)
        if total_length == 0:
            return 0.0
        return total_distance / total_length
