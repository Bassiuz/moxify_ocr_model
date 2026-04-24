"""CRNN (CNN + stacked BiLSTM + CTC head) for fixed-height text recognition.

v2 design notes
---------------
The architecture is a classic text-recognition CRNN built for TFLite on phones:

1. **Purpose-built OCR stem** — a small custom CNN (not a truncated ImageNet
   backbone). Five Conv-BN-ReLU blocks with asymmetric strides that collapse
   the height axis aggressively while preserving the width axis (time
   dimension). Final stem output is ``(B, 1, W/4, 256)`` which we reshape to
   ``(B, W/4, 256)`` — 64 timesteps for our 256-wide input, 256 channels per
   timestep. We dropped MobileNetV3-Small after v1.1: cutting a classifier
   backbone at stride 4 only surfaced 40 channels, which was too thin.
2. **Stacked 2× BiLSTM** for two-level sequence modeling — the first layer
   learns local character patterns, the second layer learns cross-character
   context (e.g. "3-digit number then a single letter = collector_number
   rarity sequence"). v1.1 had only one BiLSTM and saturated on simpler
   structural patterns (language codes, separators).
3. **Dropout on inputs + between recurrent layers** for regularization.
4. **Linear Dense head** — CTC loss applies its own log-softmax.

Inputs are ``uint8`` RGB images ``[B, H, W, 3]``; the model casts + normalizes
to [-1, 1] inside via a ``Rescaling`` layer so the saved model reloads without
``safe_mode=False``.
"""

from __future__ import annotations

import keras
import tensorflow as tf
from tensorflow.keras import Model, layers


@keras.saving.register_keras_serializable(package="moxify_ocr")
class SqueezeHeight(layers.Layer):  # type: ignore[misc]
    """Drop the height axis after it's been collapsed to 1: ``(B,1,W,C) → (B,W,C)``.

    Implemented as a registered ``Layer`` subclass (not ``Lambda``) so the model
    serializes and reloads without ``safe_mode=False``.
    """

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        return tf.squeeze(inputs, axis=1)

    def compute_output_shape(
        self, input_shape: tuple[int | None, ...]
    ) -> tuple[int | None, ...]:
        return (input_shape[0], input_shape[2], input_shape[3])


def _conv_block(
    x: tf.Tensor,
    *,
    filters: int,
    strides: tuple[int, int],
    name: str,
) -> tf.Tensor:
    """Conv 3x3 → BatchNorm → ReLU. Padding is always 'same'."""
    x = layers.Conv2D(
        filters=filters,
        kernel_size=3,
        strides=strides,
        padding="same",
        use_bias=False,  # redundant with BatchNorm
        name=f"{name}_conv",
    )(x)
    x = layers.BatchNormalization(name=f"{name}_bn")(x)
    x = layers.ReLU(name=f"{name}_relu")(x)
    return x


def build_crnn(
    *,
    input_shape: tuple[int, int, int] = (48, 256, 3),
    num_classes: int = 45,
    lstm_units: int = 256,
    dropout: float = 0.2,
) -> Model:
    """Build the v2 CRNN: custom OCR stem → stacked 2× BiLSTM → Dense.

    Args:
        input_shape: ``(H, W, C)`` for the image input. The model accepts raw
            ``uint8`` tensors of this shape; preprocessing is internal.
        num_classes: Output classes (alphabet size + 1 for the CTC blank).
        lstm_units: Units per direction in each BiLSTM. Total recurrent width
            per layer is ``2 * lstm_units``.
        dropout: Dropout rate applied before and between the recurrent layers
            and on their hidden-to-hidden connections.

    Returns:
        A :class:`tf.keras.Model` whose output is ``float32[B, T, num_classes]``
        logits, with ``T = W / 4``. For the default ``input_shape=(48, 256, 3)``
        that's ``T = 64`` timesteps — well above the CTC alignment minimum of
        ``2 * longest_label - 1 ≈ 40``.
    """
    image_input = layers.Input(shape=input_shape, dtype=tf.uint8, name="image")

    # uint8 → float32 in [-1, 1] via a proper Keras layer (no Lambda).
    x = layers.Rescaling(scale=1.0 / 127.5, offset=-1.0, name="normalize")(image_input)

    # Stem. Asymmetric strides collapse H to 1 while keeping W for the time axis.
    #   (48, 256) → (24, 128) → (12, 64) → (6, 64) → (3, 64) → (1, 64)
    x = _conv_block(x, filters=32, strides=(2, 2), name="stem1")
    x = _conv_block(x, filters=64, strides=(2, 2), name="stem2")
    x = _conv_block(x, filters=128, strides=(2, 1), name="stem3")
    x = _conv_block(x, filters=192, strides=(2, 1), name="stem4")
    x = _conv_block(x, filters=256, strides=(3, 1), name="stem5")

    # (B, 1, W/4, 256) → (B, W/4, 256)
    x = SqueezeHeight(name="squeeze_h")(x)

    # Recurrent sequence modeling.
    x = layers.Dropout(dropout, name="pre_rnn_dropout")(x)
    x = layers.Bidirectional(
        layers.LSTM(lstm_units, return_sequences=True, dropout=dropout),
        name="bilstm_1",
    )(x)
    x = layers.Dropout(dropout, name="inter_rnn_dropout")(x)
    x = layers.Bidirectional(
        layers.LSTM(lstm_units, return_sequences=True, dropout=dropout),
        name="bilstm_2",
    )(x)

    logits = layers.Dense(num_classes, name="logits")(x)

    return Model(inputs=image_input, outputs=logits, name="crnn_v2")
