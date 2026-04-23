"""Generic CRNN (CNN + BiLSTM + CTC head) for fixed-height text recognition.

The architecture is deliberately small so the model fits in a TFLite-sized
budget on a phone:

1. MobileNetV3-Small backbone (``alpha=0.5``, ``include_top=False``) truncated
   at the stride-8 block (``expanded_conv_2_add``) as a convolutional feature
   extractor. We stop there rather than run the full backbone because the CRNN
   needs ~W/8 time-steps (~32 columns from a 256-wide input) — enough for a
   one-or-two-line label with ~15 characters. Running all the way to stride-32
   would collapse the time dimension to ~8, far fewer than the longest label.
2. Collapse the spatial height of the feature map to 1 so the tensor becomes a
   sequence of per-column feature vectors.
3. Bidirectional LSTM over that sequence.
4. Dense projection to ``num_classes`` (linear logits; the CTC loss applies
   its own log-softmax).

Inputs are ``uint8`` RGB images ``[B, H, W, 3]``; normalization to the
[-1, 1] range expected by MobileNetV3 happens inside the model so the caller
can feed raw pixel tensors from :mod:`tf.data`.
"""

from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import Model, layers
from tensorflow.keras.applications import MobileNetV3Small


def build_crnn(
    *,
    input_shape: tuple[int, int, int] = (48, 256, 3),
    num_classes: int = 45,
    lstm_units: int = 96,
) -> Model:
    """Build a CRNN: MobileNetV3-Small stem -> sequence collapse -> BiLSTM -> Dense.

    Args:
        input_shape: ``(H, W, C)`` for the image input. The model accepts
            ``uint8`` tensors of this shape; preprocessing is internal.
        num_classes: Output classes (alphabet size + 1 for the CTC blank).
        lstm_units: Units per direction in the BiLSTM. Total recurrent width
            is ``2 * lstm_units``.

    Returns:
        A :class:`tf.keras.Model` whose output is ``float32[B, T, num_classes]``
        logits with ``T`` approximately ``W / 8`` (MobileNetV3-Small has three
        width-halving stages before we collapse height).
    """
    height, width, channels = input_shape

    # Raw uint8 input so the training pipeline can feed straight from tf.data
    # without an explicit cast.
    image_input = layers.Input(shape=input_shape, dtype=tf.uint8, name="image")
    x = layers.Lambda(lambda t: tf.cast(t, tf.float32), name="to_float32")(image_input)
    # MobileNetV3 expects inputs scaled to [-1, 1].
    x = layers.Lambda(lambda t: (t / 127.5) - 1.0, name="normalize")(x)

    full_backbone = MobileNetV3Small(
        input_shape=(height, width, channels),
        alpha=0.5,
        include_top=False,
        weights=None,
        include_preprocessing=False,
    )
    # Truncate at the stride-8 residual block so the output time axis keeps
    # enough columns (~W/8) for CTC decoding of the full label.
    trunk_out = full_backbone.get_layer("expanded_conv_2_add").output
    backbone = Model(
        inputs=full_backbone.inputs,
        outputs=trunk_out,
        name="mobilenetv3_small_stride8",
    )
    features = backbone(x)  # [B, H', W', C']

    # Collapse H' to 1 via global-avg-pool along the height axis. This keeps
    # one feature vector per column and preserves the time (width) dimension.
    pooled = layers.Lambda(
        lambda t: tf.reduce_mean(t, axis=1),
        name="collapse_height",
    )(features)  # [B, W', C']

    sequence = layers.Bidirectional(
        layers.LSTM(lstm_units, return_sequences=True),
        name="bilstm",
    )(pooled)

    logits = layers.Dense(num_classes, name="logits")(sequence)  # linear

    return Model(inputs=image_input, outputs=logits, name="crnn")
