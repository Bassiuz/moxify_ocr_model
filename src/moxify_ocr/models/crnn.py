"""Generic CRNN (CNN + BiLSTM + CTC head) for fixed-height text recognition.

Design notes
------------
The architecture is deliberately small so the model fits in a TFLite-sized
budget on a phone:

1. MobileNetV3-Small backbone (``alpha=0.5``, ``include_top=False``) truncated
   at the **stride-4** block (``re_lu_2``) as a convolutional feature
   extractor. We stop there rather than run the full backbone because the CRNN
   needs ``T ≥ 2L - 1`` time-steps where ``L`` is the longest label in
   characters. Our longest bottom-region labels are ~20 chars, so we need
   ``T ≥ ~40``. At ``W=256`` with stride 4 that gives ``T=64`` — safely above
   the CTC alignment minimum. (Running to stride-8 gave ``T=32`` which falls
   below the minimum and causes CTC mode collapse.)
2. Collapse the spatial height of the feature map to 1 so the tensor becomes a
   sequence of per-column feature vectors.
3. Dropout for regularization before the recurrent stage.
4. Bidirectional LSTM over that sequence.
5. Dense projection to ``num_classes`` (linear logits; the CTC loss applies
   its own log-softmax).

Inputs are ``uint8`` RGB images ``[B, H, W, 3]``; normalization to the
[-1, 1] range expected by MobileNetV3 happens inside the model via a proper
Keras ``Rescaling`` layer so the saved model can be reloaded without
``safe_mode=False``.
"""

from __future__ import annotations

import keras
import tensorflow as tf
from tensorflow.keras import Model, layers
from tensorflow.keras.applications import MobileNetV3Small

# Cut MobileNetV3-Small here. Stride is 4× the input (H/4, W/4) so a 256-wide
# input yields T=64 timesteps — enough room for CTC alignment of ~20-char labels.
# ``expanded_conv_1_expand`` is a stable layer name in MobileNetV3Small that
# sits right before the block-1 stride-2 depthwise conv, so its output is at
# stride 4 with a reasonable channel count (40 at alpha=0.5). We avoid
# ``re_lu_N`` names because Keras renames them when the backbone is
# instantiated multiple times in the same process (e.g. across tests).
_STEM_CUT_LAYER = "expanded_conv_1_expand"


@keras.saving.register_keras_serializable(package="moxify_ocr")
class CollapseHeight(layers.Layer):  # type: ignore[misc]
    """Mean-pool along the height axis: ``[B, H, W, C]`` → ``[B, W, C]``.

    Implemented as a proper ``Layer`` subclass (not ``Lambda``) so the model
    can be saved to ``.keras`` and reloaded without ``safe_mode=False``.
    """

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        return tf.reduce_mean(inputs, axis=1)

    def compute_output_shape(
        self, input_shape: tuple[int | None, ...]
    ) -> tuple[int | None, ...]:
        return (input_shape[0], input_shape[2], input_shape[3])


def build_crnn(
    *,
    input_shape: tuple[int, int, int] = (48, 256, 3),
    num_classes: int = 45,
    lstm_units: int = 96,
    dropout: float = 0.2,
) -> Model:
    """Build a CRNN: MobileNetV3-Small(stride-4) → collapse height → BiLSTM → Dense.

    Args:
        input_shape: ``(H, W, C)`` for the image input. The model accepts
            ``uint8`` tensors of this shape; preprocessing is internal.
        num_classes: Output classes (alphabet size + 1 for the CTC blank).
        lstm_units: Units per direction in the BiLSTM. Total recurrent width
            is ``2 * lstm_units``.
        dropout: Dropout rate applied between the stem and the BiLSTM.

    Returns:
        A :class:`tf.keras.Model` whose output is ``float32[B, T, num_classes]``
        logits with ``T = W / 4``.
    """
    height, width, channels = input_shape

    # Raw uint8 input so the training pipeline can feed straight from tf.data
    # without an explicit cast.
    image_input = layers.Input(shape=input_shape, dtype=tf.uint8, name="image")
    # Rescaling casts to float and scales to [-1, 1] in one proper Keras layer
    # (no Lambda → saves/loads cleanly without safe_mode=False).
    x = layers.Rescaling(scale=1.0 / 127.5, offset=-1.0, name="normalize")(image_input)

    full_backbone = MobileNetV3Small(
        input_shape=(height, width, channels),
        alpha=0.5,
        include_top=False,
        weights=None,
        include_preprocessing=False,
    )
    trunk_out = full_backbone.get_layer(_STEM_CUT_LAYER).output
    backbone = Model(
        inputs=full_backbone.inputs,
        outputs=trunk_out,
        name="mobilenetv3_small_stride4",
    )
    features = backbone(x)  # [B, H/4, W/4, C']

    pooled = CollapseHeight(name="collapse_height")(features)  # [B, W/4, C']
    pooled = layers.Dropout(dropout, name="stem_dropout")(pooled)

    sequence = layers.Bidirectional(
        layers.LSTM(lstm_units, return_sequences=True, dropout=dropout),
        name="bilstm",
    )(pooled)

    logits = layers.Dense(num_classes, name="logits")(sequence)

    return Model(inputs=image_input, outputs=logits, name="crnn")
