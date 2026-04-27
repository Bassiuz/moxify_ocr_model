"""Convert the trained name-OCR Keras model to a TFLite flatbuffer.

Usage::

    .venv/bin/python scripts/export_name_tflite.py \\
        --keras artifacts/name_v1/best.keras \\
        --out   artifacts/name_v1/name_v1.tflite

Outputs a float32 TFLite file by default (~13 MB). Pass ``--quantize`` for
dynamic-range (weight-only int8) quantization which roughly halves the
file. Note that *full* int8 isn't possible for our architecture: the
BiLSTM cells route through SELECT_TF_OPS as float, and the int8
calibration path requires the Flex delegate at calibration time which the
host TF interpreter doesn't ship with. Dynamic-range quantization
sidesteps this — weights get int8'd at conversion time, no calibration
needed, runtime arithmetic stays in float.

The output ``.tflite`` requires the **Flex delegate** at inference time
because it embeds Select-TF ops for the BiLSTM. On Android add the
``org.tensorflow:tensorflow-lite-select-tf-ops`` dependency; on iOS use
the ``TensorFlowLiteSelectTfOps`` pod.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import tensorflow as tf

from moxify_ocr.models.crnn import SqueezeHeight  # noqa: F401  used at load time


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--keras", type=Path, required=True, help="Path to the trained .keras file."
    )
    parser.add_argument(
        "--out", type=Path, required=True, help="Path to write the .tflite file."
    )
    parser.add_argument(
        "--quantize",
        action="store_true",
        help="Apply dynamic-range (weight-only int8) quantization. Roughly "
        "halves the file size; runtime activations stay float. No calibration "
        "data needed.",
    )
    args = parser.parse_args()

    if not args.keras.exists():
        raise SystemExit(f"missing keras model: {args.keras}")
    args.out.parent.mkdir(parents=True, exist_ok=True)

    print(f"loading {args.keras} ...")
    # Load via build + load_weights to avoid the custom-layer deserialization
    # issue with tf.keras.models.load_model (matches the bottom-region eval
    # harness convention).
    model = tf.keras.models.load_model(
        args.keras, custom_objects={"SqueezeHeight": SqueezeHeight}, compile=False
    )
    print(f"  input: {model.input_shape}  output: {model.output_shape}")

    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    # Our CRNN has BiLSTM cells whose ``TensorListReserve`` ops can't be
    # lowered to TFLite builtins without a static element shape. The
    # standard fix is to keep those ops as Select-TF (Flex) ops, which
    # the TFLite runtime supports as long as the consumer bundles the
    # Flex delegate. This is the recommended path for RNN-based OCR.
    base_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS,
    ]
    converter.target_spec.supported_ops = base_ops
    converter._experimental_lower_tensor_list_ops = False

    if args.quantize:
        # Dynamic-range quantization: weights → int8, activations stay
        # float at runtime. No calibration needed, and it works alongside
        # SELECT_TF_OPS (full int8 doesn't, because the calibrator needs
        # the Flex delegate to run the LSTM during calibration).
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        print("dynamic-range quantization enabled (weight-only int8)")

    tflite_bytes = converter.convert()
    args.out.write_bytes(tflite_bytes)
    size_mb = len(tflite_bytes) / 1_048_576
    print(f"wrote {args.out}  ({size_mb:.2f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
