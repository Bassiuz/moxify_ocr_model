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
    parser.add_argument(
        "--no-flex",
        action="store_true",
        help="Try to convert WITHOUT Select-TF ops by pinning batch=1 via a "
        "concrete tf.function signature. Lets the LSTM's TensorList "
        "element shape resolve to static. Use when the consumer's TFLite "
        "runtime can't ship the Flex delegate (e.g. LiteRT-only Android).",
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

    if args.no_flex:
        # Rebuild the model with ``unroll=True`` LSTMs and a fixed batch
        # of 1, then copy the trained weights over. With T=128 statically
        # known, the recurrent loop expands at graph-construction into
        # 128 separate cell invocations — no while_loop, no TensorList,
        # no resource variables. The resulting TFLite graph uses only
        # basic ops (Conv/Dense/Add/Mul/Sigmoid/Tanh/Reshape) that load
        # cleanly on conservative runtimes like LiteRT 1.4.x and iOS
        # TensorFlowLiteSwift 2.12.0. Trade-off: graph is bigger
        # (~50-80 MB) and inference is single-image (batch=1).
        from moxify_ocr.models.crnn import build_crnn  # noqa: PLC0415
        from moxify_ocr.models.name_region import (  # noqa: PLC0415
            NAME_INPUT_SHAPE,
            NAME_NUM_CLASSES,
        )

        unrolled = build_crnn(
            input_shape=NAME_INPUT_SHAPE,
            num_classes=NAME_NUM_CLASSES,
            lstm_units=256,
            unroll=True,
        )
        # Materialize variables with a forward pass, then copy weights.
        unrolled(tf.zeros((1, *NAME_INPUT_SHAPE), dtype=tf.uint8), training=False)
        unrolled.set_weights(model.get_weights())
        print(
            f"  rebuilt with unroll=True; copied {len(unrolled.weights)} weight tensors"
        )

        # Wrap in a fixed-batch input layer so the entire graph is shape-static.
        fixed_input = tf.keras.Input(
            batch_shape=(1, *NAME_INPUT_SHAPE), dtype=tf.uint8, name="image"
        )
        fixed_output = unrolled(fixed_input)
        fixed_batch_model = tf.keras.Model(
            fixed_input, fixed_output, name="name_v1_b1_unrolled"
        )

        converter = tf.lite.TFLiteConverter.from_keras_model(fixed_batch_model)
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
        print("no-flex mode: batch pinned to 1, unrolled LSTMs, builtins-only target")
    else:
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        # Our CRNN has BiLSTM cells whose ``TensorListReserve`` ops can't be
        # lowered to TFLite builtins without a static element shape. The
        # standard fix is to keep those ops as Select-TF (Flex) ops, which
        # the TFLite runtime supports as long as the consumer bundles the
        # Flex delegate. This is the recommended path for RNN-based OCR.
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,
            tf.lite.OpsSet.SELECT_TF_OPS,
        ]
        converter._experimental_lower_tensor_list_ops = False

    if args.quantize:
        # Dynamic-range quantization: weights → int8, activations stay
        # float at runtime. No calibration needed, and it works alongside
        # SELECT_TF_OPS (full int8 doesn't, because the calibrator needs
        # the Flex delegate to run the LSTM during calibration).
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        print("dynamic-range quantization enabled (weight-only int8)")
        if args.no_flex:
            # Compat-mode quantization: use the legacy dynamic-range
            # quantizer + per-tensor (not per-channel) scales. This pulls
            # CONV_2D / FULLY_CONNECTED op versions down from v5/v12 to
            # v2/v3, which TFLite runtimes from the 2.12-2.16 era ship
            # kernels for. Without these knobs the converter emits
            # FULLY_CONNECTED:v12 which iOS TFLiteSwift 2.12 / LiteRT
            # 1.4.x reject with "Unable to create interpreter".
            converter.experimental_new_dynamic_range_quantizer = False
            converter._experimental_disable_per_channel = True
            print("  + compat quantization (legacy dyn-range + per-tensor scales)")

    tflite_bytes = converter.convert()
    args.out.write_bytes(tflite_bytes)
    size_mb = len(tflite_bytes) / 1_048_576
    print(f"wrote {args.out}  ({size_mb:.2f} MB)")

    # Verify the op set + op versions so we don't ship a model that an
    # older TFLite runtime can't load. Two failure modes worth catching:
    # (a) forbidden op kinds (WHILE / resource variables / TensorList),
    # (b) op versions newer than the consumer's runtime supports.
    from collections import Counter

    from tensorflow.lite.python import schema_py_generated as schema_fb  # noqa: PLC0415

    interp = tf.lite.Interpreter(model_path=str(args.out))
    op_counts = Counter(o["op_name"] for o in interp._get_ops_details())
    # Versions live in the FlatBuffer's operator_codes table.
    with args.out.open("rb") as f:
        flatbuf = f.read()
    fb_model = schema_fb.Model.GetRootAsModel(flatbuf, 0)
    op_versions: dict[str, int] = {}
    for i in range(fb_model.OperatorCodesLength()):
        oc = fb_model.OperatorCodes(i)
        name = next(
            (
                k
                for k, v in vars(schema_fb.BuiltinOperator).items()
                if v == oc.BuiltinCode()
            ),
            f"UNK({oc.BuiltinCode()})",
        )
        op_versions[name] = max(op_versions.get(name, 0), oc.Version())
    print("op set + versions:")
    for name in sorted(op_counts):
        v = op_versions.get(name, 1)
        marker = "" if v == 1 else f"  v{v}"
        print(f"  {name}: count={op_counts[name]}{marker}")

    if args.no_flex:
        forbidden = {
            "WHILE",
            "VAR_HANDLE",
            "READ_VARIABLE",
            "ASSIGN_VARIABLE",
            "CALL_ONCE",
        }
        bad = forbidden & set(op_counts.keys())
        if bad:
            raise SystemExit(
                f"--no-flex export contains ops conservative runtimes won't "
                f"load: {sorted(bad)}. The LSTM didn't fold to "
                f"UNIDIRECTIONAL_SEQUENCE_LSTM. Try the unroll path."
            )
        max_v = max(op_versions.values())
        # Ops above this threshold may be too new for older runtimes
        # (iOS TFLiteSwift 2.12 supports through ~v7 for most ops; LiteRT
        # 1.4.x supports through ~v9). v3 is comfortably within both.
        if max_v <= 3:
            print(f"✓ all op versions ≤ v{max_v} — broadly compatible")
        else:
            high = sorted(
                f"{name}:v{v}" for name, v in op_versions.items() if v > 3
            )
            print(
                f"WARN: max op version v{max_v} (high: {high}). "
                "May not load on TFLiteSwift 2.12 / LiteRT 1.4.x — test before ship."
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
