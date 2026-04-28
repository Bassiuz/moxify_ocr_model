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
        help="Convert without Select-TF ops. Default sub-mode (--modern, "
        "implicit) produces a small model that needs a current-gen TFLite "
        "runtime (modern LiteRT / flutter_litert) — uses WHILE + resource-"
        "variable builtins for the LSTM, modern op versions, smallest size. "
        "Pass --legacy-runtime to instead unroll the LSTM and clamp op "
        "versions to v3 for older runtimes (TFLiteSwift 2.12, LiteRT 1.4.x).",
    )
    parser.add_argument(
        "--legacy-runtime",
        action="store_true",
        help="With --no-flex: unroll the LSTM + clamp quantization op "
        "versions to v3 for compat with TFLiteSwift 2.12.0 / LiteRT 1.4.x. "
        "Bigger graph (~5 MB vs ~2 MB), uses only basic builtins. Skip this "
        "if you're on flutter_litert / current-gen LiteRT.",
    )
    args = parser.parse_args()
    if args.legacy_runtime and not args.no_flex:
        raise SystemExit("--legacy-runtime requires --no-flex")

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
        from moxify_ocr.models.name_region import NAME_INPUT_SHAPE  # noqa: PLC0415

        if args.legacy_runtime:
            # Legacy-runtime path: rebuild the model with ``unroll=True``
            # LSTMs and a fixed batch of 1. With T=128 statically known,
            # the recurrent loop expands at graph-construction into 128
            # separate cell invocations — no while_loop, no TensorList, no
            # resource variables. Loads cleanly on TFLiteSwift 2.12.0 /
            # LiteRT 1.4.x. Trade-off: graph is bigger (~5 MB quantized
            # vs ~2 MB modern); inference is single-image (batch=1).
            from moxify_ocr.models.crnn import build_crnn  # noqa: PLC0415
            from moxify_ocr.models.name_region import (  # noqa: PLC0415
                NAME_NUM_CLASSES,
            )

            unrolled = build_crnn(
                input_shape=NAME_INPUT_SHAPE,
                num_classes=NAME_NUM_CLASSES,
                lstm_units=256,
                unroll=True,
            )
            unrolled(tf.zeros((1, *NAME_INPUT_SHAPE), dtype=tf.uint8), training=False)
            unrolled(tf.zeros((1, *NAME_INPUT_SHAPE), dtype=tf.uint8), training=False)
            unrolled.set_weights(model.get_weights())
            print(
                f"  rebuilt with unroll=True; copied {len(unrolled.weights)} weight tensors"
            )
            fixed_input = tf.keras.Input(
                batch_shape=(1, *NAME_INPUT_SHAPE), dtype=tf.uint8, name="image"
            )
            target_model = tf.keras.Model(
                fixed_input, unrolled(fixed_input), name="name_v1_b1_unrolled"
            )
            print(
                "no-flex/legacy-runtime: batch=1, unrolled, v3-clamped quant"
            )
        else:
            # Modern-runtime path: keep the BiLSTM as-is (no unroll). The
            # converter implements it via WHILE + VAR_HANDLE / READ_VARIABLE
            # / ASSIGN_VARIABLE / TensorList ops — all TFLite builtins, but
            # at op versions that current-gen LiteRT / flutter_litert
            # supports natively. Smaller, faster graph than the unrolled
            # path. Inputs still pinned to batch=1 to match the on-device
            # one-shot inference contract.
            fixed_input = tf.keras.Input(
                batch_shape=(1, *NAME_INPUT_SHAPE), dtype=tf.uint8, name="image"
            )
            target_model = tf.keras.Model(
                fixed_input, model(fixed_input), name="name_v1_b1_modern"
            )
            print("no-flex/modern: batch=1, dynamic LSTM, builtins-only target")

        converter = tf.lite.TFLiteConverter.from_keras_model(target_model)
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
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
        if args.no_flex and args.legacy_runtime:
            # Legacy-runtime quantization: use the legacy dynamic-range
            # quantizer + per-tensor (not per-channel) scales. This pulls
            # CONV_2D / FULLY_CONNECTED op versions down from v5/v12 to
            # v2/v3, which TFLite runtimes from the 2.12-2.16 era ship
            # kernels for. Without these knobs the converter emits
            # FULLY_CONNECTED:v12 which iOS TFLiteSwift 2.12 / LiteRT
            # 1.4.x reject with "Unable to create interpreter".
            converter.experimental_new_dynamic_range_quantizer = False
            converter._experimental_disable_per_channel = True
            print("  + legacy quantization (legacy dyn-range + per-tensor scales)")

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

    if args.no_flex and args.legacy_runtime:
        # The legacy-runtime path must NOT contain control-flow / resource-
        # variable ops (the unrolled LSTM avoids them by design). It must
        # also keep op versions ≤ v3 to fit TFLiteSwift 2.12.0 / LiteRT
        # 1.4.x's kernel registry.
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
                f"--legacy-runtime export contains ops older runtimes won't "
                f"load: {sorted(bad)}. The LSTM didn't unroll cleanly."
            )
        max_v = max(op_versions.values())
        if max_v <= 3:
            print(f"✓ all op versions ≤ v{max_v} — TFLiteSwift 2.12 / LiteRT 1.4.x compatible")
        else:
            high = sorted(
                f"{name}:v{v}" for name, v in op_versions.items() if v > 3
            )
            print(
                f"WARN: max op version v{max_v} (high: {high}). "
                "May not load on TFLiteSwift 2.12 / LiteRT 1.4.x — test before ship."
            )
    elif args.no_flex:
        # Modern-runtime path: WHILE / resource-variable ops are EXPECTED
        # and fine on current-gen LiteRT / flutter_litert. We only flag
        # SELECT_TF_OPS leakage (which would mean Flex is needed after all).
        if "FlexOp" in op_counts or any("Flex" in n for n in op_counts):
            raise SystemExit(
                "--no-flex export still contains Select-TF / Flex ops"
            )
        print(
            f"✓ modern-runtime export: max op v{max(op_versions.values())} — "
            "requires current-gen LiteRT / flutter_litert (no Flex needed)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
