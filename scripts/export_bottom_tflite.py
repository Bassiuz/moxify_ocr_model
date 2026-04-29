"""Convert the trained bottom-region OCR Keras model to a TFLite flatbuffer.

Usage::

    .venv/bin/python scripts/export_bottom_tflite.py \\
        --keras "best_v3(1).keras" \\
        --out artifacts/bottom_region_v3/bottom_region_v3.tflite

Mirrors :mod:`scripts.export_name_tflite` — same architecture (CRNN + BiLSTM
+ CTC head), just different input shape (48x256x3 vs the name model's
larger crop) and num_classes (45 vs the name model's alphabet). All the
runtime / quantization knobs work identically. See that file's docstring
for the full explanation of `--no-flex` / `--legacy-runtime` / `--quantize`.

Common invocations:

    # Float32 with Flex delegate (smallest convert effort, needs Flex on
    # the consumer)
    --keras X.keras --out X.tflite

    # Dynamic-range int8 weights, still Flex-delegated runtime
    --keras X.keras --out X_q.tflite --quantize

    # No Flex, modern LiteRT consumer (current-gen runtime)
    --keras X.keras --out X_modern_q.tflite --quantize --no-flex

    # No Flex, legacy iOS / older LiteRT
    --keras X.keras --out X_legacy_q.tflite --quantize --no-flex --legacy-runtime
"""

from __future__ import annotations

import argparse
from pathlib import Path

import tensorflow as tf

from moxify_ocr.models.crnn import SqueezeHeight  # noqa: F401  used at load time

_BOTTOM_INPUT_SHAPE = (48, 256, 3)
_BOTTOM_NUM_CLASSES = 45
_BOTTOM_LSTM_UNITS = 256


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--keras", type=Path, required=True, help="Path to the trained .keras file.")
    parser.add_argument("--out", type=Path, required=True, help="Path to write the .tflite file.")
    parser.add_argument(
        "--quantize",
        action="store_true",
        help="Apply dynamic-range (weight-only int8) quantization.",
    )
    parser.add_argument(
        "--no-flex",
        action="store_true",
        help="Convert without Select-TF ops. See export_name_tflite for details.",
    )
    parser.add_argument(
        "--legacy-runtime",
        action="store_true",
        help="With --no-flex: unroll the LSTM + clamp op versions to v3 for "
        "TFLiteSwift 2.12 / LiteRT 1.4.x compatibility.",
    )
    args = parser.parse_args()
    if args.legacy_runtime and not args.no_flex:
        raise SystemExit("--legacy-runtime requires --no-flex")

    if not args.keras.exists():
        raise SystemExit(f"missing keras model: {args.keras}")
    args.out.parent.mkdir(parents=True, exist_ok=True)

    print(f"loading {args.keras} ...")
    model = tf.keras.models.load_model(
        args.keras, custom_objects={"SqueezeHeight": SqueezeHeight}, compile=False
    )
    print(f"  input: {model.input_shape}  output: {model.output_shape}")

    if args.no_flex:
        if args.legacy_runtime:
            from moxify_ocr.models.crnn import build_crnn  # noqa: PLC0415

            unrolled = build_crnn(
                input_shape=_BOTTOM_INPUT_SHAPE,
                num_classes=_BOTTOM_NUM_CLASSES,
                lstm_units=_BOTTOM_LSTM_UNITS,
                unroll=True,
            )
            unrolled(tf.zeros((1, *_BOTTOM_INPUT_SHAPE), dtype=tf.uint8), training=False)
            unrolled(tf.zeros((1, *_BOTTOM_INPUT_SHAPE), dtype=tf.uint8), training=False)
            unrolled.set_weights(model.get_weights())
            print(f"  rebuilt with unroll=True; copied {len(unrolled.weights)} weight tensors")
            fixed_input = tf.keras.Input(
                batch_shape=(1, *_BOTTOM_INPUT_SHAPE), dtype=tf.uint8, name="image"
            )
            target_model = tf.keras.Model(
                fixed_input, unrolled(fixed_input), name="bottom_v3_b1_unrolled"
            )
            print("no-flex/legacy-runtime: batch=1, unrolled, v3-clamped quant")
        else:
            fixed_input = tf.keras.Input(
                batch_shape=(1, *_BOTTOM_INPUT_SHAPE), dtype=tf.uint8, name="image"
            )
            target_model = tf.keras.Model(
                fixed_input, model(fixed_input), name="bottom_v3_b1_modern"
            )
            print("no-flex/modern: batch=1, dynamic LSTM, builtins-only target")

        converter = tf.lite.TFLiteConverter.from_keras_model(target_model)
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    else:
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,
            tf.lite.OpsSet.SELECT_TF_OPS,
        ]
        converter._experimental_lower_tensor_list_ops = False

    if args.quantize:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        print("dynamic-range quantization enabled (weight-only int8)")
        if args.no_flex and args.legacy_runtime:
            converter.experimental_new_dynamic_range_quantizer = False
            converter._experimental_disable_per_channel = True
            print("  + legacy quantization (legacy dyn-range + per-tensor scales)")

    tflite_bytes = converter.convert()
    args.out.write_bytes(tflite_bytes)
    size_mb = len(tflite_bytes) / 1_048_576
    print(f"wrote {args.out}  ({size_mb:.2f} MB)")

    # Op-set sanity check (mirrors export_name_tflite).
    from collections import Counter

    from tensorflow.lite.python import schema_py_generated as schema_fb  # noqa: PLC0415

    interp = tf.lite.Interpreter(model_path=str(args.out))
    op_counts = Counter(o["op_name"] for o in interp._get_ops_details())
    with args.out.open("rb") as f:
        flatbuf = f.read()
    fb_model = schema_fb.Model.GetRootAsModel(flatbuf, 0)
    op_versions: dict[str, int] = {}
    for i in range(fb_model.OperatorCodesLength()):
        oc = fb_model.OperatorCodes(i)
        name = next(
            (k for k, v in vars(schema_fb.BuiltinOperator).items() if v == oc.BuiltinCode()),
            f"UNK({oc.BuiltinCode()})",
        )
        op_versions[name] = max(op_versions.get(name, 0), oc.Version())
    print("op set + versions:")
    for name in sorted(op_counts):
        v = op_versions.get(name, 1)
        marker = "" if v == 1 else f"  v{v}"
        print(f"  {name}: count={op_counts[name]}{marker}")

    if args.no_flex and args.legacy_runtime:
        forbidden = {"WHILE", "VAR_HANDLE", "READ_VARIABLE", "ASSIGN_VARIABLE", "CALL_ONCE"}
        bad = forbidden & set(op_counts.keys())
        if bad:
            raise SystemExit(
                f"--legacy-runtime export contains ops older runtimes won't load: {sorted(bad)}"
            )
        max_v = max(op_versions.values())
        if max_v <= 3:
            print(f"✓ all op versions ≤ v{max_v} — TFLiteSwift 2.12 / LiteRT 1.4.x compatible")
        else:
            high = sorted(f"{name}:v{v}" for name, v in op_versions.items() if v > 3)
            print(f"WARN: max op version v{max_v} (high: {high}). Test before ship.")
    elif args.no_flex:
        if "FlexOp" in op_counts or any("Flex" in n for n in op_counts):
            raise SystemExit("--no-flex export still contains Select-TF / Flex ops")
        print(
            f"✓ modern-runtime export: max op v{max(op_versions.values())} — "
            "requires current-gen LiteRT / flutter_litert (no Flex needed)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
