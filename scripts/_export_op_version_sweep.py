"""Try multiple converter configurations and report the resulting TFLite op
versions for each. Goal: find a combination that emits only v1 ops so the
model loads on TFLiteSwift 2.12.0 / LiteRT 1.4.1."""

from __future__ import annotations

from pathlib import Path

import tensorflow as tf
from tensorflow.lite.python import schema_py_generated as schema_fb

from moxify_ocr.models.crnn import SqueezeHeight, build_crnn  # noqa: F401
from moxify_ocr.models.name_region import NAME_INPUT_SHAPE, NAME_NUM_CLASSES

KERAS = Path("artifacts/name_v1/best.keras")
OUT_DIR = Path("/tmp/op_version_sweep")
OUT_DIR.mkdir(exist_ok=True)


def build_unrolled() -> tf.keras.Model:
    src = tf.keras.models.load_model(
        KERAS, custom_objects={"SqueezeHeight": SqueezeHeight}, compile=False
    )
    rebuilt = build_crnn(
        input_shape=NAME_INPUT_SHAPE,
        num_classes=NAME_NUM_CLASSES,
        lstm_units=256,
        unroll=True,
    )
    rebuilt(tf.zeros((1, *NAME_INPUT_SHAPE), dtype=tf.uint8), training=False)
    rebuilt.set_weights(src.get_weights())
    fixed_in = tf.keras.Input(
        batch_shape=(1, *NAME_INPUT_SHAPE), dtype=tf.uint8, name="image"
    )
    return tf.keras.Model(fixed_in, rebuilt(fixed_in), name="name_v1_b1_unrolled")


def op_versions(path: Path) -> dict[str, int]:
    with path.open("rb") as f:
        buf = f.read()
    model = schema_fb.Model.GetRootAsModel(buf, 0)
    out: dict[str, int] = {}
    for i in range(model.OperatorCodesLength()):
        oc = model.OperatorCodes(i)
        name = next(
            (k for k, v in vars(schema_fb.BuiltinOperator).items() if v == oc.BuiltinCode()),
            f"UNK({oc.BuiltinCode()})",
        )
        out[name] = oc.Version()
    return out


def configs():
    yield "baseline_fp32", {}
    yield "baseline_quantized", {"quantize": True}
    yield "fp32_no_per_channel", {"_experimental_disable_per_channel": True}
    yield "fp32_no_per_channel_dense", {
        "_experimental_disable_per_channel_quantization_for_dense_layers": True
    }
    yield "fp32_no_fuse_mul_fc", {"_experimental_disable_fuse_mul_and_fc": True}
    yield "fp32_no_resource_vars", {"experimental_enable_resource_variables": False}
    yield "q_legacy_quantizer", {
        "quantize": True,
        "experimental_new_quantizer": False,
    }
    yield "q_legacy_dyn_range", {
        "quantize": True,
        "experimental_new_dynamic_range_quantizer": False,
    }
    yield "q_no_per_channel", {
        "quantize": True,
        "_experimental_disable_per_channel": True,
    }
    yield "q_legacy_quantizer_no_per_channel", {
        "quantize": True,
        "experimental_new_quantizer": False,
        "_experimental_disable_per_channel": True,
    }
    yield "q_legacy_dyn_range_no_per_channel", {
        "quantize": True,
        "experimental_new_dynamic_range_quantizer": False,
        "_experimental_disable_per_channel": True,
    }


def convert(model: tf.keras.Model, knobs: dict) -> bytes:
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    quantize = knobs.pop("quantize", False)
    if quantize:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
    for k, v in knobs.items():
        setattr(converter, k, v)
    return converter.convert()


def main() -> int:
    print("building unrolled fixed-batch model ...")
    m = build_unrolled()
    rows: list[tuple[str, int, dict[str, int]]] = []
    for name, knobs in configs():
        try:
            data = convert(m, dict(knobs))
            path = OUT_DIR / f"{name}.tflite"
            path.write_bytes(data)
            ovs = op_versions(path)
            print(f"\n[{name}] {len(data)/1_048_576:.2f} MB")
            for op, ver in sorted(ovs.items()):
                marker = "" if ver == 1 else f"  ⚠️ v{ver} > v1"
                print(f"  {op}: v{ver}{marker}")
            max_v = max(ovs.values())
            rows.append((name, max_v, ovs))
        except Exception as e:
            print(f"\n[{name}] FAILED: {type(e).__name__}: {str(e)[:200]}")
    print("\n=== summary (max op version per config) ===")
    rows.sort(key=lambda r: r[1])
    for name, max_v, _ in rows:
        marker = "✓" if max_v == 1 else "✗"
        print(f"  {marker} {name:50s}  max_v={max_v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
