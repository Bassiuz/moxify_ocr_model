"""Convert the trained name-OCR Keras model to a TFLite flatbuffer.

Usage::

    .venv/bin/python scripts/export_name_tflite.py \\
        --keras artifacts/name_v1/best.keras \\
        --out   artifacts/name_v1/name_v1.tflite

Outputs a float32 TFLite file by default. Pass ``--int8-with-pool POOL_DIR``
to produce an int8-quantized model using ~256 calibration images sampled
from the rendered pool — typically 4× smaller and faster on phone CPUs.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import tensorflow as tf

from moxify_ocr.data.name_dataset import NamePool
from moxify_ocr.models.crnn import SqueezeHeight
from moxify_ocr.models.name_region import NAME_INPUT_SHAPE


def _representative_dataset(pool_root: Path, n: int = 256, seed: int = 0):
    """Yield ``(image,)`` tensors from the rendered pool for int8 calibration."""
    pool = NamePool.load(pool_root)
    if not pool.entries:
        raise SystemExit(f"empty pool at {pool_root} — cannot calibrate int8")
    rng = random.Random(seed)
    sample = rng.sample(pool.entries, min(n, len(pool.entries)))

    def gen():
        from PIL import Image

        for entry in sample:
            img = np.asarray(
                Image.open(entry.image_path).convert("RGB"), dtype=np.uint8
            )
            assert img.shape == NAME_INPUT_SHAPE, (
                f"unexpected pool image shape: {img.shape}"
            )
            yield [img[None].astype(np.float32)]

    return gen


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--keras", type=Path, required=True, help="Path to the trained .keras file."
    )
    parser.add_argument(
        "--out", type=Path, required=True, help="Path to write the .tflite file."
    )
    parser.add_argument(
        "--int8-with-pool",
        type=Path,
        default=None,
        help="Quantize to int8 using calibration samples from this rendered pool.",
    )
    parser.add_argument(
        "--n-calib",
        type=int,
        default=256,
        help="Number of calibration samples (only used with --int8-with-pool).",
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

    if args.int8_with_pool is not None:
        print(f"int8 calibration from {args.int8_with_pool} ({args.n_calib} samples)")
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = _representative_dataset(
            args.int8_with_pool, n=args.n_calib
        )
        # Force fully-int8 ops; the input/output remain uint8 / int8 too so
        # the consumer can skip the float32 cast on phone.
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.uint8
        converter.inference_output_type = tf.float32  # keep logits in float for CTC

    tflite_bytes = converter.convert()
    args.out.write_bytes(tflite_bytes)
    size_mb = len(tflite_bytes) / 1_048_576
    print(f"wrote {args.out}  ({size_mb:.2f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
