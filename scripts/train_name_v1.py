"""v1 training entrypoint for the card-name OCR CRNN.

Self-contained — does not share the bottom-region training pipeline because
that one is wired to the bottom-region manifest format and ALPHABET. As name
OCR matures we may consolidate; for now this script is the source of truth.

Usage::

    .venv/bin/python scripts/train_name_v1.py --config configs/name_v1.yaml

The script reads a pre-rendered :class:`NamePool` from
``data/synth_names/v1/``, splits it train/val by label hash so no name
appears in both, builds a tf.data pipeline, and trains
:func:`build_name_region_model` with CTC loss.

Validation against real card scans is deferred to v2 once we have a labeled
real-name manifest. For v1, val_cer measures generalization to held-out
*synthetic* names — a useful but imperfect proxy for real-world OCR quality.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import editdistance
import tensorflow as tf
import yaml

from moxify_ocr.data.dataset import BLANK_INDEX, decode_label, encode_label
from moxify_ocr.data.name_alphabet import NAME_ALPHABET
from moxify_ocr.data.name_dataset import NamePool, build_tf_dataset, split_pool
from moxify_ocr.models.name_region import (
    NAME_NUM_CLASSES,
    build_name_region_model,
    ctc_loss,
)


class _NameCERCallback(tf.keras.callbacks.Callback):  # type: ignore[misc]
    """End-of-epoch CER for the name-OCR pipeline.

    Differs from :class:`moxify_ocr.train.callbacks.CERCallback` in two ways:
    it expects 2-tuple ``(image, label)`` batches (no ``label_length``), and
    it decodes against :data:`NAME_ALPHABET` instead of the bottom-region
    alphabet.
    """

    def __init__(self, val_ds: tf.data.Dataset) -> None:
        super().__init__()
        self.val_ds = val_ds

    def on_epoch_end(self, epoch: int, logs: dict | None = None) -> None:
        total_distance = 0
        total_length = 0
        for images, labels in self.val_ds:
            logits = self.model(images, training=False)
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
            dense = tf.sparse.to_dense(decoded_sparse[0], default_value=-1).numpy()
            true_dense = labels.numpy()
            for pred_row, true_row in zip(dense, true_dense, strict=True):
                pred_ids = [int(v) for v in pred_row if int(v) >= 1]
                true_ids = [int(v) for v in true_row if int(v) != BLANK_INDEX]
                pred_str = decode_label(pred_ids, alphabet=NAME_ALPHABET)
                true_str = decode_label(true_ids, alphabet=NAME_ALPHABET)
                total_distance += int(editdistance.eval(pred_str, true_str))
                total_length += max(len(true_str), 1)
        cer = total_distance / total_length if total_length else 0.0
        if logs is not None:
            logs["val_cer"] = cer
        print(f"  val_cer = {cer:.4f}")


def _load_config(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return yaml.safe_load(f)


def _build_lr_schedule(
    lr: float, warmup_steps: int, decay_steps: int
) -> tf.keras.optimizers.schedules.LearningRateSchedule:
    cosine = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=lr,
        decay_steps=max(decay_steps, 1),
        alpha=0.0,
    )

    class _WarmupCosine(tf.keras.optimizers.schedules.LearningRateSchedule):  # type: ignore[misc]
        def __call__(self, step: tf.Tensor) -> tf.Tensor:
            step_f = tf.cast(step, tf.float32)
            warmup_f = tf.cast(max(warmup_steps, 1), tf.float32)
            warmup_lr = lr * step_f / warmup_f
            decayed = cosine(tf.maximum(step - warmup_steps, 0))
            return tf.where(step_f < warmup_f, warmup_lr, decayed)

        def get_config(self) -> dict[str, Any]:
            return {"lr": lr, "warmup_steps": warmup_steps, "decay_steps": decay_steps}

    return _WarmupCosine()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    cfg = _load_config(args.config)

    seed = int(cfg["train"]["seed"])
    tf.keras.utils.set_random_seed(seed)

    pool_root = Path(cfg["data"]["pool_root"])
    pool = NamePool.load(pool_root)
    if not pool.entries:
        raise SystemExit(
            f"pool at {pool_root} is empty — run scripts/render_name_pool.py first"
        )
    print(f"loaded pool: {len(pool):,} samples")

    val_fraction = float(cfg["data"].get("val_fraction", 0.1))
    train_pool, val_pool = split_pool(pool, val_fraction=val_fraction, seed=seed)
    print(
        f"split: {len(train_pool):,} train / {len(val_pool):,} val "
        f"(holdout by label, val_fraction={val_fraction})"
    )

    def encode(label: str) -> list[int]:
        return encode_label(label, alphabet=NAME_ALPHABET)

    batch_size = int(cfg["data"]["batch_size"])
    augment = bool(cfg["data"].get("augment", False))
    if augment:
        print("augmentation enabled (photo-realistic synth→real gap bridge)")
    train_ds = build_tf_dataset(
        train_pool,
        encode_fn=encode,
        batch_size=batch_size,
        shuffle_buffer=int(cfg["data"]["shuffle_buffer"]),
        repeat=True,  # required when model.fit gets steps_per_epoch
        augment=augment,
        seed=seed,
    )
    val_ds = build_tf_dataset(
        val_pool,
        encode_fn=encode,
        batch_size=batch_size,
        shuffle_buffer=0,
        repeat=False,  # val must iterate each sample exactly once per epoch
        augment=False,  # val never augments — measures generalization, not robustness
        seed=seed,
    )

    model = build_name_region_model()
    model.summary()
    print(f"output classes: {NAME_NUM_CLASSES}")

    epochs = int(cfg["train"]["epochs"])
    steps_per_epoch = max(len(train_pool) // batch_size, 1)
    total_steps = max(epochs * steps_per_epoch, 1)
    lr_schedule = _build_lr_schedule(
        lr=float(cfg["train"]["lr"]),
        warmup_steps=int(cfg["train"]["warmup_steps"]),
        decay_steps=total_steps,
    )
    optimizer = tf.keras.optimizers.AdamW(
        learning_rate=lr_schedule,
        clipnorm=1.0,
    )
    model.compile(optimizer=optimizer, loss=ctc_loss)

    output_dir = Path(cfg["train"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    best_path = output_dir / "best.keras"

    callbacks: list[tf.keras.callbacks.Callback] = [
        _NameCERCallback(val_ds),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_cer", mode="min", patience=5, restore_best_weights=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(best_path),
            monitor="val_cer",
            mode="min",
            save_best_only=True,
        ),
    ]

    model.fit(
        train_ds,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_ds,
        callbacks=callbacks,
    )

    model.save(str(best_path))
    print(f"done; best model at {best_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
