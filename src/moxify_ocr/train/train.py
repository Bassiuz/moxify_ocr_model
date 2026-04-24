"""Training entrypoint for the bottom-region OCR CRNN.

Usage::

    uv run python -m moxify_ocr.train.train --config configs/bottom_region_v1.yaml

The entrypoint is deliberately thin: it parses the YAML config, builds the
train/val datasets via :mod:`moxify_ocr.data.dataset`, builds the model via
:mod:`moxify_ocr.models.bottom_region`, compiles it with AdamW + a
warmup-then-cosine-decay LR schedule, and calls :meth:`tf.keras.Model.fit`
with CER/EarlyStopping/ModelCheckpoint callbacks.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import tensorflow as tf
import yaml

from moxify_ocr.data.dataset import DatasetConfig, build_dataset, is_trainable
from moxify_ocr.data.manifest import read_manifest
from moxify_ocr.data.splits import assign_split
from moxify_ocr.models.bottom_region import build_bottom_region_model, ctc_loss
from moxify_ocr.train.callbacks import CERCallback


def _load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        cfg: dict[str, Any] = yaml.safe_load(handle)
    return cfg


def _apply_overrides(cfg: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    """Apply ``section.key=value`` dotted-path overrides to a nested config dict.

    Values are parsed as YAML scalars so ``"1e-3"`` becomes a float and
    ``"true"`` becomes a bool — same rules as the config file itself.
    """
    for spec in overrides:
        if "=" not in spec:
            raise ValueError(f"override must be 'key=value', got {spec!r}")
        dotted, raw = spec.split("=", 1)
        value = yaml.safe_load(raw)
        parts = dotted.split(".")
        node: dict[str, Any] = cfg
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value
    return cfg


def _build_lr_schedule(
    lr: float,
    warmup_steps: int,
    decay_steps: int,
) -> tf.keras.optimizers.schedules.LearningRateSchedule:
    """Linear warmup to ``lr`` over ``warmup_steps``, then cosine decay to 0."""
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


def _count_steps(cfg: dict[str, Any], *, split: str) -> int:
    """Approximate steps-per-epoch from the filtered manifest size.

    Counts rows that pass the ``is_trainable`` + split-assignment filters
    without actually loading images — orders of magnitude faster than
    materializing the tf.data pipeline once.
    """
    data_cfg = cfg["data"]
    seed = int(cfg["train"]["seed"])
    min_release = str(data_cfg["min_release"])
    batch_size = int(data_cfg["batch_size"])
    count = sum(
        1
        for entry in read_manifest(Path(data_cfg["manifest"]))
        if is_trainable(entry, min_release=min_release)
        and assign_split(entry.set_code, seed=seed) == split
    )
    return max(count // batch_size, 1)


def _build_datasets(
    cfg: dict[str, Any],
) -> tuple[tf.data.Dataset, tf.data.Dataset]:
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    target_size = (int(model_cfg["input_width"]), int(model_cfg["input_height"]))
    seed = int(cfg["train"]["seed"])

    train_cfg = DatasetConfig(
        manifest_path=Path(data_cfg["manifest"]),
        images_root=Path(data_cfg["images_root"]),
        split="train",
        target_size=target_size,
        batch_size=int(data_cfg["batch_size"]),
        shuffle_buffer=int(data_cfg["shuffle_buffer"]),
        augment=True,
        seed=seed,
        min_release=str(data_cfg["min_release"]),
    )
    val_cfg = DatasetConfig(
        manifest_path=Path(data_cfg["manifest"]),
        images_root=Path(data_cfg["images_root"]),
        split="val",
        target_size=target_size,
        batch_size=int(data_cfg["batch_size"]),
        shuffle_buffer=0,
        augment=False,
        seed=seed,
        min_release=str(data_cfg["min_release"]),
    )
    return build_dataset(train_cfg), build_dataset(val_cfg)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train the bottom-region OCR CRNN.")
    parser.add_argument("--config", type=Path, required=True, help="Path to YAML config.")
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Dotted-path override, e.g. train.epochs=2. May be repeated.",
    )
    args = parser.parse_args(argv)

    cfg = _load_config(args.config)
    cfg = _apply_overrides(cfg, args.override)

    tf.keras.utils.set_random_seed(int(cfg["train"]["seed"]))

    train_ds, val_ds = _build_datasets(cfg)

    model = build_bottom_region_model()

    epochs = int(cfg["train"]["epochs"])
    # Count trainable train-split rows directly from the manifest so we don't
    # have to materialize the whole image-loading pipeline just to count
    # steps. This count slightly overestimates the real step count because
    # the generator skips rows with unencodable labels or missing images, but
    # it's close enough for LR scheduling.
    steps_per_epoch = _count_steps(cfg, split="train")
    total_steps = max(epochs * steps_per_epoch, 1)
    lr_schedule = _build_lr_schedule(
        lr=float(cfg["train"]["lr"]),
        warmup_steps=int(cfg["train"]["warmup_steps"]),
        decay_steps=total_steps,
    )

    optimizer = tf.keras.optimizers.AdamW(
        learning_rate=lr_schedule,
        clipnorm=1.0,  # gradient clipping — CTC gradients can spike early
    )
    model.compile(optimizer=optimizer, loss=ctc_loss)

    output_dir = Path(cfg["train"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    best_path = output_dir / "best.keras"

    callbacks: list[tf.keras.callbacks.Callback] = [
        CERCallback(val_ds),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_cer",
            mode="min",
            patience=5,
            restore_best_weights=True,
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
