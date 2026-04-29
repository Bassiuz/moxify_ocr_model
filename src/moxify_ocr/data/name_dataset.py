"""Pre-rendered name-region pool reader for the dataset pipeline.

Mirrors :mod:`moxify_ocr.data.cardconjurer_dataset` for the bottom-region
pipeline: the renderer writes ``{root}/labels.jsonl`` + ``{root}/images/``
and this module exposes :class:`NamePool` + :func:`sample_from_pool` that
the dataset builder calls one row at a time.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class _PoolEntry:
    image_path: Path  # absolute
    label: str


@dataclass
class NamePool:
    """In-memory index of ``(image_path, label)`` pairs from a name-region pool."""

    entries: list[_PoolEntry]

    @classmethod
    def load(cls, root: Path) -> NamePool:
        """Read ``{root}/labels.jsonl`` and resolve relative image paths.

        ``root`` is the directory the renderer wrote to — contains
        ``labels.jsonl`` and an ``images/`` subdir. Returns an empty pool
        if the manifest doesn't exist (the dataset builder treats an empty
        pool as a configuration error and raises a clearer message).
        """
        manifest = root / "labels.jsonl"
        if not manifest.exists():
            return cls(entries=[])
        entries: list[_PoolEntry] = []
        with manifest.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                entries.append(
                    _PoolEntry(
                        image_path=root / row["image_path"],
                        label=row["label"],
                    )
                )
        return cls(entries=entries)

    def __len__(self) -> int:
        return len(self.entries)


def sample_from_pool(pool: NamePool, *, seed: int) -> tuple[np.ndarray, str]:
    """Draw one ``(image, label)`` pair deterministically from the pool."""
    if not pool.entries:
        raise ValueError("NamePool is empty — nothing to sample from")
    rng = random.Random(seed)
    entry = rng.choice(pool.entries)
    img = np.asarray(Image.open(entry.image_path).convert("RGB"), dtype=np.uint8)
    return img, entry.label


def split_pool(
    pool: NamePool, *, val_fraction: float = 0.1, seed: int = 0
) -> tuple[NamePool, NamePool]:
    """Deterministically split a pool into (train, val) by hashing the label.

    Splitting by *label* (not by image index) means the same card name never
    appears in both train and val — so val genuinely measures generalization
    to held-out names, not memorization of seen ones rendered with a
    different frame.
    """
    import hashlib

    train: list[_PoolEntry] = []
    val: list[_PoolEntry] = []
    val_threshold = int(val_fraction * 2**32)
    for entry in pool.entries:
        h = hashlib.md5(f"{seed}:{entry.label}".encode()).digest()
        bucket = int.from_bytes(h[:4], "big")
        (val if bucket < val_threshold else train).append(entry)
    return NamePool(entries=train), NamePool(entries=val)


def build_tf_dataset(
    pool: NamePool,
    *,
    encode_fn,  # type: ignore[no-untyped-def]  # local arg, see _make_train_step in caller
    batch_size: int,
    shuffle_buffer: int,
    repeat: bool = False,
    augment: bool = False,
    seed: int = 0,
) -> "tf.data.Dataset":  # noqa: F821 — quoted; tf imported lazily to keep this module light
    """Build a `tf.data.Dataset` of `(image_uint8, label_dense_int32)` batches.

    ``encode_fn(label_str) -> list[int]`` maps a label to 1-based class
    indices; passed in so this module doesn't need to know about a specific
    alphabet.

    Set ``repeat=True`` for the *train* pipeline when ``model.fit`` is called
    with ``steps_per_epoch``: Keras consumes ``steps_per_epoch * epochs``
    batches and a finite dataset only yields ``len(pool) / batch_size`` of
    them — without ``.repeat()`` Keras emits a "ran out of data" warning at
    epoch 2 and silently no-ops the remaining epochs. Validation pipelines
    must stay finite (``repeat=False``) so end-of-epoch evaluation iterates
    each val sample once.

    Set ``augment=True`` to apply the photometric/geometric augmentation
    pipeline from :mod:`moxify_ocr.data.name_augment` per sample before it
    reaches the model. Bridges the synthetic-real gap measured on the v1
    shipped model (99.74% real-world CER vs. 0.03% synthetic-val CER). Only
    enable for train; val should stay deterministic.

    The dense label tensor is zero-padded to the max length in each batch
    (zero is the CTC blank, also doubling as padding — the loss strips it).
    Images are passed through as ``uint8``; the CRNN normalizes internally.
    """
    import numpy as np
    import tensorflow as tf

    if not pool.entries:
        raise ValueError("cannot build a tf.data.Dataset from an empty pool")

    paths = np.array([str(e.image_path) for e in pool.entries], dtype=object)
    labels = [encode_fn(e.label) for e in pool.entries]
    label_lens = np.array([len(lbl) for lbl in labels], dtype=np.int32)
    max_len = int(label_lens.max())
    padded = np.zeros((len(labels), max_len), dtype=np.int32)
    for i, lbl in enumerate(labels):
        padded[i, : len(lbl)] = lbl

    def _load(path_t: "tf.Tensor", lbl_t: "tf.Tensor") -> tuple["tf.Tensor", "tf.Tensor"]:
        raw = tf.io.read_file(path_t)
        img = tf.io.decode_png(raw, channels=3)  # uint8
        img.set_shape([48, 512, 3])
        return img, lbl_t

    ds = tf.data.Dataset.from_tensor_slices(
        (tf.constant(paths.astype(str)), tf.constant(padded))
    )
    if shuffle_buffer > 0:
        ds = ds.shuffle(shuffle_buffer, seed=seed, reshuffle_each_iteration=True)
    ds = ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE)
    if repeat:
        ds = ds.repeat()
    if augment:
        # Albumentations runs in numpy / Python — wrap via tf.numpy_function.
        # Per-sample seed derived from the global step to keep determinism in
        # the face of shuffle + repeat.
        from moxify_ocr.data.name_augment import (  # noqa: PLC0415
            apply_name_augmentation,
            build_name_augmentation_pipeline,
        )

        pipeline = build_name_augmentation_pipeline(seed=seed)
        step_counter = tf.Variable(0, trainable=False, dtype=tf.int64)

        def _augment_py(img_np: np.ndarray, step: int) -> np.ndarray:
            return apply_name_augmentation(
                img_np, pipeline, seed=int(step) ^ seed
            )

        def _augment_tf(
            img_t: "tf.Tensor", lbl_t: "tf.Tensor"
        ) -> tuple["tf.Tensor", "tf.Tensor"]:
            step = step_counter.assign_add(1)
            aug = tf.numpy_function(_augment_py, [img_t, step], tf.uint8)
            aug.set_shape([48, 512, 3])
            return aug, lbl_t

        ds = ds.map(_augment_tf, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds
