"""tf.data.Dataset builder for CTC OCR training.

Wires together manifest reading, set-aware splits, label synthesis, bottom-region
cropping, and optional augmentation into a single :class:`tf.data.Dataset`.

Foil overlay integration (Tasks 14–15 in the plan) is deferred to v2. For v0 we
train on non-foil labels only.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf
from PIL import Image

from moxify_ocr.data.augment import apply_augmentation, build_augmentation_pipeline
from moxify_ocr.data.crop import crop_bottom_region
from moxify_ocr.data.labels import PLANESWALKER_ICON, make_label
from moxify_ocr.data.manifest import ManifestEntry, read_manifest
from moxify_ocr.data.splits import assign_split

# CTC alphabet: 44 characters. Index 0 is reserved for the CTC blank; the
# characters below occupy indices 1..44.
ALPHABET: str = (
    "0123456789"  # digits (10)
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"  # uppercase (26)
    " /-."  # punctuation + space (4)
    "\n"  # line separator (1)
    "•★"  # foil glyphs (2)
    + PLANESWALKER_ICON  # planeswalker icon (1)
)
assert len(ALPHABET) == 44, f"alphabet must be 44 chars, got {len(ALPHABET)}"

BLANK_INDEX: int = 0

# Char → 1-based index mapping (0 is reserved for the CTC blank).
_CHAR_TO_INDEX: dict[str, int] = {ch: i + 1 for i, ch in enumerate(ALPHABET)}
_INDEX_TO_CHAR: dict[int, str] = {i + 1: ch for i, ch in enumerate(ALPHABET)}


# Layouts whose bottom region follows the standard CTC label format. Other
# layouts (art_series, token, scheme, plane, phenomenon, emblem, ...) either
# lack a printed collector number region or print it in a format the CRNN
# cannot reliably read.
TRAINABLE_LAYOUTS: frozenset[str] = frozenset(
    {
        "normal",
        "split",
        "transform",
        "modal_dfc",
        "flip",
        "leveler",
        "class",
        "saga",
        "adventure",
        "prototype",
    }
)


def is_trainable(entry: ManifestEntry, *, min_release: str = "2008-01-01") -> bool:
    """Filter cards that have reliable black bottom identifiers.

    - Exclude cards released before ``min_release`` (older cards often have no
      collector number printed, or printed in colors hard for CRNN to learn).
    - Exclude non-standard layouts (``art_series``, ``token``, ``scheme``,
      ``plane``, ``phenomenon``, ...) that don't follow the bottom-region
      format.
    """
    if entry.released_at == "" or entry.released_at < min_release:
        return False
    return entry.layout in TRAINABLE_LAYOUTS


@dataclass(frozen=True)
class DatasetConfig:
    """Configuration for :func:`build_dataset`."""

    manifest_path: Path
    images_root: Path
    split: str  # "train" | "val" | "test"
    target_size: tuple[int, int] = (256, 48)  # (W, H) — PIL convention
    batch_size: int = 192
    shuffle_buffer: int = 1024
    augment: bool = True
    seed: int = 0
    holdout_sets: frozenset[str] = frozenset()
    min_release: str = "2008-01-01"


def encode_label(text: str, alphabet: str = ALPHABET) -> list[int]:
    """Encode a string as a list of class indices (1-based; 0 = CTC blank).

    Raises :class:`ValueError` if ``text`` contains any character not in
    ``alphabet`` — we want loud failures on bad data rather than silent drops.
    """
    if alphabet is ALPHABET:
        mapping = _CHAR_TO_INDEX
    else:
        mapping = {ch: i + 1 for i, ch in enumerate(alphabet)}
    indices: list[int] = []
    for ch in text:
        idx = mapping.get(ch)
        if idx is None:
            raise ValueError(
                f"character {ch!r} (U+{ord(ch):04X}) not in alphabet"
            )
        indices.append(idx)
    return indices


def decode_label(indices: list[int], alphabet: str = ALPHABET) -> str:
    """Decode class indices back to a string. Skips :data:`BLANK_INDEX`.

    Raises :class:`ValueError` on any out-of-range non-blank index.
    """
    if alphabet is ALPHABET:
        reverse = _INDEX_TO_CHAR
    else:
        reverse = {i + 1: ch for i, ch in enumerate(alphabet)}
    chars: list[str] = []
    for idx in indices:
        if idx == BLANK_INDEX:
            continue
        ch = reverse.get(idx)
        if ch is None:
            raise ValueError(f"index {idx} not in alphabet range")
        chars.append(ch)
    return "".join(chars)


def _entry_to_card(entry: ManifestEntry) -> dict[str, Any]:
    """Convert a manifest row to the card dict shape expected by :func:`make_label`."""
    card: dict[str, Any] = {
        "collector_number": entry.collector_number,
        "set": entry.set_code,
        "lang": entry.lang,
        "rarity": entry.rarity,
        "type_line": entry.type_line,
        "released_at": entry.released_at,
    }
    if entry.printed_size is not None:
        card["printed_size"] = entry.printed_size
    return card


def _load_and_crop(
    image_path: Path,
    target_size: tuple[int, int],
) -> np.ndarray:
    """Load a card image from disk, crop the bottom region, return uint8 HWC RGB."""
    with Image.open(image_path) as raw:
        image = raw.convert("RGB")
        cropped = crop_bottom_region(image, target_size=target_size)
        return np.asarray(cropped, dtype=np.uint8)


def _filter_entries(
    manifest_path: Path,
    split: str,
    seed: int,
    holdout_sets: frozenset[str],
    min_release: str,
) -> list[ManifestEntry]:
    """Read the manifest, keep only trainable rows that land in ``split``."""
    return [
        entry
        for entry in read_manifest(manifest_path)
        if is_trainable(entry, min_release=min_release)
        and assign_split(entry.set_code, seed=seed, holdout_sets=holdout_sets) == split
    ]


def _make_generator(
    entries: list[ManifestEntry],
    images_root: Path,
    target_size: tuple[int, int],
    augment: bool,
    seed: int,
) -> Callable[[], Iterator[tuple[np.ndarray, np.ndarray, np.int32]]]:
    """Build a Python generator that yields (image, label_ids, label_length).

    The generator owns a single augmentation pipeline and re-seeds it per row
    so each (entry, epoch) combination is reproducible.
    """
    pipeline = build_augmentation_pipeline(seed=seed) if augment else None

    def gen() -> Iterator[tuple[np.ndarray, np.ndarray, np.int32]]:
        for idx, entry in enumerate(entries):
            card = _entry_to_card(entry)
            try:
                label = make_label(card, is_foil=False)
                label_ids = np.asarray(encode_label(label), dtype=np.int32)
            except (ValueError, KeyError):
                # Skip rows we can't encode: unsupported languages (e.g. 'qya'
                # Quenya), promo collector-number suffixes like '268p', or
                # other characters outside the uppercase CTC alphabet.
                continue
            image_path = images_root / entry.image_path
            try:
                image = _load_and_crop(image_path, target_size)
            except FileNotFoundError:
                continue
            if pipeline is not None:
                image = apply_augmentation(image, pipeline, seed=seed * 1_000_003 + idx)
            yield image, label_ids, np.int32(len(label_ids))

    return gen


def build_dataset(config: DatasetConfig) -> tf.data.Dataset:
    """Build a :class:`tf.data.Dataset` of OCR training samples for ``config.split``.

    Each batch element is ``(image, label_ids, label_length)`` where
    ``image`` is ``uint8[B, H, W, 3]``, ``label_ids`` is ``int32[B, L]``
    (zero-padded to the batch's max label length), and ``label_length`` is
    ``int32[B]`` (true length of each row's label).

    Pipeline:
      1. Read + filter manifest by split.
      2. For each entry, synthesize a non-foil label via :func:`make_label`.
      3. Load image, crop bottom region via :func:`crop_bottom_region`.
      4. Optionally augment (train only in practice; gated by ``config.augment``).
      5. Encode label to int ids.
      6. Shuffle on train, batch with zero-padding, prefetch.
    """
    entries = _filter_entries(
        config.manifest_path,
        config.split,
        config.seed,
        config.holdout_sets,
        config.min_release,
    )
    target_w, target_h = config.target_size
    gen = _make_generator(
        entries,
        config.images_root,
        config.target_size,
        config.augment,
        config.seed,
    )

    output_signature = (
        tf.TensorSpec(shape=(target_h, target_w, 3), dtype=tf.uint8),
        tf.TensorSpec(shape=(None,), dtype=tf.int32),
        tf.TensorSpec(shape=(), dtype=tf.int32),
    )

    dataset = tf.data.Dataset.from_generator(gen, output_signature=output_signature)

    if config.split == "train":
        # Train generator must yield indefinitely — Keras's model.fit with
        # steps_per_epoch=N expects N*epochs total batches across the whole
        # fit() call. Without .repeat() the generator exhausts after one pass
        # and epochs 2+ train on no data.
        dataset = dataset.repeat()

    if config.split == "train" and config.shuffle_buffer > 0:
        dataset = dataset.shuffle(
            buffer_size=config.shuffle_buffer,
            seed=config.seed,
            reshuffle_each_iteration=True,
        )

    dataset = dataset.padded_batch(
        batch_size=config.batch_size,
        padded_shapes=(
            (target_h, target_w, 3),
            (None,),
            (),
        ),
        padding_values=(
            tf.constant(0, dtype=tf.uint8),
            tf.constant(BLANK_INDEX, dtype=tf.int32),
            tf.constant(0, dtype=tf.int32),
        ),
    )
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset
