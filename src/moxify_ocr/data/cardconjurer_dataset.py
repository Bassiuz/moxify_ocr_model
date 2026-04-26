"""Pre-rendered CardConjurer pool reader for the dataset pipeline."""

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
class CardConjurerPool:
    """In-memory index of (image_path, label) pairs from a pre-rendered pool."""

    entries: list[_PoolEntry]

    @classmethod
    def load(cls, root: Path) -> CardConjurerPool:
        """Read ``{root}/labels.jsonl`` and resolve relative image paths.

        ``root`` is the directory the renderer wrote to — contains
        ``labels.jsonl`` and an ``images/`` subdir.
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


def sample_from_pool(pool: CardConjurerPool, *, seed: int) -> tuple[np.ndarray, str]:
    """Draw one (image, label) pair deterministically from the pool."""
    if not pool.entries:
        raise ValueError("CardConjurerPool is empty — nothing to sample from")
    rng = random.Random(seed)
    entry = rng.choice(pool.entries)
    img = np.asarray(Image.open(entry.image_path).convert("RGB"), dtype=np.uint8)
    return img, entry.label
