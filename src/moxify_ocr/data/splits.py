"""Deterministic set-aware train/val/test split.

We split by Scryfall ``set_code`` — not by card — so that the model is evaluated
on sets it has never seen, which is the realistic failure mode (a new set ships
and the OCR head must still read it). Hashing uses SHA-256 so the assignment is
stable across Python versions (unlike ``hash()``, which is salted).

See design doc §5.3.
"""

from __future__ import annotations

import hashlib

_Split = str  # Literal["train", "val", "test"] — kept as plain str for return flexibility.

_TRAIN_CUTOFF: int = 85  # [0, 85) → train
_VAL_CUTOFF: int = 95  # [85, 95) → val; [95, 100) → test


def assign_split(
    set_code: str,
    *,
    seed: int = 0,
    holdout_sets: frozenset[str] = frozenset(),
) -> _Split:
    """Deterministic set-aware split. Returns ``"train"``, ``"val"``, or ``"test"``.

    - If ``set_code`` (lowercased) is in ``holdout_sets`` → always ``"test"``.
    - Else SHA-256 hash of ``f"{seed}:{set_code.lower()}"``, mod 100:
      ``< 85`` → ``"train"``, ``< 95`` → ``"val"``, else ``"test"``.

    ``holdout_sets`` membership is checked case-insensitively by lowercasing
    ``set_code`` before comparison; callers are expected to pass already-lowercase
    codes in the frozenset.
    """
    key = set_code.lower()
    if key in holdout_sets:
        return "test"

    digest = hashlib.sha256(f"{seed}:{key}".encode()).digest()
    bucket = int.from_bytes(digest[:8], "big") % 100
    if bucket < _TRAIN_CUTOFF:
        return "train"
    if bucket < _VAL_CUTOFF:
        return "val"
    return "test"
