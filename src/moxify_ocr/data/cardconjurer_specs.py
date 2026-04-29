"""Random card-spec generator for the CardConjurer renderer.

Produces a stream of :class:`CardSpec` objects with realistic distributions
across set code, language, rarity, collector-number format, and foil
treatment. Each call seeded for determinism so the generated training pool
is reproducible.

The spec captures only what the renderer drives — not what the OCR labels
read. The corresponding label is derived deterministically from the spec by
the renderer (see ``scripts/render_cardconjurer_pool.py``).
"""

from __future__ import annotations

import hashlib
import random
import string
from collections.abc import Iterator
from dataclasses import dataclass

from moxify_ocr.data._real_set_codes import (
    CHAR_FREQ,
    REAL_LENGTH_FREQ,
    REAL_SET_CODES,
)

# Realistic distributions tuned to match what the manifest already contains.
_LANG_WEIGHTS = {
    "EN": 0.45, "DE": 0.10, "FR": 0.10, "ES": 0.08, "IT": 0.07,
    "PT": 0.05, "JA": 0.07, "KO": 0.04, "ZH": 0.03, "RU": 0.01,
}
_RARITY_WEIGHTS = {"C": 0.50, "U": 0.30, "R": 0.15, "M": 0.04, "S": 0.005, "P": 0.005}

# Probability of drawing a real Scryfall code (vs frequency-weighted random).
# Hybrid sampling balances: 70% real → train distribution matches production
# letter/digit/length patterns; 30% random → forces character-level OCR for
# generalization to future sets Wizards hasn't shipped yet.
_REAL_SET_CODE_PROB = 0.70

# Pre-compute the random-set-code parameters so we don't rebuild them per spec.
_RANDOM_LENGTH_KEYS = list(REAL_LENGTH_FREQ.keys())
_RANDOM_LENGTH_WEIGHTS = list(REAL_LENGTH_FREQ.values())
_RANDOM_CHAR_KEYS = list(CHAR_FREQ.keys())
_RANDOM_CHAR_WEIGHTS = list(CHAR_FREQ.values())


@dataclass(frozen=True)
class CardSpec:
    """One card to render.

    ``foil`` controls whether the pip between set and lang is ★ or •. All
    other fields are passed verbatim to CardConjurer's #info-* DOM inputs.
    """

    info_set: str
    info_language: str
    info_rarity: str
    info_number: str
    foil: bool


def _weighted_choice(rng: random.Random, weights: dict[str, float]) -> str:
    return rng.choices(list(weights.keys()), weights=list(weights.values()), k=1)[0]


def _gen_set_code(rng: random.Random) -> str:
    """Generate a hybrid set code: 70% real Scryfall code, 30% weighted-random.

    The v3_v1 model used a closed list of ~104 codes and collapsed to 0.7%
    test set_code_accuracy because it memorized strings. v3_v2 used pure
    uniform-random alpha codes and reached only 23% — the visual letter
    distribution didn't match real cards (uniform vs reality's heavy P/T/M),
    none of the codes were alphanumeric (vs real's 33% with-digit rate),
    and length was ~85% 3-char (vs reality's 43% 3-char / 54% 4-char).

    This hybrid fixes all three: 70% of synthetic codes are pulled directly
    from REAL_SET_CODES (1031 entries; matches reality exactly in length,
    character, and alphanumeric distribution); 30% are random codes whose
    length and characters are weighted by ``REAL_LENGTH_FREQ`` and
    ``CHAR_FREQ`` (so even random codes look plausibly Magic-shaped).
    """
    if rng.random() < _REAL_SET_CODE_PROB:
        return rng.choice(REAL_SET_CODES)
    length = rng.choices(_RANDOM_LENGTH_KEYS, weights=_RANDOM_LENGTH_WEIGHTS, k=1)[0]
    chars = rng.choices(_RANDOM_CHAR_KEYS, weights=_RANDOM_CHAR_WEIGHTS, k=length)
    return "".join(chars)


def _gen_collector_number(rng: random.Random) -> str:
    """Numeric NNN/NNN, bare NNN, X-prefixed, hyphen-separated, alpha-suffixed.

    The rate-tuned distribution covers the formats that show up in real
    Scryfall data. v3 model failures on held-out test cards traced back to
    the synthetic data missing hyphenated and alpha-suffixed numbers.
    """
    r = rng.random()
    if r < 0.04:
        # X-prefixed (promo): "X12", "X007"
        return f"X{rng.randint(1, 999):0{rng.choice([2, 3])}d}"
    if r < 0.07:
        # Hyphen-separated: "INV-23", "PMTG-007", "2025-1"
        prefix_kind = rng.random()
        if prefix_kind < 0.5:
            # Letter prefix: 2-4 uppercase letters then -NN
            prefix = "".join(rng.choices(string.ascii_uppercase, k=rng.choice([2, 3, 4])))
            return f"{prefix}-{rng.randint(1, 99):02d}"
        # Year prefix: 4-digit year then -N
        year = rng.choice([2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025])
        return f"{year}-{rng.randint(1, 9)}"
    if r < 0.10:
        # Alpha-suffixed variant: "042A", "118Z", "276B"
        digits = rng.randint(1, 999)
        suffix = rng.choice(string.ascii_uppercase)
        return f"{digits:0{rng.choice([2, 3])}d}{suffix}"
    if r < 0.15:
        # Bare collector number: "042"
        return f"{rng.randint(1, 999):03d}"
    # Numeric NNN/NNN — by far the common case.
    total = rng.choice([100, 150, 200, 250, 277, 280, 302, 350])
    num = rng.randint(1, total)
    return f"{num:03d}/{total:03d}"


def make_spec(seed: int) -> CardSpec:
    """Build one deterministic CardSpec from a seed."""
    rng = random.Random(seed)
    return CardSpec(
        info_set=_gen_set_code(rng),
        info_language=_weighted_choice(rng, _LANG_WEIGHTS),
        info_rarity=_weighted_choice(rng, _RARITY_WEIGHTS),
        info_number=_gen_collector_number(rng),
        foil=rng.random() < 0.20,  # ~20% foil — realistic for booster mix
    )


def _derive_seed(base: int, i: int) -> int:
    """Deterministically combine a base seed and an index into a per-spec seed.

    Uses SHA-256 so the result is stable across processes and Python versions
    (built-in ``hash()`` is salted). Returns a 32-bit unsigned int — plenty of
    range for ``random.Random``.
    """
    digest = hashlib.sha256(f"{base}:{i}".encode()).digest()
    return int.from_bytes(digest[:4], "big")


def generate_specs(n: int, seed: int = 0) -> Iterator[CardSpec]:
    """Yield ``n`` specs derived deterministically from a base seed."""
    for i in range(n):
        yield make_spec(seed=_derive_seed(seed, i))
