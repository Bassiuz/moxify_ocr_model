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

# Realistic distributions tuned to match what the manifest already contains.
_LANG_WEIGHTS = {
    "EN": 0.45, "DE": 0.10, "FR": 0.10, "ES": 0.08, "IT": 0.07,
    "PT": 0.05, "JA": 0.07, "KO": 0.04, "ZH": 0.03, "RU": 0.01,
}
_RARITY_WEIGHTS = {"C": 0.50, "U": 0.30, "R": 0.15, "M": 0.04, "S": 0.005, "P": 0.005}

# Length distribution for randomly-generated set codes. Real Magic set codes
# are 3 letters in the vast majority of cases; a few historical sets have 2
# (LEA / LEB) or 4 (10E, MH1, MH2, etc — number-led but treated similarly).
# We weight 3-letter heaviest to match real-world distribution.
_SET_CODE_LENGTH_WEIGHTS = {2: 0.05, 3: 0.85, 4: 0.10}


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
    """Generate a random 2-4 letter uppercase set code.

    NOT drawn from a fixed list. The v3 model collapsed to ~0.7% set-code
    accuracy on held-out test sets because a closed vocabulary teaches the
    model to memorize set strings rather than read characters. Random codes
    force character-level OCR, which generalizes to any new set.
    """
    length = _weighted_choice_int(rng, _SET_CODE_LENGTH_WEIGHTS)
    return "".join(rng.choices(string.ascii_uppercase, k=length))


def _weighted_choice_int(rng: random.Random, weights: dict[int, float]) -> int:
    return rng.choices(list(weights.keys()), weights=list(weights.values()), k=1)[0]


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
