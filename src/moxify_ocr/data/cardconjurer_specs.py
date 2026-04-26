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

import random
from collections.abc import Iterator
from dataclasses import dataclass

# Realistic distributions tuned to match what the manifest already contains.
_LANG_WEIGHTS = {
    "EN": 0.45, "DE": 0.10, "FR": 0.10, "ES": 0.08, "IT": 0.07,
    "PT": 0.05, "JA": 0.07, "KO": 0.04, "ZH": 0.03, "RU": 0.01,
}
_RARITY_WEIGHTS = {"C": 0.50, "U": 0.30, "R": 0.15, "M": 0.04, "S": 0.005, "P": 0.005}
# A small pool of plausible 3-4 letter set codes — enough variety without
# pretending to be exhaustive (the OCR doesn't need to know real set semantics).
# Note: ``10E`` appears twice on purpose — duplicates double the draw weight in
# ``random.choice``.
_SET_CODES = [
    "LEA", "LEB", "2ED", "3ED", "4ED", "5ED", "6ED", "7ED", "8ED", "9ED", "10E",
    "MIR", "VIS", "WTH", "TMP", "STH", "EXO", "USG", "ULG", "UDS", "MMQ", "NEM", "PCY",
    "ICE", "ALL", "CSP", "TSP", "PLC", "FUT", "10E", "LRW", "MOR", "SHM", "EVE", "ALA",
    "CON", "ARB", "ZEN", "WWK", "ROE", "M11", "SOM", "MBS", "NPH", "CMD", "M12", "ISD",
    "DKA", "AVR", "M13", "RTR", "GTC", "DGM", "M14", "THS", "BNG", "JOU", "M15", "KTK",
    "FRF", "DTK", "ORI", "BFZ", "OGW", "SOI", "EMN", "KLD", "AER", "AKH", "HOU", "XLN",
    "RIX", "DOM", "M19", "GRN", "RNA", "WAR", "MH1", "M20", "ELD", "THB", "IKO", "M21",
    "ZNR", "KHM", "STX", "MH2", "AFR", "MID", "VOW", "NEO", "SNC", "HBG", "DMU", "BRO",
    "ONE", "MOM", "MAT", "WOE", "LCI", "MKM", "OTJ", "BIG", "MH3",
]


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


def _gen_collector_number(rng: random.Random) -> str:
    """Most numeric NNN/NNN, some bare NNN, a few X-prefixed, occasional pre-modern."""
    r = rng.random()
    if r < 0.05:
        # X-prefixed (promo): "X12", "X007"
        return f"X{rng.randint(1, 999):0{rng.choice([2, 3])}d}"
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
        info_set=rng.choice(_SET_CODES),
        info_language=_weighted_choice(rng, _LANG_WEIGHTS),
        info_rarity=_weighted_choice(rng, _RARITY_WEIGHTS),
        info_number=_gen_collector_number(rng),
        foil=rng.random() < 0.20,  # ~20% foil — realistic for booster mix
    )


def generate_specs(n: int, seed: int = 0) -> Iterator[CardSpec]:
    """Yield ``n`` specs derived deterministically from a base seed."""
    for i in range(n):
        yield make_spec(seed=seed * 1_000_003 + i)
