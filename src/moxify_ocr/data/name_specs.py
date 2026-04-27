"""Random name-spec generator for the card-name OCR renderer.

Produces a stream of :class:`NameSpec` objects that the renderer turns into
synthetic name-region crops. Each spec is deterministic from a seed so the
generated training pool is reproducible.

A spec captures *what to render* (name string, frame style, color, font
jitter, foil flag) but not *how to render it* — that's the renderer's job.
The renderer holds a style table mapping ``spec.style`` to concrete frame
asset paths, name-bbox coordinates, font choice, and rotate-after flag.

Frame style coverage mirrors the layouts called out in
``docs/plans/2026-04-27-card-name-ocr-design.md``:

- Modern (2015 frame) and its treatments: regular, legendary, extended-art,
  borderless, and a sample of named showcase packs (storybook, kaldheim,
  tarkir, neo).
- Pre-modern (1997, 2003) and the original 1993 frame.
- Future Sight (2007 sci-fi frames).
- Saga (vertical reading frame; name still horizontal at top).
- Battle and Split (rotated layouts; the renderer handles the 90° rotate).
- Adventure (primary name only).
- DFC (transform / modal_dfc) front and back.
"""

from __future__ import annotations

import json
import random
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

from moxify_ocr.data.name_alphabet import NAME_ALPHABET

# Layouts whose primary card-name region we want to train on. See the design
# doc for the rationale on each inclusion / exclusion.
ALLOWED_LAYOUTS: frozenset[str] = frozenset(
    {
        "normal",
        "transform",
        "modal_dfc",
        "adventure",
        "saga",
        "split",
        "mutate",
        "class",
        "leveler",
        "prototype",
        "meld",
        "flip",
        "aftermath",
        "battle",
    }
)

# Latin-script languages only for v1; CJK / Cyrillic deferred.
ALLOWED_LANGS: frozenset[str] = frozenset(
    {"en", "es", "fr", "it", "de", "pt", "la", "ph"}
)

_NAME_ALPHABET_SET: frozenset[str] = frozenset(NAME_ALPHABET)

# Style identifiers — keep stable; the renderer's STYLE_TABLE keys must
# match exactly. Any change here is a renderer migration.
STYLES: tuple[str, ...] = (
    # Modern (2015) — by far the dominant prior in real card distribution.
    "modern_regular",
    "modern_legendary",
    "modern_extended",
    "modern_borderless",
    "modern_showcase_storybook",
    "modern_showcase_kaldheim",
    "modern_showcase_tarkir",
    "modern_showcase_neo",
    # Pre-modern eras.
    "premodern_2003",
    "premodern_1997",
    "old_1993",
    "old_8th",
    # Future Sight.
    "future_sight",
    # Special layouts.
    "saga",
    "battle",
    "split_left",
    "split_right",
    "adventure",
    "transform_front",
    "transform_back",
    "modal_dfc_front",
    "modal_dfc_back",
    "planeswalker",
    "aftermath",
    "flip",
)

# Sampling weights tuned to roughly match real-print distribution. Modern
# eras dominate because the production OCR sees mostly modern cards; we
# still want every other layout to be sampled often enough that the model
# generalizes to it.
STYLE_WEIGHTS: dict[str, float] = {
    "modern_regular": 0.25,
    "modern_legendary": 0.10,
    "modern_extended": 0.05,
    "modern_borderless": 0.05,
    "modern_showcase_storybook": 0.03,
    "modern_showcase_kaldheim": 0.02,
    "modern_showcase_tarkir": 0.02,
    "modern_showcase_neo": 0.02,
    "premodern_2003": 0.10,
    "premodern_1997": 0.07,
    "old_1993": 0.04,
    "old_8th": 0.02,
    "future_sight": 0.01,
    "saga": 0.02,
    "battle": 0.02,
    "split_left": 0.02,
    "split_right": 0.02,
    "adventure": 0.02,
    "transform_front": 0.03,
    "transform_back": 0.02,
    "modal_dfc_front": 0.02,
    "modal_dfc_back": 0.01,
    "planeswalker": 0.02,
    "aftermath": 0.005,
    "flip": 0.005,
}

# Frame color codes the renderer understands. WUBRG = the five colors;
# M = multicolor (gold), A = artifact (silver/brown), L = land.
_FRAME_COLORS: tuple[str, ...] = ("W", "U", "B", "R", "G", "M", "A", "L")
_FRAME_COLOR_WEIGHTS: tuple[float, ...] = (
    0.13, 0.13, 0.13, 0.13, 0.13,  # WUBRG roughly equal
    0.15, 0.10, 0.10,              # M / A / L slightly under each color
)

# Mana symbol identifiers. Lowercase to match the m21-pack filenames
# (``m21<id>.png``). The renderer treats these as opaque strings.
_MANA_COLORED: tuple[str, ...] = ("w", "u", "b", "r", "g")
_MANA_GENERIC: tuple[str, ...] = (
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "x"
)
# Distribution roughly matches modern card costs (mode = 1-3).
_MANA_GENERIC_WEIGHTS: tuple[float, ...] = (
    0.04, 0.18, 0.22, 0.18, 0.12, 0.08, 0.05, 0.03, 0.02, 0.02, 0.06
)
_MANA_COLORED_COUNT_WEIGHTS: tuple[float, ...] = (
    0.40, 0.30, 0.15, 0.10, 0.05
)  # P(1 / 2 / 3 / 4 / 5 colored symbols)


@dataclass(frozen=True)
class NameSpec:
    """One name-region crop to render.

    Attributes:
        name: The actual text to render. Drawn from a real Scryfall name pool;
            the renderer never invents names.
        style: Style identifier from :data:`STYLES`. Drives frame asset choice,
            font, name-bbox, and rotate-after behavior in the renderer.
        frame_color: One letter from "WUBRGMAL" picking a color variant of the
            frame asset. Real cards have a frame color even if the rendered
            name strip doesn't always show it (e.g., legendary filigree
            overlays the color).
        font_size_jitter: Small per-sample multiplier on the base font size,
            sampled in roughly [0.92, 1.08]. Keeps the model robust to small
            sizing variation introduced by camera distance / scan resolution.
        foil: Whether to apply a holographic-shimmer overlay during rendering.
            Augmentation handles most of this variance; the spec just flags
            it.
    """

    name: str
    style: str
    frame_color: str
    font_size_jitter: float
    foil: bool
    # Tuple of mana symbol ids (e.g. ``("3", "r", "r")`` → "{3}{R}{R}"). Empty
    # for cards without a printed cost (e.g. lands). The renderer paints
    # these on the right edge of the name region; the OCR target is still
    # the name only — the symbols are noise the model must learn to ignore,
    # mimicking what a real upstream cropper feeds at inference time.
    mana_cost: tuple[str, ...] = ()


def _weighted_choice(
    rng: random.Random, options: Sequence[str], weights: Sequence[float]
) -> str:
    return rng.choices(options, weights=weights, k=1)[0]


def _random_mana_cost(rng: random.Random) -> tuple[str, ...]:
    """Sample a plausible mana cost.

    ~15% of cards have no printed cost (lands). Otherwise the cost is
    optional generic (0/1/2/.../9/X) followed by 1-5 colored symbols, with
    distributions tuned to look like modern playables — the long-tail of
    rare 7-mana costs still appears, just rarely.
    """
    if rng.random() < 0.15:
        return ()
    cost: list[str] = []
    if rng.random() < 0.60:
        cost.append(
            rng.choices(_MANA_GENERIC, weights=_MANA_GENERIC_WEIGHTS, k=1)[0]
        )
    n_colored = rng.choices(
        [1, 2, 3, 4, 5], weights=_MANA_COLORED_COUNT_WEIGHTS, k=1
    )[0]
    for _ in range(n_colored):
        cost.append(rng.choice(_MANA_COLORED))
    return tuple(cost)


def make_spec(*, names: Sequence[str], seed: int) -> NameSpec:
    """Build one deterministic :class:`NameSpec` from ``seed``.

    ``names`` is the pool of card-name strings to draw from — usually the
    output of :func:`load_card_names` but tests pass a small fixed list.
    """
    if not names:
        raise ValueError("names pool is empty")
    rng = random.Random(seed)
    return NameSpec(
        name=rng.choice(names),
        style=_weighted_choice(
            rng,
            list(STYLE_WEIGHTS.keys()),
            list(STYLE_WEIGHTS.values()),
        ),
        frame_color=_weighted_choice(rng, _FRAME_COLORS, _FRAME_COLOR_WEIGHTS),
        font_size_jitter=rng.uniform(0.92, 1.08),
        foil=rng.random() < 0.20,
        mana_cost=_random_mana_cost(rng),
    )


def generate_specs(
    *, names: Sequence[str], n: int, seed: int = 0
) -> Iterator[NameSpec]:
    """Yield ``n`` specs derived deterministically from ``seed``."""
    for i in range(n):
        yield make_spec(names=names, seed=seed * 1_000_003 + i)


def load_card_names(scryfall_path: Path) -> list[str]:
    """Read Scryfall's ``default-cards.json`` and return name strings the
    renderer can use.

    Filters:
        - ``layout`` must be in :data:`ALLOWED_LAYOUTS`.
        - ``lang`` must be in :data:`ALLOWED_LANGS`.
        - Every character in the name must be in :data:`NAME_ALPHABET` —
          names with one stray glyph the model can't encode are dropped
          (these are typically corner cases like the U+A689 lookalike colon
          or a stray ``®``).

    Multi-face card names (e.g. split cards stored as ``"Foo // Bar"``) are
    split into their individual face names — the renderer already targets
    one face per crop.
    """
    with scryfall_path.open() as f:
        cards = json.load(f)
    names: list[str] = []
    seen: set[str] = set()
    for card in cards:
        if card.get("layout") not in ALLOWED_LAYOUTS:
            continue
        if card.get("lang") not in ALLOWED_LANGS:
            continue
        raw = card.get("printed_name") or card.get("name") or ""
        if not raw:
            continue
        for face in raw.split(" // "):
            face = face.strip()
            if not face or face in seen:
                continue
            if not all(ch in _NAME_ALPHABET_SET for ch in face):
                continue
            seen.add(face)
            names.append(face)
    return names
