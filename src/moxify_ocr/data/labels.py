"""Helpers for synthesising OCR labels from Scryfall card metadata.

Encodes domain knowledge about MTG card printing conventions:
- which letter appears in the bottom-left rarity slot, and
- whether the collector number is printed as ``N/TOTAL`` or bare ``N``.
"""

from __future__ import annotations

import re
from datetime import date
from typing import Any

# Scryfall rarity string → printed letter on card.
# "special" and "bonus" are intentionally absent — those cards don't print a
# distinct letter on the bottom-left.
RARITY_LETTER: dict[str, str] = {
    "common": "C",
    "uncommon": "U",
    "rare": "R",
    "mythic": "M",
}

# Scryfall lang code → printed language code on card.
LANG_CODE: dict[str, str] = {
    "en": "EN",
    "de": "DE",
    "fr": "FR",
    "it": "IT",
    "es": "ES",
    "pt": "PT",
    "ja": "JA",
    "ru": "RU",
    "ko": "KO",
    "zhs": "ZH",  # Simplified Chinese prints as ZH
    "zht": "ZH",  # Traditional Chinese also prints as ZH
    "ph": "PH",
    "la": "LA",
}

# Private Use Area codepoint standing for the planeswalker-icon class used on
# "The List" (set code ``plst``) reprints. Never a real text glyph, so it
# can't collide with any printed character in the CTC alphabet.
PLANESWALKER_ICON: str = ""

# First set to print "collector_number/set_total" was Shards of Alara
# (2008-10-03); the rule is "released on or after 2008-01-01".
_SLASH_TOTAL_CUTOFF = date(2008, 1, 1)

# Scryfall set code identifying "The List" reprints, which print a
# planeswalker icon ALONGSIDE the original printed info (not instead of it).
_PLST_SET_CODE = "plst"

# PLST collector numbers encode the original set + number as ``<SET>-<NUM>``
# (e.g. ``"KTK-86"``). The physical card actually shows the original set's
# collector info with a planeswalker icon added — NOT "PLST" or the raw
# ``KTK-86`` string. We salvage both halves for training labels.
_PLST_REPRINT_RE = re.compile(r"^([A-Za-z0-9]{2,6})-(\d+)$")


def _rarity_letter(card: dict[str, Any]) -> str | None:
    """Return the rarity letter printed on a card's bottom-left.

    - ``"L"`` for Basic Lands (regardless of Scryfall's rarity, which is ``"common"``).
    - ``RARITY_LETTER[card["rarity"]]`` for common/uncommon/rare/mythic.
    - ``None`` for ``"special"`` / ``"bonus"`` / unknown rarities.
    """
    type_line = card.get("type_line", "")
    if isinstance(type_line, str) and type_line.startswith("Basic"):
        return "L"
    return RARITY_LETTER.get(card["rarity"])


def _era_has_slash_total(card: dict[str, Any]) -> bool:
    """Return ``True`` if the card's era prints ``"collector_number/set_total"``.

    Rule: cards released on or after 2008-01-01. Uses ``card["released_at"]``
    as a ``YYYY-MM-DD`` string. Cards without ``"released_at"`` return ``False``.
    """
    released_at = card.get("released_at")
    if not released_at:
        return False
    return date.fromisoformat(released_at) >= _SLASH_TOTAL_CUTOFF


def _salvage_plst_fields(card: dict[str, Any]) -> tuple[str, str, bool]:
    """For PLST reprints, recover the printed (original-set) fields.

    PLST cards are stored by Scryfall with ``set = "plst"`` and
    ``collector_number`` like ``"KTK-86"``. But the card physically shows the
    **original** set code and collector number (``KTK`` and ``86``) with a
    small planeswalker icon added — never "PLST" — so that's what the OCR
    model sees and what we should train it to read.

    Returns ``(set_code, collector_number, is_the_list)``. For non-PLST cards
    returns ``(card["set"], card["collector_number"], False)`` unchanged.
    """
    set_code = card["set"]
    number = card["collector_number"]
    if set_code != _PLST_SET_CODE:
        return set_code, number, False
    match = _PLST_REPRINT_RE.match(number)
    if match is None:
        # Unrecognized PLST format — fall back to raw fields; caller may
        # later skip this row if the label can't be encoded cleanly.
        return set_code, number, True
    return match.group(1), match.group(2), True


def make_label(card: dict[str, Any], is_foil: bool) -> str:
    """Synthesize the expected CTC label for a card's bottom-left region.

    Label format::

        "<maybe planeswalker icon> <num>[/<total>] [<rarity letter>]\\n<SET> <foil glyph> <LANG>"

    Rules:
    - For PLST reprints (``card["set"] == "plst"``) we salvage the original
      set + number from ``collector_number`` (see :func:`_salvage_plst_fields`),
      prepend :data:`PLANESWALKER_ICON`, and skip the ``/total`` suffix (PLST
      cards don't print a set total).
    - Otherwise, when :func:`_era_has_slash_total` is true AND ``printed_size``
      is present, print ``"num/total"``; else just ``"num"``.
    - Append the rarity letter from :func:`_rarity_letter` when non-``None``.
      The letter is still printed on PLST cards (it belongs to the original
      printing), so we keep it unless ``rarity`` maps to ``None``.
    - Line 2: ``"<UPPER_SET> <glyph> <LANG>"`` where glyph is ``"★"`` for foil,
      ``"•"`` for non-foil. Language via :data:`LANG_CODE`.

    Unsupported language codes will raise :class:`KeyError` — by design.
    """
    set_code, number, is_the_list = _salvage_plst_fields(card)

    # Line 1: [planeswalker icon] <num>[/<total>] [<rarity letter>]
    line1_parts: list[str] = []
    if is_the_list:
        line1_parts.append(PLANESWALKER_ICON)

    printed_size = card.get("printed_size")
    if not is_the_list and _era_has_slash_total(card) and printed_size is not None:
        line1_parts.append(f"{number}/{printed_size}")
    else:
        line1_parts.append(number)

    letter = _rarity_letter(card)
    if letter is not None:
        line1_parts.append(letter)

    line1 = " ".join(line1_parts)

    # Line 2: "SET <glyph> LANG".
    foil_glyph = "★" if is_foil else "•"
    lang_code = LANG_CODE[card["lang"]]
    line2 = f"{set_code.upper()} {foil_glyph} {lang_code}"

    return f"{line1}\n{line2}"
