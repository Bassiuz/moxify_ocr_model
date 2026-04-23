"""Helpers for synthesising OCR labels from Scryfall card metadata.

Encodes domain knowledge about MTG card printing conventions:
- which letter appears in the bottom-left rarity slot, and
- whether the collector number is printed as ``N/TOTAL`` or bare ``N``.
"""

from __future__ import annotations

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
# planeswalker icon in place of the rarity letter.
_PLST_SET_CODE = "plst"


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


def make_label(card: dict[str, Any], is_foil: bool) -> str:
    """Synthesize the expected CTC label for a card's bottom-left region.

    Label format::

        "<maybe planeswalker icon> <num>[/<total>] [<rarity letter>]\\n<SET> <foil glyph> <LANG>"

    Rules:
    - If ``card["set"] == "plst"``, prepend :data:`PLANESWALKER_ICON` to the
      collector number, skip the rarity letter, and skip the ``/total`` suffix
      even when ``printed_size`` is present (The List uses a distinct format).
    - Otherwise, when :func:`_era_has_slash_total` is true AND ``printed_size``
      is present, print ``"num/total"``; else just ``"num"``.
    - Append the rarity letter from :func:`_rarity_letter` when non-``None``
      (and not a PLST card).
    - Line 2: ``"<UPPER_SET> <glyph> <LANG>"`` where glyph is ``"★"`` for foil,
      ``"•"`` for non-foil. Language via :data:`LANG_CODE`.

    Unsupported language codes will raise :class:`KeyError` — by design.
    """
    set_code = card["set"]
    is_plst = set_code == _PLST_SET_CODE

    # Line 1: collector number, optional /total, optional rarity letter.
    line1_parts: list[str] = []
    if is_plst:
        line1_parts.append(PLANESWALKER_ICON)

    number = card["collector_number"]
    printed_size = card.get("printed_size")
    if not is_plst and _era_has_slash_total(card) and printed_size is not None:
        line1_parts.append(f"{number}/{printed_size}")
    else:
        line1_parts.append(number)

    if not is_plst:
        letter = _rarity_letter(card)
        if letter is not None:
            line1_parts.append(letter)

    line1 = " ".join(line1_parts)

    # Line 2: "SET <glyph> LANG".
    foil_glyph = "★" if is_foil else "•"
    lang_code = LANG_CODE[card["lang"]]
    line2 = f"{set_code.upper()} {foil_glyph} {lang_code}"

    return f"{line1}\n{line2}"
