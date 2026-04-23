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

# First set to print "collector_number/set_total" was Shards of Alara
# (2008-10-03); the rule is "released on or after 2008-01-01".
_SLASH_TOTAL_CUTOFF = date(2008, 1, 1)


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
