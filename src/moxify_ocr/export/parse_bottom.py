"""Deterministic parser for CTC-decoded bottom-region strings.

Inverse of :func:`moxify_ocr.data.labels.make_label`: takes the string output
by the trained OCR model and turns it into structured fields for downstream
card matching. Tolerant by design — never raises on garbled input, just
returns ``None`` for fields it can't resolve.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from moxify_ocr.data.labels import PLANESWALKER_ICON

# Compiled once at module load — keeps the hot parse loop tight.
_SLASH_TOTAL_RE = re.compile(r"^(\d+)/(\d+)$")
_DIGITS_RE = re.compile(r"^\d+$")
# Set codes can include digits (e.g. "M13", "MB1"); languages + rarity letters
# are alpha-only, but we handle the narrower shapes after this broad match.
_UPPER_ALNUM_RE = re.compile(r"^[A-Z0-9]+$")
_UPPER_ALPHA_RE = re.compile(r"^[A-Z]+$")

# Single-letter rarity glyphs printed on MTG cards.
_RARITY_LETTERS: frozenset[str] = frozenset({"C", "U", "R", "M", "L"})

# Glyphs used to indicate foil/non-foil on the bottom-region label.
_FOIL_GLYPH = "★"
_NONFOIL_GLYPH = "•"


@dataclass(frozen=True, slots=True)
class BottomRegionResult:
    """Structured result of parsing a CTC-decoded bottom-region string."""

    collector_number: str | None
    set_total: int | None
    rarity: str | None  # 'C' | 'U' | 'R' | 'M' | 'L' | None
    set_code: str | None
    language: str | None
    foil_detected: bool | None  # True for ★, False for •, None if neither glyph seen
    on_the_list_detected: bool  # True iff PLANESWALKER_ICON seen
    raw: str  # input string, verbatim (for debugging)


def parse_bottom(
    raw: str,
    *,
    known_set_codes: set[str],
    known_languages: set[str],
) -> BottomRegionResult:
    """Parse a CTC-decoded bottom-region string into structured fields.

    Line-agnostic: handles both ``"A\\nB"`` and ``"A B"`` single-line inputs.
    Uses vocab matching for set codes and language codes (rejects unknowns →
    ``None``). Uses digit-run regexes for collector number and slash-total.
    Tolerates extra whitespace. Returns all-``None`` fields (plus ``raw``) if
    the string is empty. Does NOT raise for unparseable input.
    """
    collector_number: str | None = None
    set_total: int | None = None
    rarity: str | None = None
    set_code: str | None = None
    language: str | None = None
    foil_detected: bool | None = None
    on_the_list_detected = False

    # Split on any whitespace (including newlines); drop empty tokens.
    for token in raw.split():
        if token == PLANESWALKER_ICON:
            on_the_list_detected = True
            continue

        if token == _FOIL_GLYPH:
            foil_detected = True
            continue

        if token == _NONFOIL_GLYPH:
            # Only record non-foil if we haven't already locked in foil=True.
            if foil_detected is not True:
                foil_detected = False
            continue

        slash_match = _SLASH_TOTAL_RE.match(token)
        if slash_match is not None:
            if collector_number is None:
                collector_number = slash_match.group(1)
            if set_total is None:
                set_total = int(slash_match.group(2))
            continue

        if _DIGITS_RE.match(token) is not None:
            if collector_number is None:
                collector_number = token
            continue

        # Uppercase alphanumeric tokens: could be set code, language, or rarity letter.
        if _UPPER_ALNUM_RE.match(token) is not None:
            length = len(token)
            is_alpha = _UPPER_ALPHA_RE.match(token) is not None

            # 1-char uppercase token: always a rarity letter (set codes are 2+ chars).
            if length == 1 and is_alpha:
                if rarity is None and token in _RARITY_LETTERS:
                    rarity = token
                continue

            # 2-char uppercase alpha: prefer language over set code.
            if length == 2 and is_alpha:
                if language is None and token in known_languages:
                    language = token
                    continue
                if set_code is None and token in known_set_codes:
                    set_code = token
                continue

            # 2-6 char uppercase alphanumeric (including digits like "M13"):
            # set code only. The alpha-only 2-char path above handles pure
            # letter language codes; anything with digits falls through here.
            if 2 <= length <= 6:
                if set_code is None and token in known_set_codes:
                    set_code = token
                continue
            # Longer uppercase tokens: ignore.
            continue
        # Any other token: ignore.

    return BottomRegionResult(
        collector_number=collector_number,
        set_total=set_total,
        rarity=rarity,
        set_code=set_code,
        language=language,
        foil_detected=foil_detected,
        on_the_list_detected=on_the_list_detected,
        raw=raw,
    )
