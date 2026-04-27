"""CTC alphabet for the card-name OCR model (Latin-script v1).

Derived from a one-shot scan of `data/scryfall/default-cards.json` filtered
to ``layout in ALLOWED_LAYOUTS`` and ``lang in ALLOWED_LANGS`` (see
docs/plans/2026-04-27-card-name-ocr-design.md). All characters with
≥2 observed occurrences are included; very-rare or non-glyph characters
(``_``, ``|``, ``;``, ``®``, the U+A689 ``꞉`` lookalike) are excluded —
those represent internal-noise artifacts of the metadata, not glyphs that
get printed on physical cards.

Index 0 is reserved for the CTC blank (matches the convention in
:mod:`moxify_ocr.data.dataset`); the characters below occupy indices 1..N.
"""

from __future__ import annotations

NAME_ALPHABET: str = (
    "0123456789"  # digits (10)
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"  # uppercase (26)
    "abcdefghijklmnopqrstuvwxyz"  # lowercase (26)
    " ,'-:.&!\"?()%/$+"  # punctuation (16)
    "éóáíñûúÉâèöàÁïüêîôōä"  # diacritics (20)
)

# Sanity-check on import — must stay in sync with the docstring above.
assert len(NAME_ALPHABET) == 98, (
    f"NAME_ALPHABET must be 98 chars, got {len(NAME_ALPHABET)}"
)
assert len(set(NAME_ALPHABET)) == 98, "NAME_ALPHABET has duplicates"
