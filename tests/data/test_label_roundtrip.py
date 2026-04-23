"""End-to-end check that ``make_label`` and ``parse_bottom`` are inverses.

Runs against the cached smoke manifest if one exists locally. On CI / fresh
clones with no ``data/scryfall/manifest.jsonl`` present the test is skipped
so it doesn't fail the build.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from moxify_ocr.data.labels import LANG_CODE, make_label
from moxify_ocr.data.manifest import read_manifest
from moxify_ocr.export.parse_bottom import parse_bottom

SMOKE_MANIFEST = Path("data/scryfall/manifest.jsonl")

# The first set to print "collector_number/set_total" was Shards of Alara
# (2008-10-03); the label synthesizer uses a 2008-01-01 cutoff. Cards older
# than this cutoff don't encode ``printed_size`` in the label, so parse_bottom
# cannot recover it — not a bug, just a property of pre-modern-era printings.
_SLASH_TOTAL_CUTOFF = date(2008, 1, 1)


@pytest.mark.skipif(
    not SMOKE_MANIFEST.exists(),
    reason="Requires data/scryfall/manifest.jsonl (run scripts/ingest_scryfall.py).",
)
def test_roundtrip_on_real_manifest() -> None:
    # Build known-vocab sets from the data we have.
    entries = list(read_manifest(SMOKE_MANIFEST))
    known_set_codes = {e.set_code.upper() for e in entries}
    known_languages = set(LANG_CODE.values())

    for entry in entries:
        card: dict[str, object] = {
            "collector_number": entry.collector_number,
            "set": entry.set_code,
            "lang": entry.lang,
            "rarity": entry.rarity,
            "type_line": entry.type_line,
            "released_at": entry.released_at,
        }
        if entry.printed_size is not None:
            card["printed_size"] = entry.printed_size

        label = make_label(card, is_foil=False)
        result = parse_bottom(
            label,
            known_set_codes=known_set_codes,
            known_languages=known_languages,
        )

        assert result.collector_number == entry.collector_number, f"id={entry.scryfall_id}"
        assert result.set_code == entry.set_code.upper(), f"id={entry.scryfall_id}"
        assert result.language == LANG_CODE[entry.lang], f"id={entry.scryfall_id}"
        # set_total is only encoded when the card is both post-2008 AND has a
        # known printed_size. Otherwise the label omits it and the parser
        # correctly returns None.
        if entry.printed_size is not None and _is_modern_era(entry.released_at):
            assert result.set_total == entry.printed_size, f"id={entry.scryfall_id}"
        else:
            assert result.set_total is None, f"id={entry.scryfall_id}"
        assert result.foil_detected is False, f"id={entry.scryfall_id}"


def _is_modern_era(released_at: str) -> bool:
    if not released_at:
        return False
    return date.fromisoformat(released_at) >= _SLASH_TOTAL_CUTOFF
