"""End-to-end check that ``make_label`` and ``parse_bottom`` are inverses.

Runs against the cached smoke manifest if one exists locally. On CI / fresh
clones with no ``data/scryfall/manifest.jsonl`` present the test is skipped
so it doesn't fail the build.
"""

from __future__ import annotations

import re
from datetime import date
from pathlib import Path

import pytest

from moxify_ocr.data.dataset import encode_label
from moxify_ocr.data.labels import (
    LANG_CODE,
    _salvage_plst_fields,
    _salvage_promo_set_code,
    make_label,
)
from moxify_ocr.data.manifest import read_manifest
from moxify_ocr.export.parse_bottom import parse_bottom

# Collector numbers the parser can round-trip are bare digit runs. After PLST
# salvage ("KTK-86" → "86") this is all normal cards. Rows whose collector
# number is intrinsically non-digit (promo "46★", SLD "2020-7", lowercase
# "227b") round-trip lossily and are skipped — the training pipeline handles
# them via the same try/except skip in the dataset generator.
_ROUND_TRIPPABLE_NUMBER = re.compile(r"^\d+$")

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
    # Build known-vocab sets from the data we have. Include PLST-salvaged set
    # codes (e.g. "KTK" from "KTK-86") so parse_bottom recognizes them.
    entries = list(read_manifest(SMOKE_MANIFEST))
    known_set_codes: set[str] = set()
    for e in entries:
        known_set_codes.add(e.set_code.upper())
        card_for_salvage = {"set": e.set_code, "collector_number": e.collector_number}
        salvaged_set, _, _ = _salvage_plst_fields(card_for_salvage)
        known_set_codes.add(salvaged_set.upper())
        # Also include promo-salvaged codes (FJMP→JMP etc) so parse_bottom can
        # match them when comparing predictions.
        known_set_codes.add(_salvage_promo_set_code(salvaged_set).upper())
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

        try:
            label = make_label(card, is_foil=False)
            encode_label(label)  # also verify the label is in our alphabet
        except (KeyError, ValueError):
            # Unsupported language, lowercase promo suffixes like "227b" / "268p",
            # or other chars outside the uppercase CTC alphabet. The dataset
            # generator skips these rows at training time too.
            continue
        result = parse_bottom(
            label,
            known_set_codes=known_set_codes,
            known_languages=known_languages,
        )

        # Compute expected fields accounting for PLST salvage AND promo salvage
        # (FJMP→JMP, PONE→ONE etc — Scryfall's prefixed promo/memorabilia codes
        # vs the actual code printed on the card).
        expected_set, expected_num, is_list = _salvage_plst_fields(
            {"set": entry.set_code, "collector_number": entry.collector_number}
        )
        expected_set = _salvage_promo_set_code(expected_set)
        if not _ROUND_TRIPPABLE_NUMBER.match(expected_num):
            # Non-digit collector numbers (e.g. "46★" prerelease promo,
            # foreign glyph-suffixed) don't round-trip cleanly through the
            # vocab-based parser. Skip — not a make_label/parse_bottom bug.
            continue

        assert result.collector_number == expected_num, f"id={entry.scryfall_id}"
        assert result.set_code == expected_set.upper(), f"id={entry.scryfall_id}"
        assert result.language == LANG_CODE[entry.lang], f"id={entry.scryfall_id}"
        assert result.on_the_list_detected is is_list, f"id={entry.scryfall_id}"
        # set_total is only encoded for non-PLST, modern-era cards with a
        # known printed_size.
        if (
            not is_list
            and entry.printed_size is not None
            and _is_modern_era(entry.released_at)
        ):
            assert result.set_total == entry.printed_size, f"id={entry.scryfall_id}"
        else:
            assert result.set_total is None, f"id={entry.scryfall_id}"
        assert result.foil_detected is False, f"id={entry.scryfall_id}"


def _is_modern_era(released_at: str) -> bool:
    if not released_at:
        return False
    return date.fromisoformat(released_at) >= _SLASH_TOTAL_CUTOFF
