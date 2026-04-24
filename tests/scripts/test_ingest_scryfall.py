"""Tests for the Scryfall bulk-ingest CLI.

The CLI wires together `fetch_default_cards_path`, `download_card_image`, and
the manifest helpers. These tests mock the network-touching pieces so the
wiring can be verified deterministically.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest.mock import patch

from moxify_ocr.data.manifest import ManifestEntry, append_manifest_entry, read_manifest
from scripts.ingest_scryfall import main

_DUMMY_JPEG = b"0123456789"  # 10 bytes — content doesn't matter, just its hash.
_DUMMY_JPEG_SHA256 = hashlib.sha256(_DUMMY_JPEG).hexdigest()

# Small fixture mirroring the shape of Scryfall's /sets response.
_SETS_FIXTURE: dict[str, Any] = {
    "data": [
        {"code": "clu", "printed_size": 500},
        {"code": "sld", "printed_size": 100},
        # A set with no printed_size — e.g. a digital-only or token product.
        {"code": "digi"},
    ]
}


def _normal_card(card_id: str, *, set_code: str = "clu", collector_number: str = "0001") -> dict[str, Any]:
    return {
        "id": card_id,
        "layout": "normal",
        "lang": "en",
        "set": set_code,
        "collector_number": collector_number,
        "rarity": "common",
        "type_line": "Creature — Elf",
        "finishes": ["nonfoil", "foil"],
        "image_uris": {"large": f"https://scry/{card_id}-l.jpg"},
        "released_at": "2024-02-09",
        "frame": "2015",  # modern frame — required by the ingest filter
        "digital": False,
        "promo_types": [],
    }


def _art_series_card(card_id: str) -> dict[str, Any]:
    return {
        "id": card_id,
        "layout": "art_series",
        "lang": "en",
        "set": "sld",
        "collector_number": "A1",
        "rarity": "common",
        "type_line": "Art Series",
        "finishes": ["nonfoil"],
        "image_uris": {"large": f"https://scry/{card_id}-l.jpg"},
        "frame": "2015",
        "digital": False,
        "promo_types": [],
    }


def _write_bulk_fixture(path: Path, cards: list[dict[str, Any]]) -> None:
    path.write_text(json.dumps(cards))


def _fake_fetch(bulk_json: Path) -> Any:
    def _inner(cache_dir: Path, max_age_days: int) -> Path:
        cache_dir.mkdir(parents=True, exist_ok=True)
        return bulk_json

    return _inner


def _fake_fetch_sets(sets_json: Path) -> Any:
    def _inner(cache_dir: Path, max_age_days: int) -> Path:
        cache_dir.mkdir(parents=True, exist_ok=True)
        return sets_json

    return _inner


def _write_sets_fixture(path: Path) -> None:
    path.write_text(json.dumps(_SETS_FIXTURE))


def _fake_download_factory(out_dir: Path) -> Any:
    """Write a dummy JPEG for non-skipped cards, return None for art_series."""

    def _inner(card: dict[str, Any], cache_dir: Path) -> Path | None:
        if card.get("layout") == "art_series":
            return None
        card_id = card["id"]
        target = cache_dir / card_id[:2] / f"{card_id}.jpg"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(_DUMMY_JPEG)
        return target

    # `out_dir` closure is just for symmetry with future extensions.
    del out_dir
    return _inner


def test_cli_end_to_end_with_mocks(tmp_path: Path) -> None:
    bulk_json = tmp_path / "default-cards.json"
    sets_json = tmp_path / "sets.json"
    cards = [
        _normal_card("aaaa1111", collector_number="0001"),
        _normal_card("bbbb2222", collector_number="0002"),
        _art_series_card("cccc3333"),
        _normal_card("dddd4444", collector_number="0004"),
        _normal_card("eeee5555", collector_number="0005"),
    ]
    _write_bulk_fixture(bulk_json, cards)
    _write_sets_fixture(sets_json)

    with (
        patch(
            "scripts.ingest_scryfall.fetch_default_cards_path",
            side_effect=_fake_fetch(bulk_json),
        ),
        patch(
            "scripts.ingest_scryfall.fetch_sets_path",
            side_effect=_fake_fetch_sets(sets_json),
        ),
        patch(
            "scripts.ingest_scryfall.download_card_image",
            side_effect=_fake_download_factory(tmp_path),
        ),
    ):
        rc = main(["--out", str(tmp_path), "--limit", "100"])

    assert rc == 0
    manifest_path = tmp_path / "manifest.jsonl"
    assert manifest_path.is_file()
    rows: list[ManifestEntry] = list(read_manifest(manifest_path))
    assert len(rows) == 4
    ids = [row.scryfall_id for row in rows]
    assert ids == ["aaaa1111", "bbbb2222", "dddd4444", "eeee5555"]
    for row in rows:
        assert row.image_sha256 == _DUMMY_JPEG_SHA256
        # image_path is relative to out_dir, pointing at images/<xx>/<id>.jpg
        assert row.image_path.endswith(f"{row.scryfall_id}.jpg")
        assert row.lang == "en"
        assert row.set_code == "clu"
        assert row.rarity == "common"
        assert row.layout == "normal"
        assert row.finishes == ["nonfoil", "foil"]
        assert row.type_line == "Creature — Elf"
        # New schema fields: released_at comes from the card dict,
        # printed_size is looked up by set_code in the sets fixture.
        assert row.released_at == "2024-02-09"
        assert row.printed_size == 500


def test_cli_populates_printed_size_from_sets_index(tmp_path: Path) -> None:
    """Cards whose set has no ``printed_size`` in the index get ``None``."""
    bulk_json = tmp_path / "default-cards.json"
    sets_json = tmp_path / "sets.json"
    cards = [
        _normal_card("aaaa1111", set_code="clu", collector_number="0001"),
        _normal_card("bbbb2222", set_code="digi", collector_number="0002"),
        _normal_card("cccc3333", set_code="unknown", collector_number="0003"),
    ]
    _write_bulk_fixture(bulk_json, cards)
    _write_sets_fixture(sets_json)

    with (
        patch(
            "scripts.ingest_scryfall.fetch_default_cards_path",
            side_effect=_fake_fetch(bulk_json),
        ),
        patch(
            "scripts.ingest_scryfall.fetch_sets_path",
            side_effect=_fake_fetch_sets(sets_json),
        ),
        patch(
            "scripts.ingest_scryfall.download_card_image",
            side_effect=_fake_download_factory(tmp_path),
        ),
    ):
        rc = main(["--out", str(tmp_path)])

    assert rc == 0
    rows = {row.scryfall_id: row for row in read_manifest(tmp_path / "manifest.jsonl")}
    assert rows["aaaa1111"].printed_size == 500
    assert rows["bbbb2222"].printed_size is None  # in sets index but no printed_size.
    assert rows["cccc3333"].printed_size is None  # set code not in sets index at all.


def test_cli_respects_limit(tmp_path: Path) -> None:
    bulk_json = tmp_path / "default-cards.json"
    sets_json = tmp_path / "sets.json"
    cards = [_normal_card(f"id{i:06d}aaaa", collector_number=str(i)) for i in range(5)]
    _write_bulk_fixture(bulk_json, cards)
    _write_sets_fixture(sets_json)

    with (
        patch(
            "scripts.ingest_scryfall.fetch_default_cards_path",
            side_effect=_fake_fetch(bulk_json),
        ),
        patch(
            "scripts.ingest_scryfall.fetch_sets_path",
            side_effect=_fake_fetch_sets(sets_json),
        ),
        patch(
            "scripts.ingest_scryfall.download_card_image",
            side_effect=_fake_download_factory(tmp_path),
        ),
    ):
        rc = main(["--out", str(tmp_path), "--limit", "2"])

    assert rc == 0
    rows = list(read_manifest(tmp_path / "manifest.jsonl"))
    assert len(rows) == 2


def test_cli_skips_already_in_manifest(tmp_path: Path) -> None:
    bulk_json = tmp_path / "default-cards.json"
    sets_json = tmp_path / "sets.json"
    cards = [
        _normal_card("id000001aaaa"),
        _normal_card("id000002bbbb"),
        _normal_card("id000003cccc"),
    ]
    _write_bulk_fixture(bulk_json, cards)
    _write_sets_fixture(sets_json)

    # Pre-seed the manifest with the second card.
    manifest_path = tmp_path / "manifest.jsonl"
    seed_entry = ManifestEntry(
        scryfall_id="id000002bbbb",
        image_path="id/id000002bbbb.jpg",
        lang="en",
        set_code="clu",
        collector_number="9999",
        rarity="common",
        type_line="Already ingested",
        layout="normal",
        finishes=["nonfoil"],
        image_sha256="ff" * 32,
    )
    append_manifest_entry(manifest_path, seed_entry)

    download_calls: list[str] = []

    def tracking_download(card: dict[str, Any], cache_dir: Path) -> Path | None:
        download_calls.append(card["id"])
        return _fake_download_factory(tmp_path)(card, cache_dir)

    with (
        patch(
            "scripts.ingest_scryfall.fetch_default_cards_path",
            side_effect=_fake_fetch(bulk_json),
        ),
        patch(
            "scripts.ingest_scryfall.fetch_sets_path",
            side_effect=_fake_fetch_sets(sets_json),
        ),
        patch(
            "scripts.ingest_scryfall.download_card_image",
            side_effect=tracking_download,
        ),
    ):
        rc = main(["--out", str(tmp_path)])

    assert rc == 0
    # download_card_image was NOT called for the pre-manifested card.
    assert "id000002bbbb" not in download_calls
    assert set(download_calls) == {"id000001aaaa", "id000003cccc"}

    rows = list(read_manifest(manifest_path))
    assert len(rows) == 3
    # Original seed preserved; two new rows appended.
    assert rows[0] == seed_entry
    new_ids = {row.scryfall_id for row in rows[1:]}
    assert new_ids == {"id000001aaaa", "id000003cccc"}


def test_cli_skipped_layouts_not_in_manifest(tmp_path: Path) -> None:
    bulk_json = tmp_path / "default-cards.json"
    sets_json = tmp_path / "sets.json"
    cards = [_art_series_card("zzzz9999")]
    _write_bulk_fixture(bulk_json, cards)
    _write_sets_fixture(sets_json)

    with (
        patch(
            "scripts.ingest_scryfall.fetch_default_cards_path",
            side_effect=_fake_fetch(bulk_json),
        ),
        patch(
            "scripts.ingest_scryfall.fetch_sets_path",
            side_effect=_fake_fetch_sets(sets_json),
        ),
        patch(
            "scripts.ingest_scryfall.download_card_image",
            side_effect=_fake_download_factory(tmp_path),
        ),
    ):
        rc = main(["--out", str(tmp_path)])

    assert rc == 0
    manifest_path = tmp_path / "manifest.jsonl"
    # Either no manifest file at all, or a zero-row manifest — both are valid.
    if manifest_path.exists():
        rows: Iterator[ManifestEntry] = read_manifest(manifest_path)
        assert list(rows) == []
