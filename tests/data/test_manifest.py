from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest

from moxify_ocr.data.manifest import (
    ManifestEntry,
    append_manifest_entry,
    manifest_has,
    read_manifest,
)


def _make_entry(
    scryfall_id: str = "abc123def456",
    image_path: str = "ab/abc123def456.jpg",
) -> ManifestEntry:
    return ManifestEntry(
        scryfall_id=scryfall_id,
        image_path=image_path,
        lang="en",
        set_code="clu",
        collector_number="0280",
        rarity="common",
        type_line="Basic Land — Mountain",
        layout="normal",
        finishes=["foil", "nonfoil"],
        image_sha256="deadbeef" * 8,
    )


def test_append_creates_file_and_parent_dir(tmp_path: Path) -> None:
    manifest_path = tmp_path / "ingest" / "manifest.jsonl"
    assert not manifest_path.parent.exists()

    append_manifest_entry(manifest_path, _make_entry())

    assert manifest_path.parent.is_dir()
    assert manifest_path.is_file()


def test_append_writes_valid_jsonl(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.jsonl"
    entry_a = _make_entry(scryfall_id="id-a", image_path="aa/id-a.jpg")
    entry_b = _make_entry(scryfall_id="id-b", image_path="bb/id-b.jpg")

    append_manifest_entry(manifest_path, entry_a)
    append_manifest_entry(manifest_path, entry_b)

    lines = manifest_path.read_text().splitlines()
    assert len(lines) == 2
    parsed_a = json.loads(lines[0])
    parsed_b = json.loads(lines[1])
    assert parsed_a["scryfall_id"] == "id-a"
    assert parsed_a["image_path"] == "aa/id-a.jpg"
    assert parsed_b["scryfall_id"] == "id-b"
    assert parsed_b["image_path"] == "bb/id-b.jpg"


def test_read_returns_entries_in_order(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.jsonl"
    entries = [
        _make_entry(scryfall_id=f"id-{i}", image_path=f"xx/id-{i}.jpg") for i in range(3)
    ]
    for entry in entries:
        append_manifest_entry(manifest_path, entry)

    read_back = list(read_manifest(manifest_path))

    assert read_back == entries


def test_read_raises_if_missing(tmp_path: Path) -> None:
    manifest_path = tmp_path / "does-not-exist.jsonl"
    with pytest.raises(FileNotFoundError):
        list(read_manifest(manifest_path))


def test_read_is_lazy(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.jsonl"
    append_manifest_entry(manifest_path, _make_entry())

    result = read_manifest(manifest_path)

    assert inspect.isgenerator(result)
    first = next(result)
    assert isinstance(first, ManifestEntry)


def test_manifest_has_true_when_present(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.jsonl"
    entry = _make_entry(scryfall_id="present-id")
    append_manifest_entry(manifest_path, entry)

    assert manifest_has(manifest_path, "present-id") is True


def test_manifest_has_false_when_absent(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.jsonl"
    append_manifest_entry(manifest_path, _make_entry(scryfall_id="present-id"))

    assert manifest_has(manifest_path, "absent-id") is False


def test_manifest_has_returns_false_for_missing_file(tmp_path: Path) -> None:
    manifest_path = tmp_path / "never-created.jsonl"
    assert not manifest_path.exists()

    assert manifest_has(manifest_path, "anything") is False


def test_roundtrip_preserves_all_fields(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.jsonl"
    entry = ManifestEntry(
        scryfall_id="abc123def456",
        image_path="ab/abc123def456.jpg",
        lang="ja",
        set_code="neo",
        collector_number="0042",
        rarity="mythic",
        type_line="Legendary Creature — Human Samurai",
        layout="transform",
        finishes=["foil", "nonfoil", "etched"],
        image_sha256="0123456789abcdef" * 4,
        released_at="2022-02-18",
        printed_size=512,
    )

    append_manifest_entry(manifest_path, entry)
    read_back = list(read_manifest(manifest_path))

    assert len(read_back) == 1
    got = read_back[0]
    assert got.scryfall_id == "abc123def456"
    assert got.image_path == "ab/abc123def456.jpg"
    assert got.lang == "ja"
    assert got.set_code == "neo"
    assert got.collector_number == "0042"
    assert got.rarity == "mythic"
    assert got.type_line == "Legendary Creature — Human Samurai"
    assert got.layout == "transform"
    assert got.finishes == ["foil", "nonfoil", "etched"]
    assert got.image_sha256 == "0123456789abcdef" * 4
    assert got.released_at == "2022-02-18"
    assert got.printed_size == 512
    assert got == entry


def test_roundtrip_preserves_new_fields_with_none_printed_size(tmp_path: Path) -> None:
    """``printed_size`` can legitimately be ``None`` (e.g. digital-only sets)."""
    manifest_path = tmp_path / "manifest.jsonl"
    entry = _make_entry()
    # _make_entry defaults to released_at="" and printed_size=None.
    entry_with_date = ManifestEntry(
        scryfall_id=entry.scryfall_id,
        image_path=entry.image_path,
        lang=entry.lang,
        set_code=entry.set_code,
        collector_number=entry.collector_number,
        rarity=entry.rarity,
        type_line=entry.type_line,
        layout=entry.layout,
        finishes=entry.finishes,
        image_sha256=entry.image_sha256,
        released_at="2024-02-09",
        printed_size=None,
    )

    append_manifest_entry(manifest_path, entry_with_date)
    [got] = list(read_manifest(manifest_path))

    assert got.released_at == "2024-02-09"
    assert got.printed_size is None
    assert got == entry_with_date


def test_read_defaults_missing_new_fields(tmp_path: Path) -> None:
    """Rows written before the schema gained ``released_at``/``printed_size``
    must still parse, with the new fields defaulting to ``""`` and ``None``.
    """
    manifest_path = tmp_path / "manifest.jsonl"
    legacy_row = {
        "scryfall_id": "legacy-id",
        "image_path": "le/legacy-id.jpg",
        "lang": "en",
        "set_code": "clu",
        "collector_number": "0001",
        "rarity": "common",
        "type_line": "Creature — Elf",
        "layout": "normal",
        "finishes": ["nonfoil"],
        "image_sha256": "ab" * 32,
    }
    manifest_path.write_text(json.dumps(legacy_row) + "\n", encoding="utf-8")

    [got] = list(read_manifest(manifest_path))

    assert got.scryfall_id == "legacy-id"
    assert got.released_at == ""
    assert got.printed_size is None
