"""JSONL manifest of ingested card records.

One JSON object per line. Append-only. Used to make ingestion reproducible and
to let incremental runs skip cards that were already fetched.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class ManifestEntry:
    """One row of the ingestion manifest — immutable metadata for a cached card."""

    scryfall_id: str
    image_path: str
    lang: str
    set_code: str
    collector_number: str
    rarity: str
    type_line: str
    layout: str
    finishes: list[str]
    image_sha256: str


def append_manifest_entry(manifest_path: Path, entry: ManifestEntry) -> None:
    """Append one JSON line to ``manifest_path``.

    Creates the file (and any missing parent directories) if it does not exist.
    Flushes after the write so a crash does not lose the record.
    """
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(asdict(entry)) + "\n"
    with manifest_path.open("a", encoding="utf-8") as handle:
        handle.write(line)
        handle.flush()


def read_manifest(manifest_path: Path) -> Iterator[ManifestEntry]:
    """Yield ``ManifestEntry`` objects, one per JSONL line.

    Raises ``FileNotFoundError`` if ``manifest_path`` does not exist.
    """
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            yield ManifestEntry(**payload)


def manifest_has(manifest_path: Path, scryfall_id: str) -> bool:
    """Return ``True`` iff any row in the manifest has this ``scryfall_id``.

    Returns ``False`` — not an error — if the manifest file does not exist.
    """
    if not manifest_path.exists():
        return False
    return any(entry.scryfall_id == scryfall_id for entry in read_manifest(manifest_path))
