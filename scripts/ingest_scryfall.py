"""Bulk-ingest Scryfall cards into a local image cache + JSONL manifest.

Wires together the Scryfall bulk-data downloader, the per-card image
downloader, and the manifest writer. Resumable: cards already present in
``<out>/manifest.jsonl`` are skipped on subsequent runs.

Usage::

    python scripts/ingest_scryfall.py --out <dir> [--limit N] [--max-age-days D]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from moxify_ocr.data.manifest import ManifestEntry, append_manifest_entry, manifest_has
from moxify_ocr.data.scryfall import (
    download_card_image,
    fetch_default_cards_path,
    fetch_sets_path,
)

#: Default cache freshness in days — matches the Scryfall bulk-data refresh cadence.
_DEFAULT_MAX_AGE_DAYS = 7

#: Print a progress line every N processed cards.
_PROGRESS_EVERY = 100

#: Only ingest cards released on or after this date (M15 core set, 2014-07-18).
#: Older cards often lack the modern bottom-identifier format.
_MIN_RELEASE_DATE = "2014-07-18"


def _is_ingestable(card: dict[str, Any]) -> bool:
    """Return True iff this card has the modern bottom-identifier format.

    Excludes:
    - Digital-only cards (Arena/MTGO exclusives) — ``digital == True``
    - Cards released before M15 (2014-07-18)
    """
    if card.get("digital") is True:
        return False
    released_at = card.get("released_at")
    if not isinstance(released_at, str) or released_at < _MIN_RELEASE_DATE:
        return False
    return True


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns the process exit code."""
    parser = argparse.ArgumentParser(
        description="Download Scryfall bulk data + per-card images, writing a JSONL manifest."
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Root output directory for the bulk JSON, image cache, and manifest.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of cards to add to the manifest this run (default: all).",
    )
    parser.add_argument(
        "--max-age-days",
        type=int,
        default=_DEFAULT_MAX_AGE_DAYS,
        help=f"Bulk-data cache freshness window in days (default: {_DEFAULT_MAX_AGE_DAYS}).",
    )
    args = parser.parse_args(argv)
    return _run_ingest(
        out_dir=args.out,
        limit=args.limit,
        max_age_days=args.max_age_days,
    )


def _run_ingest(out_dir: Path, limit: int | None, max_age_days: int) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    images_dir = out_dir / "images"
    manifest_path = out_dir / "manifest.jsonl"

    bulk_path = fetch_default_cards_path(cache_dir=out_dir, max_age_days=max_age_days)
    sets_path = fetch_sets_path(cache_dir=out_dir, max_age_days=max_age_days)
    sets_index = _load_sets_index(sets_path)
    with bulk_path.open("r", encoding="utf-8") as handle:
        cards: list[dict[str, Any]] = json.load(handle)

    total = len(cards)
    added = 0
    skipped = 0

    for index, card in enumerate(cards, start=1):
        if limit is not None and added >= limit:
            break

        card_id = card.get("id")
        if not isinstance(card_id, str) or not card_id:
            skipped += 1
            continue

        if not _is_ingestable(card):
            skipped += 1
            continue

        if manifest_has(manifest_path, card_id):
            skipped += 1
            _maybe_print_progress(index, total, skipped)
            continue

        image_path = download_card_image(card, images_dir)
        if image_path is None:
            skipped += 1
            _maybe_print_progress(index, total, skipped)
            continue

        entry = _build_entry(card, image_path, out_dir, sets_index)
        append_manifest_entry(manifest_path, entry)
        added += 1
        _maybe_print_progress(index, total, skipped)

    print(
        f"done: added={added} skipped={skipped} of {total}",
        file=sys.stderr,
    )
    return 0


def _build_entry(
    card: dict[str, Any],
    image_path: Path,
    out_dir: Path,
    sets_index: dict[str, int],
) -> ManifestEntry:
    finishes_raw = card.get("finishes", [])
    finishes: list[str] = (
        [f for f in finishes_raw if isinstance(f, str)] if isinstance(finishes_raw, list) else []
    )
    try:
        rel_image = image_path.relative_to(out_dir)
    except ValueError:
        rel_image = image_path
    set_code = _as_str(card.get("set"))
    printed_size = sets_index.get(set_code.lower()) if set_code else None
    return ManifestEntry(
        scryfall_id=_as_str(card.get("id")),
        image_path=str(rel_image),
        lang=_as_str(card.get("lang")),
        set_code=set_code,
        collector_number=_as_str(card.get("collector_number")),
        rarity=_as_str(card.get("rarity")),
        type_line=_as_str(card.get("type_line")),
        layout=_as_str(card.get("layout")),
        finishes=finishes,
        image_sha256=_sha256_file(image_path),
        released_at=_as_str(card.get("released_at")),
        printed_size=printed_size,
    )


def _as_str(value: Any) -> str:
    return value if isinstance(value, str) else ""


def _load_sets_index(sets_path: Path) -> dict[str, int]:
    """Parse the cached ``sets.json`` into a ``{set_code_lower: printed_size}`` map.

    Entries missing ``printed_size`` are omitted (not stored as ``None``); the
    caller treats an absent key as "unknown", matching Scryfall's semantics for
    digital-only sets and token products.
    """
    with sets_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    entries = payload.get("data", []) if isinstance(payload, dict) else []
    result: dict[str, int] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        code = entry.get("code")
        printed_size = entry.get("printed_size")
        if isinstance(code, str) and isinstance(printed_size, int):
            result[code.lower()] = printed_size
    return result


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _maybe_print_progress(processed: int, total: int, skipped: int) -> None:
    if processed % _PROGRESS_EVERY == 0:
        print(
            f"processed {processed} / {total} skipped {skipped}",
            file=sys.stderr,
        )


if __name__ == "__main__":
    raise SystemExit(main())
