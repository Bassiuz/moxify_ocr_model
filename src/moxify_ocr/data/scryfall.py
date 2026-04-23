"""Scryfall bulk-data downloader with a simple mtime-based disk cache."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, cast

import requests

#: Scryfall bulk-data index endpoint.
BULK_DATA_INDEX_URL = "https://api.scryfall.com/bulk-data"

#: Polite User-Agent identifying this project to Scryfall (per their API guidelines).
USER_AGENT = "moxify-ocr/0.1 (https://github.com/Bassiuz/moxify_ocr_model)"

#: File name used for the cached `default_cards` bulk JSON.
DEFAULT_CARDS_FILENAME = "default-cards.json"

#: Chunk size for streaming the bulk JSON (~1 MiB).
_DOWNLOAD_CHUNK_SIZE = 1 << 20

#: Short delay between the index request and the bulk download.
_RATE_LIMIT_SLEEP_S = 0.1

_REQUEST_HEADERS = {
    "User-Agent": USER_AGENT,
    "Accept": "application/json",
}

#: Layouts we cannot meaningfully train on — skip their image downloads.
_SKIP_LAYOUTS = frozenset(
    {"art_series", "token", "double_faced_token", "scheme", "plane", "phenomenon"}
)


def fetch_default_cards_path(cache_dir: Path, max_age_days: int) -> Path:
    """Return the local path to Scryfall's `default_cards` bulk JSON.

    Downloads the file into ``cache_dir`` if the cached copy is missing or older
    than ``max_age_days``. The download is streamed to disk in chunks so it is
    safe for the ~500 MB bulk payload.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    target = cache_dir / DEFAULT_CARDS_FILENAME

    if _is_cache_fresh(target, max_age_days):
        return target

    download_uri = _find_default_cards_uri()
    time.sleep(_RATE_LIMIT_SLEEP_S)
    _stream_download(download_uri, target)
    return target


def _is_cache_fresh(path: Path, max_age_days: int) -> bool:
    if not path.exists():
        return False
    age_seconds = time.time() - path.stat().st_mtime
    return age_seconds < max_age_days * 86400


def _find_default_cards_uri() -> str:
    response = requests.get(BULK_DATA_INDEX_URL, headers=_REQUEST_HEADERS, timeout=30)
    response.raise_for_status()
    payload = response.json()
    entries = payload.get("data", []) if isinstance(payload, dict) else []
    for entry in entries:
        if isinstance(entry, dict) and entry.get("type") == "default_cards":
            uri = entry.get("download_uri")
            if isinstance(uri, str) and uri:
                return uri
            raise LookupError(
                "Scryfall bulk-data index contains a default_cards entry without a download_uri."
            )
    raise LookupError("Scryfall bulk-data index has no default_cards entry.")


def download_card_image(card: dict[str, Any], cache_dir: Path) -> Path | None:
    """Download a Scryfall card's front-face image into ``cache_dir``.

    Returns the local path to the cached JPG, or ``None`` if the card's layout
    is skipped or no usable image URL is present. The file is written to
    ``<cache_dir>/<first-two-of-id>/<id>.jpg`` and the download is skipped if
    that path already exists.
    """
    if card.get("layout") in _SKIP_LAYOUTS:
        return None

    card_id = card.get("id")
    if not isinstance(card_id, str) or not card_id:
        return None

    image_url = _pick_image_url(card)
    if image_url is None:
        return None

    target = cache_dir / card_id[:2] / f"{card_id}.jpg"
    if target.exists():
        return target

    target.parent.mkdir(parents=True, exist_ok=True)
    time.sleep(_RATE_LIMIT_SLEEP_S)
    response = requests.get(image_url, headers=_REQUEST_HEADERS, timeout=30)
    response.raise_for_status()
    target.write_bytes(response.content)
    return target


def _pick_image_url(card: dict[str, Any]) -> str | None:
    image_uris = card.get("image_uris")
    if not isinstance(image_uris, dict):
        faces = card.get("card_faces")
        if isinstance(faces, list) and faces and isinstance(faces[0], dict):
            front = cast(dict[str, Any], faces[0])
            image_uris = front.get("image_uris")
    if not isinstance(image_uris, dict):
        return None
    for key in ("large", "normal"):
        value = image_uris.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def _stream_download(url: str, dest: Path) -> None:
    with requests.get(url, headers=_REQUEST_HEADERS, stream=True, timeout=60) as response:
        response.raise_for_status()
        with dest.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=_DOWNLOAD_CHUNK_SIZE):
                if chunk:
                    handle.write(chunk)
