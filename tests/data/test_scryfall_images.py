from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch

from moxify_ocr.data.scryfall import _RATE_LIMIT_SLEEP_S, download_card_image

from .conftest import _mock_response


def _standard_card() -> dict[str, Any]:
    return {
        "id": "abc123def456",
        "layout": "normal",
        "image_uris": {
            "small": "https://scry/s.jpg",
            "normal": "https://scry/n.jpg",
            "large": "https://scry/l.jpg",
        },
    }


def test_skip_by_layout(tmp_path: Path) -> None:
    card: dict[str, Any] = {
        "id": "xyz789",
        "layout": "art_series",
        "image_uris": {"large": "https://scry/l.jpg"},
    }
    with patch("moxify_ocr.data.scryfall.requests.get") as mock_get:
        result = download_card_image(card, cache_dir=tmp_path)
    assert result is None
    mock_get.assert_not_called()


def test_skip_when_no_images(tmp_path: Path) -> None:
    card: dict[str, Any] = {"id": "noimg", "layout": "normal"}
    with patch("moxify_ocr.data.scryfall.requests.get") as mock_get:
        result = download_card_image(card, cache_dir=tmp_path)
    assert result is None
    mock_get.assert_not_called()


def test_card_faces_fallback(tmp_path: Path) -> None:
    card: dict[str, Any] = {
        "id": "def456",
        "layout": "transform",
        "card_faces": [
            {"name": "Front", "image_uris": {"large": "https://scry/front-l.jpg"}},
            {"name": "Back", "image_uris": {"large": "https://scry/back-l.jpg"}},
        ],
    }
    with patch("moxify_ocr.data.scryfall.requests.get") as mock_get:
        mock_get.return_value = _mock_response(content=b"JPEGDATA")
        result = download_card_image(card, cache_dir=tmp_path)

    assert result is not None
    assert result.exists()
    assert result.read_bytes() == b"JPEGDATA"
    assert mock_get.call_count == 1
    called_url = mock_get.call_args.args[0]
    assert called_url == "https://scry/front-l.jpg"


def test_prefers_large_over_normal(tmp_path: Path) -> None:
    card = _standard_card()
    with patch("moxify_ocr.data.scryfall.requests.get") as mock_get:
        mock_get.return_value = _mock_response(content=b"LARGEIMG")
        result = download_card_image(card, cache_dir=tmp_path)

    assert result is not None
    assert mock_get.call_count == 1
    assert mock_get.call_args.args[0] == "https://scry/l.jpg"


def test_falls_back_to_normal_when_no_large(tmp_path: Path) -> None:
    card: dict[str, Any] = {
        "id": "normalonly",
        "layout": "normal",
        "image_uris": {"normal": "https://scry/n.jpg"},
    }
    with patch("moxify_ocr.data.scryfall.requests.get") as mock_get:
        mock_get.return_value = _mock_response(content=b"NORMALIMG")
        result = download_card_image(card, cache_dir=tmp_path)

    assert result is not None
    assert mock_get.call_count == 1
    assert mock_get.call_args.args[0] == "https://scry/n.jpg"


def test_idempotent_when_file_exists(tmp_path: Path) -> None:
    card = _standard_card()
    card_id = card["id"]
    target = tmp_path / card_id[:2] / f"{card_id}.jpg"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"EXISTING")

    with patch("moxify_ocr.data.scryfall.requests.get") as mock_get:
        result = download_card_image(card, cache_dir=tmp_path)

    assert result == target
    assert target.read_bytes() == b"EXISTING"
    mock_get.assert_not_called()


def test_path_layout_uses_id_prefix(tmp_path: Path) -> None:
    card = _standard_card()
    with patch("moxify_ocr.data.scryfall.requests.get") as mock_get:
        mock_get.return_value = _mock_response(content=b"X")
        result = download_card_image(card, cache_dir=tmp_path)

    assert result is not None
    card_id = card["id"]
    expected = tmp_path / card_id[:2] / f"{card_id}.jpg"
    assert result == expected
    assert result.exists()


def test_rate_limit_sleeps_before_request(tmp_path: Path) -> None:
    card = _standard_card()
    calls: list[str] = []

    def fake_sleep(seconds: float) -> None:
        calls.append(f"sleep:{seconds}")

    def fake_get(*args: Any, **kwargs: Any) -> Any:
        calls.append("get")
        return _mock_response(content=b"X")

    with (
        patch("moxify_ocr.data.scryfall.time.sleep", side_effect=fake_sleep),
        patch("moxify_ocr.data.scryfall.requests.get", side_effect=fake_get),
    ):
        download_card_image(card, cache_dir=tmp_path)

    assert f"sleep:{_RATE_LIMIT_SLEEP_S}" in calls
    sleep_idx = calls.index(f"sleep:{_RATE_LIMIT_SLEEP_S}")
    get_idx = calls.index("get")
    assert sleep_idx < get_idx
