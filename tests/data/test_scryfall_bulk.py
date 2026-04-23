from __future__ import annotations

import json
import os
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from moxify_ocr.data.scryfall import fetch_default_cards_path

from .conftest import _mock_response


def test_fetch_uses_cache_when_fresh(tmp_path: Path) -> None:
    cached = tmp_path / "default-cards.json"
    cached.write_text(json.dumps([{"id": "x"}]))
    # The file was just written; well within 7 days.
    with patch("moxify_ocr.data.scryfall.requests.get") as mock_get:
        result = fetch_default_cards_path(cache_dir=tmp_path, max_age_days=7)
    mock_get.assert_not_called()
    assert result == cached
    assert json.loads(result.read_text()) == [{"id": "x"}]


def test_fetch_downloads_when_missing(tmp_path: Path) -> None:
    index_response = {
        "data": [
            {
                "type": "default_cards",
                "download_uri": "https://example.com/default.json",
            }
        ]
    }
    with patch("moxify_ocr.data.scryfall.requests.get") as mock_get:
        mock_get.side_effect = [
            _mock_response(json_data=index_response),
            _mock_response(content=b'[{"id": "abc"}]'),
        ]
        path = fetch_default_cards_path(cache_dir=tmp_path, max_age_days=7)
    assert path.exists()
    assert path == tmp_path / "default-cards.json"
    assert json.loads(path.read_text()) == [{"id": "abc"}]
    assert mock_get.call_count == 2


def test_fetch_downloads_when_stale(tmp_path: Path) -> None:
    cached = tmp_path / "default-cards.json"
    cached.write_text(json.dumps([{"id": "old"}]))
    # Mark file as 10 days old (older than max_age_days=7).
    ten_days_ago = time.time() - 10 * 86400
    os.utime(cached, (ten_days_ago, ten_days_ago))

    index_response = {
        "data": [
            {
                "type": "default_cards",
                "download_uri": "https://example.com/default.json",
            }
        ]
    }
    with patch("moxify_ocr.data.scryfall.requests.get") as mock_get:
        mock_get.side_effect = [
            _mock_response(json_data=index_response),
            _mock_response(content=b'[{"id": "new"}]'),
        ]
        path = fetch_default_cards_path(cache_dir=tmp_path, max_age_days=7)
    assert path.exists()
    assert json.loads(path.read_text()) == [{"id": "new"}]
    assert mock_get.call_count == 2


def test_fetch_raises_if_no_default_cards_entry(tmp_path: Path) -> None:
    index_response = {
        "data": [
            {
                "type": "oracle_cards",
                "download_uri": "https://example.com/oracle.json",
            }
        ]
    }
    with patch("moxify_ocr.data.scryfall.requests.get") as mock_get:
        mock_get.side_effect = [_mock_response(json_data=index_response)]
        with pytest.raises(LookupError, match="default_cards"):
            fetch_default_cards_path(cache_dir=tmp_path, max_age_days=7)


def test_fetch_sends_polite_user_agent(tmp_path: Path) -> None:
    index_response = {
        "data": [
            {
                "type": "default_cards",
                "download_uri": "https://example.com/default.json",
            }
        ]
    }
    with patch("moxify_ocr.data.scryfall.requests.get") as mock_get:
        mock_get.side_effect = [
            _mock_response(json_data=index_response),
            _mock_response(content=b"[]"),
        ]
        fetch_default_cards_path(cache_dir=tmp_path, max_age_days=7)

    for call in mock_get.call_args_list:
        headers = call.kwargs.get("headers") or (call.args[1] if len(call.args) > 1 else None)
        assert headers is not None, "request must pass headers"
        user_agent = headers.get("User-Agent")
        assert user_agent is not None
        assert "moxify-ocr" in user_agent.lower()
