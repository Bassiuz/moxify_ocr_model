"""Shared fixtures and helpers for `tests/data/` test modules."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock


def _mock_response(
    *,
    json_data: Any | None = None,
    content: bytes | None = None,
) -> MagicMock:
    """Build a minimal fake `requests.Response` suitable for the downloader."""
    response = MagicMock()
    response.raise_for_status = MagicMock()
    if json_data is not None:
        response.json = MagicMock(return_value=json_data)
    if content is not None:
        response.content = content
        # iter_content yields one chunk equal to the full content.
        response.iter_content = MagicMock(return_value=iter([content]))
    else:
        response.iter_content = MagicMock(return_value=iter([]))
    # Support `with requests.get(...) as r:` usage for streaming downloads.
    response.__enter__ = MagicMock(return_value=response)
    response.__exit__ = MagicMock(return_value=None)
    return response
