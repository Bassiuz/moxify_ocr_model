"""Tests for the cardconjurer pool renderer's resume logic.

Most of ``render_cardconjurer_pool.py`` is integration code that needs a
running CardConjurer container + headless Chromium, so we don't unit-test
it. The resume helper is pure I/O and worth a real test — getting it wrong
silently re-renders ~20h of work.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# scripts/ is not a package; import the helper by path-mangling.
_SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(_SCRIPT_PATH))

from render_cardconjurer_pool import _read_completed_indices  # noqa: E402


def _write_labels(path: Path, indices: list[int]) -> None:
    rows = [{"image_path": f"images/{i:08d}.png", "label": "FOO"} for i in indices]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")


def test_returns_empty_when_no_labels_file(tmp_path: Path) -> None:
    assert _read_completed_indices(tmp_path / "labels.jsonl") == set()


def test_returns_empty_for_empty_file(tmp_path: Path) -> None:
    p = tmp_path / "labels.jsonl"
    p.touch()
    assert _read_completed_indices(p) == set()


def test_parses_indices_from_image_paths(tmp_path: Path) -> None:
    p = tmp_path / "labels.jsonl"
    _write_labels(p, [0, 1, 7, 12345, 99999])
    assert _read_completed_indices(p) == {0, 1, 7, 12345, 99999}


def test_tolerates_partial_last_line(tmp_path: Path) -> None:
    """A hard kill mid-write leaves a truncated last line — must skip cleanly."""
    p = tmp_path / "labels.jsonl"
    p.write_text(
        json.dumps({"image_path": "images/00000000.png"}) + "\n"
        + json.dumps({"image_path": "images/00000001.png"}) + "\n"
        + '{"image_path": "images/00000002.pn'  # truncated, no closing brace
    )
    assert _read_completed_indices(p) == {0, 1}


def test_ignores_rows_without_image_path(tmp_path: Path) -> None:
    p = tmp_path / "labels.jsonl"
    p.write_text(
        json.dumps({"image_path": "images/00000005.png"}) + "\n"
        + json.dumps({"label": "no image_path field"}) + "\n"
        + json.dumps({"image_path": 12345}) + "\n"  # wrong type
        + json.dumps({"image_path": "images/00000007.png"}) + "\n"
    )
    assert _read_completed_indices(p) == {5, 7}


def test_ignores_rows_with_unexpected_path_format(tmp_path: Path) -> None:
    p = tmp_path / "labels.jsonl"
    p.write_text(
        json.dumps({"image_path": "images/00000001.png"}) + "\n"
        + json.dumps({"image_path": "/absolute/00000002.png"}) + "\n"
        + json.dumps({"image_path": "images/abc.png"}) + "\n"  # not numeric
        + json.dumps({"image_path": "images/00000003.jpg"}) + "\n"  # wrong ext
        + json.dumps({"image_path": "images/00000004.png"}) + "\n"
    )
    assert _read_completed_indices(p) == {1, 4}
