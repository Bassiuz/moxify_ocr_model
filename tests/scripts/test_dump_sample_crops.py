"""Tests for the bottom-crop dump CLI (``scripts/dump_sample_crops.py``).

Smoke-checks that the CLI loads manifest rows, runs each card image through
``crop_bottom_region``, and writes the result to disk. Covers the limit arg,
missing-file warning, and output size/mode invariants.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from moxify_ocr.data.manifest import ManifestEntry, append_manifest_entry
from scripts.dump_sample_crops import main

# Scryfall "large" images are ~672x936; we use that here so the crop math
# produces realistic ~256x48 outputs.
_CARD_SIZE: tuple[int, int] = (672, 936)


def _write_card(path: Path, color: tuple[int, int, int] = (255, 255, 255)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", _CARD_SIZE, color).save(path, format="JPEG", quality=95)


def _entry(scryfall_id: str, image_path: str) -> ManifestEntry:
    return ManifestEntry(
        scryfall_id=scryfall_id,
        image_path=image_path,
        lang="en",
        set_code="tst",
        collector_number="0001",
        rarity="common",
        type_line="Creature",
        layout="normal",
        finishes=["nonfoil"],
        image_sha256="ff" * 32,
    )


def _build_manifest(
    tmp_path: Path,
    n: int,
    *,
    missing_indices: set[int] | None = None,
) -> tuple[Path, Path]:
    """Create a manifest with ``n`` rows and write matching JPEGs under ``images_root``.

    Rows whose index is in ``missing_indices`` are still listed in the manifest
    but have no JPEG on disk.
    """
    missing_indices = missing_indices or set()
    images_root = tmp_path / "scryfall"
    manifest_path = images_root / "manifest.jsonl"
    for i in range(n):
        scryfall_id = f"card{i:04d}"
        image_rel = f"images/{scryfall_id[:2]}/{scryfall_id}.jpg"
        append_manifest_entry(manifest_path, _entry(scryfall_id, image_rel))
        if i not in missing_indices:
            _write_card(images_root / image_rel)
    return manifest_path, images_root


def test_dump_creates_expected_files(tmp_path: Path) -> None:
    manifest, images_root = _build_manifest(tmp_path, n=3)
    out = tmp_path / "crops"
    rc = main(
        [
            "--manifest",
            str(manifest),
            "--images-root",
            str(images_root),
            "--out",
            str(out),
        ]
    )
    assert rc == 0
    outputs = sorted(out.glob("*.jpg"))
    assert len(outputs) == 3
    for p in outputs:
        with Image.open(p) as img:
            assert img.mode == "RGB"
            assert img.size == (256, 48)


def test_dump_respects_limit(tmp_path: Path) -> None:
    manifest, images_root = _build_manifest(tmp_path, n=5)
    out = tmp_path / "crops"
    rc = main(
        [
            "--manifest",
            str(manifest),
            "--images-root",
            str(images_root),
            "--out",
            str(out),
            "--limit",
            "2",
        ]
    )
    assert rc == 0
    outputs = sorted(out.glob("*.jpg"))
    assert len(outputs) == 2
    # First N entries, not shuffled.
    names = {p.stem for p in outputs}
    assert names == {"card0000", "card0001"}


def test_dump_warns_on_missing_image(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # 2 rows, second is missing on disk.
    manifest, images_root = _build_manifest(tmp_path, n=2, missing_indices={1})
    out = tmp_path / "crops"
    rc = main(
        [
            "--manifest",
            str(manifest),
            "--images-root",
            str(images_root),
            "--out",
            str(out),
        ]
    )
    assert rc == 0
    captured = capsys.readouterr()
    assert "warning" in captured.err.lower()
    assert "card0001" in captured.err
    # Only the existing image was dumped.
    outputs = sorted(out.glob("*.jpg"))
    assert [p.stem for p in outputs] == ["card0000"]


def test_dump_summary_line_to_stderr(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    manifest, images_root = _build_manifest(tmp_path, n=2)
    out = tmp_path / "crops"
    rc = main(
        [
            "--manifest",
            str(manifest),
            "--images-root",
            str(images_root),
            "--out",
            str(out),
        ]
    )
    assert rc == 0
    captured = capsys.readouterr()
    assert f"dumped 2 crops to {out}" in captured.err


def test_dump_output_is_rgb_and_correct_size(tmp_path: Path) -> None:
    manifest, images_root = _build_manifest(tmp_path, n=1)
    out = tmp_path / "crops"
    rc = main(
        [
            "--manifest",
            str(manifest),
            "--images-root",
            str(images_root),
            "--out",
            str(out),
        ]
    )
    assert rc == 0
    (output,) = list(out.glob("*.jpg"))
    with Image.open(output) as img:
        assert img.mode == "RGB"
        assert img.size == (256, 48)


def test_dump_handles_non_rgb_source_jpeg(tmp_path: Path) -> None:
    """Some source JPEGs are grayscale ('L'); script should convert transparently."""
    images_root = tmp_path / "scryfall"
    manifest_path = images_root / "manifest.jsonl"
    entry = _entry("gray0001", "images/gr/gray0001.jpg")
    append_manifest_entry(manifest_path, entry)
    gray_path = images_root / entry.image_path
    gray_path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("L", _CARD_SIZE, 200).save(gray_path, format="JPEG", quality=95)

    out = tmp_path / "crops"
    rc = main(
        [
            "--manifest",
            str(manifest_path),
            "--images-root",
            str(images_root),
            "--out",
            str(out),
        ]
    )
    assert rc == 0
    (output,) = list(out.glob("*.jpg"))
    with Image.open(output) as img:
        assert img.mode == "RGB"
        assert img.size == (256, 48)


def test_dump_creates_out_dir_if_missing(tmp_path: Path) -> None:
    manifest, images_root = _build_manifest(tmp_path, n=1)
    out = tmp_path / "deeply" / "nested" / "crops"
    assert not out.exists()
    rc = main(
        [
            "--manifest",
            str(manifest),
            "--images-root",
            str(images_root),
            "--out",
            str(out),
        ]
    )
    assert rc == 0
    assert out.is_dir()
    assert len(list(out.glob("*.jpg"))) == 1


# Sanity: confirm we're actually reading back the right manifest shape so that
# a future refactor of the manifest format breaks this test loudly.
def test_manifest_fixture_roundtrip(tmp_path: Path) -> None:
    manifest, _ = _build_manifest(tmp_path, n=2)
    with manifest.open() as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    assert [row["scryfall_id"] for row in rows] == ["card0000", "card0001"]
