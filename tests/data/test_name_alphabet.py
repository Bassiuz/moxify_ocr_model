"""Tests for the name-OCR CTC alphabet."""

from __future__ import annotations

from moxify_ocr.data.dataset import decode_label, encode_label
from moxify_ocr.data.name_alphabet import NAME_ALPHABET


def test_alphabet_size() -> None:
    assert len(NAME_ALPHABET) == 98
    assert len(set(NAME_ALPHABET)) == 98


def test_alphabet_contains_expected_categories() -> None:
    # Every digit, every Latin letter (both cases), and the most common
    # diacritics must be present — these are non-negotiable.
    for ch in "0123456789":
        assert ch in NAME_ALPHABET
    for ch in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        assert ch in NAME_ALPHABET
    for ch in "abcdefghijklmnopqrstuvwxyz":
        assert ch in NAME_ALPHABET
    for ch in " ,'-":  # whitespace + the very-common name punctuation
        assert ch in NAME_ALPHABET
    for ch in "éóáíñûú":  # top-7 diacritics in real card names
        assert ch in NAME_ALPHABET


def test_alphabet_excludes_metadata_noise() -> None:
    # These characters appear in scryfall metadata but are not real card glyphs:
    # `_` and `|` are internal separators, `꞉` (U+A689) is a Unicode lookalike
    # for `:`, `®` only shows up once in a non-card line.
    for ch in "_|;®ꚉ":
        assert ch not in NAME_ALPHABET, f"{ch!r} should be excluded"


def test_alphabet_round_trips_via_encode_decode() -> None:
    sample = "Lim-Dûl's Vault"  # mixed case, apostrophe, hyphen, diacritic
    # Note: û is in NAME_ALPHABET. If this fails, the alphabet drifted.
    indices = encode_label(sample, alphabet=NAME_ALPHABET)
    decoded = decode_label(indices, alphabet=NAME_ALPHABET)
    assert decoded == sample


def test_alphabet_rejects_unknown_chars() -> None:
    import pytest

    with pytest.raises(ValueError, match="not in alphabet"):
        encode_label("草薙", alphabet=NAME_ALPHABET)  # CJK — out of scope for v1
