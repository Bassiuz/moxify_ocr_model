"""Tests for the synthetic bottom-region renderer."""

from __future__ import annotations

from collections import Counter

import numpy as np
import pytest

from moxify_ocr.data.synthetic import generate_synthetic_crop


def test_output_shape_and_dtype() -> None:
    image, label = generate_synthetic_crop(seed=0)
    assert image.dtype == np.uint8
    assert image.shape == (94, 268, 3)
    assert isinstance(label, str)
    assert len(label) > 0


def test_deterministic_given_seed() -> None:
    a_img, a_label = generate_synthetic_crop(seed=42)
    b_img, b_label = generate_synthetic_crop(seed=42)
    assert a_label == b_label
    assert np.array_equal(a_img, b_img)


def test_different_seeds_produce_different_output() -> None:
    a_img, a_label = generate_synthetic_crop(seed=1)
    b_img, b_label = generate_synthetic_crop(seed=2)
    assert a_label != b_label or not np.array_equal(a_img, b_img)


def test_background_is_dark() -> None:
    """The rendered background should mostly be near-black (<30 luminance)."""
    image, _ = generate_synthetic_crop(seed=7)
    lum = image.mean(axis=2)
    dark_frac = (lum < 30).mean()
    assert dark_frac > 0.7, f"expected >70% dark pixels, got {dark_frac:.2%}"


def test_text_is_light() -> None:
    """Some pixels should be near-white (text)."""
    image, _ = generate_synthetic_crop(seed=7)
    lum = image.mean(axis=2)
    light_frac = (lum > 200).mean()
    assert light_frac > 0.01, f"expected at least 1% light pixels, got {light_frac:.2%}"


def test_label_format_well_formed() -> None:
    """Label has exactly one \\n separating two lines, and line 2 has the lang code."""
    for seed in range(20):
        _, label = generate_synthetic_crop(seed=seed)
        assert label.count("\n") == 1, label
        line1, line2 = label.split("\n")
        # Line 2 must end with a known 2-char language code (uppercase)
        tail = line2.split(" ")[-1]
        assert len(tail) == 2 and tail.isupper(), f"bad lang tail in {label!r}"


def test_language_distribution_is_uniform_ish_over_many_samples() -> None:
    """Over many samples, we should see every language code we support, not just EN."""
    langs = Counter()
    for seed in range(500):
        _, label = generate_synthetic_crop(seed=seed)
        lang = label.split("\n")[1].split(" ")[-1]
        langs[lang] += 1
    # Expect at least 6 different languages across 500 samples (we have 13 supported).
    assert len(langs) >= 6, f"only saw languages {dict(langs)}"
    # English should NOT be ≥50% (it would be in a real-data sample).
    en_frac = langs["EN"] / sum(langs.values())
    assert en_frac < 0.25, f"EN should be uniformly ~1/13, got {en_frac:.2%}"


@pytest.mark.parametrize("seed", range(5))
def test_label_contains_only_alphabet_chars(seed: int) -> None:
    """Every synthetic label's characters are in the CTC alphabet."""
    from moxify_ocr.data.dataset import encode_label

    _, label = generate_synthetic_crop(seed=seed)
    # encode_label raises ValueError on any char not in ALPHABET.
    ids = encode_label(label)
    assert len(ids) == len(label)
