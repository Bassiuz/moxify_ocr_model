"""Tests for the random card-spec generator."""

from __future__ import annotations

from collections import Counter

from moxify_ocr.data.cardconjurer_specs import generate_specs, make_spec


def test_make_spec_is_deterministic_with_seed() -> None:
    a = make_spec(seed=42)
    b = make_spec(seed=42)
    assert a == b


def test_make_spec_produces_valid_label() -> None:
    spec = make_spec(seed=0)
    # Set is a 3-letter or 4-letter code from a known list.
    assert 2 <= len(spec.info_set) <= 5
    # Rarity is a single uppercase letter.
    assert spec.info_rarity in {"C", "U", "R", "M", "S", "P"}
    # Language is from our supported set.
    assert spec.info_language in {"EN", "DE", "FR", "ES", "IT", "PT", "JA", "KO", "ZH", "RU"}


def test_generate_specs_yields_n_specs() -> None:
    specs = list(generate_specs(n=100, seed=0))
    assert len(specs) == 100
    # Should be diverse — at least 3 different languages in 100 draws.
    langs = Counter(s.info_language for s in specs)
    assert len(langs) >= 3


def test_collector_number_formats_are_diverse() -> None:
    """Realistic distribution: most numeric, a few X-prefixed, some no /total."""
    specs = list(generate_specs(n=500, seed=0))
    has_slash = sum("/" in s.info_number for s in specs)
    has_x = sum(s.info_number.upper().startswith("X") for s in specs)
    # Majority should have /total (matches modern manifest distribution).
    assert has_slash > 250
    # X-prefix is rare but must exist (>0, <10% so not over-represented).
    assert 0 < has_x < 50


def test_foil_distribution_is_realistic() -> None:
    specs = list(generate_specs(n=1000, seed=0))
    foil_count = sum(s.foil for s in specs)
    # Real MTG distribution: foils are minority but not negligible.
    assert 100 < foil_count < 400


def test_generate_specs_with_adjacent_base_seeds_does_not_overlap() -> None:
    """Adjacent base seeds must produce essentially-disjoint streams.

    Regression test for a prior linear seed-mix bug where ``generate_specs(n, seed=0)``
    and ``generate_specs(n, seed=1)`` overlapped in 99% of specs.
    """
    a = list(generate_specs(n=100, seed=0))
    b = list(generate_specs(n=100, seed=1))
    overlap = sum(1 for spec_a, spec_b in zip(a, b, strict=True) if spec_a == spec_b)
    # Random chance of two specs being identical is roughly (1/N_sets) × (1/N_langs)
    # × (1/N_rarities) × (1/N_collectors) × 1/2 — vanishingly small. Allow up to 5
    # accidental collisions out of 100 to leave room for distribution-edge cases.
    assert overlap < 5, f"adjacent base seeds overlap {overlap}/100 — seed mixing is broken"
