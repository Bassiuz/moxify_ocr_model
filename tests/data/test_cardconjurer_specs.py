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
    # Set is a 2-4 letter uppercase string. Random generation, not a fixed list —
    # the model needs to learn character-level OCR rather than memorize a closed
    # vocabulary, so it generalizes to new sets it has never seen.
    assert 2 <= len(spec.info_set) <= 4
    assert spec.info_set.isupper()
    assert spec.info_set.isalpha()
    # Rarity is a single uppercase letter.
    assert spec.info_rarity in {"C", "U", "R", "M", "S", "P"}
    # Language is from our supported set.
    assert spec.info_language in {"EN", "DE", "FR", "ES", "IT", "PT", "JA", "KO", "ZH", "RU"}


def test_set_codes_are_diverse_across_specs() -> None:
    """Set codes must be drawn fresh each spec, not from a closed list.

    Regression test for v3 set_code_accuracy collapsing to ~0.7% on held-out
    sets. The fix is generating random alphabetic codes so the model never sees
    the same closed vocabulary twice.
    """
    specs = list(generate_specs(n=1000, seed=0))
    unique_set_codes = {s.info_set for s in specs}
    # With ~26 letters × 3-letter combinations there are ~17k possibilities;
    # 1000 draws should yield at least 500 unique codes.
    assert len(unique_set_codes) > 500, (
        f"only {len(unique_set_codes)} unique set codes in 1000 specs — "
        "set codes are still drawn from a small closed pool"
    )


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
    assert has_slash > 200
    # X-prefix is rare but must exist (>0, <10% so not over-represented).
    assert 0 < has_x < 50


def test_collector_numbers_include_hyphen_format() -> None:
    """Promo + masterpiece cards use NNNN-N or LLL-NN style numbers.

    v3 model failed on these (e.g. predicted "201" for ground-truth "2025-1").
    Spec generator must emit them at a non-trivial rate so the model learns
    that hyphens are valid in the collector-number position.
    """
    specs = list(generate_specs(n=2000, seed=0))
    has_hyphen = sum("-" in s.info_number for s in specs)
    # Realistic distribution: ~5% of cards have hyphenated collector numbers.
    # We want at least 50 examples in 2000 specs to ensure the model sees them.
    assert has_hyphen > 50, f"only {has_hyphen}/2000 collector numbers contain '-'"


def test_collector_numbers_include_alpha_suffix() -> None:
    """Variant cards use NNNA suffix style (e.g. 042A, 118Z).

    v3 model failed on these. Spec generator must emit uppercase suffix forms
    so the model learns to read trailing letters in the collector-number slot.
    """
    specs = list(generate_specs(n=2000, seed=0))
    # Match `<digits><uppercase letter>` at the end of the collector-number string,
    # without a slash before it (we don't want to count rarity letters).
    import re

    pattern = re.compile(r"^[0-9]+[A-Z]$")
    has_alpha_suffix = sum(1 for s in specs if pattern.match(s.info_number))
    assert has_alpha_suffix > 50, (
        f"only {has_alpha_suffix}/2000 collector numbers have alpha-suffix form"
    )


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
