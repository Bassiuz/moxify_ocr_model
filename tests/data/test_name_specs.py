"""Tests for the name-OCR spec generator."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from moxify_ocr.data.name_specs import (
    STYLE_WEIGHTS,
    STYLES,
    NameSpec,
    generate_specs,
    load_card_names,
    make_spec,
)

_SAMPLE_NAMES: list[str] = [
    "Lightning Bolt",
    "Tarmogoyf",
    "Lim-Dûl's Vault",
    "Æther Vial",
    "Will-o'-the-Wisp",
    "Dryad of the Ilysian Grove",
    "Asmoranomardicadaistinaculdacar",
    "Sol Ring",
    "Black Lotus",
    "Force of Will",
]


def test_make_spec_is_deterministic_with_seed() -> None:
    a = make_spec(names=_SAMPLE_NAMES, seed=42)
    b = make_spec(names=_SAMPLE_NAMES, seed=42)
    assert a == b


def test_make_spec_returns_name_from_pool() -> None:
    spec = make_spec(names=_SAMPLE_NAMES, seed=0)
    assert spec.name in _SAMPLE_NAMES


def test_make_spec_style_is_in_table() -> None:
    spec = make_spec(names=_SAMPLE_NAMES, seed=0)
    assert spec.style in STYLES


def test_make_spec_frame_color_is_valid() -> None:
    spec = make_spec(names=_SAMPLE_NAMES, seed=0)
    # WUBRGMAL = white/blue/black/red/green/multicolor/artifact/land
    assert spec.frame_color in set("WUBRGMAL")


def test_styles_table_matches_weights_keys() -> None:
    assert set(STYLES) == set(STYLE_WEIGHTS.keys())
    assert all(w > 0 for w in STYLE_WEIGHTS.values())


def test_generate_specs_yields_n_specs() -> None:
    specs = list(generate_specs(names=_SAMPLE_NAMES, n=200, seed=0))
    assert len(specs) == 200
    # Diversity checks: names, styles, and colors should all be exercised.
    names = Counter(s.name for s in specs)
    styles = Counter(s.style for s in specs)
    colors = Counter(s.frame_color for s in specs)
    assert len(names) >= 5  # at least half the sample names appear
    assert len(styles) >= 4  # multiple style buckets sampled
    assert len(colors) >= 5  # at least half the colors


def test_modern_style_dominates_distribution() -> None:
    """Modern (2015 frame) is ~70% of real cards; weights should reflect that."""
    specs = list(generate_specs(names=_SAMPLE_NAMES, n=2000, seed=0))
    modern_count = sum(1 for s in specs if s.style.startswith("modern"))
    # Allow generous slack — 50%+ is enough to confirm the prior is sane.
    assert modern_count > 1000


def test_namespec_immutable() -> None:
    """NameSpec is frozen — attempts to mutate fields raise."""
    import dataclasses

    spec = make_spec(names=_SAMPLE_NAMES, seed=0)
    try:
        spec.name = "different"  # type: ignore[misc]
    except dataclasses.FrozenInstanceError:
        return
    raise AssertionError("NameSpec should be frozen")


def test_load_card_names_filters_layouts_and_langs(tmp_path: Path) -> None:
    fake = [
        {"layout": "normal", "lang": "en", "name": "Lightning Bolt"},
        {"layout": "normal", "lang": "ja", "name": "稲妻"},  # CJK — drop
        {"layout": "token", "lang": "en", "name": "Goblin"},  # token — drop
        {"layout": "split", "lang": "en", "name": "Fire // Ice"},  # split into both
        {"layout": "saga", "lang": "en", "name": "The Eldest Reborn"},
        {"layout": "normal", "lang": "en", "name": "Has©glyph"},  # drop: © not in alphabet
    ]
    path = tmp_path / "scryfall.json"
    path.write_text(json.dumps(fake))
    names = load_card_names(path)
    assert "Lightning Bolt" in names
    assert "Fire" in names
    assert "Ice" in names
    assert "The Eldest Reborn" in names
    assert "稲妻" not in names
    assert "Goblin" not in names
    assert "Has©glyph" not in names


def test_mana_cost_distribution_is_realistic() -> None:
    """Roughly mimic real-card cost distribution: most have a cost, some don't,
    and lengths are mostly 1-4 symbols."""
    from collections import Counter

    specs = list(generate_specs(names=_SAMPLE_NAMES, n=2000, seed=0))
    no_cost = sum(1 for s in specs if not s.mana_cost)
    # ~15% are land-style with no cost; allow generous slack.
    assert 200 < no_cost < 500
    lengths = Counter(len(s.mana_cost) for s in specs if s.mana_cost)
    # Lengths 1-3 should be the bulk. 7+ rare but allowed.
    assert sum(lengths[k] for k in (1, 2, 3, 4)) > 0.6 * sum(lengths.values())


def test_mana_cost_uses_known_symbol_ids() -> None:
    """Every emitted symbol must be a valid m21-pack id."""
    valid = set("wubrg") | set("0123456789x")
    specs = list(generate_specs(names=_SAMPLE_NAMES, n=500, seed=0))
    for s in specs:
        for sym in s.mana_cost:
            assert sym in valid, f"unknown mana symbol id: {sym!r}"


def test_load_card_names_dedupes() -> None:
    """A name printed in multiple sets / languages must appear only once."""
    fake = [
        {"layout": "normal", "lang": "en", "name": "Sol Ring"},
        {"layout": "normal", "lang": "fr", "name": "Sol Ring"},  # same name, dedup
        {"layout": "normal", "lang": "en", "name": "Sol Ring"},
    ]
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
        json.dump(fake, f)
        f.flush()
        names = load_card_names(Path(f.name))
    assert names == ["Sol Ring"]
