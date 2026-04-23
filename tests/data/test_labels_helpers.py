from __future__ import annotations

from typing import Any

from moxify_ocr.data.labels import RARITY_LETTER, _era_has_slash_total, _rarity_letter


def _card(
    *,
    rarity: str = "common",
    type_line: str = "Creature — Elf",
    released_at: str | None = "2024-02-09",
) -> dict[str, Any]:
    d: dict[str, Any] = {"rarity": rarity, "type_line": type_line}
    if released_at is not None:
        d["released_at"] = released_at
    return d


# Rarity tests
def test_rarity_letter_common() -> None:
    assert _rarity_letter(_card(rarity="common", type_line="Creature — Wizard")) == "C"


def test_rarity_letter_uncommon() -> None:
    assert _rarity_letter(_card(rarity="uncommon", type_line="Creature — Wizard")) == "U"


def test_rarity_letter_rare() -> None:
    assert _rarity_letter(_card(rarity="rare", type_line="Creature — Wizard")) == "R"


def test_rarity_letter_mythic() -> None:
    assert _rarity_letter(_card(rarity="mythic", type_line="Creature — Wizard")) == "M"


def test_rarity_letter_basic_land_returns_L() -> None:
    assert _rarity_letter(_card(rarity="common", type_line="Basic Land — Mountain")) == "L"


def test_rarity_letter_snow_covered_basic_land_still_L() -> None:
    # "Basic Snow Land — Plains" — the "Basic" keyword is what matters
    assert _rarity_letter(_card(rarity="common", type_line="Basic Snow Land — Plains")) == "L"


def test_rarity_letter_non_basic_land_not_L() -> None:
    # Dual lands, fetches, etc. — "Land — Island Mountain" NOT "Basic"
    assert _rarity_letter(_card(rarity="rare", type_line="Land — Island Mountain")) == "R"


def test_rarity_letter_special_returns_none() -> None:
    assert _rarity_letter(_card(rarity="special", type_line="Creature — Wizard")) is None


def test_rarity_letter_bonus_returns_none() -> None:
    assert _rarity_letter(_card(rarity="bonus", type_line="Instant")) is None


def test_rarity_letter_unknown_rarity_returns_none() -> None:
    assert _rarity_letter(_card(rarity="made_up", type_line="Creature — Elf")) is None


# Era tests
def test_era_has_slash_total_modern_yes() -> None:
    assert _era_has_slash_total(_card(released_at="2024-02-09")) is True


def test_era_has_slash_total_2008_boundary_yes() -> None:
    # Shards of Alara launched 2008-10-03, first set with /total printing
    assert _era_has_slash_total(_card(released_at="2008-10-03")) is True


def test_era_has_slash_total_2008_01_01_yes() -> None:
    # Cutoff: 2008-01-01 inclusive is the rule; verify the boundary
    assert _era_has_slash_total(_card(released_at="2008-01-01")) is True


def test_era_has_slash_total_2007_12_31_no() -> None:
    assert _era_has_slash_total(_card(released_at="2007-12-31")) is False


def test_era_has_slash_total_pre_2008_no() -> None:
    assert _era_has_slash_total(_card(released_at="2005-07-15")) is False


def test_era_has_slash_total_no_released_at_returns_false() -> None:
    assert _era_has_slash_total(_card(released_at=None)) is False


# Constant export test
def test_rarity_letter_constant_has_exactly_four_keys() -> None:
    assert set(RARITY_LETTER.keys()) == {"common", "uncommon", "rare", "mythic"}
    assert RARITY_LETTER == {"common": "C", "uncommon": "U", "rare": "R", "mythic": "M"}
