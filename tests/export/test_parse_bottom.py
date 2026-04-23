from __future__ import annotations

from moxify_ocr.data.labels import PLANESWALKER_ICON
from moxify_ocr.export.parse_bottom import BottomRegionResult, parse_bottom

KNOWN_SETS = {"CLU", "M13", "MMQ", "TSB", "MB1", "CMD", "LEA", "PLST", "SLD"}
KNOWN_LANGS = {"EN", "DE", "FR", "IT", "ES", "PT", "JA", "RU", "KO", "ZH", "PH", "LA"}


def test_parse_modern_rare_english_nonfoil() -> None:
    result = parse_bottom(
        "0280/286 R\nCLU • EN",
        known_set_codes=KNOWN_SETS,
        known_languages=KNOWN_LANGS,
    )
    assert result.collector_number == "0280"
    assert result.set_total == 286
    assert result.rarity == "R"
    assert result.set_code == "CLU"
    assert result.language == "EN"
    assert result.foil_detected is False
    assert result.on_the_list_detected is False


def test_parse_modern_mythic_german_foil() -> None:
    result = parse_bottom(
        "0128/249 M\nM13 ★ DE",
        known_set_codes=KNOWN_SETS,
        known_languages=KNOWN_LANGS,
    )
    assert result.set_total == 249
    assert result.rarity == "M"
    assert result.foil_detected is True
    assert result.language == "DE"


def test_parse_basic_land_L() -> None:
    result = parse_bottom(
        "270/286 L\nCLU • EN",
        known_set_codes=KNOWN_SETS,
        known_languages=KNOWN_LANGS,
    )
    assert result.rarity == "L"


def test_parse_single_line_tolerated() -> None:
    # Same fields, newline replaced with space
    result = parse_bottom(
        "0280/286 R CLU • EN",
        known_set_codes=KNOWN_SETS,
        known_languages=KNOWN_LANGS,
    )
    assert result.collector_number == "0280"
    assert result.set_code == "CLU"


def test_parse_pre_2008_no_slash_total() -> None:
    result = parse_bottom(
        "128 C\nMMQ • EN",
        known_set_codes=KNOWN_SETS,
        known_languages=KNOWN_LANGS,
    )
    assert result.collector_number == "128"
    assert result.set_total is None
    assert result.rarity == "C"


def test_parse_the_list() -> None:
    input_str = f"{PLANESWALKER_ICON} 0041\nPLST ★ EN"
    result = parse_bottom(
        input_str,
        known_set_codes=KNOWN_SETS,
        known_languages=KNOWN_LANGS,
    )
    assert result.on_the_list_detected is True
    assert result.collector_number == "0041"
    assert result.set_code == "PLST"
    assert result.foil_detected is True


def test_parse_unknown_set_code_becomes_none() -> None:
    result = parse_bottom(
        "1 C\nZZZ • EN",
        known_set_codes=KNOWN_SETS,
        known_languages=KNOWN_LANGS,
    )
    assert result.set_code is None
    assert result.rarity == "C"


def test_parse_unknown_language_becomes_none() -> None:
    result = parse_bottom(
        "1 C\nCLU • XX",
        known_set_codes=KNOWN_SETS,
        known_languages=KNOWN_LANGS,
    )
    assert result.language is None


def test_parse_tolerates_extra_whitespace() -> None:
    result = parse_bottom(
        "   0280/286   R   CLU   •   EN   ",
        known_set_codes=KNOWN_SETS,
        known_languages=KNOWN_LANGS,
    )
    assert result.collector_number == "0280"
    assert result.language == "EN"


def test_parse_empty_string_all_none() -> None:
    result = parse_bottom(
        "",
        known_set_codes=KNOWN_SETS,
        known_languages=KNOWN_LANGS,
    )
    assert result.collector_number is None
    assert result.set_total is None
    assert result.rarity is None
    assert result.set_code is None
    assert result.language is None
    assert result.foil_detected is None
    assert result.on_the_list_detected is False


def test_parse_no_foil_glyph_leaves_foil_none() -> None:
    # Hypothetical: OCR missed the glyph
    result = parse_bottom(
        "0280/286 R\nCLU EN",
        known_set_codes=KNOWN_SETS,
        known_languages=KNOWN_LANGS,
    )
    assert result.foil_detected is None


def test_parse_preserves_raw() -> None:
    text = "0280/286 R\nCLU • EN"
    result = parse_bottom(
        text,
        known_set_codes=KNOWN_SETS,
        known_languages=KNOWN_LANGS,
    )
    assert result.raw == text


def test_parse_rarity_letter_not_confused_with_single_char_set() -> None:
    # If 'L' would parse as a 1-char set code (impossible — set codes are 2+), rarity wins
    result = parse_bottom(
        "270/286 L\nCLU • EN",
        known_set_codes=KNOWN_SETS,
        known_languages=KNOWN_LANGS,
    )
    assert result.rarity == "L"
    assert result.set_code == "CLU"


def test_parse_special_guest_SLD_set() -> None:
    # Secret Lair cards use 'SLD' — confirm vocab matching lets it through
    result = parse_bottom(
        "0127 C\nSLD • JA",
        known_set_codes=KNOWN_SETS,
        known_languages=KNOWN_LANGS,
    )
    assert result.set_code == "SLD"
    assert result.language == "JA"


def test_parse_mb1_3letter_set() -> None:
    result = parse_bottom(
        "45\nMB1 • EN",
        known_set_codes=KNOWN_SETS,
        known_languages=KNOWN_LANGS,
    )
    assert result.collector_number == "45"
    assert result.set_code == "MB1"


def test_parse_garbage_input_never_raises() -> None:
    # Completely garbled input — should return raw + mostly Nones
    garbage = "lkjsadflk $$$ !!! 💀💀💀"
    result = parse_bottom(
        garbage,
        known_set_codes=KNOWN_SETS,
        known_languages=KNOWN_LANGS,
    )
    assert result.raw == garbage
    assert result.collector_number is None
    assert result.set_total is None
    assert result.rarity is None
    assert result.set_code is None
    assert result.language is None
    assert result.foil_detected is None
    assert result.on_the_list_detected is False


def test_parse_returns_frozen_dataclass() -> None:
    # Dataclass is frozen — confirm we can't mutate it
    result = parse_bottom(
        "0280/286 R\nCLU • EN",
        known_set_codes=KNOWN_SETS,
        known_languages=KNOWN_LANGS,
    )
    assert isinstance(result, BottomRegionResult)
    try:
        result.collector_number = "9999"  # type: ignore[misc]
    except (AttributeError, Exception):
        pass
    else:
        raise AssertionError("BottomRegionResult should be frozen")


def test_parse_foil_bullet_after_star_does_not_flip() -> None:
    # If we see ★ first, a later • should NOT flip foil_detected back to False
    result = parse_bottom(
        "0280/286 R\nCLU ★ • EN",
        known_set_codes=KNOWN_SETS,
        known_languages=KNOWN_LANGS,
    )
    assert result.foil_detected is True


def test_parse_lowercase_set_code_rejected() -> None:
    # Our known_set_codes are uppercase; lowercase 'clu' should NOT match
    result = parse_bottom(
        "0280/286 R\nclu • EN",
        known_set_codes=KNOWN_SETS,
        known_languages=KNOWN_LANGS,
    )
    assert result.set_code is None


def test_parse_first_collector_number_wins() -> None:
    # If two bare digit tokens appear, only the first should become collector_number
    result = parse_bottom(
        "128 C MMQ 999 • EN",
        known_set_codes=KNOWN_SETS,
        known_languages=KNOWN_LANGS,
    )
    assert result.collector_number == "128"


def test_parse_slash_total_takes_priority_over_bare_digit() -> None:
    # If a N/T token appears first, collector = N, total = T
    result = parse_bottom(
        "0280/286 R\nCLU • EN",
        known_set_codes=KNOWN_SETS,
        known_languages=KNOWN_LANGS,
    )
    assert result.collector_number == "0280"
    assert result.set_total == 286
