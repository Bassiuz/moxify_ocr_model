from __future__ import annotations

from typing import Any

from moxify_ocr.data.labels import LANG_CODE, PLANESWALKER_ICON, make_label


def _card(
    *,
    collector_number: str,
    printed_size: int | None,
    rarity: str,
    set_code: str,
    lang: str,
    type_line: str,
    released_at: str,
) -> dict[str, Any]:
    d: dict[str, Any] = {
        "collector_number": collector_number,
        "set": set_code,  # NB: Scryfall key is "set" (the code), not "set_code"
        "lang": lang,
        "rarity": rarity,
        "type_line": type_line,
        "released_at": released_at,
    }
    if printed_size is not None:
        d["printed_size"] = printed_size
    return d


def test_make_label_modern_rare_english_nonfoil() -> None:
    card = _card(
        collector_number="0280",
        printed_size=286,
        rarity="rare",
        set_code="clu",
        lang="en",
        type_line="Land — Island Mountain",
        released_at="2024-02-09",
    )
    assert make_label(card, is_foil=False) == "0280/286 R\nCLU • EN"


def test_make_label_modern_mythic_german_foil() -> None:
    card = _card(
        collector_number="0128",
        printed_size=249,
        rarity="mythic",
        set_code="m13",
        lang="de",
        type_line="Creature — Dragon",
        released_at="2012-07-13",
    )
    assert make_label(card, is_foil=True) == "0128/249 M\nM13 ★ DE"


def test_make_label_basic_land_prints_L() -> None:
    card = _card(
        collector_number="270",
        printed_size=286,
        rarity="common",
        set_code="clu",
        lang="en",
        type_line="Basic Land — Mountain",
        released_at="2024-02-09",
    )
    assert make_label(card, is_foil=False) == "270/286 L\nCLU • EN"


def test_make_label_the_list_salvages_original_set_and_number() -> None:
    # PLST cards store collector_number like "KTK-86" — the card actually
    # shows KTK/86 with a planeswalker icon, NOT "PLST" or "KTK-86".
    card = _card(
        collector_number="KTK-86",
        printed_size=None,
        rarity="common",
        set_code="plst",
        lang="en",
        type_line="Sorcery",
        released_at="2019-11-07",
    )
    assert make_label(card, is_foil=False) == f"{PLANESWALKER_ICON} 86 C\nKTK • EN"


def test_make_label_the_list_foil_with_rarity() -> None:
    card = _card(
        collector_number="M20-195",
        printed_size=None,
        rarity="mythic",
        set_code="plst",
        lang="en",
        type_line="Creature — Dragon",
        released_at="2022-03-18",
    )
    assert make_label(card, is_foil=True) == f"{PLANESWALKER_ICON} 195 M\nM20 ★ EN"


def test_make_label_the_list_unrecognized_number_format_falls_through() -> None:
    # If collector_number doesn't match <SET>-<NUM>, keep raw fields + PW icon.
    card = _card(
        collector_number="42",  # no set prefix
        printed_size=None,
        rarity="rare",
        set_code="plst",
        lang="en",
        type_line="Creature",
        released_at="2022-03-18",
    )
    assert make_label(card, is_foil=False) == f"{PLANESWALKER_ICON} 42 R\nPLST • EN"


def test_make_label_pre_2008_no_slash_total() -> None:
    card = _card(
        collector_number="128",
        printed_size=None,
        rarity="common",
        set_code="mmq",
        lang="en",
        type_line="Creature — Goblin",
        released_at="1999-10-04",
    )
    assert make_label(card, is_foil=False) == "128 C\nMMQ • EN"


def test_make_label_special_rarity_omits_letter() -> None:
    # Time Spiral Timeshifted = rarity 'special' → no letter printed
    card = _card(
        collector_number="123",
        printed_size=None,
        rarity="special",
        set_code="tsb",
        lang="en",
        type_line="Creature — Wizard",
        released_at="2006-10-06",
    )
    assert make_label(card, is_foil=False) == "123\nTSB • EN"


def test_make_label_bonus_rarity_omits_letter() -> None:
    card = _card(
        collector_number="45",
        printed_size=None,
        rarity="bonus",
        set_code="mb1",
        lang="en",
        type_line="Instant",
        released_at="2021-11-19",
    )
    assert make_label(card, is_foil=False) == "45\nMB1 • EN"


def test_make_label_japanese_foil() -> None:
    card = _card(
        collector_number="0015",
        printed_size=286,
        rarity="rare",
        set_code="clu",
        lang="ja",
        type_line="Creature — Elf",
        released_at="2024-02-09",
    )
    assert make_label(card, is_foil=True) == "0015/286 R\nCLU ★ JA"


def test_make_label_zhs_maps_to_ZH() -> None:
    card = _card(
        collector_number="0015",
        printed_size=286,
        rarity="rare",
        set_code="clu",
        lang="zhs",
        type_line="Creature — Elf",
        released_at="2024-02-09",
    )
    assert make_label(card, is_foil=False) == "0015/286 R\nCLU • ZH"


def test_make_label_zht_maps_to_ZH() -> None:
    card = _card(
        collector_number="0015",
        printed_size=286,
        rarity="rare",
        set_code="clu",
        lang="zht",
        type_line="Creature — Elf",
        released_at="2024-02-09",
    )
    assert make_label(card, is_foil=False) == "0015/286 R\nCLU • ZH"


def test_make_label_modern_card_without_printed_size_drops_slash_total() -> None:
    # Modern era but set didn't print a size — just collector_number
    card = _card(
        collector_number="045",
        printed_size=None,
        rarity="uncommon",
        set_code="cmd",
        lang="en",
        type_line="Creature — Wizard",
        released_at="2011-06-17",
    )
    assert make_label(card, is_foil=False) == "045 U\nCMD • EN"


def test_make_label_plst_drops_printed_size_even_when_present() -> None:
    # PLST collector numbers don't encode /total even when a size is around.
    # Here the collector_number is the salvageable "KTK-86" format.
    card = _card(
        collector_number="KTK-86",
        printed_size=500,
        rarity="rare",
        set_code="plst",
        lang="en",
        type_line="Creature — Elf",
        released_at="2022-03-18",
    )
    # Salvaged: KTK set, 86 collector, rarity R, no /total.
    assert make_label(card, is_foil=False) == f"{PLANESWALKER_ICON} 86 R\nKTK • EN"


def test_make_label_uppercases_set_code() -> None:
    # Scryfall gives lowercase; we print uppercase
    card = _card(
        collector_number="1",
        printed_size=None,
        rarity="common",
        set_code="lea",
        lang="en",
        type_line="Creature — Goblin",
        released_at="1993-08-05",
    )
    assert make_label(card, is_foil=False) == "1 C\nLEA • EN"


def test_make_label_pre_2008_with_printed_size_still_no_slash_total() -> None:
    # Era check dominates: even if printed_size exists, pre-2008 cards never printed /total
    card = _card(
        collector_number="128",
        printed_size=350,
        rarity="rare",
        set_code="mmq",
        lang="en",
        type_line="Creature — Goblin",
        released_at="1999-10-04",
    )
    assert make_label(card, is_foil=False) == "128 R\nMMQ • EN"


def test_make_label_french_nonfoil() -> None:
    card = _card(
        collector_number="0200",
        printed_size=286,
        rarity="uncommon",
        set_code="clu",
        lang="fr",
        type_line="Creature — Elf",
        released_at="2024-02-09",
    )
    assert make_label(card, is_foil=False) == "0200/286 U\nCLU • FR"


def test_lang_code_has_all_expected_mappings() -> None:
    assert LANG_CODE == {
        "en": "EN",
        "de": "DE",
        "fr": "FR",
        "it": "IT",
        "es": "ES",
        "pt": "PT",
        "ja": "JA",
        "ru": "RU",
        "ko": "KO",
        "zhs": "ZH",
        "zht": "ZH",
        "ph": "PH",
        "la": "LA",
    }


def test_planeswalker_icon_is_private_use_area() -> None:
    # U+E100 is in the Private Use Area (U+E000–U+F8FF)
    assert PLANESWALKER_ICON == ""
    assert len(PLANESWALKER_ICON) == 1
    assert 0xE000 <= ord(PLANESWALKER_ICON) <= 0xF8FF
