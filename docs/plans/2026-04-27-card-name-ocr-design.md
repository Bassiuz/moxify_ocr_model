# Card Name OCR — Design

**Goal:** synthetic-data pipeline + CRNN model for reading the *primary* card name from a pre-cropped name region. Mirrors the bottom-region pipeline architecturally; differs in renderer (PIL compositor of CardConjurer asset PNGs, no Playwright), alphabet (Latin-script names), input shape, and layout coverage.

**Scope (in):**

- Latin-script printings only: `en, es, fr, it, de, pt, la, ph`. CJK / Cyrillic deferred to v2.
- Layouts (Scryfall `layout` distribution from `data/scryfall/default-cards.json`, total 113 776 cards):
  - `normal` (104 413), `transform` (1 082, per face), `adventure` (407, primary name only),
    `saga` (364), `split` (345, per face — already rotated to horizontal upstream),
    `modal_dfc` (310, per face), `mutate` (146), `class` (66), `leveler` (63),
    `prototype` (47), `meld` (72, per face), plus **battle** layout (currently
    classified as `transform` in the dump; rotated upstream), **flip** and **aftermath**
    (primary name only — no second-name).
- Frame eras (Scryfall `frame`): `2015` (77 508), `2003` (18 079), `1997` (12 295),
  `1993` (5 653), `future` (241).
- Frame effects: `legendary, inverted, extendedart, showcase, enchantment, etched,
  fullart, snow, miracle, colorshifted` etc. — sampled by drawing from the
  CardConjurer frame-pack library.
- Frame treatments via CardConjurer asset packs: `m15, old, future, storybook,
  kaldheim, tarkir, neo, iko, enchantingTales, mysticalArchive, extended, borderless,
  etched, expedition, invocation, planeswalker, saga, adventure, akh, aftermath,
  flip, mid, snc, ravnica, dmu, mh2, snow, breakingNews, doubleFeature, dossier, lotr`.

**Scope (out):**

- CJK / Cyrillic printings (v2).
- Secret Lair custom borders.
- Second names on Adventure / Aftermath / Flip — caller takes a separate crop for those.
- Token / art_series / emblem / vanguard / scheme / planar — these aren't typical
  MTG playables and have idiosyncratic name regions.
- Online injection during training; we pre-render to disk like bottom-region.

## Architecture

**Renderer:** pure PIL, no Playwright. Per sample:

1. Sample a `NameSpec` (layout, frame-pack, font, name string, foil, art crop).
2. Composite a stand-in *full card*: random art behind, random frame PNG on top
   (loaded from `/tmp/cardconjurer-master/img/frames/<pack>/`). The frame PNGs
   already include the name-region typographic accents (legendary filigree,
   extended-art trim, etc.).
3. Draw the name with `PIL.ImageDraw` at the per-layout name-bbox coordinates
   using a font from `/tmp/cardconjurer-master/fonts/` (Beleren for 2015+,
   Goudy Medieval for 1993/1997, custom for showcase packs that ship one).
4. Crop the name strip.
5. For Battle / Saga-rotated / Split layouts whose name reads sideways: rotate
   90° to horizontal *here* (the model only ever sees horizontal text, per the
   contract with the upstream cropper).
6. Resize to `(48, 512, 3)`.
7. Save PNG + JSONL row.

**Specs generator:** mirrors [src/moxify_ocr/data/cardconjurer_specs.py](src/moxify_ocr/data/cardconjurer_specs.py).
Random sampling weighted by realistic distributions; names drawn from
`data/scryfall/default-cards.json` filtered to `layout in ALLOWED_LAYOUTS` and
`lang in ALLOWED_LANGS`. Each spec deterministic from a seed.

**Pool:** pre-rendered to `data/synth_names/v1/` matching the format produced by
[scripts/render_cardconjurer_pool.py](scripts/render_cardconjurer_pool.py)
(`images/{seed:08d}.png` + one `labels.jsonl`).

**Dataset reader:** clone of [src/moxify_ocr/data/cardconjurer_dataset.py](src/moxify_ocr/data/cardconjurer_dataset.py).

**Model:** clone of [src/moxify_ocr/models/bottom_region.py](src/moxify_ocr/models/bottom_region.py)
with input shape `(48, 512, 3)` and a names-specific alphabet. The CRNN trunk
in [crnn.py](src/moxify_ocr/models/crnn.py) is shape-flexible — only the input
height needs to match the existing time-axis-pooling design (height=48 ✓).

**Alphabet:** built once by enumerating characters that appear in Scryfall's
`name` and `printed_name` for the allowed-language slice. Saved to
`src/moxify_ocr/data/name_alphabet.py` as a frozen sorted string. Expected
~120 chars (A-Z, a-z, 0-9, common punctuation, Æ, accented vowels).

**Training:** new config `configs/name_v1.yaml` reusing the v3 training entrypoint.

## File layout

```
src/moxify_ocr/data/
  name_alphabet.py        # frozen Latin-script alphabet
  name_specs.py           # NameSpec + generate_specs(n, seed)
  name_dataset.py         # NamePool + sample_from_pool
  name_renderer.py        # PIL compositor (importable; the script wraps it)
src/moxify_ocr/models/
  name_region.py          # CRNN entrypoint for names
scripts/
  render_name_pool.py     # CLI wrapper around name_renderer
  build_name_alphabet.py  # one-shot: scan scryfall, freeze the alphabet
configs/
  name_v1.yaml
data/synth_names/v1/      # gitignored
  images/
  labels.jsonl
```

## Risks called out up-front

1. **Frame-pack coverage gaps.** CardConjurer's frame library is rich for
   modern + showcase but thin for some pre-modern eras (`old/abu`, `seventh`,
   `8th`). If the smoke contact sheet shows an era looking visually
   un-MTG-ish, we add hand-extracted backgrounds as a fallback.
2. **Name-region geometry varies per pack.** Each frame's name slot is in a
   slightly different y-position. We hand-tune a `name_bbox` per pack (a small
   table inside `name_renderer.py`). Smoke test exists to validate this.
3. **`(48, 512, 3)` may clip the longest names.** "Asmoranomardicadaistinaculdacar"
   at 31 chars * 17 px/char = 527 px — borderline. We track CER vs name-length
   and bump width to 576 if needed.
4. **Real foreign-language Latin-script printings are rare** (es=1207, fr=430,
   it=194, de=9, pt=3 in Scryfall). The name distribution is overwhelmingly
   English even when we sample broadly. This is fine — Latin-script names look
   the same regardless of language tag — but it does mean diacritic coverage
   has to be augmented (we'll synthesize names with random diacritic
   substitution at sample time).
5. **Image-reading hang.** Per memory: never `Read` an image. Inspect via
   PIL/numpy + a saved contact sheet, hand the path to the user.
6. **Multi-hour renders / trainings.** Per memory: print the command and stop;
   the user runs it.

## Out of scope (deliberately)

- Token / emblem / planar / scheme / vanguard / art_series layouts.
- Foreign-script printings (CJK, Cyrillic, Hebrew, etc.).
- Adventure subname / Aftermath bottom name / Flip rotated name — primary
  only. Caller crops the secondary names separately and (for v1) feeds them
  to the same model after rotation, accepting whatever degraded CER results.
- Reading anything besides the name (mana cost, art credit, set symbol).

## Acceptance criteria

The plan is complete when:

1. `scripts/render_name_pool.py --n 100 --out-dir artifacts/name_smoke_pool` produces 100 PNGs + a contact sheet, and the user signs off on visual realism.
2. `data/synth_names/v1/` contains ≥49 500 valid PNGs after the user kicks off the full render.
3. A v1 training run hits val_cer ≤ some target the user picks after seeing initial loss curves.
4. All tests in `tests/data/test_name_*.py` pass.

## Implementation order

Each task is a separate commit. TDD where it makes sense (modules with logic);
smoke-test instead of unit-test for one-shot scripts.

1. Build the name alphabet from real Scryfall data — one-shot script,
   commits a frozen string constant.
2. `name_specs.py` + tests — random spec generator.
3. `name_renderer.py` + a smoke driver — PIL compositor; visual smoke test
   (the user inspects the contact sheet).
4. `name_dataset.py` + tests — pool reader.
5. `name_region.py` — model entrypoint, mirrors `bottom_region.py`.
6. `configs/name_v1.yaml` — training config; hand off the 50K render and the
   training run to the user (multi-hour each).
