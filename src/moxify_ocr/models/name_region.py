"""Name-region model wiring: CRNN builder + reused CTC loss.

The same CRNN trunk that the bottom-region OCR uses, with two changes:

1. Wider input: ``(48, 512, 3)`` (vs. 256 wide) to fit the longer card-name
   strings. The time-axis output is ``T = W/4 = 128`` — well above the
   CTC alignment floor for our 31-char longest name.
2. Larger output alphabet: 98 Latin-script characters + 1 CTC blank = 99
   classes (vs. 45 for the bottom-region model).

The CTC loss in :mod:`moxify_ocr.models.bottom_region` is alphabet-agnostic
— blank index 0, sparse from-dense, time-major False — so we re-export it
here to make the entrypoint self-contained.
"""

from __future__ import annotations

from tensorflow.keras import Model

from moxify_ocr.data.name_alphabet import NAME_ALPHABET
from moxify_ocr.models.bottom_region import ctc_loss as ctc_loss
from moxify_ocr.models.crnn import build_crnn

NAME_NUM_CLASSES: int = len(NAME_ALPHABET) + 1  # +1 for the CTC blank.
NAME_INPUT_SHAPE: tuple[int, int, int] = (48, 512, 3)


def build_name_region_model() -> Model:
    """Return the v1 name-region CRNN (48x512 RGB → ``[B, 128, 99]`` logits).

    Reuses the v2 bottom-region trunk verbatim. Width is 2× the bottom-region
    model so we have 128 timesteps for sequences up to ~31 chars; class count
    is 99 to cover the Latin-script alphabet (see
    :mod:`moxify_ocr.data.name_alphabet`).
    """
    return build_crnn(
        input_shape=NAME_INPUT_SHAPE,
        num_classes=NAME_NUM_CLASSES,
        lstm_units=256,
    )
