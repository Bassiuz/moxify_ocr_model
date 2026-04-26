"""Augmentation pipeline for bottom-region OCR crops.

Tuned for v3 synthetic-only training: pure white CardConjurer text → real
Scryfall ink-bleed/JPEG-y crops is a wider domain gap than the previous
real-only setup, so the pipeline is stronger on the photometric and geometric
axes most likely to bridge it (rotation, perspective, brightness, blur, JPEG
compression, plus a HueSaturationValue cast for color drift).

The pipeline is seeded per-call via :meth:`albumentations.Compose.set_random_seed`,
which is the supported path in albumentations >=2.x.
"""

from __future__ import annotations

import albumentations as A
import numpy as np

# Max gaussian noise σ in uint8 intensity units. Albumentations 2.x accepts
# ``std_range`` as a fraction of 255, so we convert. The first v3 pass at 12/255
# combined with low JPEG + heavy blur made 6 of 16 sample variants illegible;
# back off to a modest bump above the v2 baseline of 5/255.
_GAUSS_NOISE_MAX_UINT8: float = 7.0
_GAUSS_NOISE_STD_RANGE: tuple[float, float] = (0.0, _GAUSS_NOISE_MAX_UINT8 / 255.0)


def build_augmentation_pipeline(*, seed: int = 0) -> A.Compose:
    """Build an Albumentations pipeline tuned for 48x256 bottom-region crops.

    Geometric:
      - rotate ±4°
      - perspective warp scale 1-3.5%
      - affine scale 0.94-1.06

    Photometric:
      - brightness/contrast ±25%
      - hue/sat/val drift (p=0.4) — synthetic is pure white, real has color cast
      - gaussian noise σ ∈ [0, 7] (uint8 intensity units)
      - JPEG compression quality ∈ [70, 95]
      - gaussian blur σ ∈ [0, 1.1]
      - motion blur (p=0.15)

    Operates on uint8 HWC RGB images. Returns a :class:`A.Compose` whose per-call
    randomness is controlled via :func:`apply_augmentation`.
    """
    pipeline = A.Compose(
        [
            A.Affine(rotate=(-4, 4), scale=(0.94, 1.06), p=1.0),
            A.Perspective(scale=(0.01, 0.035), p=1.0),
            A.RandomBrightnessContrast(
                brightness_limit=0.25,
                contrast_limit=0.25,
                p=1.0,
            ),
            A.HueSaturationValue(
                hue_shift_limit=5,
                sat_shift_limit=10,
                val_shift_limit=10,
                p=0.4,
            ),
            A.GaussNoise(std_range=_GAUSS_NOISE_STD_RANGE, p=1.0),
            A.ImageCompression(quality_range=(70, 95), p=1.0),
            A.GaussianBlur(sigma_limit=(0.0, 1.1), p=1.0),
            A.MotionBlur(p=0.15),
        ],
        seed=seed,
    )
    return pipeline


def apply_augmentation(
    image: np.ndarray,
    pipeline: A.Compose,
    *,
    seed: int,
) -> np.ndarray:
    """Apply ``pipeline`` to ``image`` with a per-call seed.

    Re-seeds the pipeline (via ``set_random_seed``) so repeated calls with the
    same seed produce byte-identical output. Input and output are uint8 HWC RGB.
    """
    pipeline.set_random_seed(seed)
    out = pipeline(image=image)["image"]
    # Albumentations preserves dtype, but be defensive — CTC training requires uint8.
    if out.dtype != np.uint8:
        out = out.astype(np.uint8)
    assert isinstance(out, np.ndarray)
    return out
