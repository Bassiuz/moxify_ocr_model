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
# ``std_range`` as a fraction of 255, so we convert. Was 5/255; bump for the
# synthetic→real gap.
_GAUSS_NOISE_MAX_UINT8: float = 12.0
_GAUSS_NOISE_STD_RANGE: tuple[float, float] = (0.0, _GAUSS_NOISE_MAX_UINT8 / 255.0)


def build_augmentation_pipeline(*, seed: int = 0) -> A.Compose:
    """Build an Albumentations pipeline tuned for 48x256 bottom-region crops.

    Geometric:
      - rotate ±5°
      - perspective warp scale 1-5%
      - affine scale 0.92-1.08

    Photometric:
      - brightness/contrast ±30%
      - hue/sat/val drift (p=0.5) — synthetic is pure white, real has color cast
      - gaussian noise σ ∈ [0, 12] (uint8 intensity units)
      - JPEG compression quality ∈ [60, 95]
      - gaussian blur σ ∈ [0, 1.5]
      - motion blur (p=0.20)

    Operates on uint8 HWC RGB images. Returns a :class:`A.Compose` whose per-call
    randomness is controlled via :func:`apply_augmentation`.
    """
    pipeline = A.Compose(
        [
            A.Affine(rotate=(-5, 5), scale=(0.92, 1.08), p=1.0),
            A.Perspective(scale=(0.01, 0.05), p=1.0),
            A.RandomBrightnessContrast(
                brightness_limit=0.30,
                contrast_limit=0.30,
                p=1.0,
            ),
            A.HueSaturationValue(
                hue_shift_limit=8,
                sat_shift_limit=15,
                val_shift_limit=15,
                p=0.5,
            ),
            A.GaussNoise(std_range=_GAUSS_NOISE_STD_RANGE, p=1.0),
            A.ImageCompression(quality_range=(60, 95), p=1.0),
            A.GaussianBlur(sigma_limit=(0.0, 1.5), p=1.0),
            A.MotionBlur(p=0.20),
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
