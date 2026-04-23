"""Augmentation pipeline for bottom-region OCR crops.

Design doc §5.4 called for more aggressive augmentation; we've scaled back
because the crop is only 48x256 and aggressive geometric/photometric noise
destroys readability. The current parameters are a conservative starting
point — we can revisit once a baseline model exists.

The pipeline is seeded per-call via :meth:`albumentations.Compose.set_random_seed`,
which is the supported path in albumentations >=2.x.
"""

from __future__ import annotations

import albumentations as A
import numpy as np

# Max gaussian noise σ in uint8 intensity units. Albumentations 2.x accepts
# ``std_range`` as a fraction of 255, so we convert.
_GAUSS_NOISE_MAX_UINT8: float = 5.0
_GAUSS_NOISE_STD_RANGE: tuple[float, float] = (0.0, _GAUSS_NOISE_MAX_UINT8 / 255.0)


def build_augmentation_pipeline(*, seed: int = 0) -> A.Compose:
    """Build an Albumentations pipeline tuned for 48x256 bottom-region crops.

    Geometric:
      - rotate ±3°
      - perspective warp scale 1-3%
      - affine scale 0.95-1.05

    Photometric:
      - brightness/contrast ±20%
      - gaussian noise σ ∈ [0, 5] (uint8 intensity units)
      - JPEG compression quality ∈ [70, 95]
      - gaussian blur σ ∈ [0, 1.0]
      - motion blur (p=0.15)

    Operates on uint8 HWC RGB images. Returns a :class:`A.Compose` whose per-call
    randomness is controlled via :func:`apply_augmentation`.
    """
    pipeline = A.Compose(
        [
            A.Affine(rotate=(-3, 3), scale=(0.95, 1.05), p=1.0),
            A.Perspective(scale=(0.01, 0.03), p=1.0),
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=1.0,
            ),
            A.GaussNoise(std_range=_GAUSS_NOISE_STD_RANGE, p=1.0),
            A.ImageCompression(quality_range=(70, 95), p=1.0),
            A.GaussianBlur(sigma_limit=(0.0, 1.0), p=1.0),
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
