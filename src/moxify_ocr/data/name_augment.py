"""Augmentation pipeline for synthetic name-region OCR crops.

Built specifically to bridge the synthetic-real gap measured on the v1
shipped model: 99.74% CER on real screen-captured frames, with the model
emitting mostly empty strings or single hallucinated capital letters
(textbook CTC-converged-into-blank-bias on out-of-distribution input).

Differs from :mod:`moxify_ocr.data.augment` (the bottom-region pipeline) in
emphasis: real-world failure was on **screen-photographed** card images
(phone camera looking at a desktop monitor), so we lean harder on photometric
distortions (glare gradients, color cast, JPEG, blur) and stronger
perspective skew than the bottom-region pipeline does. Geometric rotation
is kept modest because the upstream cropper has already de-warped the card
quad — extra rotation is mostly camera-induced sub-degree wobble.

Operates on uint8 HWC RGB images of shape (48, 512, 3).
"""

from __future__ import annotations

import albumentations as A
import numpy as np


def build_name_augmentation_pipeline(*, seed: int = 0) -> A.Compose:
    """Tuned for 48x512 name-region crops captured from screen-photo frames.

    Geometric:
      - rotate ±2° (sub-degree camera wobble)
      - perspective warp scale 0.5-4.5% (heavier than bottom-region; phone
        cameras photographing a screen rarely sit perfectly perpendicular)
      - affine scale 0.92-1.08

    Photometric (the bridge to real-world):
      - brightness/contrast ±35%
      - hue/sat/val cast — monitor white point ≠ training white (high p=0.7)
      - random shadow — simulates monitor glare strip across the crop
      - gauss noise σ ∈ [0, 9] (uint8) — camera ISP read noise
      - JPEG compression quality ∈ [55, 92] — camera + scan-loop transit
      - gaussian blur σ ∈ [0, 1.4] — camera focus + screen AA
      - motion blur (p=0.20) — handshake on phone capture
      - downscale 0.5-0.95 then upscale — simulates the on-screen card
        being smaller than the placement guide

    The blank-bias observed on v1 should weaken under augmentation: the model
    can no longer rely on "perfect synthetic background" as a signal, so the
    blank-when-uncertain shortcut is forced into competition with real glyph
    shape features.
    """
    pipeline = A.Compose(
        [
            A.Affine(rotate=(-2, 2), scale=(0.92, 1.08), p=1.0),
            A.Perspective(scale=(0.005, 0.045), p=1.0),
            A.RandomBrightnessContrast(
                brightness_limit=0.35,
                contrast_limit=0.35,
                p=1.0,
            ),
            A.HueSaturationValue(
                hue_shift_limit=8,
                sat_shift_limit=18,
                val_shift_limit=15,
                p=0.7,
            ),
            A.RandomShadow(
                shadow_roi=(0.0, 0.0, 1.0, 1.0),
                num_shadows_limit=(1, 2),
                shadow_dimension=4,
                shadow_intensity_range=(0.25, 0.55),
                p=0.35,
            ),
            A.GaussNoise(std_range=(0.0, 9.0 / 255.0), p=1.0),
            A.ImageCompression(quality_range=(55, 92), p=1.0),
            A.GaussianBlur(sigma_limit=(0.0, 1.4), p=1.0),
            A.MotionBlur(p=0.20),
            A.Downscale(scale_range=(0.5, 0.95), p=0.4),
        ],
        seed=seed,
    )
    return pipeline


def apply_name_augmentation(
    image: np.ndarray,
    pipeline: A.Compose,
    *,
    seed: int,
) -> np.ndarray:
    """Apply ``pipeline`` to ``image`` with a per-call seed for determinism."""
    pipeline.set_random_seed(seed)
    out = pipeline(image=image)["image"]
    if out.dtype != np.uint8:
        out = out.astype(np.uint8)
    assert isinstance(out, np.ndarray)
    return out
