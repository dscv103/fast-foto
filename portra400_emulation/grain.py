import numpy as np
from scipy.ndimage import gaussian_filter

from preprocessing.color_space import linear_to_srgb, srgb_to_linear
from . import config


def add_grain(rgb_linear: np.ndarray, seed=None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    rgb_srgb = linear_to_srgb(np.clip(rgb_linear, 0.0, 1.0))

    h, w, _ = rgb_srgb.shape
    base = rng.normal(0.0, config.GRAIN_SIGMA, size=(h, w))
    noise = np.stack(
        [
            config.GRAIN_CORRELATION * base
            + (1 - config.GRAIN_CORRELATION)
            * rng.normal(0.0, config.GRAIN_SIGMA, size=(h, w))
            for _ in range(3)
        ],
        axis=-1,
    )

    blurred = np.empty_like(noise)
    for c in range(3):
        blurred[:, :, c] = gaussian_filter(
            noise[:, :, c], sigma=config.GRAIN_BLUR_SIGMA
        )

    rgb_noisy = np.clip(rgb_srgb + blurred, 0.0, 1.0)
    return srgb_to_linear(rgb_noisy)
