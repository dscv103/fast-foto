import numpy as np
from scipy.interpolate import CubicSpline

from . import config


def normalize_midtones(rgb_linear: np.ndarray) -> np.ndarray:
    luminance = (
        config.LUMA_WEIGHTS[0] * rgb_linear[:, :, 0]
        + config.LUMA_WEIGHTS[1] * rgb_linear[:, :, 1]
        + config.LUMA_WEIGHTS[2] * rgb_linear[:, :, 2]
    )
    median_lum = float(np.median(luminance))
    if median_lum < 0.001:
        return rgb_linear.astype(np.float32)

    scale = config.TARGET_MIDTONE / median_lum
    scale = float(
        np.clip(scale, config.MIDTONE_SCALE_CLAMP[0], config.MIDTONE_SCALE_CLAMP[1])
    )

    rgb_normalized = rgb_linear.astype(np.float32) * scale
    return np.clip(rgb_normalized, 0.0, 1.0)


def enhance_contrast_mild(rgb_linear: np.ndarray, strength: float = None) -> np.ndarray:
    s = config.DEFAULT_CONTRAST_STRENGTH if strength is None else strength
    curve = CubicSpline(
        config.CONTRAST_CURVE_INPUT, config.CONTRAST_CURVE_OUTPUT, bc_type="natural"
    )

    rgb_contrast = rgb_linear.astype(np.float32).copy()
    for c in range(3):
        channel = rgb_contrast[:, :, c].ravel()
        channel_curved = curve(channel)
        channel_blended = (1.0 - s) * channel + s * channel_curved
        rgb_contrast[:, :, c] = channel_blended.reshape(rgb_contrast[:, :, c].shape)

    return np.clip(rgb_contrast, 0.0, 1.0)
