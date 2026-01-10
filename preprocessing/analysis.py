import numpy as np

from .config import LUMA_WEIGHTS


def analyze_histogram(rgb_linear: np.ndarray) -> dict:
    luminance = (
        LUMA_WEIGHTS[0] * rgb_linear[:, :, 0]
        + LUMA_WEIGHTS[1] * rgb_linear[:, :, 1]
        + LUMA_WEIGHTS[2] * rgb_linear[:, :, 2]
    )

    brightness_mean = float(luminance.mean())
    brightness_median = float(np.median(luminance))
    brightness_min = float(luminance.min())
    brightness_max = float(luminance.max())

    shadows_ratio = float((luminance < 0.2).sum() / luminance.size)
    highlights_ratio = float((luminance > 0.8).sum() / luminance.size)

    histogram, _ = np.histogram(luminance, bins=256, range=(0.0, 1.0))

    return {
        "brightness_mean": brightness_mean,
        "brightness_median": brightness_median,
        "shadows_ratio": shadows_ratio,
        "highlights_ratio": highlights_ratio,
        "dynamic_range": brightness_max - brightness_min,
        "is_underexposed": shadows_ratio > 0.30,
        "is_overexposed": highlights_ratio > 0.20,
        "histogram": histogram,
        "brightness_min": brightness_min,
        "brightness_max": brightness_max,
    }
