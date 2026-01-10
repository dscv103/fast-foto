import numpy as np

from . import config
from .config import LUMA_WEIGHTS


def detect_shadow_crush(rgb_linear: np.ndarray, threshold: float = None) -> dict:
    thr = config.SHADOW_CRUSH_THRESHOLD if threshold is None else threshold
    luminance = (
        LUMA_WEIGHTS[0] * rgb_linear[:, :, 0]
        + LUMA_WEIGHTS[1] * rgb_linear[:, :, 1]
        + LUMA_WEIGHTS[2] * rgb_linear[:, :, 2]
    )

    crushed = (luminance < thr).sum()
    total_pixels = luminance.size
    crushed_percent = (crushed / total_pixels) * 100.0
    detail_lost = crushed_percent > config.SHADOW_CRUSH_ACTION_THRESHOLD

    return {
        "crushed_pixels_percent": crushed_percent,
        "detail_lost": detail_lost,
        "recommendation": "Apply shadow lift" if detail_lost else "Shadows look good",
    }


def lift_shadows(rgb_linear: np.ndarray, strength: float = 0.5) -> np.ndarray:
    luminance = (
        LUMA_WEIGHTS[0] * rgb_linear[:, :, 0]
        + LUMA_WEIGHTS[1] * rgb_linear[:, :, 1]
        + LUMA_WEIGHTS[2] * rgb_linear[:, :, 2]
    )

    shadow_threshold = 0.25
    safe_lum = np.maximum(luminance, 0.01)
    lift_amount = np.where(
        luminance < shadow_threshold,
        strength * 0.25 * np.log1p(shadow_threshold / safe_lum),
        0.0,
    )
    lift_amount = np.clip(lift_amount, 0.0, 0.3 * strength)

    rgb_lifted = rgb_linear.astype(np.float32).copy()
    for c in range(3):
        rgb_lifted[:, :, c] = np.minimum(1.0, rgb_linear[:, :, c] + lift_amount)

    return rgb_lifted
