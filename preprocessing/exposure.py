import numpy as np

from . import config
from .config import (
    TARGET_MEAN_LUMINANCE,
    HIGHLIGHT_PENALTY_THRESHOLD,
    SHADOW_PENALTY_THRESHOLD,
)


def calculate_exposure_correction(histogram_stats: dict) -> dict:
    brightness_mean = histogram_stats["brightness_mean"]
    highlights_ratio = histogram_stats["highlights_ratio"]
    shadows_ratio = histogram_stats["shadows_ratio"]

    target = TARGET_MEAN_LUMINANCE
    current_ratio = brightness_mean / target if brightness_mean > 0 else 1.0

    ev_correction = np.clip(
        np.log2(1.0 / current_ratio),
        config.EV_CLAMP[0],
        config.EV_CLAMP[1],
    )

    if highlights_ratio > HIGHLIGHT_PENALTY_THRESHOLD:
        ev_correction -= config.HIGHLIGHT_PENALTY_EV

    if shadows_ratio > SHADOW_PENALTY_THRESHOLD:
        ev_correction += config.SHADOW_PENALTY_EV

    exposure_factor = 2.0**ev_correction

    return {
        "ev_correction": float(ev_correction),
        "exposure_factor": float(exposure_factor),
        "reason": f"Current brightness: {brightness_mean:.3f}, Target: {target:.3f}",
    }


def soft_clip_highlights(
    rgb: np.ndarray, max_value: float = 1.0, toe_strength: float = None
) -> np.ndarray:
    toe = config.SOFT_CLIP_TOE_STRENGTH if toe_strength is None else toe_strength
    threshold = config.SOFT_CLIP_THRESHOLD
    rgb = rgb.astype(np.float32)
    mask = rgb > threshold
    out = rgb.copy()
    # Apply log-based toe only where above threshold
    out[mask] = threshold + toe * np.log1p((rgb[mask] - threshold) / toe)
    return np.clip(out, 0.0, max_value)


def apply_exposure_correction(
    rgb_linear: np.ndarray, exposure_factor: float
) -> np.ndarray:
    rgb_corrected = rgb_linear.astype(np.float32) * exposure_factor
    rgb_corrected = soft_clip_highlights(rgb_corrected)
    return np.clip(rgb_corrected, 0.0, 1.0)
