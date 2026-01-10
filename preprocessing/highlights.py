import numpy as np

from . import config
from .config import LUMA_WEIGHTS


def detect_clipping(rgb_linear: np.ndarray, tolerance: float = None) -> dict:
    tol = config.CLIP_TOLERANCE if tolerance is None else tolerance
    clipped_r = (rgb_linear[:, :, 0] > tol).sum()
    clipped_g = (rgb_linear[:, :, 1] > tol).sum()
    clipped_b = (rgb_linear[:, :, 2] > tol).sum()
    total_pixels = rgb_linear.shape[0] * rgb_linear.shape[1]

    channel_clipping = {
        "red": (clipped_r / total_pixels) * 100.0,
        "green": (clipped_g / total_pixels) * 100.0,
        "blue": (clipped_b / total_pixels) * 100.0,
    }
    overall_clipping = max(channel_clipping.values())
    action_needed = overall_clipping > config.CLIP_ACTION_THRESHOLD

    return {
        "overall_clipping_percent": overall_clipping,
        "channel_clipping": channel_clipping,
        "action_needed": action_needed,
        "recommendation": "Apply highlight recovery"
        if action_needed
        else "No action needed",
    }


def recover_highlight_detail(
    rgb_linear: np.ndarray, strength: float = 1.0
) -> np.ndarray:
    luminance = (
        LUMA_WEIGHTS[0] * rgb_linear[:, :, 0]
        + LUMA_WEIGHTS[1] * rgb_linear[:, :, 1]
        + LUMA_WEIGHTS[2] * rgb_linear[:, :, 2]
    )
    threshold = 0.8
    highlight_mask = np.maximum(0.0, (luminance - threshold) / (1.0 - threshold))
    recovery = 1.0 - (strength * 0.3 * highlight_mask)

    rgb_recovered = rgb_linear.astype(np.float32).copy()
    for c in range(3):
        rgb_recovered[:, :, c] *= recovery

    return np.clip(rgb_recovered, 0.0, 1.0)
