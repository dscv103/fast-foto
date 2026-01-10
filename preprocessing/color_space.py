import numpy as np


def srgb_to_linear(rgb_srgb: np.ndarray) -> np.ndarray:
    """Convert sRGB to linear RGB (float32)."""
    rgb = rgb_srgb.astype(np.float32)
    return np.where(
        rgb <= 0.04045,
        rgb / 12.92,
        ((rgb + 0.055) / 1.055) ** 2.4,
    )


def linear_to_srgb(rgb_linear: np.ndarray) -> np.ndarray:
    """Convert linear RGB to sRGB (float32)."""
    rgb = rgb_linear.astype(np.float32)
    return np.where(
        rgb <= 0.0031308,
        rgb * 12.92,
        1.055 * np.power(rgb, 1 / 2.4) - 0.055,
    )
