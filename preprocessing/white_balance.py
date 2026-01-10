import numpy as np


def auto_white_balance_gray_world(rgb_linear: np.ndarray) -> np.ndarray:
    r_mean = float(rgb_linear[:, :, 0].mean())
    g_mean = float(rgb_linear[:, :, 1].mean())
    b_mean = float(rgb_linear[:, :, 2].mean())

    average_mean = (r_mean + g_mean + b_mean) / 3.0

    r_gain = average_mean / r_mean if r_mean > 0.001 else 1.0
    g_gain = average_mean / g_mean if g_mean > 0.001 else 1.0
    b_gain = average_mean / b_mean if b_mean > 0.001 else 1.0

    rgb_wb = rgb_linear.astype(np.float32).copy()
    rgb_wb[:, :, 0] *= r_gain
    rgb_wb[:, :, 1] *= g_gain
    rgb_wb[:, :, 2] *= b_gain

    return np.clip(rgb_wb, 0.0, 1.0)


def auto_white_balance_white_patch(
    rgb_linear: np.ndarray, percentile: float = 99.5
) -> np.ndarray:
    r_max = float(np.percentile(rgb_linear[:, :, 0], percentile))
    g_max = float(np.percentile(rgb_linear[:, :, 1], percentile))
    b_max = float(np.percentile(rgb_linear[:, :, 2], percentile))

    r_scale = 1.0 / r_max if r_max > 0.001 else 1.0
    g_scale = 1.0 / g_max if g_max > 0.001 else 1.0
    b_scale = 1.0 / b_max if b_max > 0.001 else 1.0

    rgb_wb = rgb_linear.astype(np.float32).copy()
    rgb_wb[:, :, 0] *= r_scale
    rgb_wb[:, :, 1] *= g_scale
    rgb_wb[:, :, 2] *= b_scale

    return np.clip(rgb_wb, 0.0, 1.0)


def auto_white_balance_hybrid(rgb_linear: np.ndarray) -> np.ndarray:
    rgb_wp = auto_white_balance_white_patch(rgb_linear, percentile=98.0)
    rgb_gw = auto_white_balance_gray_world(rgb_wp)

    blend_factor = 0.7
    rgb_hybrid = (1.0 - blend_factor) * rgb_wp + blend_factor * rgb_gw
    return np.clip(rgb_hybrid, 0.0, 1.0)
