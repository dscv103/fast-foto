import numpy as np
from scipy.interpolate import CubicSpline

from . import config


def _build_curve(y_points: np.ndarray) -> CubicSpline:
    return CubicSpline(config.MASTER_CURVE_X, y_points, bc_type="natural")


def apply_tone_curves(rgb_linear: np.ndarray) -> np.ndarray:
    r_curve = _build_curve(config.RED_CURVE_Y)
    g_curve = _build_curve(config.GREEN_CURVE_Y)
    b_curve = _build_curve(config.BLUE_CURVE_Y)
    master = _build_curve(config.MASTER_CURVE_Y)

    out = rgb_linear.astype(np.float32).copy()
    # apply per-channel curves then master
    out[:, :, 0] = r_curve(out[:, :, 0])
    out[:, :, 1] = g_curve(out[:, :, 1])
    out[:, :, 2] = b_curve(out[:, :, 2])

    out = master(np.clip(out, 0.0, 1.0))
    return np.clip(out, 0.0, 1.0)
