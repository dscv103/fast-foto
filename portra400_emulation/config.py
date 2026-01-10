import numpy as np

# Warm shift multipliers (linear domain, milder to reduce white tint)
WARM_MULTIPLIERS = np.array([1.01, 1.0, 0.95], dtype=np.float32)

# HSL sector adjustments (degrees, scales)
HSL_SECTORS = [
    {
        "center": 120.0,
        "width": 30.0,
        "hue_shift": 20.0,
        "sat_scale": 0.80,
        "light_delta": 0.02,
    },
    {
        "center": 210.0,
        "width": 30.0,
        "hue_shift": -8.0,
        "sat_scale": 0.90,
        "light_delta": 0.01,
    },
    {
        "center": 30.0,
        "width": 35.0,
        "hue_shift": 0.0,
        "sat_scale": 1.10,
        "light_delta": 0.03,
    },
    {
        "center": 300.0,
        "width": 20.0,
        "hue_shift": 0.0,
        "sat_scale": 0.85,
        "light_delta": 0.0,
    },
]

# Master tone curve control points (x->y)
MASTER_CURVE_X = np.array([0.0, 0.10, 0.25, 0.50, 0.75, 0.90, 1.0], dtype=np.float32)
MASTER_CURVE_Y = np.array([0.0, 0.14, 0.32, 0.52, 0.74, 0.92, 1.0], dtype=np.float32)

# Per-channel curves
RED_CURVE_Y =   np.array([0.0, 0.16, 0.34, 0.54, 0.75, 0.92, 1.0], dtype=np.float32)
GREEN_CURVE_Y = np.array([0.0, 0.14, 0.32, 0.52, 0.74, 0.92, 1.0], dtype=np.float32)
BLUE_CURVE_Y =  np.array([0.0, 0.12, 0.29, 0.48, 0.70, 0.90, 1.0], dtype=np.float32)

GLOBAL_SAT_SCALE = 0.90

# Grain defaults
GRAIN_SIGMA = 0.01
GRAIN_BLUR_SIGMA = 2.5
GRAIN_CORRELATION = 0.3

DEFAULT_PARAMS = {
    "warm_shift": True,
    "hsl_adjust": True,
    "tone_curves": True,
    "global_saturation": GLOBAL_SAT_SCALE,
    "grain": False,
    "grain_seed": None,
}
