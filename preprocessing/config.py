import numpy as np

# Luminance weights (BT.709)
LUMA_WEIGHTS = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)

# Exposure targets
TARGET_MEAN_LUMINANCE = 0.22
EV_CLAMP = (-2.0, 1.0)
HIGHLIGHT_PENALTY_THRESHOLD = 0.15
HIGHLIGHT_PENALTY_EV = 0.5
SHADOW_PENALTY_THRESHOLD = 0.40
SHADOW_PENALTY_EV = 0.3

# Soft clipping
SOFT_CLIP_THRESHOLD = 0.8
SOFT_CLIP_TOE_STRENGTH = 0.1

# Clipping detection
CLIP_TOLERANCE = 0.98
CLIP_ACTION_THRESHOLD = 5.0  # percent

# Shadow detection
SHADOW_CRUSH_THRESHOLD = 0.05
SHADOW_CRUSH_ACTION_THRESHOLD = 5.0  # percent

# Midtone normalization (lowered target to avoid overbrightening in linear space)
TARGET_MIDTONE = 0.18
MIDTONE_SCALE_CLAMP = (0.7, 1.4)

# Contrast (milder S-curve to avoid harsh contrast)
CONTRAST_CURVE_INPUT = np.array([0.0, 0.2, 0.5, 0.8, 1.0], dtype=np.float32)
CONTRAST_CURVE_OUTPUT = np.array([0.0, 0.12, 0.5, 0.88, 1.0], dtype=np.float32)
DEFAULT_CONTRAST_STRENGTH = 0.3

# Defaults for preprocess pipeline
DEFAULT_PARAMS = {
    "auto_exposure": True,
    "auto_white_balance": True,
    "highlight_recovery_strength": 0.7,
    "shadow_lift_strength": 0.4,
    "contrast_enhancement": 0.3,
    "noise_reduction": False,
    "verbose": False,
}

PORTRAIT_CONFIG = {
    "auto_exposure": True,
    "auto_white_balance": True,
    "highlight_recovery_strength": 0.8,
    "shadow_lift_strength": 0.5,
    "contrast_enhancement": 0.3,
    "noise_reduction": False,
}

LANDSCAPE_CONFIG = {
    "auto_exposure": True,
    "auto_white_balance": True,
    "highlight_recovery_strength": 0.6,
    "shadow_lift_strength": 0.3,
    "contrast_enhancement": 0.5,
    "noise_reduction": False,
}

HIGH_ISO_CONFIG = {
    "auto_exposure": True,
    "auto_white_balance": True,
    "highlight_recovery_strength": 0.5,
    "shadow_lift_strength": 0.6,
    "contrast_enhancement": 0.2,
    "noise_reduction": True,
}

PRESETS = {
    "portrait": PORTRAIT_CONFIG,
    "landscape": LANDSCAPE_CONFIG,
    "high-iso": HIGH_ISO_CONFIG,
}
