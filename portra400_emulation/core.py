from typing import Any, Dict, Tuple

import numpy as np

from preprocessing.color_space import linear_to_srgb, srgb_to_linear

from . import config
from .grain import add_grain
from .hsl import apply_hsl_sectors
from .tone_curves import apply_tone_curves

ArrayLike = np.ndarray


def apply_portra400_emulation(
    rgb_linear: ArrayLike,
    *,
    warm_shift: bool = config.DEFAULT_PARAMS["warm_shift"],
    hsl_adjust: bool = config.DEFAULT_PARAMS["hsl_adjust"],
    tone_curves: bool = config.DEFAULT_PARAMS["tone_curves"],
    global_saturation: float = config.DEFAULT_PARAMS["global_saturation"],
    grain: bool = config.DEFAULT_PARAMS["grain"],
    grain_seed: int | None = config.DEFAULT_PARAMS["grain_seed"],
) -> Tuple[ArrayLike, Dict[str, Any]]:
    meta: Dict[str, Any] = {}
    out = rgb_linear.astype(np.float32)

    if warm_shift:
        out = np.clip(out * config.WARM_MULTIPLIERS, 0.0, 1.0)
        meta["warm_shift"] = True
    else:
        meta["warm_shift"] = False

    if hsl_adjust:
        rgb_srgb = linear_to_srgb(np.clip(out, 0.0, 1.0))
        rgb_srgb = apply_hsl_sectors(rgb_srgb, config.HSL_SECTORS)
        out = srgb_to_linear(rgb_srgb)
        meta["hsl_adjust"] = True
    else:
        meta["hsl_adjust"] = False

    if tone_curves:
        out = apply_tone_curves(out)
        meta["tone_curves"] = True
    else:
        meta["tone_curves"] = False

    if global_saturation != 1.0:
        rgb_srgb = linear_to_srgb(np.clip(out, 0.0, 1.0))
        from .hsl import rgb_to_hsl, hsl_to_rgb

        h, s, l = rgb_to_hsl(rgb_srgb)
        s = np.clip(s * global_saturation, 0.0, 1.0)
        rgb_srgb = hsl_to_rgb(h, s, l)
        out = srgb_to_linear(rgb_srgb)
    meta["global_saturation"] = global_saturation

    if grain:
        out = add_grain(out, seed=grain_seed)
        meta["grain"] = {"enabled": True, "seed": grain_seed}
    else:
        meta["grain"] = {"enabled": False}

    out = np.clip(out, 0.0, 1.0)
    return out, meta
