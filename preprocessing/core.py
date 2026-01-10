from typing import Any, Dict, Tuple, Union

import numpy as np

from .analysis import analyze_histogram
from .config import DEFAULT_PARAMS
from .contrast import enhance_contrast_mild, normalize_midtones
from .exposure import apply_exposure_correction, calculate_exposure_correction
from .highlights import detect_clipping, recover_highlight_detail
from .io import load_image_linear
from .shadows import detect_shadow_crush, lift_shadows
from .white_balance import auto_white_balance_hybrid
from . import config
from . import noise as noise_module


ArrayLike = np.ndarray


def preprocess_for_portra_emulation(
    input_data: Union[str, ArrayLike],
    *,
    auto_exposure: bool = DEFAULT_PARAMS["auto_exposure"],
    auto_white_balance: bool = DEFAULT_PARAMS["auto_white_balance"],
    highlight_recovery_strength: float = DEFAULT_PARAMS["highlight_recovery_strength"],
    shadow_lift_strength: float = DEFAULT_PARAMS["shadow_lift_strength"],
    contrast_enhancement: float = DEFAULT_PARAMS["contrast_enhancement"],
    noise_reduction: bool = DEFAULT_PARAMS["noise_reduction"],
    verbose: bool = DEFAULT_PARAMS["verbose"],
) -> Tuple[ArrayLike, Dict[str, Any]]:
    if verbose:
        print("[1/8] Loading image...")
    rgb_linear, load_meta = load_image_linear(input_data)

    if verbose:
        print("[2/8] Analyzing histogram...")
    hist_stats = analyze_histogram(rgb_linear)
    # ensure JSON-safe histogram
    hist_stats["histogram"] = hist_stats["histogram"].astype(int).tolist()
    if verbose:
        print(
            f"    Mean brightness: {hist_stats['brightness_mean']:.3f} | "
            f"Shadows: {hist_stats['shadows_ratio'] * 100:.1f}% | "
            f"Highlights: {hist_stats['highlights_ratio'] * 100:.1f}%"
        )

    if auto_exposure:
        if verbose:
            print("[3/8] Correcting exposure...")
        exp_corr = calculate_exposure_correction(hist_stats)
        rgb_linear = apply_exposure_correction(rgb_linear, exp_corr["exposure_factor"])
        if verbose:
            print(f"    EV correction: {exp_corr['ev_correction']:+.2f} stops")
    else:
        if verbose:
            print("[3/8] Skipping exposure correction")
        exp_corr = None

    if auto_white_balance:
        if verbose:
            print("[4/8] Correcting white balance...")
        rgb_linear = auto_white_balance_hybrid(rgb_linear)
    else:
        if verbose:
            print("[4/8] Skipping white balance")

    if verbose:
        print("[5/8] Recovering highlights...")
    clipping_info = detect_clipping(rgb_linear)
    if clipping_info["action_needed"] and highlight_recovery_strength > 0:
        rgb_linear = recover_highlight_detail(
            rgb_linear, strength=highlight_recovery_strength
        )
        if verbose:
            print(
                f"    Clipping: {clipping_info['overall_clipping_percent']:.1f}% | "
                f"Applied {highlight_recovery_strength:.1f} strength"
            )
    else:
        if verbose:
            print(
                f"    Clipping: {clipping_info['overall_clipping_percent']:.1f}% (OK)"
            )

    if verbose:
        print("[6/8] Lifting shadows...")
    shadow_info = detect_shadow_crush(rgb_linear)
    if shadow_info["detail_lost"] and shadow_lift_strength > 0:
        rgb_linear = lift_shadows(rgb_linear, strength=shadow_lift_strength)
        if verbose:
            print(
                f"    Crush: {shadow_info['crushed_pixels_percent']:.1f}% | "
                f"Applied {shadow_lift_strength:.1f} strength"
            )
    else:
        if verbose:
            print(f"    Crush: {shadow_info['crushed_pixels_percent']:.1f}% (OK)")

    if verbose:
        print("[7/8] Noise reduction...")
    if noise_reduction:
        rgb_linear = noise_module.reduce_noise_bilateral(rgb_linear)
    else:
        if verbose:
            print("    Skipped")

    if verbose:
        print("[8/8] Normalizing tones...")
    rgb_linear = normalize_midtones(rgb_linear)
    if contrast_enhancement > 0:
        rgb_linear = enhance_contrast_mild(rgb_linear, strength=contrast_enhancement)
        if verbose:
            print(f"    Applied {contrast_enhancement:.1f} contrast")

    metadata = {
        "load": load_meta,
        "histogram_stats": hist_stats,
        "exposure_correction": exp_corr,
        "clipping_info": clipping_info,
        "shadow_crush_info": shadow_info,
        "parameters_used": {
            "auto_exposure": auto_exposure,
            "auto_white_balance": auto_white_balance,
            "highlight_recovery_strength": highlight_recovery_strength,
            "shadow_lift_strength": shadow_lift_strength,
            "contrast_enhancement": contrast_enhancement,
            "noise_reduction": noise_reduction,
        },
    }

    return rgb_linear, metadata
