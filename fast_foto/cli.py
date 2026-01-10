import argparse
import json
import os
from typing import Any, Dict

import numpy as np

from preprocessing import config as pre_config
from preprocessing.core import preprocess_for_portra_emulation
from preprocessing.io import prepare_srgb_output, save_png16
from portra400_emulation.core import apply_portra400_emulation


def _merge_params(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    out.update(override)
    return out


def _json_safe(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return obj


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fast Foto: preprocess + Portra 400 emulation"
    )
    parser.add_argument("input", help="Input image path (JPG/PNG/RAW/DNG)")
    parser.add_argument("--out", required=True, help="Output image path (PNG16)")
    parser.add_argument(
        "--preset", choices=list(pre_config.PRESETS.keys()), help="Preprocessing preset"
    )
    parser.add_argument(
        "--no-preprocess", action="store_true", help="Skip preprocessing stage"
    )
    parser.add_argument(
        "--no-portra", action="store_true", help="Skip Portra emulation stage"
    )
    parser.add_argument("--grain", action="store_true", help="Enable film grain")
    parser.add_argument("--seed", type=int, help="Random seed for grain")
    parser.add_argument(
        "--metadata-out", help="Path to write metadata JSON (default: OUT + .json)"
    )
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    pre_params = dict(pre_config.DEFAULT_PARAMS)
    if args.preset:
        pre_params = _merge_params(pre_params, pre_config.PRESETS[args.preset])
    pre_params["verbose"] = args.verbose

    # Run preprocessing
    metadata: Dict[str, Any] = {}
    if args.no_preprocess:
        from preprocessing.io import load_image_linear

        rgb_linear, load_meta = load_image_linear(args.input)
        metadata["preprocess"] = {"skipped": True, "load": load_meta}
    else:
        rgb_linear, pre_meta = preprocess_for_portra_emulation(args.input, **pre_params)
        metadata["preprocess"] = pre_meta

    # Portra emulation
    if args.no_portra:
        portra_meta = {"skipped": True}
    else:
        portra_out, portra_meta = apply_portra400_emulation(
            rgb_linear,
            grain=args.grain,
            grain_seed=args.seed,
        )
        rgb_linear = portra_out
    metadata["portra400"] = portra_meta

    # Convert and save sRGB PNG16
    rgb_srgb = prepare_srgb_output(rgb_linear)
    save_png16(args.out, rgb_srgb)

    # Write metadata
    metadata_path = args.metadata_out or f"{args.out}.json"
    metadata["output"] = {
        "path": args.out,
        "metadata_path": metadata_path,
        "format": "png16",
    }
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(_json_safe(metadata), f, indent=2)

    if args.verbose:
        print(f"Saved image to {args.out}")
        print(f"Saved metadata to {metadata_path}")


if __name__ == "__main__":  # pragma: no cover
    main()
