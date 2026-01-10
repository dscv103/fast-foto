import os
from typing import Tuple, Dict, Any, Union

import numpy as np
from PIL import Image

from .color_space import srgb_to_linear, linear_to_srgb

try:
    import rawpy
except ImportError:  # pragma: no cover
    rawpy = None


ArrayLike = np.ndarray


def _load_raw_linear(path: str) -> ArrayLike:
    if rawpy is None:
        raise ImportError("rawpy is required to load RAW/DNG files")

    with rawpy.imread(path) as raw:
        rgb = raw.postprocess(
            use_camera_wb=False,
            use_auto_wb=False,
            user_wb=(1.0, 1.0, 1.0, 1.0),
            output_color=rawpy.ColorSpace.sRGB,
            no_auto_bright=True,
            gamma=(1, 1),
            bright=1.0,
            output_bps=16,
            demosaic_algorithm=rawpy.DemosaicAlgorithm.AHD,
        )
    rgb_linear = (rgb.astype(np.float32)) / 65535.0
    return rgb_linear


def load_image_linear(
    path_or_array: Union[str, ArrayLike],
) -> Tuple[ArrayLike, Dict[str, Any]]:
    """Load image as linear RGB float32 in [0,1]."""
    if isinstance(path_or_array, np.ndarray):
        arr = path_or_array.astype(np.float32)
        return arr, {"source": "array", "shape": arr.shape}

    path = str(path_or_array)
    ext = os.path.splitext(path)[1].lower()

    if ext in {".dng", ".raw", ".cr2", ".nef", ".arw"}:
        rgb_linear = _load_raw_linear(path)
        source = "raw"
    else:
        img = Image.open(path).convert("RGB")
        rgb_srgb = np.array(img).astype(np.float32) / 255.0
        rgb_linear = srgb_to_linear(rgb_srgb)
        source = "srgb"

    meta = {
        "source": source,
        "path": path,
        "shape": rgb_linear.shape,
        "dtype": str(rgb_linear.dtype),
    }
    return rgb_linear, meta


def save_png16(path: str, rgb_srgb: ArrayLike) -> None:
    """Save sRGB float image to 16-bit PNG using Pillow raw encoder."""
    clipped = np.clip(rgb_srgb, 0.0, 1.0)
    quantized = np.round(clipped * 65535.0).astype(">u2")  # big-endian uint16
    h, w, _ = quantized.shape
    # Pillow raw encoder expects big-endian for RGB; use frombytes with raw mode
    img = Image.frombytes("RGB", (w, h), quantized.tobytes(), "raw", "RGB;16B")
    img.save(path, format="PNG")


def prepare_srgb_output(rgb_linear: ArrayLike) -> ArrayLike:
    """Convert linear to sRGB float32 and clamp."""
    rgb_srgb = linear_to_srgb(np.clip(rgb_linear, 0.0, 1.0))
    return np.clip(rgb_srgb, 0.0, 1.0)
