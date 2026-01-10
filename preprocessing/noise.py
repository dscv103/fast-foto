import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None


def reduce_noise_bilateral(
    rgb_linear: np.ndarray,
    kernel_size: int = 5,
    sigma_spatial: float = 10,
    sigma_range: float = 0.1,
) -> np.ndarray:
    if cv2 is None:
        raise ImportError("opencv-python is required for bilateral filtering")

    rgb_8bit = (np.clip(rgb_linear, 0.0, 1.0) * 255.0).astype(np.uint8)
    filtered = cv2.bilateralFilter(
        rgb_8bit,
        d=kernel_size,
        sigmaColor=sigma_range * 255.0,
        sigmaSpace=sigma_spatial,
    )
    return filtered.astype(np.float32) / 255.0
