import math
import unittest
import numpy as np

from preprocessing.analysis import analyze_histogram
from preprocessing.exposure import (
    apply_exposure_correction,
    calculate_exposure_correction,
)
from preprocessing.white_balance import auto_white_balance_hybrid
from preprocessing.highlights import detect_clipping, recover_highlight_detail
from preprocessing.shadows import detect_shadow_crush, lift_shadows
from preprocessing.contrast import normalize_midtones


class TestPreprocessing(unittest.TestCase):
    def test_exposure_correction_lifts_dark_image(self):
        img = np.full((10, 10, 3), 0.05, dtype=np.float32)
        stats = analyze_histogram(img)
        corr = calculate_exposure_correction(stats)
        adjusted = apply_exposure_correction(img, corr["exposure_factor"])
        self.assertGreater(adjusted.mean(), img.mean())

    def test_white_balance_reduces_cast(self):
        img = np.zeros((4, 4, 3), dtype=np.float32)
        img[:, :, 0] = 0.6  # red cast
        img[:, :, 1] = 0.3
        img[:, :, 2] = 0.2
        balanced = auto_white_balance_hybrid(img)
        r_mean, g_mean, b_mean = balanced.mean(axis=(0, 1))
        self.assertLess(abs(r_mean - g_mean), 0.1)
        self.assertLess(abs(b_mean - g_mean), 0.1)

    def test_highlight_recovery_reduces_clipping(self):
        img = np.full((10, 10, 3), 0.5, dtype=np.float32)
        img[:5, :5, :] = 1.05  # clipped area
        clip_info = detect_clipping(img)
        self.assertTrue(clip_info["action_needed"])
        recovered = recover_highlight_detail(img, strength=0.7)
        clip_info_after = detect_clipping(recovered)
        self.assertLessEqual(
            clip_info_after["overall_clipping_percent"],
            clip_info["overall_clipping_percent"],
        )

    def test_shadow_lift_reduces_crush(self):
        img = np.full((10, 10, 3), 0.01, dtype=np.float32)
        crush_info = detect_shadow_crush(img)
        self.assertTrue(crush_info["detail_lost"])
        lifted = lift_shadows(img, strength=0.5)
        crush_after = detect_shadow_crush(lifted)
        self.assertLess(
            crush_after["crushed_pixels_percent"], crush_info["crushed_pixels_percent"]
        )

    def test_normalize_midtone_moves_median(self):
        img = np.linspace(0.0, 1.0, num=100, dtype=np.float32).reshape(10, 10, 1)
        img = np.repeat(img, 3, axis=2)
        normalized = normalize_midtones(img)
        median_before = np.median(img[:, :, 0])
        median_after = np.median(normalized[:, :, 0])
        # median should move toward target (currently 0.18)
        # if median_before > target, median_after should be lower
        if median_before > 0.18:
            self.assertLess(median_after, median_before)
        else:
            self.assertGreater(median_after, median_before)


if __name__ == "__main__":
    unittest.main()
