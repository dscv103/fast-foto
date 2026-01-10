import unittest
import numpy as np

from portra400_emulation.config import (
    MASTER_CURVE_Y,
    RED_CURVE_Y,
    GREEN_CURVE_Y,
    BLUE_CURVE_Y,
)


class TestPortraToneCurves(unittest.TestCase):
    """
    Regression test to ensure Portra 400 tone curves remain anchored at true black/white.
    
    This prevents accidental re-introduction of lifted black point / lowered white point
    (e.g., mapping 0→0.05 and 1→0.98), which causes flat-looking results with compressed
    dynamic range.
    """

    def test_master_curve_anchored_at_black(self):
        """MASTER_CURVE_Y should start at 0.0 (true black)."""
        self.assertTrue(
            np.isclose(MASTER_CURVE_Y[0], 0.0, atol=1e-6),
            f"MASTER_CURVE_Y[0] = {MASTER_CURVE_Y[0]} is not anchored at 0.0"
        )

    def test_master_curve_anchored_at_white(self):
        """MASTER_CURVE_Y should end at 1.0 (true white)."""
        self.assertTrue(
            np.isclose(MASTER_CURVE_Y[-1], 1.0, atol=1e-6),
            f"MASTER_CURVE_Y[-1] = {MASTER_CURVE_Y[-1]} is not anchored at 1.0"
        )

    def test_red_curve_anchored_at_black(self):
        """RED_CURVE_Y should start at 0.0 (true black)."""
        self.assertTrue(
            np.isclose(RED_CURVE_Y[0], 0.0, atol=1e-6),
            f"RED_CURVE_Y[0] = {RED_CURVE_Y[0]} is not anchored at 0.0"
        )

    def test_red_curve_anchored_at_white(self):
        """RED_CURVE_Y should end at 1.0 (true white)."""
        self.assertTrue(
            np.isclose(RED_CURVE_Y[-1], 1.0, atol=1e-6),
            f"RED_CURVE_Y[-1] = {RED_CURVE_Y[-1]} is not anchored at 1.0"
        )

    def test_green_curve_anchored_at_black(self):
        """GREEN_CURVE_Y should start at 0.0 (true black)."""
        self.assertTrue(
            np.isclose(GREEN_CURVE_Y[0], 0.0, atol=1e-6),
            f"GREEN_CURVE_Y[0] = {GREEN_CURVE_Y[0]} is not anchored at 0.0"
        )

    def test_green_curve_anchored_at_white(self):
        """GREEN_CURVE_Y should end at 1.0 (true white)."""
        self.assertTrue(
            np.isclose(GREEN_CURVE_Y[-1], 1.0, atol=1e-6),
            f"GREEN_CURVE_Y[-1] = {GREEN_CURVE_Y[-1]} is not anchored at 1.0"
        )

    def test_blue_curve_anchored_at_black(self):
        """BLUE_CURVE_Y should start at 0.0 (true black)."""
        self.assertTrue(
            np.isclose(BLUE_CURVE_Y[0], 0.0, atol=1e-6),
            f"BLUE_CURVE_Y[0] = {BLUE_CURVE_Y[0]} is not anchored at 0.0"
        )

    def test_blue_curve_anchored_at_white(self):
        """BLUE_CURVE_Y should end at 1.0 (true white)."""
        self.assertTrue(
            np.isclose(BLUE_CURVE_Y[-1], 1.0, atol=1e-6),
            f"BLUE_CURVE_Y[-1] = {BLUE_CURVE_Y[-1]} is not anchored at 1.0"
        )


if __name__ == "__main__":
    unittest.main()
