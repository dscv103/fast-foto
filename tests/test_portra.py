import unittest
import numpy as np

from portra400_emulation.core import apply_portra400_emulation


class TestPortra400(unittest.TestCase):
    def test_tone_curve_preserves_full_range(self):
        ramp = np.linspace(0.0, 1.0, num=256, dtype=np.float32).reshape(16, 16, 1)
        img = np.repeat(ramp, 3, axis=2)
        out, meta = apply_portra400_emulation(img, grain=False)
        # Curves anchor at true black (0.0) and true white (1.0)
        self.assertLessEqual(out.min(), 0.01)
        self.assertGreaterEqual(out.max(), 0.99)

    def test_grain_deterministic_with_seed(self):
        img = np.full((8, 8, 3), 0.5, dtype=np.float32)
        out1, _ = apply_portra400_emulation(img, grain=True, grain_seed=42)
        out2, _ = apply_portra400_emulation(img, grain=True, grain_seed=42)
        self.assertTrue(np.allclose(out1, out2))


if __name__ == "__main__":
    unittest.main()
