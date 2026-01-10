import unittest
import numpy as np

from portra400_emulation.core import apply_portra400_emulation


class TestPortra400(unittest.TestCase):
    def test_tone_curve_lifts_blacks(self):
        ramp = np.linspace(0.0, 1.0, num=256, dtype=np.float32).reshape(16, 16, 1)
        img = np.repeat(ramp, 3, axis=2)
        out, meta = apply_portra400_emulation(img, grain=False)
        self.assertGreaterEqual(out.min(), 0.049)
        self.assertLessEqual(out.max(), 0.981)

    def test_grain_deterministic_with_seed(self):
        img = np.full((8, 8, 3), 0.5, dtype=np.float32)
        out1, _ = apply_portra400_emulation(img, grain=True, grain_seed=42)
        out2, _ = apply_portra400_emulation(img, grain=True, grain_seed=42)
        self.assertTrue(np.allclose(out1, out2))


if __name__ == "__main__":
    unittest.main()
