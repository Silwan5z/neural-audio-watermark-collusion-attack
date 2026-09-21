from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from registry import int_to_bits  # noqa: E402
from target_bit_margin import optimize_softmin  # noqa: E402


class PayloadAndOptimizerTest(unittest.TestCase):
    def test_payload_bits_are_lsb_first(self) -> None:
        self.assertEqual(int_to_bits(5, 4).tolist(), [1, 0, 1, 0])

    def test_optimizer_respects_weight_constraints(self) -> None:
        coalition = np.asarray([
            [0, 0, 0, 0],
            [1, 1, 0, 0],
            [1, 0, 1, 0],
            [1, 0, 0, 1],
            [0, 1, 1, 1],
        ], dtype=np.int8)
        target = np.asarray([1, 1, 1, 1], dtype=np.int8)
        weights, score, success = optimize_softmin(
            coalition, target, beta=8.0, minimum_effective_k=3.0,
            entropy_weight=0.05)
        self.assertTrue(success)
        self.assertTrue(np.all(weights >= 0.0))
        self.assertAlmostEqual(float(weights.sum()), 1.0, places=10)
        self.assertGreaterEqual(1.0 / float(np.sum(weights ** 2)),
                                3.0 - 1e-6)
        self.assertTrue(np.isfinite(score))


if __name__ == "__main__":
    unittest.main()
