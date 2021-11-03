import unittest
import numpy as np
from models import rmspe


class TestRMSPE(unittest.TestCase):

    def test_rmspe_returns_0_for_identical_arrays(self):
        prediction = np.ones((5))
        target = np.ones((5))
        self.assertAlmostEqual(rmspe(prediction, target), 0.0)

    def test_rmspe_returns_100_for_different_by_1_array(self):
        prediction = np.ones((5))
        prediction *= 2
        target = np.ones((5))
        self.assertAlmostEqual(rmspe(prediction, target), 100.0)

    def test_rmspe_returns_44_for_1_in_5_different_by_1_array(self):
        prediction = np.ones((5))
        prediction[0] = 2
        target = np.ones((5))
        self.assertAlmostEqual(rmspe(prediction, target), 44.72135954999579)

    def test_rmspe_returns_numpy_float64(self):
        prediction = np.ones((5))
        target = np.ones((5))
        self.assertAlmostEqual(type(rmspe(prediction, target)), np.float64)


if __name__ == "__main__":
    unittest.main()
