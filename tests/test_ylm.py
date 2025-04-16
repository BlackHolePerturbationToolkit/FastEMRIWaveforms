import numpy as np

from few.tests.base import FewTest
from few.utils.ylm import GetYlms


class ModuleTest(FewTest):
    @classmethod
    def name(self) -> str:
        return "YLM"

    def testYlms(self):
        ylm = GetYlms()

        # int inputs
        self.assertAlmostEqual(
            ylm(2, 2, 1.0, 0.3), 0.30878955111651396 + 0.2112542979501157j
        )
        self.assertAlmostEqual(
            ylm(4, -3, 2.0, 0.1), 0.11564282752815112 - 0.03577251856181077j
        )

        # array inputs for l, m
        ylms = ylm(np.array([5, 4]), np.array([2, -1]), 1.0, 0.3)
        self.assertTrue(
            np.allclose(
                ylms,
                np.array(
                    [
                        -0.12267424266796564 - 0.08392596484459626j,
                        0.37975916907591606 - 0.11747327711681067j,
                    ]
                ),
            )
        )
