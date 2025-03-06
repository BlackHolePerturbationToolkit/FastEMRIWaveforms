import unittest

from few.utils.utility import ELQ_to_pex,get_kerr_geo_constants_of_motion

from few.utils.globals import get_logger, get_first_backend

few_logger = get_logger()

few_logger.warning(
    "Utility test is running"
)
class UtilityTest(unittest.TestCase):
    def test_Constants_of_Motion(self):
        a = 0.9
        p = 10.
        e = 0.3
        x = 0.5

        E, L, Q = get_kerr_geo_constants_of_motion(a, p, e, x)

        p_new, e_new, x_new = ELQ_to_pex(a, E, L, Q)

        self.assertLess(abs(p_new - p), 1e-13)
        self.assertLess(abs(e_new - e), 1e-13)
        self.assertLess(abs(x_new - x), 1e-13)

    



