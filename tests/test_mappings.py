import numpy as np
from few.tests.base import FewTest
from few.utils.mappings.kerrecceq import kerrecceq_backward_map, kerrecceq_forward_map
from few.utils.mappings.pn import Y_to_xI, xI_to_Y
from few.utils.geodesic import get_separatrix

class MappingsTest(FewTest):
    @classmethod
    def name(self) -> str:
        return "Mappings"

    def test_kerrecceq_mapping(self):
        # test the mapping is invertible

        # regionA
        a = 0.3
        p = 8.0
        e = 0.4
        xI = 1.0

        for kind in ["flux", "amplitude"]:
            u, w, y, z = kerrecceq_forward_map(a, p, e, xI)
            a_new, p_new, e_new, xI_new = kerrecceq_backward_map(u, w, y, z)

            np.testing.assert_allclose(
                [a, p, e, xI],
                [a_new.item(), p_new.item(), e_new.item(), xI_new.item()],
                rtol=1e-10,
            )

        # regionB
        p = 45.0
        for kind in ["flux", "amplitude"]:
            u, w, y, z = kerrecceq_forward_map(a, p, e, xI, kind=kind)
            a_new, p_new, e_new, xI_new = kerrecceq_backward_map(
                u, w, y, z, regionA=False, kind=kind
            )

            np.testing.assert_allclose(
                [a, p, e, xI],
                [a_new.item(), p_new.item(), e_new.item(), xI_new.item()],
                rtol=1e-10,
            )

        # check the mapping at the inner p boundary for NaNs due to logarithmic scaling
        p = get_separatrix(a, e, xI) + 1e-3
        u, w, y, z = kerrecceq_forward_map(a, p, e, xI)
        self.assertFalse(np.isnan(u))

    def test_PN_mapping(self):
        # Check whether Y->xI and xI->Y are inverses
        a = 0.3
        p = 8.0
        e = 0.4
        Y = 0.3

        xI = Y_to_xI(a, p, e, Y)
        Y_new = xI_to_Y(a, p, e, xI)

        np.testing.assert_allclose(Y_new, Y, rtol=1e-10)

        # check that equatorial maps to equatorial
        Y = 1.0

        xI = Y_to_xI(a, p, e, Y)
        Y_new = xI_to_Y(a, p, e, xI)

        np.testing.assert_allclose(Y_new, Y, rtol=1e-10)
        np.testing.assert_allclose(xI, Y, rtol=1e-10)

        # check that polar maps to polar
        Y = 0.0

        xI = Y_to_xI(a, p, e, Y)

        np.testing.assert_allclose(xI, Y, rtol=1e-10)

        # check integer arguments doesn't break things
        np.testing.assert_allclose(
            Y_to_xI(0.3, 8, 0.4, 1), Y_to_xI(0.3, 8.0, 0.4, 1.0), rtol=1e-10
        )
