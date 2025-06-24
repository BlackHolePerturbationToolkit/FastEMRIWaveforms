import numpy as np

from few.tests.base import FewTest
from few.utils.mappings.pn import Y_to_xI, xI_to_Y


class PnMappingsTest(FewTest):
    @classmethod
    def name(self) -> str:
        return "PN Mappings"

    def test_back_and_forth_scalar_xI(self):
        # When a, p, and e are scalar values
        a = 0.5
        p = 10.0
        e = 0.0
        # Given a xI value in the range [-1, 1]
        xI_inputs = np.linspace(-1.0, 1.0, 10, endpoint=True)
        self.logger.info(f"{xI_inputs=}")
        # Let's map xI values to corresponding Y
        Y_inputs = np.array([xI_to_Y(a, p, e, xI) for xI in xI_inputs])
        self.logger.info(f"{Y_inputs=}")

        # Then map result Y back to xI
        xI_outputs = np.array([Y_to_xI(a, p, e, Y) for Y in Y_inputs])
        self.logger.info(f"{xI_outputs=}")
        # And check that resulting values are close to input ones
        np.testing.assert_allclose(xI_outputs, xI_inputs)

        # And map once again to Y
        Y_outputs = np.array([xI_to_Y(a, p, e, xI) for xI in xI_outputs])
        # And check values match
        np.testing.assert_allclose(Y_outputs, Y_inputs)
