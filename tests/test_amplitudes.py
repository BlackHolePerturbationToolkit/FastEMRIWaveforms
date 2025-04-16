import numpy as np

from few.amplitude.ampinterp2d import AmpInterpKerrEccEq
from few.tests.base import FewBackendTest

KERRECCEQ_AMP_TEST_POINTS = [
    {
        "lmn": (3, 2, 4),
        "ape": (0.94071396, 3.03167504, 0.23666111),
        "Almn": 0.002753336140076555 + 0.004143260615912325j,
    },
    {
        "lmn": (2, -2, 5),
        "ape": (0.77593827, 5.44830674, 0.39950367),
        "Almn": -0.0001975310651632842 + 4.806588136412258e-05j,
    },
    {
        "lmn": (2, 2, 0),
        "ape": (0.99090693, 1.81708312, 0.25313127),
        "Almn": -0.09712303568244392 + 0.0004771647068539275j,
    },
    {
        "lmn": (4, -3, 1),
        "ape": (0.87600927, 43.70360964, 0.478125),
        "Almn": -9.89242565478408e-07 + 1.750469365836169e-05j,
    },
    {
        "lmn": (3, 3, 0),
        "ape": (0.36158837, 72.94420732, 0.39375),
        "Almn": 5.869887391976143e-05 + 0.00136010620358342j,
    },
]


class AmplitudesTest(FewBackendTest):
    @classmethod
    def name(self) -> str:
        return "Amplitudes"

    @classmethod
    def parallel_class(self):
        return AmpInterpKerrEccEq

    def test_kerrecceq(self):
        amp_module = AmpInterpKerrEccEq(force_backend=self.backend)

        # test if amplitude generation gives same values for scalar vs array inputs
        a = 0.3
        p = 8
        e = 0.4
        xI = 1.0
        amplitudes = amp_module(a, p, e, xI)
        amplitudes_array = amp_module(a, np.array([p]), np.array([e]), np.array([xI]))
        if self.backend.uses_gpu:
            np.testing.assert_allclose(
                amplitudes.get(), amplitudes_array.get(), rtol=1e-10
            )
        else:
            np.testing.assert_allclose(amplitudes, amplitudes_array, rtol=1e-10)

        # test the amplitudes at some random points against their numerical values
        for test_point in KERRECCEQ_AMP_TEST_POINTS:
            a, p, e = test_point["ape"]
            mode_interp = amp_module(a, p, e, xI, specific_modes=[test_point["lmn"]])[
                test_point["lmn"]
            ].item()
        np.testing.assert_allclose(mode_interp, test_point["Almn"], rtol=0, atol=1e-9)

        # test the mode symmetry
        specific_modes = [(3, 2, 1), (3, -2, -1)]
        specific_amplitudes = amp_module(a, p, e, xI, specific_modes=specific_modes)
        if self.backend.uses_gpu:
            np.testing.assert_allclose(
                specific_amplitudes[(3, 2, 1)].get(),
                -specific_amplitudes[(3, -2, -1)].conj().get(),
                rtol=1e-10,
            )
        else:
            np.testing.assert_allclose(
                specific_amplitudes[(3, 2, 1)],
                -specific_amplitudes[(3, -2, -1)].conj(),
                rtol=1e-10,
            )
