from few.tests.base import FewTest
from few.trajectory.inspiral import EMRIInspiral
from few.trajectory.ode import KerrEccEqFlux
from few.utils.constants import YRSID_SI
from few.utils.utility import (
    get_m2_at_t,
    get_p_at_t,
)
from few.utils.geodesic import (
    ELQ_to_pex,
    get_fundamental_frequencies,
    get_kerr_geo_constants_of_motion,
    get_separatrix,   
)


def get_traj_module() -> EMRIInspiral:
    return EMRIInspiral(func=KerrEccEqFlux)


class UtilityTest(FewTest):
    @classmethod
    def name(self) -> str:
        return "Utility"

    def test_Constants_of_Motion(self):
        a = 0.9
        p = 10.0
        e = 0.3
        x = 0.5

        E, L, Q = get_kerr_geo_constants_of_motion(a, p, e, x)

        p_new, e_new, x_new = ELQ_to_pex(a, E, L, Q)

        self.assertLess(abs(p_new - p), 1e-13)
        self.assertLess(abs(e_new - e), 1e-13)
        self.assertLess(abs(x_new - x), 1e-13)

    def test_get_m2_at_t(self):
        m1 = 1e6

        a = 0.9
        p0 = 10.0
        e0 = 0.3
        x0 = 1.0

        traj_args = [m1, a, p0, e0, x0]
        traj_kwargs = {}
        index_of_m2 = 1

        t_out = 1.0

        traj_module = get_traj_module()

        m2 = get_m2_at_t(
            traj_module,
            t_out,
            traj_args,
            index_of_m2=index_of_m2,
            traj_kwargs=traj_kwargs,
        )

        traj_args = [m1, m2, a, p0, e0, x0]

        t, p, e, xI, Phi_phi, Phi_theta, Phi_r = traj_module(*traj_args, T=t_out)
        diff = 1 - t[-1] / (t_out * YRSID_SI)

        self.assertLess(abs(diff), 1e-10)

    def test_get_p_at_t(self):
        m1 = 1e6
        m2 = 10
        a = 0.9
        e0 = 0.3
        x0 = 1.0

        traj_args = [m1, m2, a, e0, x0]
        traj_kwargs = {}
        index_of_p = 3

        t_out = 1.0

        traj_module = get_traj_module()

        p0 = get_p_at_t(
            traj_module,
            t_out,
            traj_args,
            index_of_p=index_of_p,
            traj_kwargs=traj_kwargs,
        )

        traj_args = [m1, m2, a, p0, e0, x0]

        t, p, e, xI, Phi_phi, Phi_theta, Phi_r = traj_module(*traj_args, T=t_out)
        diff = 1 - t[-1] / (t_out * YRSID_SI)

        self.assertLess(abs(diff), 1e-10)

    def test_get_separatrix_generic(self):
        a = 0.9
        e = 0.3
        x = 0.5

        p_sep = get_separatrix(a, e, x)

        p_sep_KerrGeo = 4.100908189793339

        diff = p_sep_KerrGeo - p_sep

        self.assertLess(abs(diff), 1e-14)

    def test_get_separatrix_Schwarz(self):
        a = 0.0
        e = 0.5
        x = 0.5

        p_sep = get_separatrix(a, e, x)

        # p_sep_KerrGeo = 4.100908189793339

        diff = p_sep - (6.0 + 2.0 * e)

        self.assertLess(abs(diff), 1e-14)

    def test_get_fundamental_frequencies(self):
        a = 0.9
        e = 0.3
        p = 10.0
        x = 0.5

        OmegaPhi, OmegaTheta, OmegaR = get_fundamental_frequencies(a, p, e, x)

        # Taken from the KerrGeodesics Mathematica package
        OmegaPhiKerrGeo, OmegaThetaKerrGeo, OmegaRKerrGeo = (
            0.028478026708595002,
            0.027032450748133277,
            0.020053165349083846,
        )

        [
            abs(OmegaPhi - OmegaPhiKerrGeo),
            abs(OmegaTheta - OmegaThetaKerrGeo),
            abs(OmegaR - OmegaRKerrGeo),
        ]

        self.assertLess(abs(OmegaPhi - OmegaPhiKerrGeo), 1e-13)
        self.assertLess(abs(OmegaTheta - OmegaThetaKerrGeo), 1e-13)
        self.assertLess(abs(OmegaR - OmegaRKerrGeo), 1e-13)
