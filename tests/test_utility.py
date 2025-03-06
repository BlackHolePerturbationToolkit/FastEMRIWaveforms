import unittest

from few.utils.utility import ELQ_to_pex,get_kerr_geo_constants_of_motion,get_mu_at_t,get_p_at_t,get_separatrix
from few.trajectory.inspiral import EMRIInspiral
from few.utils.constants import YRSID_SI
from few.trajectory.ode import KerrEccEqFlux

traj_module = EMRIInspiral(func=KerrEccEqFlux)

from few.utils.globals import get_logger, get_first_backend

few_logger = get_logger()

few_logger.warning(
    "Utility tests are running"
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

    def test_get_mu_at_t(self):
        m1 = 1e6

        a = 0.9
        p0 = 10.
        e0 = 0.3
        x0 = 1.

        traj_args = [m1, a, p0, e0, x0]
        traj_kwargs = {}
        index_of_mu = 1

        t_out = 1.

        m2 = get_mu_at_t(traj_module,t_out,traj_args,index_of_mu=index_of_mu,traj_kwargs=traj_kwargs)



        traj_args = [m1, m2, a, p0, e0, x0]

        t, p, e, xI, Phi_phi, Phi_theta, Phi_r = traj_module(*traj_args, T=t_out)
        diff = 1 - t[-1]/ (t_out*YRSID_SI)

        self.assertLess(abs(diff), 1e-10)

    def test_get_p_at_t(self):
        m1 = 1e6
        m2 = 10
        a = 0.9
        e0 = 0.3
        x0 = 1.

        traj_args = [m1,m2,a, e0, x0]
        traj_kwargs = {}
        index_of_p = 3

        t_out = 1.

        p0 = get_p_at_t(traj_module,t_out,traj_args,index_of_p=index_of_p,traj_kwargs=traj_kwargs)



        traj_args = [m1, m2, a, p0, e0, x0]

        t, p, e, xI, Phi_phi, Phi_theta, Phi_r = traj_module(*traj_args, T=t_out)
        diff = 1 - t[-1]/ (t_out*YRSID_SI)

        self.assertLess(abs(diff), 1e-10)
    def test_get_separatrix_generic(self):
        
            a= 0.9
            e = 0.3
            x = 0.5

            p_sep = get_separatrix(a, e, x)

            p_sep_KerrGeo = 4.100908189793339

            diff = p_sep_KerrGeo - p_sep

            self.assertLess(abs(diff), 1e-14)

    

    



