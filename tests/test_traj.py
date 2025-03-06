# python -m unittest few/tests/test_traj.py
import unittest
import numpy as np
import time
np.random.seed(42)
import matplotlib.pyplot as plt
from few.trajectory.inspiral import EMRIInspiral
from few.utils.constants import YRSID_SI
from few.trajectory.ode import KerrEccEqFlux, PN5, SchwarzEccFlux

from few.utils.globals import get_logger

few_logger = get_logger()

few_logger.warning("Traj Test is running")

insp_kw = {
    "T": 10.,
    "dt": 1.0,
    "err": 1e-9,
    "DENSE_STEPPING": 0,
    "buffer_length": int(1e4),
}



def run_forward_back(traj_module, M, mu, a, p0, e0, xI0, forwards_kwargs):
    """
    Run a trajectory forward, then run a trajectory backward from the finish point and return both result sets.
    """

    forwards_result = traj_module(M, mu, a, p0, e0, xI0, **forwards_kwargs)

    # Now test backwards integration
    final_p = forwards_result[1][-1]
    final_e = forwards_result[2][-1]
    final_x = forwards_result[3][-1]

    insp_kw_back = forwards_kwargs.copy()
    insp_kw_back.update({"integrate_backwards": True})
    insp_kw_back.update({"T": forwards_result[0][-1] / YRSID_SI})

    backwards_result = traj_module(M, mu, a, final_p, final_e, final_x, **insp_kw_back)

    return forwards_result, backwards_result

N_TESTS = 50

class ModuleTest(unittest.TestCase):
    def test_trajectory_pn5(self):
        few_logger.info("Testing pn5")
        # initialize trajectory class
        traj = EMRIInspiral(func=PN5)

        # set initial parameters
        M = 1e5
        mu = 1e1
        for i in range(N_TESTS):
            p0 = np.random.uniform(9.0, 15)
            e0 = np.random.uniform(0.0, 0.6)
            a = np.random.uniform(0.0, 1.0)
            x0 = np.random.uniform(-1.0, 1.0)

            # do not want to be too close to polar
            if np.abs(x0) < 1e-2:
                x0 = np.sign(x0) * 1e-2
            few_logger.info(f" Test {i}/{N_TESTS} with {p0=}, {e0=}, {a=} and {x0=}")
            forwards, backwards = run_forward_back(traj, M, mu, a, p0, e0, x0, forwards_kwargs=insp_kw)
            for i in range(1, 3):
                self.assertAlmostEqual(backwards[i][-1], forwards[i][0], places=3)

    def test_trajectory_kerr(self):
        few_logger.info("Testing kerr")
        # initialize trajectory class
        traj = EMRIInspiral(func=KerrEccEqFlux)

        # set initial parameters
        M = 1e5
        mu = 1e1
        for i in range(N_TESTS):
            p0 = np.random.uniform(9.0, 15)
            e0 = np.random.uniform(0.0, 0.6)
            a = np.random.uniform(0.0, 0.999)
            x0 = 1.0

            few_logger.info(f" Test {i}/{N_TESTS} with {p0=}, {e0=}, {a=} and {x0=}")
            forwards, backwards = run_forward_back(traj, M, mu, a, p0, e0, x0, forwards_kwargs=insp_kw)
            for i in range(1, 6):
                self.assertAlmostEqual(backwards[i][-1], forwards[i][0], places=3)

    def test_trajectory_schwarz(self):
        few_logger.info("Testing schwarz")
        # initialize trajectory class
        traj = EMRIInspiral(func=SchwarzEccFlux)

        # set initial parameters
        M = 1e5
        mu = 1e1
        for i in range(N_TESTS):
            p0 = np.random.uniform(9.0, 15)
            e0 = np.random.uniform(0.0, 0.6)
            a = 0.0
            x0 = 1.0

            few_logger.info(f" Test {i}/{N_TESTS} with {p0=}, {e0=}, {a=} and {x0=}")
            forwards, backwards = run_forward_back(traj, M, mu, a, p0, e0, x0, forwards_kwargs=insp_kw)
            for i in range(1, 6):
                self.assertAlmostEqual(backwards[i][-1], forwards[i][0], places=3)


    # def test_trajectory_KerrEccentricEquatorial(self):
        
    #     # test against Schwarz
    #     traj_Schw = EMRIInspiral(func=SchwarzEccFlux)
    #     a = 0.0
    #     M = 1e5
    #     mu = 1e1

    #     for flux_output_convention in ["ELQ", "pex"]:
    #         traj = EMRIInspiral(func=KerrEccEqFlux, flux_output_convention="ELQ")
    #         few_logger.info(f"testing kerr {flux_output_convention} against schwarzschild")
    #         for i in range(N_TESTS):
    #             p0 = np.random.uniform(9.0, 15)
    #             e0 = np.random.uniform(0.0, 0.5)
                
    #             tS, pS, eS, xS, Phi_phiS, Phi_thetaS, Phi_rS = traj_Schw(M, mu, 0.0, p0, e0, 1.0, **insp_kw)
    #             t, p, e, x, Phi_phi, Phi_theta, Phi_r = traj(M, mu, 0.0, p0, e0, 1.0, new_t=tS, upsample=True, **insp_kw)
                
    #             diff = np.abs(Phi_phi[-1] - Phi_phiS[-1])

    #             self.assertLess(np.max(diff), 10.0, msg=f"Failed for {p0=}, {e0=}, {diff=}")

    # def test_backwards_trajectory(self):
    #     # initialize trajectory class
    #     list_func = [
    #         PN5,
    #         SchwarzEccFlux,
    #         KerrEccEqFlux,
    #     ]
    #     for el in list_func:
    #         few_logger.info("testing backwards {}".format(el))
    #         traj = EMRIInspiral(func=el)

    #         # set initial parameters
    #         M = 1e6
    #         mu = 10.0

    #         # plt.figure()
    #         # tic = time.perf_counter()
    #         for i in range(N_TESTS):
    #             p0 = np.random.uniform(9.0, 12.0)
    #             e0 = np.random.uniform(0.1, 0.5)
    #             a = np.random.uniform(0.01, 0.98)

    #             if el is SchwarzEccFlux:
    #                 a = 0.0
    #                 x0 = 1.0
    #             elif el is KerrEccEqFlux:
    #                 x0 = 1.0
    #             else:
    #                 x0 = np.random.uniform(0.2, 0.8)

    #             # print(a,p0,e0)
    #             # run trajectory forwards
    #             insp_kw["T"] = 0.1
    #             t, p_forward, e_forward, x_forward, _, _, _ = traj(
    #                 M, mu, np.abs(a), p0, e0, x0, **insp_kw
    #             )

    #             p_final = p_forward[-1]
    #             e_final = e_forward[-1]
    #             x_final = x_forward[-1]

    #             # run trajectory backwards
    #             insp_kw_back = insp_kw.copy()
    #             insp_kw_back.update({"integrate_backwards": True})

    #             t, p_back, e_back, x_back, _, _, _ = traj(
    #                 M, mu, np.abs(a), p_final, e_final, x_final, **insp_kw_back
    #             )

    #             self.assertAlmostEqual(p_back[-1], p_forward[0], places=8)
    #             self.assertAlmostEqual(e_back[-1], e_forward[0], places=8)
    #             self.assertAlmostEqual(x_back[-1], x_forward[0], places=8)
