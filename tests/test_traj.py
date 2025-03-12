# python -m unittest few/tests/test_traj.py
import unittest
import numpy as np
import time
np.random.seed(42)
import matplotlib.pyplot as plt
from few.trajectory.inspiral import EMRIInspiral
from few.utils.constants import YRSID_SI, MTSUN_SI
from few.trajectory.ode import KerrEccEqFlux, PN5, SchwarzEccFlux
from few.utils.utility import get_separatrix
from few.utils.globals import get_logger
from scipy.integrate import solve_ivp
few_logger = get_logger()

few_logger.warning("Traj Test is running")

insp_kw = {
    "T": 10.,
    "dt": 0.01,
    "err": 1e-10,
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
            p0 = np.random.uniform(11, 15)
            e0 = np.random.uniform(0.0, 0.6)
            a = np.random.uniform(0.0, 1.0)
            x0 = np.random.uniform(-1.0, 1.0)
            # print(f"For {p0=}, {e0=}, {a=}, {x0=}")
            # do not want to be too close to polar
            if np.abs(x0) < 1e-1:
                x0 = np.sign(x0) * 1e-1
            few_logger.info(f" Test {i}/{N_TESTS} with {p0=}, {e0=}, {a=} and {x0=}")
            forwards, backwards = run_forward_back(traj, M, mu, a, p0, e0, x0, forwards_kwargs=insp_kw)
            for i in range(1, 3):
                self.assertAlmostEqual(backwards[i][-1], forwards[i][0], places=3, msg=f"Failed for {p0=}, {e0=}, {a=}, {x0=}")

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
                self.assertAlmostEqual(backwards[i][-1], forwards[i][0], places=3, msg=f"Failed for {p0=}, {e0=}, {a=}, {x0=}")

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
                self.assertAlmostEqual(backwards[i][-1], forwards[i][0], places=3, msg=f"Failed for {p0=}, {e0=}, {a=}, {x0=}")

    def test_scipy_solve_ivp_vs_trajectory_kerr(self):
        few_logger.info("Testing scipy solve_ivp vs trajectory kerr")
        # initialize trajectory class
        traj = EMRIInspiral(func=KerrEccEqFlux)

        for i in range(10):
            print(f"Test {i}/{N_TESTS}")
            # set initial parameters
            M = 1e6
            mu = 1e1
            a = np.random.uniform(0.0, 0.999)
            p0 = np.random.uniform(9.0, 15)
            e0 = np.random.uniform(0.0, 0.6)
            x0 = 1.0

            few_logger.info(f" Test {i}/{N_TESTS} with {p0=}, {e0=}, {a=} and {x0=}")

            # Define ODE flux and parameters
            ode_flux = KerrEccEqFlux()
            pars = [M, mu, a, p0, e0, x0]
            ode_flux.add_fixed_parameters(*pars[:3])

            def sep_stop(t, y):
                sep = get_separatrix(pars[2], y[1], y[2])
                return y[0] - sep - 2e-3

            sep_stop.terminal = True

            start_time_traj = time.time()
            forwards_result = traj(M, mu, a, p0, e0, x0, **insp_kw)
            end_time_traj = time.time()
            traj_duration = end_time_traj - start_time_traj
            print(f"Trajectory module duration: {traj_duration:.4f} seconds")

            # Solve using scipy solve_ivp
            final_t = pars[1] * forwards_result[0][-1]/ (pars[0]**2*MTSUN_SI) #pars[1] * 4. * YRSID_SI / (pars[0]**2*MTSUN_SI)
            start_time_scipy = time.time()
            res = solve_ivp(lambda t, y: ode_flux(y), (0, final_t), 
                np.asarray([pars[3], pars[4], pars[5], 0., 0., 0.]), atol=insp_kw["err"], rtol=0.0, method='DOP853', dense_output=True, events=sep_stop)
            end_time_scipy = time.time()
            scipy_duration = end_time_scipy - start_time_scipy
            print(f"Scipy solve_ivp duration: {scipy_duration:.4f} seconds")
            print(f"Ratio scipy/traj: {scipy_duration/traj_duration:.4f}")
            print(f"Number of steps scipy: {res.t.size}, Number of steps traj: {len(forwards_result[0])}, Ratio: {res.t.size/len(forwards_result[0]):.4f}")
            new_time = pars[1] * forwards_result[0]/ (pars[0]**2*MTSUN_SI)
            
            abs_diff_p = np.abs(res.sol(new_time)[0] - forwards_result[1])
            abs_diff_e = np.abs(res.sol(new_time)[1] - forwards_result[2])
            abs_diff_phi = np.abs(res.sol(new_time)[3]/(mu/M) - forwards_result[4])
            abs_diff_phir = np.abs(res.sol(new_time)[5]/(mu/M) - forwards_result[6])

            # plot delta p of the two integrators
            # plt.figure()
            # plt.plot(np.diff(res.y[0]), '.', label="scipy ")
            # plt.plot(np.diff(forwards_result[1]), '.', label="trajectory")
            # plt.legend()
            # plt.title('Delta p Comparison')
            # plt.savefig("delta_p.png")

            # plt.figure(figsize=(18, 12))

            # # First subplot with the two trajectories for Phi_phi
            # plt.subplot(3, 2, 1)
            # plt.plot(new_time, res.sol(new_time)[3]/(mu/M), label="scipy solve_ivp")
            # plt.plot(new_time, forwards_result[4],'--', label="trajectory")
            # plt.xlabel('Time')
            # plt.ylabel('Phi_phi')
            # plt.legend()
            # plt.title('Phi_phi Comparison')

            # # Second subplot with the absolute difference for Phi_phi
            # plt.subplot(3, 2, 2)
            # plt.plot(new_time, abs_diff_phi, label="Absolute Difference")
            # plt.xlabel('Time')
            # plt.ylabel('Absolute Difference')
            # plt.legend()
            # plt.title('Absolute Difference in Phi_phi')

            # # Third subplot with the two trajectories for p
            # plt.subplot(3, 2, 3)
            # plt.plot(new_time, res.sol(new_time)[0], label="scipy solve_ivp")
            # plt.plot(new_time, forwards_result[1],'--', label="trajectory")
            # plt.xlabel('Time')
            # plt.ylabel('p')
            # plt.legend()
            # plt.title('p Comparison')

            # # Fourth subplot with the absolute difference for p
            # plt.subplot(3, 2, 4)
            
            # plt.plot(new_time, abs_diff_p, label="Absolute Difference")
            # plt.xlabel('Time')
            # plt.ylabel('Absolute Difference')
            # plt.legend()
            # plt.title('Absolute Difference in p')

            # # Fifth subplot with the two trajectories for e
            # plt.subplot(3, 2, 5)
            # plt.plot(new_time, res.sol(new_time)[1], label="scipy solve_ivp")
            # plt.plot(new_time, forwards_result[2],'--', label="trajectory")
            # plt.xlabel('Time')
            # plt.ylabel('e')
            # plt.legend()
            # plt.title('e Comparison')

            # # Sixth subplot with the absolute difference for e
            # plt.subplot(3, 2, 6)
            # plt.plot(new_time, abs_diff_e, label="Absolute Difference")
            # plt.xlabel('Time')
            # plt.ylabel('Absolute Difference')
            # plt.legend()
            # plt.title('Absolute Difference in e')

            # plt.tight_layout()
            # plt.savefig("scipy_vs_traj.png")
            # plt.close()
            
            self.assertAlmostEqual(res.sol(new_time)[0][-1], forwards_result[1][-1], places=2, msg=f"Failed for scipy solve_ivp vs trajectory kerr with {p0=}, {e0=}, {a=}, {x0=}")
            self.assertAlmostEqual(res.sol(new_time)[1][-1], forwards_result[2][-1], places=2, msg=f"Failed for scipy solve_ivp vs trajectory kerr with {p0=}, {e0=}, {a=}, {x0=}")
            # self.assertAlmostEqual(res.y[3][-1]/(mu/M), forwards_result[4][-1], places=2, msg=f"Failed for scipy solve_ivp vs trajectory kerr with {p0=}, {e0=}, {a=}, {x0=}")
            # self.assertAlmostEqual(res.y[5][-1]/(mu/M), forwards_result[6][-1], places=2, msg=f"Failed for scipy solve_ivp vs trajectory kerr with {p0=}, {e0=}, {a=}, {x0=}")
            self.assertLess(abs_diff_phir[-1], 0.01, msg=f"Failed for scipy solve_ivp vs trajectory kerr with {p0=}, {e0=}, {a=}, {x0=}")
            self.assertLess(abs_diff_phir[-1], 0.01, msg=f"Failed for scipy solve_ivp vs trajectory kerr with {p0=}, {e0=}, {a=}, {x0=}")
            

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

# rhs = KerrEccEqFlux()
# M = 1e6
# mu = 1e2
# a = 0.7
# p0 = 10.0
# e0 = 0.8
# xI0 = 1.0  # +1 for prograde, -1 for retrograde inspirals
# rhs.add_fixed_parameters(M, mu, a)
# rhs([p0, e0, xI0], scale_by_eps=False)

# # Define the file path
# file_path = '../data_for_FEW/fluxes/a0.70_xI1.000.flux'
# scott_data = np.loadtxt(file_path)

# p = scott_data[:, 1]
# e = scott_data[:, 2]
# pdot_scott = scott_data[:, 14] + scott_data[:, 15]
# edot_scott = scott_data[:, 16] + scott_data[:, 17]

# plt.figure()
# for pp,ee,pdot_test,edot_test in zip(p, e,pdot_scott, edot_scott):
#     try:
#         pdot, edot = rhs([pp, ee, xI0], scale_by_eps=False)[:2]
#         rel_diff_pdot = np.abs(1-pdot_test/pdot)
#         # print("pdot relative difference",rel_diff_pdot)
#         if ee > 0.0:
#             rel_diff_edot = np.abs(1-edot_test/edot)
#             # print("edot relative difference",rel_diff_edot)
#         plt.semilogy(pp, rel_diff_pdot, 'r.')
#         plt.semilogy(pp, rel_diff_edot, 'b.')
#     except:
#         print(f"Out of bounds for {pp=}, {ee=}")
# plt.semilogy(pp, rel_diff_pdot, 'r.', label="edot")
# plt.semilogy(pp, rel_diff_edot, 'b.', label="pdot")
# plt.xlabel('p')
# plt.ylabel('Relative Difference')
# plt.title('Flux Comparison')
# plt.legend()
# plt.savefig("flux_comparison_p.png")

# plt.figure()
# for pp,ee,pdot_test,edot_test in zip(p, e,pdot_scott, edot_scott):
#     try:
#         pdot, edot = rhs([pp, ee, xI0], scale_by_eps=False)[:2]
#         rel_diff_pdot = np.abs(1-pdot_test/pdot)
#         # print("pdot relative difference",rel_diff_pdot)
#         if ee > 0.0:
#             rel_diff_edot = np.abs(1-edot_test/edot)
#             # print("edot relative difference",rel_diff_edot)
#         plt.semilogy(ee, rel_diff_pdot, 'r.')
#         plt.semilogy(ee, rel_diff_edot, 'b.')
#     except:
#         print(f"Out of bounds for {pp=}, {ee=}")
# plt.semilogy(ee, rel_diff_pdot, 'r.', label="edot")
# plt.semilogy(ee, rel_diff_edot, 'b.', label="pdot")
# plt.xlabel('e')
# plt.ylabel('Relative Difference')
# plt.title('Flux Comparison')
# plt.legend()
# plt.savefig("flux_comparison_e.png")
