# python -m unittest few/tests/test_traj.py
import time

import numpy as np
from scipy.integrate import solve_ivp

from few.tests.base import FewTest, tagged_test
from few.trajectory.inspiral import EMRIInspiral
from few.trajectory.ode import PN5, KerrEccEqFlux, SchwarzEccFlux
from few.utils.constants import MTSUN_SI, YRSID_SI
from few.utils.geodesic import get_separatrix

np.random.seed(42)

insp_kw = {
    "T": 10.0,
    "dt": 0.01,
    "err": 1e-10,
    "DENSE_STEPPING": 0,
    "buffer_length": int(1e4),
}


def compute_traj_1_2(traj_1, traj_2, m1, m2, a, p0, e0, xI0, T=0.01):
    """
    Inputs: primary mass M, secondary mass mu, primary spin a, eccentricity e0,
            observation time T (optional)

    outputs: two separate trajectories from two specified trajectory modules
    """

    # Compute trajectories for ELQ and pex
    out_1 = traj_1(m1, m2, a, p0, e0, 1.0, T=T)  # trajectory module 1
    out_2 = traj_2(
        m1, m2, a, p0, e0, 1.0, T=T, new_t=out_1[0], upsample=True
    )  # trajectory module 2
    # NOTE: using out_1 time array.

    return out_1, out_2


def run_forward_back(traj_module, m1, m2, a, p0, e0, xI0, forwards_kwargs):
    """
    Run a trajectory forward, then run a trajectory backward from the finish point and return both result sets.
    """

    forwards_result = traj_module(m1, m2, a, p0, e0, xI0, **forwards_kwargs)

    # Now test backwards integration
    final_p = forwards_result[1][-1]
    final_e = forwards_result[2][-1]
    final_x = forwards_result[3][-1]

    insp_kw_back = forwards_kwargs.copy()
    insp_kw_back.update({"integrate_backwards": True})
    insp_kw_back.update({"T": forwards_result[0][-1] / YRSID_SI})

    backwards_result = traj_module(m1, m2, a, final_p, final_e, final_x, **insp_kw_back)

    return forwards_result, backwards_result


N_TESTS = 10  # Perform 10 tests.


class ModuleTest(FewTest):
    @classmethod
    def name(self) -> str:
        return "Traj"

    def test_trajectory_pn5(self):
        self.logger.info("Testing pn5")
        # initialize trajectory class
        traj = EMRIInspiral(func=PN5)

        # set initial parameters
        m1 = 1e5
        m2 = 1e1
        for i in range(N_TESTS):
            p0 = np.random.uniform(11, 15)
            e0 = np.random.uniform(0.0, 0.6)
            a = np.random.uniform(0.0, 1.0)
            x0 = np.random.uniform(-1.0, 1.0)
            # print(f"For {p0=}, {e0=}, {a=}, {x0=}")
            # do not want to be too close to polar
            if np.abs(x0) < 1e-1:
                x0 = np.sign(x0) * 1e-1
            self.logger.info(f" Test {i}/{N_TESTS} with {p0=}, {e0=}, {a=} and {x0=}")
            forwards, backwards = run_forward_back(
                traj, m1, m2, a, p0, e0, x0, forwards_kwargs=insp_kw
            )
            for i in range(1, 3):
                self.assertAlmostEqual(
                    backwards[i][-1],
                    forwards[i][0],
                    places=3,
                    msg=f"Failed for {p0=}, {e0=}, {a=}, {x0=}",
                )

    def test_trajectory_kerr(self):
        self.logger.info("Testing kerr")
        # initialize trajectory class
        traj = EMRIInspiral(func=KerrEccEqFlux)

        # set initial parameters
        m1 = 1e5
        m2 = 1e1
        for i in range(N_TESTS):
            p0 = np.random.uniform(9.0, 15)
            e0 = np.random.uniform(0.0, 0.6)
            a = np.random.uniform(0.0, 0.999)
            x0 = 1.0

            self.logger.info(f" Test {i}/{N_TESTS} with {p0=}, {e0=}, {a=} and {x0=}")
            forwards, backwards = run_forward_back(
                traj, m1, m2, a, p0, e0, x0, forwards_kwargs=insp_kw
            )
            for i in range(1, 6):
                self.assertAlmostEqual(
                    backwards[i][-1],
                    forwards[i][0],
                    places=3,
                    msg=f"Failed for {p0=}, {e0=}, {a=}, {x0=}",
                )

    def test_trajectory_schwarz(self):
        self.logger.info("Testing schwarz")
        # initialize trajectory class
        traj = EMRIInspiral(func=SchwarzEccFlux)

        # set initial parameters
        m1 = 1e5
        m2 = 1e1
        for i in range(N_TESTS):
            p0 = np.random.uniform(9.0, 15)
            e0 = np.random.uniform(0.0, 0.6)
            a = 0.0
            x0 = 1.0

            self.logger.info(f" Test {i}/{N_TESTS} with {p0=}, {e0=}, {a=} and {x0=}")
            forwards, backwards = run_forward_back(
                traj, m1, m2, a, p0, e0, x0, forwards_kwargs=insp_kw
            )
            for i in range(1, 6):
                self.assertAlmostEqual(
                    backwards[i][-1],
                    forwards[i][0],
                    places=3,
                    msg=f"Failed for {p0=}, {e0=}, {a=}, {x0=}",
                )

    def test_backward_trajectory_termination_kerr(self):
        self.logger.info("Testing kerr backward termination at grid boundaries")
        
        # Initialize trajectory class
        traj = EMRIInspiral(func=KerrEccEqFlux)
        
        # Test parameters
        m1 = 1e6
        m2 = 10.
        
        # Test 1: Termination at e boundary
        self.logger.info("Test 1: Termination at eccentricity boundary")
        a = 0.5
        x0 = 1.0
        
        # Start near the maximum eccentricity
        p0 = 12.0
        flux_obj = KerrEccEqFlux()
        flux_obj.a = a
        e_max = flux_obj._max_e(p0, x0, a)
        e0 = e_max - 0.05
        
        backwards_kwargs = insp_kw.copy()
        backwards_kwargs.update({
            "integrate_backwards": True,
            "T": 10.0,
            "dt": 10.
        })
        
        result = traj(m1, m2, a, p0, e0, x0, **backwards_kwargs)
        final_e = result[2][-1]
        
        # Check that we're near the maximum eccentricity
        flux_obj.a = a
        e_max_final = flux_obj._max_e(result[1][-1], x0, a)
        
        self.assertAlmostEqual(
            final_e, e_max_final, 
            msg=f"Failed to reach e boundary. Final e={final_e}, max e={e_max_final}"
        )
        
        # Test 2: Termination at p boundary
        self.logger.info("Test 2: Termination at p boundary")
        a = 0.0
        e0 = 0.0
        
        # Start very close to PMAX
        # PMAX = PMAX_REGIONB = 200
        PMAX = 200
        p0 = 199.5
        
        backwards_kwargs = insp_kw.copy()
        backwards_kwargs.update({
            "integrate_backwards": True,
            "T": 10_000.,  # Much longer integration time
            "dt": 10_000.  # Larger time step for slow evolution
        })
        
        result = traj(m1, m2, a, p0, e0, x0, **backwards_kwargs)
        final_p = result[1][-1]
                
        self.assertAlmostEqual(
            final_p, PMAX,
            msg=f"Failed to reach p boundary. Final p={final_p}, PMAX={PMAX}"
        )

    def test_trajectory_ELQ_vs_pex(self):
        """
        This test computes the trajectory using the ELQ and pex flux conventions. It will then
        compare the end points of the two trajectrories and pass if a certain threshold is met.

        The purpose of this test really is to check that the pex trajectory is behaving as it should.
        """
        self.logger.info("Testing pex against ELQ trajectories. Dephasing.")

        # Set flux conventions. Integrate ELQ or pex
        inspiral_kwargs_ELQ = {"flux_output_convention": "ELQ", "err": 1e-10}
        inspiral_kwargs_pex = {"flux_output_convention": "pex", "err": 1e-10}

        labels = ["t", "p", "e", "x_I0", "Phi_phi", "Phi_theta", "Phi_r"]
        # initialise classes for ELQ and pex evolution
        traj_ELQ = EMRIInspiral(func=KerrEccEqFlux, **inspiral_kwargs_ELQ)
        traj_pex = EMRIInspiral(func=KerrEccEqFlux, **inspiral_kwargs_pex)

        for i in range(N_TESTS):
            T_obs = np.random.uniform(0.01, 1.0)  # Observation time

            # set initial intrinsic parameters
            m1 = np.random.uniform(5e5, 5e6)
            m2 = np.random.uniform(5, 100)
            p0 = np.random.uniform(9.0, 15)
            e0 = np.random.uniform(0.01, 0.6)
            a = np.random.uniform(0.0, 0.999)
            xI0 = np.random.choice([1.0, -1.0])

            self.logger.info(
                f" Test {i}/{N_TESTS} with {m1=}, {m2=},, {p0=}, {e0=}, {a=} and {xI0=} for T = {T_obs} yr observation"
            )

            # Compute trajectory modules for ELQ and pex
            out_ELQ, out_pex = compute_traj_1_2(
                traj_ELQ, traj_pex, m1, m2, a, p0, e0, xI0, T=T_obs
            )

            # Test trajectories
            nu = m1 * m2 / (m1 + m2) ** 2
            for j in range(1, 4):
                # Test (p, e, xI0) <-- orbital parameters
                self.assertAlmostEqual(
                    out_ELQ[j][-1],
                    out_pex[j][-1],
                    delta=nu,  # Condition on (p,e,x_I)
                    msg=f"for parameter {labels[j]}, End points: Values differ: {out_ELQ[j][-1]} vs {out_pex[j][-1]}",
                )
                self.assertAlmostEqual(
                    out_ELQ[3 + j][-1],
                    out_pex[3 + j][-1],
                    delta=0.1,  # Tight constraint, dephasing cannot be < 1
                    msg=f"for parameter {labels[3 + j]}, End points: Values differ: {out_ELQ[3 + j][-1]} vs {out_pex[3 + j][-1]}",
                )

    @tagged_test(slow=True)
    def test_scipy_solve_ivp_vs_trajectory_kerr(self):
        self.logger.info("Testing scipy solve_ivp vs trajectory kerr")
        # initialize trajectory class
        traj = EMRIInspiral(func=KerrEccEqFlux)

        for i in range(10):
            self.logger.info(f" Test {i}/{N_TESTS}")
            # set initial parameters
            m1 = 1e6
            m2 = 1e1
            a = np.random.uniform(0.0, 0.999)
            p0 = np.random.uniform(9.0, 15)
            e0 = np.random.uniform(0.0, 0.6)
            x0 = 1.0

            nu = m1 * m2 / (m1 + m2) ** 2
            M = m1 + m2

            self.logger.info(f" Test {i}/{N_TESTS} with {p0=}, {e0=}, {a=} and {x0=}")

            # Define ODE flux and parameters
            ode_flux = KerrEccEqFlux()
            pars = [m1, m2, a, p0, e0, x0]
            ode_flux.add_fixed_parameters(*pars[:3])

            def sep_stop(t, y):
                sep = get_separatrix(pars[2], y[1], y[2])
                return y[0] - sep - 2e-3

            sep_stop.terminal = True

            start_time_traj = time.time()
            forwards_result = traj(m1, m2, a, p0, e0, x0, **insp_kw)
            end_time_traj = time.time()
            traj_duration = end_time_traj - start_time_traj
            self.logger.info(f"Trajectory module duration: {traj_duration:.4f} seconds")

            # Solve using scipy solve_ivp

            final_t = (
                nu * forwards_result[0][-1] / (M * MTSUN_SI)
            )  # pars[1] * 4. * YRSID_SI / (pars[0]**2*MTSUN_SI)
            start_time_scipy = time.time()
            res = solve_ivp(
                lambda t, y: ode_flux(y),
                (0, final_t),
                np.asarray([pars[3], pars[4], pars[5], 0.0, 0.0, 0.0]),
                atol=insp_kw["err"],
                rtol=0.0,
                method="DOP853",
                dense_output=True,
                events=sep_stop,
            )
            end_time_scipy = time.time()
            scipy_duration = end_time_scipy - start_time_scipy
            self.logger.debug("Scipy solve_ivp duration: %.4f seconds", scipy_duration)
            self.logger.debug("Ratio scipy/traj: %.4f", scipy_duration / traj_duration)
            self.logger.debug(
                "Number of steps scipy: %u, Number of steps traj: %u, Ratio: %.4f",
                res.t.size,
                len(forwards_result[0]),
                res.t.size / len(forwards_result[0]),
            )

            new_time = nu * forwards_result[0] / (M * MTSUN_SI)

            # abs_diff_p = np.abs(res.sol(new_time)[0] - forwards_result[1])
            # abs_diff_e = np.abs(res.sol(new_time)[1] - forwards_result[2])
            abs_diff_phi = np.abs(res.sol(new_time)[3] / nu - forwards_result[4])
            abs_diff_phir = np.abs(res.sol(new_time)[5] / nu - forwards_result[6])

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

            self.assertAlmostEqual(
                res.sol(new_time)[0][-1],
                forwards_result[1][-1],
                places=2,
                msg=f"Failed for scipy solve_ivp vs trajectory kerr with {p0=}, {e0=}, {a=}, {x0=}",
            )
            self.assertAlmostEqual(
                res.sol(new_time)[1][-1],
                forwards_result[2][-1],
                places=2,
                msg=f"Failed for scipy solve_ivp vs trajectory kerr with {p0=}, {e0=}, {a=}, {x0=}",
            )
            # self.assertAlmostEqual(res.y[3][-1]/(mu/M), forwards_result[4][-1], places=2, msg=f"Failed for scipy solve_ivp vs trajectory kerr with {p0=}, {e0=}, {a=}, {x0=}")
            # self.assertAlmostEqual(res.y[5][-1]/(mu/M), forwards_result[6][-1], places=2, msg=f"Failed for scipy solve_ivp vs trajectory kerr with {p0=}, {e0=}, {a=}, {x0=}")
            self.assertLess(
                abs_diff_phi[-1],
                0.01,
                msg=f"Failed for scipy solve_ivp vs trajectory kerr with {p0=}, {e0=}, {a=}, {x0=}",
            )
            self.assertLess(
                abs_diff_phir[-1],
                0.01,
                msg=f"Failed for scipy solve_ivp vs trajectory kerr with {p0=}, {e0=}, {a=}, {x0=}",
            )


# def test_trajectory_KerrEccentricEquatorial(self):

#     # test against Schwarz
#     traj_Schw = EMRIInspiral(func=SchwarzEccFlux)
#     a = 0.0
#     M = 1e5
#     mu = 1e1

#     for flux_output_convention in ["ELQ", "pex"]:
#         traj = EMRIInspiral(func=KerrEccEqFlux, flux_output_convention="ELQ")
#         self.logger.info(f"testing kerr {flux_output_convention} against schwarzschild")
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
