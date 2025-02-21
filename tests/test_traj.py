# python -m unittest few/tests/test_traj.py
import unittest
import numpy as np
import time

from few.trajectory.inspiral import EMRIInspiral
from few.utils.constants import YRSID_SI
from few.trajectory.ode import KerrEccEqFlux, PN5, SchwarzEccFlux

from few.utils.globals import get_logger

few_logger = get_logger()

few_logger.warning("Traj Test is running")

T = 1000.0
dt = 10.0

insp_kw = {
    "T": T,
    "dt": 1.0,
    "err": 1e-10,
    "DENSE_STEPPING": 0,
    "buffer_length": int(1e4),
    "upsample": False,
}

np.random.seed(42)


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


class ModuleTest(unittest.TestCase):
    def test_trajectory_pn5(self):
        few_logger.info("Testing pn5")
        # initialize trajectory class
        traj = EMRIInspiral(func=PN5)

        # set initial parameters
        M = 1e5
        mu = 1e1
        np.random.seed(42)
        N_TESTS = 10
        for i in range(N_TESTS):
            p0 = np.random.uniform(10.0, 15)
            e0 = np.random.uniform(0.0, 1.0)
            a = np.random.uniform(0.0, 1.0)
            Y0 = np.random.uniform(-1.0, 1.0)

            # do not want to be too close to polar
            if np.abs(Y0) < 1e-2:
                Y0 = np.sign(Y0) * 1e-2
            few_logger.info(f" Test {i}/{N_TESTS} with {p0=}, {e0=}, {a=} and {Y0=}")
            forwards, backwards = run_forward_back(
                traj, M, mu, a, p0, e0, Y0, forwards_kwargs=insp_kw
            )
            self.assertAlmostEqual(backwards[1][-1], forwards[1][0], places=8)

    def test_trajectory_SchwarzEccFlux(self):
        few_logger.info("Testing Schwarz traj")

        # initialize trajectory class
        traj = EMRIInspiral(func=SchwarzEccFlux)

        # set initial parameters
        M = 1e5
        mu = 1e1
        p0 = 10.0
        e0 = 0.7

        # run trajectory
        _trajectory = traj(M, mu, 0.0, p0, e0, 1.0)

    def test_trajectory_KerrEccentricEquatorial(self):
        # initialize trajectory class
        #
        list_func = [
            KerrEccEqFlux
        ]  # , 'KerrEccentricEquatorial_ELQ', 'KerrEccentricEquatorial_ELQ_nofrequencies',]
        # list_func = ['KerrEccentricEquatorial_ELQ', 'KerrEccentricEquatorial_ELQ_nofrequencies']
        for el in list_func:
            few_logger.info("testing kerr {}".format(el))
            traj = EMRIInspiral(func=el)

            # set initial parameters
            M = 1e6
            mu = 1.0

            # Np = 0
            # eval_time = []
            # N_points = []
            # last_phase = []
            # define 100 random initial conditions
            # Ntest = 200
            # pvec = np.random.uniform(9.0, 15.0, Ntest)
            # evec = np.random.uniform(0.01, 0.6, Ntest)
            # avec = np.random.uniform(0.01, 0.90, Ntest)

            # define a function that for a given error and initial conditions return the number of points, the last phase and the time to evaluate the trajectory
            def get_N_Phif_evalT(M, mu, a, p0, e0, err):
                insp_kw["err"] = err
                tic = time.perf_counter()
                t, p, e, x, Phi_phi, Phi_theta, Phi_r = traj(
                    M, mu, a, p0, e0, 1.0, **insp_kw
                )
                toc = time.perf_counter()
                return len(t), Phi_phi[-1], toc - tic

            # err_vec = 10**np.linspace(-9,-13,10)
            # muvec = np.asarray([1, 1000])
            # results = np.asarray([[[get_N_Phif_evalT(M, mu, avec[i], pvec[i], evec[i], err)  for i in range(Ntest)] for mu in muvec]for err in err_vec])
            # phase_difference = results[:-1,:,:,1] - results[-1,:,:,1]
            # timing = results[:,:,:,2]
            # N_points = results[:,:,:,0]
            # plt.figure(figsize=(10, 8))

            # # Plot 1: Mean phase difference
            # plt.subplot(3, 1, 1)
            # plt.title(f'Average over {Ntest} random initial conditions, M{M:.0e}')
            # plt.loglog(err_vec[:-1], np.median(phase_difference, axis=-1)[:,0],'-o',label=f'mu={muvec[0]}')
            # plt.loglog(err_vec[:-1], np.median(phase_difference, axis=-1)[:,1],'-x',label=f'mu={muvec[1]}')
            # plt.axhline(1.0, color='k', linestyle='--',label='1.0 rad')
            # plt.axhline(0.1, color='k', linestyle='-',label='0.1 rad')
            # plt.ylabel('Final Phase Difference')
            # plt.legend()

            # # Plot 2: Mean timing
            # plt.subplot(3, 1, 2)
            # plt.semilogx(err_vec[:-1], np.median(timing, axis=-1)[:-1,0],'-o',label=f'mu={muvec[0]}')
            # plt.semilogx(err_vec[:-1], np.median(timing, axis=-1)[:-1,1],'-x',label=f'mu={muvec[1]}')
            # plt.ylabel('Timing [seconds]')

            # # Plot 3: Mean number of points
            # plt.subplot(3, 1, 3)
            # plt.semilogx(err_vec[:-1], np.median(N_points, axis=-1)[:-1,0],'-o',label=f'mu={muvec[0]}')
            # plt.semilogx(err_vec[:-1], np.median(N_points, axis=-1)[:-1,1],'-x',label=f'mu={muvec[1]}')
            # plt.xlabel('Relative error ODE')
            # plt.ylabel('Number of Points')

            # plt.savefig(f'{el}_rootSeparatrix.png')

            # test against Schwarz
            traj_Schw = EMRIInspiral(
                func=SchwarzEccFlux
            )  # EMRIInspiral(func="SchwarzEccFlux")
            a = 0.0
            # charge = 0.0

            # check against Schwarzchild
            # plt.figure()
            for i in range(100):
                p0 = np.random.uniform(10.0, 15)
                e0 = np.random.uniform(0.1, 0.5)

                # tic = time.perf_counter()
                t, p, e, x, Phi_phi, Phi_theta, Phi_r = traj(
                    M, mu, a, p0, e0, 1.0, T=100.0, err=1e-10, buffer_length=int(1e5)
                )
                # toc = time.perf_counter()

                tS, pS, eS, xS, Phi_phiS, Phi_thetaS, Phi_rS = traj_Schw(
                    M,
                    mu,
                    0.0,
                    p0,
                    e0,
                    1.0,
                    T=100.0,
                    new_t=t,
                    upsample=True,
                    err=1e-14,
                    buffer_length=int(1e5),
                )
                mask = Phi_rS != 0.0
                diff = np.abs(Phi_phi[mask] - Phi_phiS[mask])
                # print(np.max(diff),toc-tic,len(t))
                # self.assertLess(np.max(diff),2.0)

            #     plt.plot(p[mask],e[mask])
            #     plt.plot(pS[mask],eS[mask],'--')
            # plt.show()
            # plot phases
            # plt.figure(); plt.plot(t[mask],np.abs(Phi_phi[mask] - Phi_phiS[mask])); plt.show()
            # breakpoint()

        # check against Schwarzchild
        for i in range(100):
            p0 = np.random.uniform(10.0, 15)
            e0 = np.random.uniform(0.1, 0.5)

            t, p, e, x, Phi_phi, Phi_theta, Phi_r = traj(
                M, mu, a, p0, e0, 1.0, T=2.0, buffer_length=int(1e5)
            )
            tS, pS, eS, xS, Phi_phiS, Phi_thetaS, Phi_rS = traj_Schw(
                M,
                mu,
                0.0,
                p0,
                e0,
                1.0,
                T=2.0,
                new_t=t,
                upsample=True,
                buffer_length=int(1e5),
            )
            mask = Phi_rS != 0.0
            diff = np.abs(Phi_phi[mask] - Phi_phiS[mask])
            # plt.figure(); plt.plot(tS,pS);plt.plot(t,p);plt.show()
            # plt.figure(); plt.plot(tS,Phi_phiS);plt.plot(t,Phi_phi);plt.show()

            self.assertLess(np.max(diff), 2.0)

        # t, p, e, x, Phi_phi, Phi_theta, Phi_r = traj(1e6, 100.0, 0.99, 6.0, 0.5, 1.0, T=2079375.6399400292/YRSID_SI)
        # self.assertLess(np.abs(Phi_phi[-1] - 37548.68909110543),2.0) # value from Scott

        # s_t, s_p, s_e, s_x, s_omr, s_omt, s_omph, s_r, s_th, s_ph = np.loadtxt("data_for_lorenzo/scott_data/a0.99_p0_6_e0_0.5_xI0_1.0_wl.txt").T
        # mask = (s_p>(0.1+s_p[-1]))
        # t, p, e, x, Phi_phi, Phi_theta, Phi_r = traj(1e6, 100.0, 0.99, s_p[0], s_e[0], 1.0, T=4.0, new_t=s_t[mask]*MTSUN_SI*M, upsample=True)
        # # plt.figure();  plt.plot(t,p); plt.plot(s_t*MTSUN_SI*M,s_p,'--',label='Scott'); plt.show()
        # plt.figure();  plt.plot(p,e); plt.plot(s_p[mask],s_e[mask],'--',label='Scott'); plt.show()
        # plt.figure();  plt.semilogy(t,np.abs(Phi_phi-s_ph[mask])); plt.ylabel('phase difference phi'); plt.xlabel('time [seconds]'); plt.show()

    def test_backwards_trajectory(self):
        # initialize trajectory class
        list_func = [
            PN5,
            SchwarzEccFlux,
            KerrEccEqFlux,
        ]
        for el in list_func:
            few_logger.info("testing backwards {}".format(el))
            traj = EMRIInspiral(func=el)

            # set initial parameters
            M = 1e6
            mu = 10.0

            # plt.figure()
            # tic = time.perf_counter()
            for i in range(100):
                p0 = np.random.uniform(9.0, 12.0)
                e0 = np.random.uniform(0.1, 0.5)
                a = np.random.uniform(0.01, 0.98)

                if el is SchwarzEccFlux:
                    a = 0.0
                    Y0 = 1.0
                elif el is KerrEccEqFlux:
                    Y0 = 1.0
                else:
                    Y0 = np.random.uniform(0.2, 0.8)

                # print(a,p0,e0)
                # run trajectory forwards
                insp_kw["T"] = 0.1
                t, p_forward, e_forward, x_forward, _, _, _ = traj(
                    M, mu, np.abs(a), p0, e0, Y0, **insp_kw
                )

                p_final = p_forward[-1]
                e_final = e_forward[-1]
                x_final = x_forward[-1]

                # run trajectory backwards
                insp_kw_back = insp_kw.copy()
                insp_kw_back.update({"integrate_backwards": True})

                t, p_back, e_back, x_back, _, _, _ = traj(
                    M, mu, np.abs(a), p_final, e_final, x_final, **insp_kw_back
                )

                self.assertAlmostEqual(p_back[-1], p_forward[0], places=8)
                self.assertAlmostEqual(e_back[-1], e_forward[0], places=8)
                self.assertAlmostEqual(x_back[-1], x_forward[0], places=8)
