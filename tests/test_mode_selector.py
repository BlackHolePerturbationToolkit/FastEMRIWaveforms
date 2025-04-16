import numpy as np

from few.amplitude.romannet import RomanAmplitude
from few.summation.interpolatedmodesum import CubicSplineInterpolant
from few.tests.base import FewBackendTest
from few.trajectory.inspiral import EMRIInspiral
from few.trajectory.ode import SchwarzEccFlux
from few.utils.globals import get_file_manager
from few.utils.modeselector import ModeSelector
from few.utils.utility import get_mismatch
from few.utils.ylm import GetYlms
from few.waveform import FastSchwarzschildEccentricFluxBicubic


class ModeSelectorTest(FewBackendTest):
    @classmethod
    def name(self) -> str:
        return "ModeSelector"

    @classmethod
    def parallel_class(self):
        return FastSchwarzschildEccentricFluxBicubic

    def test_mode_selector(self):
        # first, lets get amplitudes for a trajectory
        traj = EMRIInspiral(func=SchwarzEccFlux)
        ylm_gen = GetYlms(include_minus_m=True, force_backend="cpu")

        # parameters
        m1 = 1e5
        m2 = 1e1
        p0 = 10.0
        e0 = 0.3
        theta = np.pi / 3.0
        phi = np.pi / 2.0

        t, p, e, x, Phi_phi, Phi_theta, Phi_r = traj(m1, m2, 0.0, p0, e0, 1.0, T=10.0)

        # get amplitudes along trajectory
        amp = RomanAmplitude(force_backend="cpu")

        teuk_modes = amp(0.0, p, e, x)

        # get ylms
        ylms = ylm_gen(amp.unique_l, amp.unique_m, theta, phi).copy()[amp.inverse_lm]

        # select modes

        mode_selector = ModeSelector(
            amp.l_arr_no_mask, amp.m_arr_no_mask, amp.n_arr_no_mask, force_backend="cpu"
        )

        eps = 1e-2  # tolerance on mode contribution to total power

        modeinds = [amp.l_arr, amp.m_arr, amp.n_arr]
        (teuk_modes_in, ylms_in, ls, ms, ns) = mode_selector(
            teuk_modes, ylms, modeinds, mode_selection_threshold=eps
        )

        # print("We reduced the mode content from {} modes to {} modes.".format(teuk_modes.shape[1], teuk_modes_in.shape[1]))
        ls_orig = ls
        ms_orig = ms
        ns_orig = ns

        # produce sensitivity function

        noise = np.genfromtxt(
            get_file_manager().get_file("LPA.txt"),
            names=True,
        )
        f, PSD = (
            np.asarray(noise["f"], dtype=np.float64),
            np.asarray(noise["ASD"], dtype=np.float64) ** 2,
        )

        sens_fn = CubicSplineInterpolant(f, PSD, force_backend="cpu")

        # select modes with noise weighting

        # provide sensitivity function kwarg
        mode_selector_noise_weighted = ModeSelector(
            amp.l_arr_no_mask,
            amp.m_arr_no_mask,
            amp.n_arr_no_mask,
            sensitivity_fn=sens_fn,
            force_backend="cpu",
        )

        # Schwarzschild
        a = 0.0
        Y = np.zeros_like(p)  # equatorial / cos iota
        fund_freq_args = (m1, m2, a, p, e, Y, t)

        modeinds = [amp.l_arr, amp.m_arr, amp.n_arr]
        (teuk_modes_in, ylms_in, ls, ms, ns) = mode_selector_noise_weighted(
            teuk_modes, ylms, modeinds, fund_freq_args=fund_freq_args, mode_selection_threshold=eps
        )

        # print("We reduced the mode content from {} modes to {} modes when using noise-weighting.".format(teuk_modes.shape[1], teuk_modes_in.shape[1]))
        # import matplotlib.pyplot as plt
        # plt.figure(); plt.title(f'Mode selection comparison \n M={M:.1e},mu={mu:.1e},e0={e0},p0={p0},eps={eps:.2e}');
        # plt.plot(ms,ns,'o',label=f'new select, N={len(ms)}', ms=10); plt.plot(ms_orig,ns_orig,'P',label=f'old select, N={len(ms_orig)}', ms=5); plt.legend(); plt.ylabel('n'); plt.xlabel('m'); plt.show()

        # mode_selector_kwargs = {}

        # noise_weighted_mode_selector_kwargs = dict(sensitivity_fn=sens_fn)

        few_base = FastSchwarzschildEccentricFluxBicubic(force_backend=self.backend)

        m1 = 1e6
        m2 = 1e1
        p0 = 12.0
        e0 = 0.3
        theta = np.pi / 3.0
        phi = np.pi / 4.0
        dist = 1.0
        dt = 10.0
        T = 0.001
        mode_selection = [(ll, mm, nn) for ll, mm, nn in zip(ls_orig, ms_orig, ns_orig)]
        wave_base = few_base(
            m1, m2, p0, e0, theta, phi, dist, dt=dt, T=T, mode_selection=mode_selection
        )
        mode_selection = [(ll, mm, nn) for ll, mm, nn in zip(ls, ms, ns)]
        wave_weighted = few_base(
            m1, m2, p0, e0, theta, phi, dist, dt=dt, T=T, mode_selection=mode_selection
        )

        self.logger.info(
            "  mismatch: {}".format(
                mismatch := get_mismatch(
                    wave_base, wave_weighted, use_gpu=self.backend.uses_gpu
                )
            )
        )
        self.assertLess(mismatch, eps)
