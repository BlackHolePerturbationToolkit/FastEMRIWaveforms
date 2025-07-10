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
        # first, lets make a trajectory
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

        # instantiate amplitude module
        amp = RomanAmplitude(force_backend="cpu")

        # select modes

        # instantiate mode selector
        mode_selector = ModeSelector(
            amp, 
            ylm_generator=ylm_gen,
            force_backend="cpu"
        )

        eps = 1e-2  # tolerance on mode contribution to total power

        (teuk_modes_in, ylms_in, ls, ms, ks, ns) = mode_selector(
            t, 0.0, p, e, x, theta, phi, mode_selection_threshold=eps
        )

        ls_orig = ls
        ms_orig = ms
        ks_orig = ks
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
            amp,
            ylm_generator=ylm_gen, 
            sensitivity_fn=sens_fn,
            force_backend="cpu",
        )

        freqs = traj.inspiral_generator.eval_integrator_derivative_spline(t, order=1)[:,3:6] / 2 / np.pi

        online_mode_selection_args = dict(
            f_phi = freqs[:,0],
            f_theta = freqs[:,1],
            f_r = freqs[:,2],
        )

        (teuk_modes_in, ylms_in, ls_nw, ms_nw, ks_nw, ns_nw) = mode_selector_noise_weighted(
            t, 0.0, p, e, x, theta, phi,
            online_mode_selection_args=online_mode_selection_args, 
            mode_selection_threshold=eps
        )

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
        mode_selection = [(ll, mm, kk, nn) for ll, mm, kk, nn in zip(ls_orig, ms_orig, ks_orig, ns_orig)]
        wave_base = few_base(
            m1, m2, p0, e0, theta, phi, dist, dt=dt, T=T, mode_selection=mode_selection
        )
        mode_selection = [(ll, mm, kk, nn) for ll, mm, kk, nn in zip(ls, ms, ks, ns)]
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
