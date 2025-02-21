import numpy as np

from few.waveform import FastKerrEccentricEquatorialFlux
from few.utils.utility import get_mismatch
import unittest

from few.utils.globals import get_logger, get_first_backend

few_logger = get_logger()

best_backend = get_first_backend(FastKerrEccentricEquatorialFlux.supported_backends())
few_logger.warning("Physics Test is running with backend {}".format(best_backend.name))


class WaveformTest(unittest.TestCase):
    def setUp(self):
        self.waveform_generator = FastKerrEccentricEquatorialFlux(
            force_backend=best_backend
        )

    def tearDown(self):
        del self.waveform_generator

    def test_sampling_variation(self):
        # parameters
        T = 0.001
        M = 1e6
        mu = 1e1
        a = 0.6
        p0 = 8.0
        e0 = 0.3
        xI = 1.0

        theta = np.pi / 3
        phi = np.pi / 4

        wave1 = self.waveform_generator(M, mu, a, p0, e0, xI, theta, phi, T=T, dt=10.0)[
            :500
        ]
        wave2 = self.waveform_generator(M, mu, a, p0, e0, xI, theta, phi, T=T, dt=5.0)[
            :1000
        ]

        # these two waveforms should be identical once resampled to the same time axis
        assert get_mismatch(wave1, wave2[::2], use_gpu=best_backend.uses_gpu) < 1e-14, (
            "Waveforms not invariant under a change of sampling rate."
        )

        wave1 = self.waveform_generator(M, mu, a, p0, e0, xI, theta, phi, T=T, dt=10.0)[
            :1000
        ]
        wave2 = self.waveform_generator(
            M, mu, a, p0, e0, xI, theta, phi, T=2 * T, dt=10.0
        )[:1000]

        # these two waveforms should be identical once resampled to the same time axis
        assert get_mismatch(wave1, wave2, use_gpu=best_backend.uses_gpu) < 1e-14, (
            "Waveforms not invariant under a change of duration."
        )

    def test_mass_invariance(self):
        # parameters
        T = 0.001
        M = 1e6
        mu = 1e1
        a = 0.6
        p0 = 8.0
        e0 = 0.3
        xI = 1.0

        theta = np.pi / 3
        phi = np.pi / 4

        wave1 = self.waveform_generator(M, mu, a, p0, e0, xI, theta, phi, T=T, dt=10.0)[
            :500
        ]

        M2 = M * 2
        mu2 = mu * 2
        wave2 = self.waveform_generator(
            M2, mu2, a, p0, e0, xI, theta, phi, T=T, dt=10.0
        )[:1000]

        # these two waveforms should be identical up to a rescailing of the time axis
        assert get_mismatch(wave1, wave2[::2], use_gpu=best_backend.uses_gpu) < 1e-14, (
            "Waveforms not invariant under a rescaling of total mass and coordinate time."
        )

    def test_prograde_retrograde_convergence(self):
        # parameters
        T = 0.001
        M = 1e6
        mu = 1e1
        a = 0.0
        p0 = 8.0
        e0 = 0.3
        xI = 1.0
        theta = np.pi / 3
        phi = np.pi / 4

        wave1 = self.waveform_generator(M, mu, a, p0, e0, xI, theta, phi, T=T, dt=10.0)
        wave2 = self.waveform_generator(M, mu, a, p0, e0, -xI, theta, phi, T=T, dt=10.0)

        # these two waveforms should be identical as they represent the same physical system
        assert get_mismatch(wave1, wave2, use_gpu=best_backend.uses_gpu) < 1e-14, (
            "Schwarzschild waveforms not identical under a sign-flip of xI."
        )

        a = 1e-11

        wave1 = self.waveform_generator(M, mu, a, p0, e0, xI, theta, phi, T=T, dt=10.0)
        wave2 = self.waveform_generator(M, mu, a, p0, e0, -xI, theta, phi, T=T, dt=10.0)

        # these two waveforms should be identical as they represent (essentially) the same physical system
        assert get_mismatch(wave1, wave2, use_gpu=best_backend.uses_gpu) < 1e-14, (
            "Prograde-retrograde waveforms not convergent."
        )

        a = 0.3

        # check that negative spin inputs are correctly handled
        wave1 = self.waveform_generator(M, mu, a, p0, e0, xI, theta, phi, T=T, dt=10.0)
        wave2 = self.waveform_generator(
            M, mu, -a, p0, e0, -xI, theta, phi, T=T, dt=10.0
        )

        # these two waveforms should be identical as they represent the same physical system
        assert get_mismatch(wave1, wave2, use_gpu=best_backend.uses_gpu) < 1e-14, (
            "Waveforms not convergent under joint sign-flip of a and xI."
        )

        wave3 = self.waveform_generator(M, mu, -a, p0, e0, xI, theta, phi, T=T, dt=10.0)

        # these waveforms should greatly differ
        assert get_mismatch(wave1, wave3, use_gpu=best_backend.uses_gpu) > 1e-5, (
            "Waveforms not divergent under a sign-flip of xI."
        )

    def test_orientation_behaviour(self):
        # parameters
        T = 0.001
        M = 1e6
        mu = 1e1
        a = 0.6
        p0 = 8.0
        e0 = 0.3
        xI = 1.0
        theta = np.pi / 3
        phi = np.pi / 4

        # confirm the parity of the spherical harmonics
        # one can show that under the transformation (theta, phi) -> (pi - theta, phi + pi),
        # Then h -> h^* (-1)^m for all \ell modes

        # choose modes with even m
        modes_check = [(ell, 2, enn) for ell in range(2, 11) for enn in range(-50, 51)]
        wave1 = self.waveform_generator(
            M, mu, a, p0, e0, xI, theta, phi, T=T, dt=10.0, mode_selection=modes_check
        )
        wave2 = self.waveform_generator(
            M,
            mu,
            a,
            p0,
            e0,
            xI,
            np.pi - theta,
            phi + np.pi,
            T=T,
            dt=10.0,
            mode_selection=modes_check,
        )

        assert (
            get_mismatch(wave1, wave2.conj(), use_gpu=best_backend.uses_gpu) < 1e-14
        ), "Even ell-mode waveform not convergent under a parity transformation."

        # choose modes with odd m
        modes_check = [(ell, 3, enn) for ell in range(3, 11) for enn in range(-50, 51)]

        wave1 = self.waveform_generator(
            M, mu, a, p0, e0, xI, theta, phi, T=T, dt=10.0, mode_selection=modes_check
        )
        wave2 = self.waveform_generator(
            M,
            mu,
            a,
            p0,
            e0,
            xI,
            np.pi - theta,
            phi + np.pi,
            T=T,
            dt=10.0,
            mode_selection=modes_check,
        )

        assert (
            get_mismatch(wave1, -wave2.conj(), use_gpu=best_backend.uses_gpu) < 1e-14
        ), "Odd ell-mode waveform not convergent under a parity transformation."

    def test_distance_scaling(self):
        # parameters
        T = 0.001
        M = 1e6
        mu = 1e1
        a = 0.6
        p0 = 8.0
        e0 = 0.3
        xI = 1.0
        theta = np.pi / 3
        phi = np.pi / 4

        wave1 = self.waveform_generator(
            M, mu, a, p0, e0, xI, theta, phi, T=T, dt=10.0, dist=1.0
        )
        wave2 = self.waveform_generator(
            M, mu, a, p0, e0, xI, theta, phi, T=T, dt=10.0, dist=2.0
        )
        assert get_mismatch(wave1, wave2 * 2, use_gpu=best_backend.uses_gpu) < 1e-14, (
            "Waveform not invariant under a rescaling of luminosity distance."
        )

    def test_nonphysical_input_failure(self):
        T = 0.001
        M = 1e6
        mu = 1e1
        a = 0.6
        p0 = 8.0
        e0 = 0.3
        xI = 1.0
        theta = np.pi / 3
        phi = np.pi / 4
        pars = [M, mu, a, p0, e0, xI, theta, phi]
        with self.assertRaises(ValueError):
            self.waveform_generator(*pars, T=T, dt=10.0, dist=-1.0)
            for i in [0, 1, 3, 4]:
                pars_copy = pars.copy()
                pars_copy[i] *= -1
                self.waveform_generator(*pars_copy, T=T, dt=10.0, dist=1.0)

    def test_request_nonphysical_modes(self):
        T = 0.001
        M = 1e6
        mu = 1e1
        a = 0.6
        p0 = 8.0
        e0 = 0.3
        xI = 1.0
        theta = np.pi / 3
        phi = np.pi / 4
        modes = [(2, 2, 0), (2, 4, 0), (2, 3, 0)]
        with self.assertRaises(ValueError):
            self.waveform_generator(
                M, mu, a, p0, e0, xI, theta, phi, T=T, dt=10.0, mode_selection=modes
            )
