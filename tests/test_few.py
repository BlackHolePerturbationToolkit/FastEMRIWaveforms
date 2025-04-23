import pickle

import numpy as np

from few.amplitude.ampinterp2d import AmpInterpSchwarzEcc
from few.amplitude.romannet import RomanAmplitude
from few.tests.base import FewBackendTest, FewTest, tagged_test
from few.trajectory.inspiral import EMRIInspiral
from few.trajectory.ode import SchwarzEccFlux
from few.utils.utility import get_mismatch, get_overlap
from few.waveform import FastSchwarzschildEccentricFlux, SlowSchwarzschildEccentricFlux


class WaveformTest(FewBackendTest):
    @classmethod
    def name(self) -> str:
        return "FEW"

    @classmethod
    def parallel_class(self):
        return FastSchwarzschildEccentricFlux

    def test_pickle(self):
        # test ability to pickle class

        # keyword arguments for inspiral generator (RunSchwarzEccFluxInspiral)
        inspiral_kwargs = {
            "DENSE_STEPPING": 0,  # we want a sparsely sampled trajectory
            "buffer_length": int(
                1e3
            ),  # all of the trajectories will be well under len = 1000
        }

        # keyword arguments for inspiral generator (RomanAmplitude)
        amplitude_kwargs = {
            "buffer_length": int(
                1e3
            )  # all of the trajectories will be well under len = 1000
        }

        # keyword arguments for Ylm generator (GetYlms)
        Ylm_kwargs = {
            "include_minus_m": False  # if we assume positive m, it will generate negative m for all m>0
        }

        # keyword arguments for summation generator (InterpolatedModeSum)
        sum_kwargs = {}

        fast = FastSchwarzschildEccentricFlux(
            inspiral_kwargs=inspiral_kwargs,
            amplitude_kwargs=amplitude_kwargs,
            Ylm_kwargs=Ylm_kwargs,
            sum_kwargs=sum_kwargs,
            force_backend=self.backend,
        )

        check_pickle = pickle.dumps(fast)
        extracted_gen = pickle.loads(check_pickle)

        # parameters
        T = 0.001  # years
        dt = 15.0  # seconds
        m1 = 1e6
        m2 = 1e1
        p0 = 8.0
        e0 = 0.2
        theta = np.pi / 3  # polar viewing angle
        phi = np.pi / 4  # azimuthal viewing angle
        dist = 1.0  # distance

        _fast_wave = extracted_gen(m1, m2, p0, e0, theta, phi, dist=dist, T=T, dt=dt)

    @tagged_test(slow=True)
    def test_fast_and_slow(self):
        # keyword arguments for inspiral generator (RunSchwarzEccFluxInspiral)
        inspiral_kwargs = {
            "DENSE_STEPPING": 0,  # we want a sparsely sampled trajectory
            "buffer_length": int(
                1e3
            ),  # all of the trajectories will be well under len = 1000
        }

        # keyword arguments for inspiral generator (RomanAmplitude)
        amplitude_kwargs = {
            "buffer_length": int(
                1e3
            )  # all of the trajectories will be well under len = 1000
        }

        # keyword arguments for Ylm generator (GetYlms)
        Ylm_kwargs = {
            "include_minus_m": False  # if we assume positive m, it will generate negative m for all m>0
        }

        # keyword arguments for summation generator (InterpolatedModeSum)
        sum_kwargs = {}

        fast = FastSchwarzschildEccentricFlux(
            inspiral_kwargs=inspiral_kwargs,
            amplitude_kwargs=amplitude_kwargs,
            Ylm_kwargs=Ylm_kwargs,
            sum_kwargs=sum_kwargs,
            force_backend=self.backend,
        )

        # setup slow

        # keyword arguments for inspiral generator (RunSchwarzEccFluxInspiral)
        inspiral_kwargs = {
            "DENSE_STEPPING": 1,  # we want a sparsely sampled trajectory
            "buffer_length": int(3e3),  # dense stepping trajectories
        }

        # keyword arguments for amplitude generator (AmpInterpSchwarzEcc)
        amplitude_kwargs = {}

        # keyword arguments for Ylm generator (GetYlms)
        Ylm_kwargs = {
            "include_minus_m": False  # if we assume positive m, it will generate negative m for all m>0
        }

        # keyword arguments for summation generator (InterpolatedModeSum)
        sum_kwargs = {
            "force_backend": "cpu"
        }  # GPU is availabel for this type of summation
        slow = SlowSchwarzschildEccentricFlux(
            inspiral_kwargs=inspiral_kwargs,
            amplitude_kwargs=amplitude_kwargs,
            Ylm_kwargs=Ylm_kwargs,
            sum_kwargs=sum_kwargs,
            force_backend=self.backend,
        )

        # parameters
        T = 0.001  # years
        dt = 15.0  # seconds
        m1 = 1e6
        m2 = 1e1
        p0 = 7.0
        e0 = 0.2
        theta = np.pi / 3  # polar viewing angle
        phi = np.pi / 4  # azimuthal viewing angle
        dist = 1.0  # distance

        slow_wave = slow(m1, m2, p0, e0, theta, phi, dist=dist, T=T, dt=dt)

        fast_wave = fast(m1, m2, p0, e0, theta, phi, dist=dist, T=T, dt=dt)

        mm = get_mismatch(slow_wave, fast_wave, use_gpu=self.backend.uses_gpu)

        self.assertLess(mm, 1e-4)

        # check that batching the slow waveform gives the same result
        batch_size = int(1e2)
        slow_wave_batched = slow(
            m1, m2, p0, e0, theta, phi, dist=dist, T=T, dt=dt, batch_size=batch_size
        )

        self.assertTrue(np.allclose(slow_wave, slow_wave_batched, atol=0, rtol=1e-13))


def amplitude_test(amp_class):
    # initialize ROMAN class
    amp = RomanAmplitude(buffer_length=1000)  # buffer_length creates memory buffers

    p = np.linspace(10.0, 14.0, 10)
    e = np.linspace(0.1, 0.7, 10)

    p_all, e_all = np.asarray([temp.ravel() for temp in np.meshgrid(p, e)])

    teuk_modes = amp_class(0.0, p_all, e_all, np.ones_like(p_all) * 1.0)

    # (2, 2, 0) and (7, -3, 1) modes
    specific_modes = [(2, 2, 0), (7, -3, 1)]

    # notice this returns a dictionary with keys as the mode tuple and values as the mode values at all trajectory points
    specific_teuk_modes = amp_class(
        0.0, p_all, e_all, np.ones_like(p_all) * 1.0, specific_modes=specific_modes
    )

    # we can find the index to these modes to check
    inds = np.array([amp.special_index_map[lmn] for lmn in specific_modes])

    first_check = np.allclose(specific_teuk_modes[(2, 2, 0)], teuk_modes[:, inds[0]])
    second_check = np.allclose(
        specific_teuk_modes[(7, -3, 1)], (-1) ** 7 * np.conj(teuk_modes[:, inds[1]])
    )
    return first_check, second_check


class ModuleTest(FewTest):
    @classmethod
    def name(self) -> str:
        return "FEW Module"

    def test_trajectory(self):
        # initialize trajectory class
        traj = EMRIInspiral(func=SchwarzEccFlux)

        # set initial parameters
        m1 = 1e5
        m2 = 1e1
        p0 = 10.0
        e0 = 0.7

        # run trajectory
        t, p, e, x, Phi_phi, Phi_theta, Phi_r = traj(m1, m2, 0.0, p0, e0, 1.0)

    def test_amplitudes(self):
        amp = RomanAmplitude(force_backend="cpu")

        first_check, second_check = amplitude_test(amp)

        # make sure they are the same
        self.assertTrue(first_check)

        # to check -m modes we need to take the conjugate
        self.assertTrue(second_check)

    def test_amplitudes_bicubic(self):
        # initialize class
        amp2 = AmpInterpSchwarzEcc()

        first_check, second_check = amplitude_test(amp2)

        # make sure they are the same
        self.assertTrue(first_check)

        # to check -m modes we need to take the conjugate
        self.assertTrue(second_check)

    def test_mismatch(self):
        dt = 1.0
        t = np.arange(10000) * dt
        x0 = np.sin(t) + 1j * np.sin(t)

        # check 1
        x1 = np.sin(t) + 1j * np.sin(t)
        self.assertAlmostEqual(get_overlap(x0, x1), 1.0)
        self.assertAlmostEqual(1.0 - get_overlap(x0, x1), get_mismatch(x0, x1))

        # check 1
        x2 = np.sin(t) + 1j * np.cos(t)
        self.assertAlmostEqual(get_overlap(x0, x2), 0.499981442642142)
        self.assertAlmostEqual(1.0 - get_overlap(x0, x1), get_mismatch(x0, x1))
