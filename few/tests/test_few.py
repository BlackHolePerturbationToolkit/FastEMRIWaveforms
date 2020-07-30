import unittest
import numpy as np

from few.trajectory.flux import RunSchwarzEccFluxInspiral
from few.amplitude.romannet import ROMANAmplitude
from few.amplitude.interp2dcubicspline import Interp2DAmplitude
from few.waveform import FastSchwarzschildEccentricFlux, SlowSchwarzschildEccentricFlux
from few.utils.overlap import get_overlap, get_mismatch
from few.utils.ylm import GetYlms
from few.utils.modeselector import ModeSelector
from few.summation.interpolatedmodesum import CubicSplineInterpolant

try:
    import cupy as xp

    gpu_avialable = True

except (ModuleNotFoundError, ImportError) as e:
    import numpy as xp

    gpu_avialable = False


class WaveformTest(unittest.TestCase):
    def test_fast_and_slow(self):

        # keyword arguments for inspiral generator (RunSchwarzEccFluxInspiral)
        inspiral_kwargs = {
            "DENSE_STEPPING": 0,  # we want a sparsely sampled trajectory
            "max_init_len": int(
                1e3
            ),  # all of the trajectories will be well under len = 1000
        }

        # keyword arguments for inspiral generator (ROMANAmplitude)
        amplitude_kwargs = {
            "max_input_len": int(
                1e3
            ),  # all of the trajectories will be well under len = 1000
            "use_gpu": gpu_avialable,  # GPU is available in this class
        }

        # keyword arguments for Ylm generator (GetYlms)
        Ylm_kwargs = {
            "assume_positive_m": False  # if we assume positive m, it will generate negative m for all m>0
        }

        # keyword arguments for summation generator (InterpolatedModeSum)
        sum_kwargs = {
            "use_gpu": gpu_avialable
        }  # GPU is availabel for this type of summation

        fast = FastSchwarzschildEccentricFlux(
            inspiral_kwargs=inspiral_kwargs,
            amplitude_kwargs=amplitude_kwargs,
            Ylm_kwargs=Ylm_kwargs,
            sum_kwargs=sum_kwargs,
            use_gpu=gpu_avialable,
        )

        # setup slow

        # keyword arguments for inspiral generator (RunSchwarzEccFluxInspiral)
        inspiral_kwargs = {
            "DENSE_STEPPING": 1,  # we want a sparsely sampled trajectory
            "max_init_len": int(1e7),  # dense stepping trajectories
        }

        # keyword arguments for inspiral generator (ROMANAmplitude)
        amplitude_kwargs = {"max_input_len": int(1e4)}  # this must be >= batch_size

        # keyword arguments for Ylm generator (GetYlms)
        Ylm_kwargs = {
            "assume_positive_m": False  # if we assume positive m, it will generate negative m for all m>0
        }

        # keyword arguments for summation generator (InterpolatedModeSum)
        sum_kwargs = {"use_gpu": False}  # GPU is availabel for this type of summation

        slow = SlowSchwarzschildEccentricFlux(
            inspiral_kwargs=inspiral_kwargs,
            amplitude_kwargs=amplitude_kwargs,
            Ylm_kwargs=Ylm_kwargs,
            sum_kwargs=sum_kwargs,
            use_gpu=False,
        )

        # parameters
        T = 0.001  # years
        dt = 15.0  # seconds
        M = 1e6
        mu = 1e1
        p0 = 8.0
        e0 = 0.2
        theta = np.pi / 3  # polar viewing angle
        phi = np.pi / 4  # azimuthal viewing angle
        batch_size = int(1e4)

        slow_wave = slow(M, mu, p0, e0, theta, phi, T=T, dt=dt, batch_size=batch_size)

        fast_wave = fast(M, mu, p0, e0, theta, phi, T=T, dt=dt)

        mm = get_mismatch(slow_wave, fast_wave, use_gpu=gpu_avialable)

        self.assertLess(mm, 1e-4)

        # test_rk4
        fast.inspiral_kwargs["use_rk4"] = True
        fast_wave = fast(M, mu, p0, e0, theta, phi, T=T, dt=dt)


def amplitude_test(amp_class):
    # initialize ROMAN class
    amp = ROMANAmplitude(max_input_len=5000)  # max_input_len creates memory buffers

    p = np.linspace(10.0, 14.0, 10)
    e = np.linspace(0.1, 0.7, 10)

    p_all, e_all = np.asarray([temp.ravel() for temp in np.meshgrid(p, e)])

    teuk_modes = amp_class(p_all, e_all)

    # (2, 2, 0) and (7, -3, 1) modes
    specific_modes = [(2, 2, 0), (7, -3, 1)]

    # notice this returns a dictionary with keys as the mode tuple and values as the mode values at all trajectory points
    specific_teuk_modes = amp_class(p_all, e_all, specific_modes=specific_modes)

    # we can find the index to these modes to check
    inds = np.array([amp.special_index_map[lmn] for lmn in specific_modes])

    first_check = np.allclose(specific_teuk_modes[(2, 2, 0)], teuk_modes[:, inds[0]])
    second_check = np.allclose(
        specific_teuk_modes[(7, -3, 1)], np.conj(teuk_modes[:, inds[1]])
    )
    return first_check, second_check


class ModuleTest(unittest.TestCase):
    def test_trajectory(self):

        # initialize trajectory class
        traj = RunSchwarzEccFluxInspiral()

        # set initial parameters
        M = 1e5
        mu = 1e1
        p0 = 10.0
        e0 = 0.7

        # run trajectory
        t, p, e, Phi_phi, Phi_r, flux = traj(M, mu, p0, e0)

        traj.sanity_check_traj(p, e)

    def test_amplitudes(self):

        amp = ROMANAmplitude()

        first_check, second_check = amplitude_test(amp)

        # make sure they are the same
        self.assertTrue(first_check)

        # to check -m modes we need to take the conjugate
        self.assertTrue(second_check)

    def test_amplitudes_bicubic(self):
        # initialize class
        amp2 = Interp2DAmplitude()

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
