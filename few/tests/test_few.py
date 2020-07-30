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
