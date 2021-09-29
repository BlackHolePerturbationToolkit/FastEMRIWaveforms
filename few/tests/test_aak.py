import unittest
import numpy as np
import warnings

from few.waveform import Pn5AAKWaveform
from few.utils.utility import get_overlap, get_mismatch

try:
    import cupy as xp

    gpu_available = True

except (ModuleNotFoundError, ImportError) as e:
    import numpy as xp

    warnings.warn(
        "CuPy is not installed or a gpu is not available. If trying to run on a gpu, please install CuPy."
    )
    gpu_available = False


class AAKWaveformTest(unittest.TestCase):
    def test_aak(self):

        # keyword arguments for inspiral generator (RunSchwarzEccFluxInspiral)
        inspiral_kwargs = {
            "DENSE_STEPPING": 0,  # we want a sparsely sampled trajectory
            "max_init_len": int(
                1e3
            ),  # all of the trajectories will be well under len = 1000
        }

        # keyword arguments for summation generator (InterpolatedModeSum)
        sum_kwargs = {"pad_output": False}

        # set initial parameters
        M = 1e6
        mu = 1e1
        a = 0.2
        p0 = 14.0
        e0 = 0.2
        iota0 = 0.1
        Y0 = np.cos(iota0)

        qS = 0.2
        phiS = 0.2
        qK = 0.8
        phiK = 0.8
        dist = 1.0
        mich = False
        dt = 10.0
        T = 0.001

        wave_cpu = Pn5AAKWaveform(inspiral_kwargs, sum_kwargs, use_gpu=False)

        waveform_cpu = wave_cpu(
            M, mu, a, p0, e0, Y0, qS, phiS, qK, phiK, dist, mich=mich, dt=dt, T=T
        )

        self.assertTrue(
            np.all(np.abs(waveform_cpu) > 0.0)
            and np.all(np.isnan(waveform_cpu) == False)
        )

        if gpu_available:
            wave_gpu = Pn5AAKWaveform(inspiral_kwargs, sum_kwargs, use_gpu=True)

            waveform = wave_gpu(
                M, mu, a, p0, e0, Y0, qS, phiS, qK, phiK, dist, mich=mich, dt=dt, T=T
            )

            mm = get_mismatch(waveform, waveform_cpu, use_gpu=gpu_available)
            self.assertLess(mm, 1e-10)
