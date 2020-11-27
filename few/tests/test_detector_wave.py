import unittest
import numpy as np
import warnings

from few.waveform import GenerateEMRIWaveform

try:
    import cupy as xp

    gpu_available = True

except (ModuleNotFoundError, ImportError) as e:
    import numpy as xp

    warnings.warn(
        "CuPy is not installed or a gpu is not available. If trying to run on a gpu, please install CuPy."
    )
    gpu_available = False


class WaveformTest(unittest.TestCase):
    def test_detector_frame(self):

        # specific waveform kwargs
        waveform_kwargs = dict(use_gpu=gpu_available,)

        gen_wave = GenerateEMRIWaveform(
            "FastSchwarzschildEccentricFlux", **waveform_kwargs
        )

        gen_wave_aak = GenerateEMRIWaveform("Pn5AAKWaveform", **waveform_kwargs)

        # parameters
        T = 0.001  # years
        dt = 15.0  # seconds
        M = 1e6
        a = 0.1  # will be ignored in Schwarz waveform
        mu = 1e1
        p0 = 12.0
        e0 = 0.2
        Y0 = 1.0
        qK = 0.2  # polar spin angle
        phiK = np.pi / 4  # azimuthal viewing angle
        qS = 0.1  # polar sky angle
        phiS = np.pi / 3  # azimuthal viewing angle
        dist = 1.0  # distance
        Phi_phi0 = 1.0
        Phi_theta0 = 2.0
        Phi_r0 = 3.0

        h = gen_wave(
            M,
            mu,
            a,
            p0,
            e0,
            Y0,
            dist,
            qS,
            phiS,
            qK,
            phiK,
            Phi_phi0,
            Phi_theta0,
            Phi_r0,
            T=T,
            dt=dt,
        )

        h2 = gen_wave_aak(
            M,
            mu,
            a,
            p0,
            e0,
            Y0,
            dist,
            qS,
            phiS,
            qK,
            phiK,
            Phi_phi0,
            Phi_theta0,
            Phi_r0,
            T=T,
            dt=dt,
        )

        breakpoint()
