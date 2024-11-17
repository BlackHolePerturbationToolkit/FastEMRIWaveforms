import unittest
import warnings

# from few.waveform import FastSchwarzschildEccentricFlux 
from few.waveform import GenerateEMRIWaveform

from few.utils.utility import get_mismatch
from few.trajectory.ode import KerrEccEqFlux

try:
    import cupy as xp

    use_gpu = True

except (ModuleNotFoundError, ImportError) as e:
    import numpy as xp

    warnings.warn(
        "CuPy is not installed or a gpu is not available. If trying to run on a gpu, please install CuPy."
    )
    use_gpu = False


# keyword arguments for inspiral generator (Kerr Waveform)
inspiral_kwargs_Kerr = {
                "DENSE_STEPPING": 0,  # we want a sparsely sampled trajectory
                "max_init_len": int(
                    1e3
                ),  # all of the trajectories will be well under len = 1000
            "func": KerrEccEqFlux
                        }

class KerrWaveformTest(unittest.TestCase):
    def test_Kerr_vs_Schwarzchild(self):
        # Test whether the Kerr and Schwarzschild waveforms agree.  

        wave_generator_Kerr = GenerateEMRIWaveform("FastKerrEccentricEquatorialFlux",
                                                        use_gpu=use_gpu)
        wave_generator_Schwarz = GenerateEMRIWaveform("FastSchwarzschildEccentricFlux",
                                                        use_gpu=use_gpu)
                                                                                  

        # parameters
        M = 1e6
        mu = 1e1
        p0 = 10.0
        e0 = 0.4
        
        qS = 0.2
        phiS = 0.2
        qK = 0.8
        phiK = 0.8

        Phi_phi0 = 1.0
        Phi_theta0 = 2.0 
        Phi_r0 = 3.0 

        dist = 1.0
        dt = 10.0
        T = 0.1 

        Kerr_wave = wave_generator_Kerr(M, mu, 0.0, p0, e0, 1.0, dist, qS, phiS, qK, phiK, Phi_phi0, Phi_theta0, Phi_r0, T = T, dt = dt)
        Schwarz_wave = wave_generator_Schwarz(M, mu, 0.0, p0, e0, 1.0, dist, qS, phiS, qK, phiK, Phi_phi0, Phi_theta0, Phi_r0, T = T, dt = dt)

        mm = get_mismatch(Kerr_wave, Schwarz_wave, use_gpu=use_gpu)

        self.assertLess(mm, 1e-4)
    def test_retrograde_orbits(self):
        """
        Here we test that retrograde orbits and prograde orbits for a = \pm 0.7
        have large mismatches.
        """
        print("Testing retrograde orbits")

        wave_generator_Kerr = GenerateEMRIWaveform("FastKerrEccentricEquatorialFlux",
                                                        use_gpu=use_gpu)

        # parameters
        M = 1e6
        mu = 1e1
        a = 0.7
        p0 = 15.0
        e0 = 0.4
        
        qS = 0.2
        phiS = 0.2
        qK = 0.8
        phiK = 0.8

        Phi_phi0 = 1.0
        Phi_theta0 = 2.0 
        Phi_r0 = 3.0 

        dist = 1.0
        dt = 10.0
        T = 0.1 

        Kerr_wave_retrograde = wave_generator_Kerr(M, mu, abs(a), p0, e0, -1.0, dist, qS, phiS, qK, phiK, Phi_phi0, Phi_theta0, Phi_r0, T = T, dt = dt)
        Kerr_wave_prograde = wave_generator_Kerr(M, mu, abs(a), p0, e0, 1.0, dist, qS, phiS, qK, phiK, Phi_phi0, Phi_theta0, Phi_r0, T = T, dt = dt)

        mm = get_mismatch(Kerr_wave_retrograde, Kerr_wave_prograde, use_gpu=use_gpu)
        self.assertGreater(mm, 1e-3)


