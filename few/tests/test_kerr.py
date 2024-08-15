import unittest
import pickle
import numpy as np
import warnings

from few.trajectory.inspiral import EMRIInspiral

from few.amplitude.interp2dcubicspline import Interp2DAmplitude
from few.amplitude.ampinterp2d import AmpInterpKerrEqEcc
from few.amplitude.romannet import RomanAmplitude
from few.amplitude.interp2dcubicspline import Interp2DAmplitude

# from few.waveform import FastSchwarzschildEccentricFlux 
from few.waveform import GenerateEMRIWaveform, SchwarzschildEccentricWaveformBase, KerrEquatorialEccentricWaveformBase 

from few.utils.utility import get_overlap, get_mismatch
from few.utils.ylm import GetYlms
from few.utils.modeselector import ModeSelector

from few.summation.interpolatedmodesum import CubicSplineInterpolant
from few.summation.interpolatedmodesum import CubicSplineInterpolant
from few.summation.interpolatedmodesum import InterpolatedModeSum
from few.summation.directmodesum import DirectModeSum

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
            "func": "KerrEccentricEquatorial"
                        }

class KerrWaveformTest(unittest.TestCase):
    def test_Kerr_vs_Schwarzchild(self):
        # Test whether the Kerr and Schwarzschild waveforms agree.  
        amplitude_kwargs_Kerr = {"specific_spins":[-0.10, 0.0, 0.10]}

        inspiral_kwargs_Schwarz = inspiral_kwargs_Kerr.copy()
        inspiral_kwargs_Schwarz["func"] = "SchwarzEccFlux"

        # keyword arguments for summation generator (InterpolatedModeSum)
        wave_generator_Kerr = GenerateEMRIWaveform(KerrEquatorialEccentricWaveformBase,
                                                        EMRIInspiral,
                                                        AmpInterpKerrEqEcc,
                                                        InterpolatedModeSum,
                                                        ModeSelector,
                                                        inspiral_kwargs=inspiral_kwargs_Kerr,
                                                        amplitude_kwargs = amplitude_kwargs_Kerr,
                                                        use_gpu=use_gpu)
        wave_generator_Schwarz = GenerateEMRIWaveform(SchwarzschildEccentricWaveformBase,
                                                        EMRIInspiral,
                                                        Interp2DAmplitude,
                                                        InterpolatedModeSum,
                                                        ModeSelector,
                                                        inspiral_kwargs=inspiral_kwargs_Schwarz,
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

        self.assertLess(mm, 1e-5)
    def test_retrograde_orbits(self):
        """
        Here we test that retrograde orbits and prograde orbits for a = \pm 0.7
        have large mismatches.
        """
        print("Testing retrograde orbits")
        amplitude_kwargs_Kerr = {"specific_spins":[-0.7, 0.7]}

        # keyword arguments for summation generator (InterpolatedModeSum)
        wave_generator_Kerr = GenerateEMRIWaveform(KerrEquatorialEccentricWaveformBase,
                                                        EMRIInspiral,
                                                        AmpInterpKerrEqEcc,
                                                        InterpolatedModeSum,
                                                        ModeSelector,
                                                        inspiral_kwargs=inspiral_kwargs_Kerr,
                                                        amplitude_kwargs = amplitude_kwargs_Kerr,
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


