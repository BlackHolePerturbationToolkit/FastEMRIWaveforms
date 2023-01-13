#python -m unittest few/tests/test_traj.py 
import unittest
import numpy as np
import warnings

from few.trajectory.inspiral import EMRIInspiral
from few.amplitude.romannet import RomanAmplitude
from few.amplitude.interp2dcubicspline import Interp2DAmplitude
from few.waveform import FastSchwarzschildEccentricFlux, SlowSchwarzschildEccentricFlux
from few.utils.utility import get_overlap, get_mismatch, get_separatrix, get_fundamental_frequencies
from few.utils.ylm import GetYlms
from few.utils.modeselector import ModeSelector
from few.summation.interpolatedmodesum import CubicSplineInterpolant

try:
    import cupy as xp

    gpu_available = True

except (ModuleNotFoundError, ImportError) as e:
    import numpy as xp

    warnings.warn(
        "CuPy is not installed or a gpu is not available. If trying to run on a gpu, please install CuPy."
    )
    gpu_available = False


class ModuleTest(unittest.TestCase):
    def test_trajectory(self):


        err = 1e-10
        
        # initialize trajectory class
        traj = EMRIInspiral(func="KerrEccentricEquatorial")

        # set initial parameters
        M = 1e6
        mu = 1e1
        p0 = 30.0
        e0 = 0.0001
        a=0.85

        # run trajectory
        t, p, e, x, Phi_phi, Phi_theta, Phi_r = traj(M, mu, a, p0, e0, 1.0, T=10, err=err)
        # t_pn, p_pn, e_pn, x_pn, Phi_phi_pn, Phi_theta_pn, Phi_r_pn = traj(M, mu, a, p0, e0, 1.0, new_t=t, upsample=True, err=err/10)

        traj = EMRIInspiral(func="pn5")

        # run trajectory
        t_pn, p_pn, e_pn, x_pn, Phi_phi_pn, Phi_theta_pn, Phi_r_pn = traj(M, mu, a, p0, e0, 1.0, T=10)#new_t=t, upsample=True, err=err)

        import matplotlib.pyplot as plt
        plt.figure(); plt.semilogy(p, Phi_phi); plt.semilogy(p_pn, Phi_phi_pn, label='pn5'); plt.legend(); plt.show()
        plt.figure(); plt.plot(p, e); plt.plot(p_pn, e_pn, label='pn5'); plt.legend(); plt.show()
