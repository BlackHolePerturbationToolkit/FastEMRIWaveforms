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

        # initialize trajectory class
        traj = EMRIInspiral(func="KerrEccentricEquatorial")

        # set initial parameters
        M = 1e5
        mu = 1e1
        p0 = 10.0
        e0 = 0.7
        a=0.85

        # run trajectory
        t, p, e, x, Phi_phi, Phi_theta, Phi_r = traj(M, mu, a, p0, e0, 1.0)

        traj = EMRIInspiral(func="pn5")

        # set initial parameters
        M = 1e5
        mu = 1e1
        p0 = 10.0
        e0 = 0.7
        a=0.85

        # run trajectory
        t, p, e, x, Phi_phi, Phi_theta, Phi_r = traj(M, mu, a, p0, e0, 1.0)



traj = EMRIInspiral(func="KerrEccentricEquatorial")
trajpn5 = EMRIInspiral(func="pn5")
# set initial parameters
M = 1e6
mu = 5e1
p0 = 15.0
e0 = 0.4999
a=0.85

# run trajectory
err = 1e-5
insp_kw = {
    "T": 10.0,
    "dt": 10.0,
    "err": err,
    "DENSE_STEPPING": 0,
    "max_init_len": int(1e4),
    "use_rk4": False,
    # "upsample": True,
    # "fix_T": True

    }
np.random.seed(32)
import matplotlib.pyplot as plt
plt.figure()
for _ in range(50):
    # p0 = 10.021478000424167 
    # e0 = 0.29088984025761766
    p0 = np.random.uniform(10.0, 20.0)
    e0 = np.random.uniform(0.0, 0.5)
    # t, p, e, x, Phi_phi, Phi_theta, Phi_r = trajpn5(M, mu, a, p0, e0, 1.0, **insp_kw)
    t, p, e, x, Phi_phi, Phi_theta, Phi_r = traj(M, mu, a, p0, e0, 1.0, **insp_kw)
    plt.plot(p, e,'.')
plt.ylim([0.0, 0.5])
plt.xlim([2.0, 16.0])
plt.show()

print("DONE")