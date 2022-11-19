import unittest
import numpy as np
import warnings

from few.trajectory.inspiral import EMRIInspiral
from few.amplitude.romannet import RomanAmplitude
from few.amplitude.interp2dcubicspline import Interp2DAmplitude
from few.waveform import FastSchwarzschildEccentricFlux, SlowSchwarzschildEccentricFlux
from few.utils.utility import get_overlap, get_mismatch
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
        a=0.7

        # run trajectory
        t, p, e, x, Phi_phi, Phi_theta, Phi_r = traj(M, mu, a, p0, e0, 1.0)


traj = EMRIInspiral(func="KerrEccentricEquatorial")

# set initial parameters
M = 1e6
mu = 5e1
p0 = 15.0
e0 = 0.4999
a=0.989999

# run trajectory
err_vec = 10**np.linspace(-8.0, -10.0, num=1)
p_vec = []
e_vec = []
for err in err_vec:
    insp_kw = {
            "T": 5.0,
            "dt": 10.0,
            "err": err,
            "DENSE_STEPPING": 0,
            "max_init_len": int(1e4),
            "use_rk4": False,
            # "upsample": True,
            # "fix_T": True

            }

    t, p, e, x, Phi_phi, Phi_theta, Phi_r = traj(M, mu, a, p0, e0, -1.0, **insp_kw)
    print('err',err)
    print(Phi_phi[-1])
    p_vec.append(p)
    e_vec.append(e)

diff = p_vec - p_vec[-1]
diff = np.array(diff)

import matplotlib.pyplot as plt
plt.figure()
# [plt.semilogy(t, np.abs(dd) , label='err = ') for dd in diff]
[plt.plot(pp, ee) for pp,ee in zip(p_vec,e_vec)]
plt.show()

print("DONE")