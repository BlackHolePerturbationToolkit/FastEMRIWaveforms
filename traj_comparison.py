#python -m unittest few/tests/test_traj.py 
import unittest
import numpy as np
import warnings

from few.trajectory.inspiral import EMRIInspiral
from few.amplitude.romannet import RomanAmplitude
from few.amplitude.interp2dcubicspline import Interp2DAmplitude
from few.waveform import FastSchwarzschildEccentricFlux, SlowSchwarzschildEccentricFlux
from few.utils.utility import get_overlap, get_mismatch, get_separatrix, get_fundamental_frequencies, get_fundamental_frequencies_spin_corrections
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

traj = EMRIInspiral(func="KerrEccentricEquatorial")
trajpn5 = EMRIInspiral(func="pn5")
# set initial parameters
M = 1e6
mu = 8.0
p0 = 10.0
e0 = 0.45
a=0.85
Y0 = 1.0
# run trajectory
err = 1e-10
insp_kw = {
    "T": 10.0,
    "dt": 10.0,
    "err": err,
    "DENSE_STEPPING": 0,
    "max_init_len": int(1e4),
    # "upsample": True,
    # "fix_T": True

    }

np.random.seed(32)
import matplotlib.pyplot as plt
import time, os
print(os.getpid())

second_spin = 0.0

plt.figure()
plt.title(f"a={a},M={M:.1e},mu={mu:.1e}\n e0={e0:.2}, p0={p0:.2}, sigma={second_spin:.2e}")
t, p, e, x, Phi_phi, Phi_theta, Phi_r = trajpn5(M, mu, a, p0, e0, Y0, **insp_kw) 
plt.plot(p, e,'-',label=f"PN5, inspiral duration={t[-1]/3.14e7:.2} yrs")
t, p, e, x, Phi_phi, Phi_theta, Phi_r = traj(M, mu, a, p0, e0, Y0, second_spin, **insp_kw)
plt.plot(p, e,':',label=f"Fluxes, inspiral duration={t[-1]/3.14e7:.2} yrs")

plt.xlabel('p')
plt.ylabel('e')
plt.legend()
plt.show()