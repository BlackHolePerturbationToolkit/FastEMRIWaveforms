import unittest
import numpy as np
import warnings
import time 
from few.trajectory.inspiral import EMRIInspiral

try:
    import cupy as xp

    gpu_available = True

except (ModuleNotFoundError, ImportError) as e:
    import numpy as xp

    warnings.warn(
        "CuPy is not installed or a gpu is not available. If trying to run on a gpu, please install CuPy."
    )
    gpu_available = False

T=5.0
dt = 10.0

insp_kw = {
"T": T,
"dt": dt,
"err": 1e-10,
"DENSE_STEPPING": 0,
"max_init_len": int(1e4),
"use_rk4": False,
"upsample": False,
}
np.random.seed(42)

class ModuleTest(unittest.TestCase):
    def test_trajectory_pn5(self):

        # initialize trajectory class
        traj = EMRIInspiral(func="pn5")

        # set initial parameters
        M = 1e5
        mu = 1e1
        np.random.seed(42)
        for i in range(1000):
            p0 = np.random.uniform(10.0,15)
            e0 = np.random.uniform(0.0, 1.0)
            a = np.random.uniform(0.0, 1.0)
            Y0 = np.random.uniform(-1.0, 1.0)

            # do not want to be too close to polar
            if np.abs(Y0) < 1e-2:
                Y0 = np.sign(Y0) * 1e-2

            # run trajectory
            #print("start", a, p0, e0, Y0)
            t, p, e, x, Phi_phi, Phi_theta, Phi_r = traj(M, mu, a, p0, e0, Y0, **insp_kw)

    def test_trajectory_SchwarzEccFlux(self):
        # initialize trajectory class
        traj = EMRIInspiral(func="SchwarzEccFlux")

        # set initial parameters
        M = 1e5
        mu = 1e1
        p0 = 10.0
        e0 = 0.7

        # run trajectory
        t, p, e, x, Phi_phi, Phi_theta, Phi_r = traj(M, mu, 0.0, p0, e0, 1.0)