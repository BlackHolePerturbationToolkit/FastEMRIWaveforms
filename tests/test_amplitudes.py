# python -m unittest few/tests/test_traj.py
import unittest
import numpy as np
import time

from few.trajectory.inspiral import EMRIInspiral
from few.utils.constants import YRSID_SI
from few.trajectory.ode import KerrEccEqFlux, PN5, SchwarzEccFlux

from few.utils.globals import get_logger

few_logger = get_logger()

few_logger.warning("Traj Test is running")

T = 1000.0
dt = 1.0
# set initial parameters
M = 1e5
mu = 1e1
x0 = 1.0

insp_kw = {
    "T": T,
    "dt": dt,
    "err": 1e-10,
    "DENSE_STEPPING": 0,
    "buffer_length": int(1e4),
    "upsample": False,
}

np.random.seed(42)

def run_forward_back(traj_module, M, mu, a, p0, e0, xI0, forwards_kwargs):
    """
    Run a trajectory forward, then run a trajectory backward from the finish point and return both result sets.
    """

    forwards_result = traj_module(M, mu, a, p0, e0, xI0, **forwards_kwargs)

    # Now test backwards integration
    final_p = forwards_result[1][-1]
    final_e = forwards_result[2][-1]
    final_x = forwards_result[3][-1]

    insp_kw_back = forwards_kwargs.copy()
    insp_kw_back.update({"integrate_backwards": True})
    insp_kw_back.update({"T": forwards_result[0][-1] / YRSID_SI})

    backwards_result = traj_module(M, mu, a, final_p, final_e, final_x, **insp_kw_back)

    return forwards_result, backwards_result

N_TESTS = 10

class ModuleTest(unittest.TestCase):
    def test_amplitude(self):
        l,m,n = 2,2,0
        # amp1 =
        # amp2 = 
        # self.assertAlmostEqual(amp1, np.conj(amp2), places=6)
