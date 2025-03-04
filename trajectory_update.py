# python -m unittest few/tests/test_traj.py
import unittest
import numpy as np
import time

from few.trajectory.inspiral import EMRIInspiral
from few.utils.constants import YRSID_SI
from few.trajectory.ode import KerrEccEqFlux, PN5, SchwarzEccFlux, KerrEccEqFluxAPEX

from few.utils.globals import get_logger

few_logger = get_logger()

few_logger.warning("Traj Test is running")

T = 1000.0
dt = 10.0

insp_kw = {
    "T": T,
    "dt": 1.0,
    "err": 1e-10,
    "DENSE_STEPPING": 0,
    "buffer_length": int(1e4),
    "upsample": False,
}

np.random.seed(42)
import matplotlib.pyplot as plt

traj = EMRIInspiral(func=KerrEccEqFluxAPEX)
trajELQ = EMRIInspiral(func=KerrEccEqFlux)
# # set initial parameters
M = 1e6
mu = 100.0
a = 0.5
p0 = 6.0
e0 = 1e-10
t, p, e, x, Phi_phi, Phi_theta, Phi_r = traj(M, mu, a, p0, e0, 1.0, T=1.0, dt=1.0)

start_time = time.time()
t, p, e, x, Phi_phi, Phi_theta, Phi_r = traj(M, mu, a, p0, e0, 1.0, T=1.0, dt=1.0)
traj_time = time.time() - start_time
print(f"Time taken for traj: {traj_time} seconds")

# fig, axs = plt.subplots(3, 1, figsize=(10, 8))

# # Plot difference in p
# axs[0].plot(p, p - pELQ)
# axs[0].set_ylabel('Difference in p')
# axs[0].legend(['p - pELQ'])

# # Plot difference in e
# axs[1].plot(p, e - eELQ)
# axs[1].set_ylabel('Difference in e')
# axs[1].legend(['e - eELQ'])

# # Plot difference in Phi_phi
# axs[2].plot(p, Phi_phi - Phi_phiELQ)
# axs[2].set_ylabel('Difference in Phi_phi')
# axs[2].set_xlabel('Time')
# axs[2].legend(['Phi_phi - Phi_phiELQ'])

# plt.tight_layout()
# plt.show()