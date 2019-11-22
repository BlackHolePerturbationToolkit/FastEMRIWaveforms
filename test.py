import numpy as np
import time

try:
    import cupy as xp
except ImportError:
    import numpy as xp

from few.create_waveform import CreateWaveform

nn_kwargs = dict(input_str="SE_", folder="few/files/weights/", activation_kwargs={})

kwargs = dict(transform_file="few/files/reduced_basis.dat", nn_kwargs=nn_kwargs)

# Load the inspiral data. This should be in the format (p, e, Phi_phi, Phi_r)
traj = np.genfromtxt("insp_p12.5_e0.7_tspacing_1M.dat")[0::3][:100000]

batch_size = kwargs["batch_size"] = len(traj)

p = xp.asarray(traj[:, 0], dtype=xp.float32)
e = xp.asarray(traj[:, 1], dtype=xp.float32)
Phi_phi = xp.asarray(traj[:, 2], dtype=xp.float32)
Phi_r = xp.asarray(traj[:, 3], dtype=xp.float32)

l = xp.zeros(2214, dtype=int)
m = xp.zeros(2214, dtype=int)
n = xp.zeros(2214, dtype=int)

ind = 0
for l_i in range(2, 10 + 1):
    for m_i in range(1, l_i + 1):
        for n_i in range(-20, 20 + 1):
            l[ind] = l_i
            m[ind] = m_i
            n[ind] = n_i
            ind += 1

cw = CreateWaveform(**kwargs)
theta = np.pi / 4
phi = np.pi / 3

num = 10
out = cw(p, e, Phi_r, Phi_phi, l, m, n, theta, phi)
check = []
for _ in range(num):
    st = time.perf_counter()
    out = cw(p, e, Phi_r, Phi_phi, l, m, n, theta, phi)
    et = time.perf_counter()
    print("Timing:", (et - st))
