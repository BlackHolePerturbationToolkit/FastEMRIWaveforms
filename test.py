import numpy as np
import time

try:
    import cupy as xp

    print("running cupy")
except ImportError:
    import numpy as xp

    print("running numpy")

from few.create_waveform import CreateWaveform

nn_kwargs = dict(input_str="SE_", folder="few/files/weights/", activation_kwargs={})

kwargs = dict(transform_file="few/files/reduced_basis.dat", nn_kwargs=nn_kwargs)

# Load the inspiral data. This should be in the format (p, e, Phi_phi, Phi_r)
traj = np.genfromtxt("insp_p12.5_e0.7_tspacing_1M.dat")[0::3][:1000]

batch_size = kwargs["batch_size"] = len(traj)

p = xp.asarray(traj[:, 0], dtype=xp.float32)
e = xp.asarray(traj[:, 1], dtype=xp.float32)
Phi_phi = xp.asarray(traj[:, 2], dtype=xp.float32)
Phi_r = xp.asarray(traj[:, 3], dtype=xp.float32)

l = xp.zeros(2214, dtype=int)
m = xp.zeros(2214, dtype=int)
n = xp.zeros(2214, dtype=int)

ind = 0
mode_inds = {}
total_n = 41
for l_i in range(2, 10 + 1):
    for m_i in range(1, l_i + 1):
        ind_start = ind
        num_n_here = 0
        for n_i in range(-20, 20 + 1):
            l[ind] = l_i
            m[ind] = m_i
            n[ind] = n_i

            mode_inds[(l_i, m_i, n_i)] = ind
            mode_inds[(l_i, -m_i, n_i)] = ind_start + total_n - 1 - num_n_here

            ind += 1
            num_n_here += 1


kwargs["mode_inds"] = mode_inds
cw = CreateWaveform(**kwargs)
theta = np.pi / 2
phi = 0.0  # np.pi / 3

num = 1
get_modes = [(2, 2, 0), (3, 2, 2), (4, -2, -18), (7, 7, 10)]
out = cw(p, e, Phi_r, Phi_phi, l, m, n, theta, phi, get_modes=get_modes,)

import matplotlib.pyplot as plt

for mode in get_modes:
    plt.plot(out[mode].real, label=mode)

plt.legend()
plt.show()
import pdb

pdb.set_trace()
check = []
for _ in range(num):
    st = time.perf_counter()
    out = cw(p, e, Phi_r, Phi_phi, l, m, n, theta, phi)
    if xp != np:
        xp.cuda.Device(0).synchronize()
    et = time.perf_counter()
    print("Timing:", (et - st))

import matplotlib.pyplot as plt
import pdb

plt.plot(out.imag)
plt.plot(out.real)
# plt.savefig("orig_traj_zoom.pdf")
plt.show()
plt.close()

from pyNIT import NIT
from scipy.interpolate import CubicSpline

t, p, e, Phi_phi, Phi_r = NIT(p[0], e[0])
dt = 1.0
t_new = np.arange(0.0, t[-1] + dt, dt)[:10000]

p_spline = CubicSpline(t, p)
e_spline = CubicSpline(t, e)
Phi_phi_spline = CubicSpline(t, Phi_phi)
Phi_r_spline = CubicSpline(t, Phi_r)

p, e, Phi_phi, Phi_r = (
    p_spline(t_new),
    e_spline(t_new),
    Phi_phi_spline(t_new),
    Phi_r_spline(t_new),
)
batch_size = kwargs["batch_size"] = len(t_new)

cw = CreateWaveform(**kwargs)
out = cw(p, e, Phi_r, Phi_phi, l, m, n, theta, phi)

plt.plot(out.imag)
plt.plot(out.real)
plt.savefig("flux_traj_reversing_Zlmnk_bad_ylm.pdf")
plt.show()
pdb.set_trace()
