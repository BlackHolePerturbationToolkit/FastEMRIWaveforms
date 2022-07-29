import unittest
import numpy as np
import warnings
import time 
from few.trajectory.inspiral import EMRIInspiral
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
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
        traj = EMRIInspiral(func="pn5")

        # set initial parameters
        M = 1e5
        mu = 1e1
        p0 = 10.0
        e0 = 0.7
        a=0.7

        # run trajectory
        t, p, e, x, Phi_phi, Phi_theta, Phi_r = traj(M, mu, a, p0, e0, 0.8, max_init_len=int(1e6))


traj = EMRIInspiral(func="pn5")

# set initial parameters
M = 1e6
mu = 5e1
p0 = 12.0
e0 = 0.1
a=0.9
Y0 = 0.8

# run trajectory
T=1.0
t_vec = np.linspace(0.0, T*365.0*3600*24)
err_vec = 10**np.linspace(-1.0, -15.0, num=10)
dt_vec = 10**np.linspace(0.0, 4,num=10)

insp_kw = {
        "T": T,
        "dt":1e-10,
        "err": 1e-20,
        "DENSE_STEPPING": 0,
        "max_init_len": int(1e3),
        "use_rk4": False,
        "upsample": True,
        "new_t": t_vec
        }

t, p_true, e, x, Phi_phi, Phi_theta, Phi_r = traj(M, mu, a, p0, e0, Y0, **insp_kw)#err=err)

plt.figure()
for dt in dt_vec[::-1]:
    p_vec = []
    for err in err_vec:
        insp_kw = {
                "T": T,
                "dt": dt,
                "err": err,
                "DENSE_STEPPING": 0,
                "max_init_len": int(1e3),
                "use_rk4": False,
                "upsample": True,
                "new_t": t_vec
                }

        st = time.time()
        t, p, e, x, Phi_phi, Phi_theta, Phi_r = traj(M, mu, a, p0, e0, Y0, **insp_kw)#err=err)
        print("trajectory timing",time.time()-st )
        p_vec.append(p)

    diff = p_vec - p_vec[-1]#p_true#p_vec[-1]
    diff = np.array(diff)[:-1]
    # print(np.mean(diff,axis=1))
    mean_err = np.sum(np.abs(diff)**2,axis=1)
    print(dt, mean_err)
    plt.loglog(err_vec[:-1], mean_err, 'o-', label=f'dt={dt:.2e}')

plt.xlabel('mean error')
plt.xlabel('err')
plt.legend()
plt.show()



names = list(mcolors.TABLEAU_COLORS)
# breakpoint()

plt.figure()
[plt.semilogy(t, np.abs(dd), color=cc) for dd,err,cc in zip(diff,err_vec,names)]
[plt.axhline(err , label=f'err = {err:.2e}', color=cc, linestyle=':') for dd,err,cc in zip(diff,err_vec,names)]
# [plt.plot(t, pp) for pp in p_vec]
plt.legend()
plt.show()
