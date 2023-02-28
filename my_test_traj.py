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
M = 1e4
mu = 10.0
p0 = 15.0
e0 = 0.4999
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
trajpn5.get_derivatives(mu/M, a, p0, e0, 1.0, np.asarray([1.0]))
traj.get_derivatives(mu/M, a, p0, e0, 1.0, np.asarray([0.0]))
breakpoint()
np.random.seed(32)
import matplotlib.pyplot as plt
import time, os
print(os.getpid())

# (Pdb) get_fundamental_frequencies_spin_corrections(0.9, 10.0, 0.1, 1.0)
# (-0.0010196312485235952, 0.0, 0.0013532812299746403)
# (Pdb) get_fundamental_frequencies_spin_corrections(0.9, 15.0, 0.3, 1.0)
# (-0.000331335060747499, 0.0, 0.00033285096450886626)
# (Pdb) get_fundamental_frequencies_spin_corrections(0.5, 13.0, 0.6, 1.0) 
# (-0.0005431591165829322, 0.0, 0.00029071115906960485)
get_fundamental_frequencies_spin_corrections(0.850000, 9.612440, 0.112196, 1.0)
get_fundamental_frequencies(0.850000, 9.612440, 0.112196, 1.0)

# plt.figure()
# plt.title(f"Y0={Y0},M={M:.1e},mu={mu:.1e}")
# for i in range(10000):
#     # p0 = 9.6097
#     # e0 = 0.000143448 
        
#     p0 = np.random.uniform(9.0, 15.0)
#     e0 = np.random.uniform(0.0, 0.9)
#     dom1,dom2,dom3 = get_fundamental_frequencies_spin_corrections(0.850000, p0,e0, 1.0)
#     om1,om2,om3 = get_fundamental_frequencies(0.850000,  p0,e0, 1.0)
    
#     plt.semilogy(e0,np.abs(dom1),'.')
#     plt.semilogy(e0,np.abs(dom3),'x')

# plt.xlabel('sep')

# plt.legend()
# plt.show()

second_spin = 1e-7

plt.figure()
plt.title(f"a={a},M={M:.1e},mu={mu:.1e}, secondary spin={second_spin:.2e}")
count = 0
for p0 in np.linspace(9.0, 20.0,num=10):
    e0=0.5
    # p0 = 9.6097
    # e0 = 0.000143448 
    
    
    # p0 = np.random.uniform(9.0, 25.0)
    # e0 = np.random.uniform(0.0, 0.2)
    
    Y0 = np.random.uniform(-1.0, 1.0)
    print('--------------------')
    print(p0, e0, Y0)
    
    tic = time.perf_counter()
    # t, p, e, x, Phi_phi, Phi_theta, Phi_r = trajpn5(M, mu, a, p0, e0, Y0, **insp_kw) 
    t, p, e, x, Phi_phi, Phi_theta, Phi_r = traj(M, mu, a, p0, e0, 1.0, second_spin, **insp_kw)
    toc = time.perf_counter()
    print("time=",toc-tic, Phi_phi[-1])
    plt.plot(p, e,'.',label=f" e0={e0:.2e}, p0={p0:.2e}",alpha=0.1)#time = {toc-tic:.3}, N={len(t)},


# for i in range(5):
#     # p0 = 9.6097
#     # e0 = 0.000143448 
    
#     p0 = np.random.uniform(9.0, 50.0)
#     e0 = np.random.uniform(0.1, 0.9)
#     print('--------------------')
#     print(i, p0, e0)
    
#     tic = time.time()
#     t, p, e, x, Phi_phi, Phi_theta, Phi_r = trajpn5(M, mu, a, p0, e0, Y0, **insp_kw) 
#     # t, p, e, x, Phi_phi, Phi_theta, Phi_r = traj(M, mu, a, p0, e0, 1.0, **insp_kw)
#     toc = time.time()
#     print(toc-tic, Phi_phi[-1])
#     plt.plot(p, e,'.',label=f"time = {toc-tic:.3}, N={len(t)}, e0={e0:.2e}")


plt.xlabel('p')
plt.ylabel('e')
plt.legend()
plt.show()

np.savetxt("t_test.txt",t)
np.savetxt("p_test.txt",p)
np.savetxt("e_test.txt",e)
np.savetxt("PhiPhi_test.txt",Phi_phi)
np.savetxt("PhiR_test.txt",Phi_r)

print("DONE")