import numpy as np
import matplotlib.pyplot as plt
from few.trajectory.inspiral import EMRIInspiral
from few.utils.constants import YRSID_SI
import time

fn = "KerrEccentricEquatorial"

# todo: once we are happy, remove the rootfind_separatrix kwarg and delete this script
traj_new = EMRIInspiral(func=fn, rootfind_separatrix=True)
traj_old = EMRIInspiral(func=fn, rootfind_separatrix=False)

M = 1e6
mu = 1e3
a = 0.9
p0 = 3.
e0 = 0.2
xI = 1.

Phiphi0 = np.pi/3
Phir0 = np.pi/4


# T = 48892595.6247182 / YRSID_SI #10.
T = 4.
dt = 10.

pars = [M, mu, a, p0, e0, xI]

t1 = []
t2 = []
for i in range(300):
    st = time.perf_counter()
    tr1 = traj_old(*pars, Phi_phi0=Phiphi0, Phi_r0=Phir0, dt=dt, T=T, use_rk4=False)
    mt1 = time.perf_counter()
    tr2 = traj_new(*pars, Phi_phi0=Phiphi0, Phi_r0=Phir0, dt=dt, T=T, use_rk4=False)
    mt2 = time.perf_counter()
    t1.append(mt1-st)
    t2.append(mt2-mt1)

print("OLD:", np.median(t1)*1000, "ms +/-", np.std(t1[100:])*1e6,"mus", tr1[0].size, np.array(tr1)[[1,2,4],-1])
print("NEW:", np.median(t2)*1000, "ms +/-", np.std(t2[100:])*1e6,"mus", tr2[0].size, np.array(tr2)[[1,2,4],-1])

# print(tr1[0][-1], tr2[0][-1])
print("DELTA PHI: ", tr1[4][-1] - tr2[4][-1])   # this is worse for rk8 and more extreme mass ratios

plt.figure()
plt.subplot(211)
plt.semilogy(tr1[1],tr1[2])
plt.semilogy(tr2[1],tr2[2])
plt.ylabel("e")
plt.subplot(212)
plt.plot(tr1[1],tr1[4]-tr2[4])
plt.xlabel("p")
plt.ylabel("Phi_phi")
plt.savefig("test_traj.pdf", bbox_inches="tight")