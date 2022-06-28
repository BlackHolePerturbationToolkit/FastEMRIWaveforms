import numpy as np
import matplotlib.pyplot as plt


# import all FEW stuff needed

from few.trajectory.inspiral import EMRIInspiral

from few.waveform import AAKWaveformBase

from few.utils.baseclasses import *
from few.trajectory.inspiral import EMRIInspiral
from few.summation.aakwave import AAKSummation
from few.utils.utility import get_mismatch

# buld waveform class
class pn5_add_phys_AAK(AAKWaveformBase, Pn5AAK, ParallelModuleBase):
    def __init__(self, *args, inspiral_kwargs={}, **kwargs):

        inspiral_kwargs["func"] = "pn5_add_phys"

        AAKWaveformBase.__init__(
            self,
            EMRIInspiral,
            AAKSummation,
            inspiral_kwargs=inspiral_kwargs,
            **kwargs
        )

    @property
    def gpu_capability(self):
        return True

    @property
    def allow_batching(self):
        return False
        

# define parameters
M = 1e6
mu = 5e1
a = 0.3
p0 = 13.0
e0 = 0.2
Y0 = 0.8
dist = 1.0
qS = np.pi/3
phiS = np.pi/4
qK = np.pi/5.
phiK = np.pi/6.
Phi_phi0=0.0
Phi_theta0=0.0
Phi_r0=0.0
mich=False
dt=10.0
T=2.0

# initialize trajectory
add_phys_traj = EMRIInspiral(func="pn5_add_phys")

# run without effect
phys_factor1 = 0.0
phys_factor2 = 0.0
t1, p1, e1, Y1, pp1, pt1, pr1 = add_phys_traj(M, mu, a, p0, e0, Y0, phys_factor1, phys_factor2)

# run with strong effect
phys_factor1 = 1e-1
phys_factor2 = 1e-1
t2, p2, e2, Y2, pp2, pt2, pr2 = add_phys_traj(M, mu, a, p0, e0, Y0, phys_factor1, phys_factor2)

# plot difference over time
plt.plot(t1, p1)
plt.plot(t2, p2)
plt.savefig("plot1.png")
plt.close()

# initialize waveform generator
wave = pn5_add_phys_AAK(num_threads=1)

# run original wave
phys_factor1 = 0.0
phys_factor2 = 0.0
h1 = wave(M, mu, a, p0, e0, Y0, dist, qS, phiS, qK, phiK, phys_factor1, phys_factor2, Phi_phi0=Phi_phi0, Phi_theta0=Phi_theta0, Phi_r0=Phi_r0, dt=dt, T=T)

# run adjusted wave
phys_factor1 = 1e-8
phys_factor2 = 1e-8
h2 = wave(M, mu, a, p0, e0, Y0, dist, qS, phiS, qK, phiK, phys_factor1, phys_factor2, Phi_phi0=Phi_phi0, Phi_theta0=Phi_theta0, Phi_r0=Phi_r0, dt=dt, T=T)

# plot difference
plt.plot(np.arange(len(h1)) * dt, h1.real)
plt.plot(np.arange(len(h2)) * dt, h2.real, '--')
plt.savefig("plot2.png")

print(f"Mismatch: {get_mismatch(h1, h2)}")

# time CPU
import time 
num = 3
st = time.perf_counter()
for _ in range(num):
    h2 = wave(M, mu, a, p0, e0, Y0, dist, qS, phiS, qK, phiK, phys_factor1, phys_factor2, Phi_phi0=Phi_phi0, Phi_theta0=Phi_theta0, Phi_r0=Phi_r0, dt=dt, T=T)
et = time.perf_counter()
print((et - st)/num)


# time GPU
import cupy as xp
xp.cuda.runtime.setDevice(1)

wave_gpu = pn5_add_phys_AAK(use_gpu=True)
num = 50
st = time.perf_counter()
for _ in range(num):
    h2 = wave_gpu(M, mu, a, p0, e0, Y0, dist, qS, phiS, qK, phiK, Phi_phi0=Phi_phi0, Phi_theta0=Phi_theta0, Phi_r0=Phi_r0, dt=dt, T=T)
et = time.perf_counter()
print((et - st)/num)
breakpoint()
