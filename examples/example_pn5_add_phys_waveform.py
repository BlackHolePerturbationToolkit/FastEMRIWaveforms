import numpy as np
import matplotlib.pyplot as plt


# import all FEW stuff needed

from few.trajectory.inspiral import EMRIInspiral
from few.trajectory.ode.pn5 import dpdt8H_5PNe10, dedt8H_5PNe10, dYdt8H_5PNe10
from few.trajectory.ode.base import ODEBase

from few.waveform.base import AAKWaveformBase

from few.utils.baseclasses import *
from few.trajectory.inspiral import EMRIInspiral
from few.summation.aakwave import AAKSummation
from few.utils.utility import get_mismatch, get_separatrix, _get_separatrix_kernel_inner, get_fundamental_frequencies, _KerrGeoCoordinateFrequencies_kernel_inner
from few.utils.pn_map import Y_to_xI, _Y_to_xI_kernel_inner

# define trajectory RHS class

class PN5AddPhys(ODEBase):
    def evaluate_rhs(self, p, e, Y, Phi_phi, Phi_theta, Phi_r):
        # unpack additional arguments
        phys_factor1, phys_factor2 = self.additional_args

        # guard against bad integration steps
        if e < 0 or (p - 6 - 2* e) < 0:
            return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        
        xI = Y_to_xI(self.a, p, e, Y)
        p_sep = get_separatrix(self.a, e, xI)

        # or we can directly evaluate the numba kernels for speed (indicated below, not vectorised)
        # xI = _Y_to_xI_kernel_inner(self.a, p, e, Y)
        # p_sep = _get_separatrix_kernel_inner(self.a, e, xI)

        if p < p_sep:
             return [0., 0., 0., 0., 0., 0.,]

        pdot = dpdt8H_5PNe10(self.a, p, e, Y, 10, 10) * (1. + phys_factor1)
        edot = dedt8H_5PNe10(self.a, p, e, Y, 10, 8) * (1. + phys_factor1)
        Ydot = dYdt8H_5PNe10(self.a, p, e, Y, 7, 10)
        
        Omega_phi, Omega_theta, Omega_r = get_fundamental_frequencies(self.a, p, e, xI)
        # or we can directly evaluate the numba kernel for speed
        # frequencies = _KerrGeoCoordinateFrequencies_kernel_inner(self.a, p, e, xI)
        
        Omega_phi *= (1. + phys_factor2)

        return [pdot, edot, Ydot, Omega_phi, Omega_theta, Omega_r]

# buld waveform class
class pn5_add_phys_AAK(AAKWaveformBase, Pn5AAK, ParallelModuleBase):
    def __init__(self, *args, inspiral_kwargs={}, **kwargs):

        inspiral_kwargs["func"] = PN5AddPhys

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
add_phys_traj = EMRIInspiral(func=PN5AddPhys)

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
wave = pn5_add_phys_AAK()

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


# uncomment to time GPU
# import cupy as xp
# xp.cuda.runtime.setDevice(1)

# wave_gpu = pn5_add_phys_AAK(use_gpu=True)
# num = 50
# st = time.perf_counter()
# for _ in range(num):
#     h2 = wave_gpu(M, mu, a, p0, e0, Y0, dist, qS, phiS, qK, phiK, Phi_phi0=Phi_phi0, Phi_theta0=Phi_theta0, Phi_r0=Phi_r0, dt=dt, T=T)
# et = time.perf_counter()
# print((et - st)/num)
# breakpoint()
