#python -m unittest few/tests/test_traj.py 
import unittest
import numpy as np
import warnings
import matplotlib.pyplot as plt
from matplotlib import cm
import time, os
print("PID",os.getpid())
from few.trajectory.inspiral import EMRIInspiral
from few.amplitude.romannet import RomanAmplitude
from few.amplitude.interp2dcubicspline import Interp2DAmplitude
from few.waveform import FastSchwarzschildEccentricFlux, SlowSchwarzschildEccentricFlux
from few.utils.utility import get_overlap, get_mismatch, get_separatrix, get_fundamental_frequencies, get_fundamental_frequencies_spin_corrections
from few.utils.ylm import GetYlms
from few.utils.modeselector import ModeSelector
from few.summation.interpolatedmodesum import CubicSplineInterpolant
from few.utils.constants import *
np.random.seed(32)

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
trajS = EMRIInspiral(func="SchwarzEccFlux")
trajpn5 = EMRIInspiral(func="pn5")


insp_kw = {
    "T": 10.0,
    "dt": 10.0,
    "err": 1e-13,
    "DENSE_STEPPING": 0,
    "max_init_len": int(1e4),
    # "upsample": True,
    # "fix_T": True
    }

# set initial parameters
M = 1e6
mu = 50.0
epsilon = mu/M
p0 = 15.0
e0 = 0.4
a=0.85
Y0 = 1.0
second_spin = 0.0

# check fluxes
p_all, e_all = np.asarray([temp.ravel() for temp in np.meshgrid( np.linspace(4.0, 14.0), np.linspace(0.01, 0.5))])
out = np.asarray([traj.get_derivative(epsilon, a, pp, ee, 1.0, np.asarray([0.0]))  for pp,ee in zip(p_all,e_all)])
pdot = out[:,0]/epsilon
edot = out[:,1]/epsilon
Omega_phi = out[:,3]
Omega_r = out[:,5]
pdot_full = pdot.copy()

plt.figure()
cb = plt.tricontourf(p_all, e_all, np.log10(np.abs(pdot)))
plt.colorbar(cb,label=r'$log_{10} (\dot{p}) $')
plt.xlabel('p')
plt.ylabel('e')
plt.tight_layout()
plt.savefig('pdot.png')

plt.figure()
cb = plt.tricontourf(p_all, e_all, np.log10(np.abs(edot)))
plt.colorbar(cb,label=r'$log_{10} (\dot{e}) $')
plt.xlabel('p')
plt.ylabel('e')
plt.tight_layout()
plt.savefig(f'edot.png')

np.savetxt(f"p_e_pdot_edot.txt", np.asarray((p_all, e_all, pdot, edot)).T )

#################################################################
tic = time.perf_counter()
t, p, e, x, Phi_phi, Phi_theta, Phi_r = traj(M, mu, a, p0, e0, 1.0, second_spin, **insp_kw)
toc = time.perf_counter()
print("time=",toc-tic, Phi_phi[-1])

out = np.asarray([traj.get_derivative(epsilon, a, pp, ee, 1.0, np.asarray([0.0]))  for pp,ee in zip(p,e)])

pdot = out[:,0]/epsilon
edot = out[:,1]/epsilon
Omega_phi = out[:,3]
Omega_r = out[:,5]

om1,om2,om3 = get_fundamental_frequencies(a,p,e,np.ones_like(p))
print("check = ",np.all(Omega_phi==om1),np.all(Omega_r==om3))

#################################################################

fig, axes = plt.subplots(2, 3)

plt.subplots_adjust(wspace=0.3)
fig.set_size_inches(14, 8)
err = insp_kw["err"]
plt.title(f"a={a},M={M:.1},mu={mu:.1}, sec. spin={second_spin:.2e}\nprecision integrator = {err:.1e}")

axes = axes.ravel()

ylabels = [r'$e$', r'$p$', r'$e$', r'$\Phi_\phi$', r'$\Phi_r$', r'Flux']
xlabels = [r'$p$', r'$t$ [seconds]', r'$t$ [seconds]', r'$t$ [seconds]', r'$t$ [seconds]', r'$t$ [seconds]', r'$t$ [seconds]', r'$t$ [seconds]']
ys = [e, p, e, Phi_phi, Phi_r]
xs = [p, t, t, t, t]

for i, (ax, x, y, xlab, ylab) in enumerate(zip(axes, xs, ys, xlabels, ylabels)):
    ax.plot(x, y)
    ax.set_xlabel(xlab, fontsize=16)
    ax.set_ylabel(ylab, fontsize=16)
plt.tight_layout()
plt.savefig(f'trajectory_evolution_info_M={M}_mu={mu}_p0={p0}_e0={e0}_a={a}.png')
arr_out = np.asarray((t, p, e, Phi_phi, Phi_theta, Phi_r))
np.savetxt(f"t[seconds]_p_e_Phiphi_Phitheta_Phir_info_M={M}_mu={mu}_p0={p0}_e0={e0}_a={a}.txt", arr_out.T )

# plt.close('all')

# plt.figure()
# cb = plt.tricontourf(p_all, e_all, np.log10(np.abs(pdot_full)))
# plt.colorbar(cb,label=r'$log_{10} (\dot{p}) $')
# plt.plot(p,e,'r.',alpha=0.6)
# plt.xlabel('p')
# plt.ylabel('e')
# plt.show()


plt.figure()
plt.title(r'numerical derivative $\partial_p \dot{p}$ over the trajectory')
plt.semilogy(om2/om3, np.abs(np.gradient(pdot,p)),'.')
plt.xlabel('p')
plt.ylabel(r'$\frac{\partial \dot{p}}{\partial p} $')
plt.savefig('numerical_derivative_pdot_dp_info_M={M}_mu={mu}_p0={p0}_e0={e0}_a={a}.txt.png')


print("DONE")