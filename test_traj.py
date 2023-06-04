import sys
import os

import matplotlib.pyplot as plt
import numpy as np

from few.trajectory.inspiral import EMRIInspiral
from few.amplitude.romannet import RomanAmplitude
from few.amplitude.interp2dcubicspline import Interp2DAmplitude
from few.waveform import FastSchwarzschildEccentricFlux, SlowSchwarzschildEccentricFlux, GenerateEMRIWaveform
from few.utils.utility import (get_overlap, 
                               get_mismatch, 
                               get_fundamental_frequencies, 
                               get_separatrix, 
                               get_mu_at_t, 
                               get_p_at_t, 
                               get_kerr_geo_constants_of_motion,
                               xI_to_Y,
                               Y_to_xI)

from few.utils.ylm import GetYlms
from few.utils.modeselector import ModeSelector
from few.summation.interpolatedmodesum import CubicSplineInterpolant
from few.waveform import SchwarzschildEccentricWaveformBase
from few.summation.interpolatedmodesum import InterpolatedModeSum
from few.summation.directmodesum import DirectModeSum
from few.utils.constants import *
from few.summation.aakwave import AAKSummation
from few.waveform import Pn5AAKWaveform, AAKWaveformBase

import scipy.interpolate



use_gpu = False

# keyword arguments for inspiral generator (RunKerrGenericPn5Inspiral)
inspiral_kwargs = {
    "DENSE_STEPPING": 0,  # we want a sparsely sampled trajectory
    "max_init_len": int(1e3),  # all of the trajectories will be well under len = 1000
}

# keyword arguments for inspiral generator (RomanAmplitude)
amplitude_kwargs = {
    "max_init_len": int(1e3),  # all of the trajectories will be well under len = 1000
    "use_gpu": use_gpu  # GPU is available in this class
}

# keyword arguments for Ylm generator (GetYlms)
Ylm_kwargs = {
    "assume_positive_m": False  # if we assume positive m, it will generate negative m for all m>0
}

# keyword arguments for summation generator (InterpolatedModeSum)
sum_kwargs = {
    "use_gpu": use_gpu,  # GPU is availabel for this type of summation
    "pad_output": False
    }

# THE FOLLOWING THREAD COMMANDS DO NOT WORK ON THE M1 CHIP, BUT CAN BE USED WITH OLDER MODELS
# EVENTUALLY WE WILL PROBABLY REMOVE OMP WHICH NOW PARALLELIZES WITHIN ONE WAVEFORM AND LEAVE IT TO 
# THE USER TO PARALLELIZE FOR MANY WAVEFORMS ON THEIR OWN. 

# set omp threads one of two ways
# num_threads = 4

# this is the general way to set it for all computations
# from few.utils.utility import omp_set_num_threads
# omp_set_num_threads(num_threads)


traj = EMRIInspiral(func="pn5") 

M = 1e6
mu = 1e1
a = 0.5
p_f = 9.0
e_f = 0.3 
iota0_f = 0.5
Y_f = np.cos(iota0_f)
T = 1.0
Phi_phi0 = 1
Phi_theta0 = 2
Phi_r0 = 3
integrate_backwards = 1.0 # NEW PARAMETER : if integrate_backwards = 1.0 then we integrate backwards. 
                          # Otherwise we integrate forwards. MUST BE A FLOAT.

x_new = Y_to_xI(a, p_f, e_f, Y_f)
p_sep = get_separatrix(a, e_f, x_new)

print("separatrix is at",p_sep)

t_back, p_back, e_back, Y_back, Phi_phi_back, Phi_r_back, Phi_theta_back = traj(M, mu, a, p_f, e_f, Y_f, integrate_backwards, 
                                             Phi_phi0=Phi_phi0, Phi_theta0=Phi_theta0, Phi_r0=Phi_r0, T=T )

t_back_shift = max(t_back) - t_back

# fig, axes = plt.subplots(2, 4)
# plt.subplots_adjust(wspace=0.5)
# fig.set_size_inches(14, 8)
# axes = axes.ravel()

# ylabels = [r'$e$', r'$p$', r'$e$', r'$Y$', r'$\Phi_\phi$', r'$\Phi_r$', r'$\Phi_\theta$']
# xlabels = [r'$p$', r'$t$', r'$t$', r'$t$', r'$t$', r'$t$', r'$t$', r'$t$', r'$t$']
# ys = [e_back, p_back, e_back, Y_back, Phi_phi_back, Phi_r_back, Phi_theta_back]
# xs = [p_back, t_back, t_back, t_back, t_back, t_back, t_back]

# for i, (ax, x, y, xlab, ylab) in enumerate(zip(axes, xs, ys, xlabels, ylabels)):
#     ax.plot(x, y)
#     ax.set_xlabel(xlab, fontsize=16)
#     ax.set_ylabel(ylab, fontsize=16)
#     ax.set_title("integrate backwards")
# axes[-1].set_visible(False)
# plt.tight_layout()
# plt.show()
# plt.clf()

integrate_forwards = 0.0 # NEW PARAMETER : if integrate_backwards = 1.0 then we integrate backwards. 
                          # Otherwise we integrate forwards. MUST BE A FLOAT.

initial_p = p_back[-1]
initial_e = e_back[-1]
initial_Y = Y_back[-1]

t_forward, p_forward, e_forward, Y_forward, Phi_phi_forward, Phi_r_forward, Phi_theta_forward = traj(M, mu, a, initial_p, initial_e, initial_Y, integrate_forwards, 
                                             Phi_phi0=Phi_phi0, Phi_theta0=Phi_theta0, Phi_r0=Phi_r0, T=T )
breakpoint()
# plt.plot(t_forward,Y_forward,c = 'red', linestyle = 'dashed')
# plt.plot(t_back_shift,Y_back,c = 'purple',alpha = 0.5)
# plt.show()
# plt.clf()


# plt.plot(t_forward,Phi_phi_forward,c = 'red', linestyle = 'dashed', label = 'forward integration')
# plt.plot(t_back_shift, Phi_phi0 + max(Phi_phi_back) - Phi_phi_back,c = 'purple',alpha = 0.5, label = 'backward integration')
# plt.legend()
# plt.xlabel(r'time [s]')
# plt.ylabel(r'$\Phi_{\phi}$')
# plt.tight_layout()
# plt.show()
# plt.clf()
# breakpoint()
# quit()
from scipy.interpolate import interp1d
p_back_interp = interp1d(t_back_shift,p_back, kind = 'cubic')
p_new_back = p_back_interp(t_forward)

e_back_interp = interp1d(t_back_shift,e_back, kind = 'cubic')
e_new_back = e_back_interp(t_forward)

Y_back_interp = interp1d(t_back_shift,Y_back, kind = 'cubic')
Y_new_back = Y_back_interp(t_forward)

Phi_phi_back_transform = Phi_phi0 + max(Phi_phi_back) - Phi_phi_back 
Phi_phi_back_interp = interp1d(t_back_shift, Phi_phi_back_transform, kind = 'cubic')
new_Phi_phi_back = Phi_phi_back_interp(t_forward)

Phi_theta_back_transform = Phi_theta0 + max(Phi_theta_back) - Phi_theta_back 
Phi_theta_back_interp = interp1d(t_back_shift, Phi_theta_back_transform, kind = 'cubic')
new_Phi_theta_back = Phi_theta_back_interp(t_forward)

Phi_r_back_transform = Phi_r0 + max(Phi_r_back) - Phi_r_back 
Phi_r_back_interp = interp1d(t_back_shift, Phi_r_back_transform, kind = 'cubic')
new_Phi_r_back = Phi_r_back_interp(t_forward)
# plt.plot(t_forward,p_new_back, label = 'interpolant')
# plt.plot(t_forward,p_forward, label = 'intergating forward')
# plt.legend()
# plt.xlabel(r'time [s]')
# plt.ylabel(r'$p$')
# plt.tight_layout()
# plt.show()
# plt.clf()

Phi_phi_back_transform = Phi_phi0 + max(Phi_phi_back) - Phi_phi_back 
Phi_phi_back_interp = interp1d(t_back_shift, Phi_phi_back_transform, kind = 'cubic')
new_Phi_phi_back = Phi_phi_back_interp(t_forward)
# plt.plot(t_forward,new_Phi_phi_back, label = 'interpolant')
# plt.plot(t_forward,Phi_phi_forward, label = 'intergating forward')
# plt.legend()
# plt.xlabel(r'time [s]')
# plt.ylabel(r'$p$')
# plt.tight_layout()
# plt.show()
# plt.clf()
# breakpoint()
# fig, axes = plt.subplots(2, 4)
# plt.subplots_adjust(wspace=0.5)
# fig.set_size_inches(14, 8)
# axes = axes.ravel()

# ylabels = [r'$e$', r'$p$', r'$e$', r'$Y$', r'$\Phi_\phi$', r'$\Phi_r$', r'$\Phi_\theta$']
# xlabels = [r'$p$', r'$t$', r'$t$', r'$t$', r'$t$', r'$t$', r'$t$', r'$t$', r'$t$']
# ys = [e_forward, p_forward, e_forward, Y_forward, Phi_phi_forward, Phi_r_forward, Phi_theta_forward]
# xs = [p_forward, t_forward, t_forward, t_forward, t_forward, t_forward, t_forward]

# for i, (ax, x, y, xlab, ylab) in enumerate(zip(axes, xs, ys, xlabels, ylabels)):
#     ax.plot(x, y)
#     ax.set_xlabel(xlab, fontsize=16)
#     ax.set_ylabel(ylab, fontsize=16)
#     ax.set_title("integrate forwards")
# axes[-1].set_visible(False)
# plt.tight_layout()
# plt.show()
# plt.clf()
# plt.close()

# quit()
#Create splines
#rom scipy.interpolate import interp1d



qS = 0.2
phiS = 0.2
qK = 0.8
phiK = 0.8
dist = 1.0
mich = False
dt = 15.0
T = 1
integrate_backwards = 1.0 
wave_generator = Pn5AAKWaveform(inspiral_kwargs=inspiral_kwargs, sum_kwargs=sum_kwargs, use_gpu=False)
waveform_back = wave_generator(M, mu, a, p_f, e_f, Y_f, dist, qS, phiS, qK, phiK, integrate_backwards,
                          Phi_phi0=Phi_phi0, Phi_theta0=Phi_theta0, Phi_r0=Phi_r0, mich=mich, dt=dt, T=T)
h_p_back = waveform_back.real

n_t = len(h_p_back)

t = np.arange(0,n_t*dt,dt)

# plt.plot(t,h_p_back)
# plt.xlabel(r'Time $t$')
# plt.ylabel(r'$h_{p}$')
# plt.title(r'Integrate backwards')
# plt.show()
# plt.clf()
# initial_p = p_back[-1]

wave_generator = Pn5AAKWaveform(inspiral_kwargs=inspiral_kwargs, sum_kwargs=sum_kwargs, use_gpu=False)
waveform_forward = wave_generator(M, mu, a, initial_p, initial_e, initial_Y, dist, qS, phiS, qK, phiK, integrate_forwards,
                          Phi_phi0=Phi_phi0, Phi_theta0=Phi_theta0, Phi_r0=Phi_r0, mich=mich, dt=dt, T=T)

h_p_forward = waveform_forward.real

n_t = len(h_p_forward)

t = np.arange(0,n_t*dt,dt)

# plt.plot(t,h_p_forward)
# plt.xlabel(r'Time $t$')
# plt.ylabel(r'$h_{p}$')
# plt.title(r'Integrate forwards')
# plt.show()
# plt.clf()


plt.plot(t/24/60/60,h_p_back, c = 'blue', alpha = 1, linestyle = 'dashed', label = 'integrated backwards')
plt.plot(t/24/60/60,h_p_forward, c = 'red', alpha = 0.5, label = 'integrated forwards')
plt.xlabel(r'Time $t$ [days]')
plt.ylabel(r'$h_{p}$')
plt.title(r'Comparison')
plt.legend()
# plt.xlim([0, 0.01])
plt.show()
plt.clf()

print("Mismatch between two waveforms is", get_mismatch(waveform_forward,waveform_back))
