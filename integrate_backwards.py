import sys
import os

import matplotlib.pyplot as plt
import numpy as np
import cupy as cp

from few.trajectory.inspiral import EMRIInspiral
from few.amplitude.romannet import RomanAmplitude
from few.amplitude.interp2dcubicspline import Interp2DAmplitude
from few.waveform import GenerateEMRIWaveform
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

from scipy.interpolate import interp1d



use_gpu = False

# keyword arguments for inspiral 
inspiral_kwargs = {
    "DENSE_STEPPING": 0,  # Set to 1 if we want a sparsely sampled trajectory
    "max_init_len": int(1e7), # all of the trajectories will be well under len = 1000
    "err" : 1e-10,
}


# initial parameters
M = 1e6
mu = 1e1
a = 0.85
p_f = 8.50  # Final semi-latus rectum
e_f = 0.15  # Final eccentricity
iota0_f = 1.0  # Final inclination
Y_f = np.cos(iota0_f)
T = 1 
Phi_phi0 = 1.0
Phi_theta0 = 2.0
Phi_r0 = 3.0

dt = 10.0
# waveform_choice = input("Do you want Kerr inspirals? [y/n]")
# if waveform_choice == "y":
#     waveform_class = 'Pn5AAKWaveform'
#     trajectory_class = 'pn5'
#     print("5PN AAK trajectory and waveform")
# else:
#     waveform_class = 'FastSchwarzschildEccentricFlux'
#     trajectory_class = "SchwarzEccFlux"
#     print("Eccentric Schwarzschild trajectory and waveform")

# trajectory_class = "pn5"
# trajectory_class = "SchwarzEccFlux"
trajectory_class = "KerrEccentricEquatorial"

if trajectory_class == "SchwarzEccFlux" or trajectory_class == "KerrEccentricEquatorial":
    Y_f = 1.0 
    if trajectory_class == "SchwarzEccFlux":
        a = 0.0

inspiral_kwargs_back = {
    "DENSE_STEPPING": 0,  # Set to 1 if we want a sparsely sampled trajectory
    "max_init_len": int(1e3), # all of the trajectories will be well under len = 1000
    "err" : 1e-10,
    "integrate_backwards": True
}

inspiral_kwargs_forward = {
    "DENSE_STEPPING": 0,  # Set to 1 if we want a sparsely sampled trajectory
    "max_init_len": int(1e3), # all of the trajectories will be well under len = 1000
    "err" : 1e-10,
    "integrate_backwards": False
}



# Set up trajectory module for backwards integration
traj_backwards = EMRIInspiral(func = trajectory_class)

# Generate backwards integration
t_back, p_back, e_back, Y_back, Phi_phi_back, Phi_r_back, Phi_theta_back = traj_backwards(M, mu, a, p_f, e_f, Y_f, 
                                             Phi_phi0=Phi_phi0, Phi_theta0=Phi_theta0, Phi_r0=Phi_r0, dt = dt, T=T, **inspiral_kwargs_back)
# Set up trajectory module for forwards integration
traj_forwards = EMRIInspiral(func = trajectory_class)

# Start trajectory at final point of backwards integration
initial_p = p_back[-1]
initial_e = e_back[-1]
initial_Y = Y_back[-1]

# Generate forwards integration
t_forward, p_forward, e_forward, Y_forward, Phi_phi_forward, Phi_r_forward, Phi_theta_forward = traj_forwards(M, mu, a, initial_p, initial_e, initial_Y, 
                                             Phi_phi0=Phi_phi0, Phi_theta0=Phi_theta0, Phi_r0=Phi_r0, T=T, **inspiral_kwargs_forward)
print("Finished trajectories")
p_back_interp = interp1d(max(t_back) - t_back,p_back, kind = 'cubic')
p_check_forward = p_back_interp(t_forward)
# ======================== WAVEFORM GENERATION =============================

# Set extrinsic parameters for waveform generation
qS = 0.2
phiS = 0.2
qK = 0.8
phiK = 0.8
dist = 1.0
mich = False

# Set up inspiral_kwargs - note use of "integrate_backwards"
inspiral_kwargs = {
    "max_init_len": int(1e3),
    "DENSE_STEPPING": 0,
    "err": 1e-10}

# keyword arguments for inspiral generator (RomanAmplitude)
# amplitude_kwargs = {
#     "max_init_len": int(1e5),  # all of the trajectories will be well under len = 1000
#     "use_gpu": use_gpu  # GPU is available in this class
# }

amplitude_kwargs = {"specific_spins":
                    [0.80, 0.90, 0.95]}
# keyword arguments for Ylm generator (GetYlms)
Ylm_kwargs = {
    "assume_positive_m": False  # if we assume positive m, it will generate negative m for all m>0
}

# keyword arguments for summation generator (InterpolatedModeSum)
sum_kwargs = {
    "use_gpu": use_gpu,  # GPU is availabel for this type of summation
    "pad_output": False
    }
# Generate waveform using backwards integration
# waveform_class = 'Pn5AAKWaveform'
# waveform_class = 'FastSchwarzschildEccentricFlux'
waveform_class = "KerrEccentricEquatorialFlux"
wave_generator_backwards = GenerateEMRIWaveform(waveform_class, 
                                                inspiral_kwargs = inspiral_kwargs_back,
                                                sum_kwargs = sum_kwargs,
                                                amplitude_kwargs = amplitude_kwargs,
                                                use_gpu=True)

waveform_back = wave_generator_backwards(M, mu, a, p_f, e_f, Y_f, dist, qS, phiS, qK, phiK, 
                          Phi_phi0=Phi_phi0, Phi_theta0=Phi_theta0, Phi_r0=Phi_r0, dt=dt, T=T)
                        
print("Finished backwards integration")

# Extract plus polarised waveform
h_p_back_int = cp.asnumpy(waveform_back.real)

# Now change inspiral_kwargs so we integrate forwards
inspiral_kwargs['integrate_backwards'] = False

# Generate waveform using forwards integration 
wave_generator_forwards = GenerateEMRIWaveform(waveform_class,
                                               inspiral_kwargs=inspiral_kwargs_forward, 
                                               sum_kwargs=sum_kwargs, 
                                               amplitude_kwargs = amplitude_kwargs,
                                               use_gpu=True)
waveform_forward = wave_generator_forwards(M, mu, a, initial_p, initial_e, initial_Y, dist, qS, phiS, qK, phiK,
                          Phi_phi0=Phi_phi0, Phi_theta0=Phi_theta0, Phi_r0=Phi_r0, dt=dt, T=T)
                       

print("Finished forwards integration")
# Extract plus polarised waveform
h_p_forward_int = cp.asnumpy(waveform_forward.real)

# Output non-noise_weighted mismatch
# print("Mismatch between two waveforms is", get_mismatch(waveform_forward,waveform_back[::-1]))

# Extract relevant times [seconds]
n_t = len(h_p_forward_int)
t = np.arange(0,n_t*dt,dt)

# Plot the two waveforms against eachother
plot_direc = "/home/ad/burkeol/work/Current_Projects/FastEMRIWaveforms_backwards/waveform_plots"
plt.plot(t/24/60/60,h_p_back_int[::-1], c = 'blue', alpha = 1, linestyle = 'dashed', label = 'integrated backwards')
plt.plot(t[0:-1]/24/60/60,h_p_forward_int[1:], c = 'red', alpha = 0.5, label = 'integrated forwards')
plt.xlabel(r'Time $t$ [days]')
plt.ylabel(r'$h_{p}$')
plt.title(r'Comparison')
# plt.xlim([365-0.01,365])
plt.xlim([365 - 0.01,365])
plt.legend()
plt.grid()
plt.show()
plt.savefig(plot_direc + "/waveform_back_forward_Schwarz.pdf", bbox_inches="tight")
plt.clf()

breakpoint()