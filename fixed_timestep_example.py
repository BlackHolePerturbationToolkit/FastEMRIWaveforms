import sys
import os

import matplotlib.pyplot as plt
import numpy as np

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
                               Y_to_xI,
                               interpolate_trajectories_backwards_integration)

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

sum_kwargs = {
        "use_gpu": use_gpu,  # GPU is availabel for this type of summation
        "pad_output": True
        }

# These kwargs used for Schwarzschild model
amplitude_kwargs = {
"max_init_len": int(1e2),  
"use_gpu": use_gpu  # GPU is available in this class
}
# keyword arguments for Ylm generator (GetYlms)
Ylm_kwargs = {
    "assume_positive_m": False  # if we assume positive m, it will generate negative m for all m>0
}

# keyword arguments for inspiral 

# initial parameters
M = 1e6
mu = 1e1
Phi_phi0 = 1
Phi_r0 = 3

waveform_choice = input("Do you want Kerr inspirals? [y/n]")
if waveform_choice == "y":
    waveform_class = 'Pn5AAKWaveform'
    trajectory_class = 'pn5'
    
    # Parameters for AAK model
    p0 = 8.1
    e0 = 0.2 
    iota0 = 1.0; Y0 = np.cos(iota0)   
    a = 0.85
    Phi_theta0 = 2
    

    # these kwargs not used for AAK model
    print("5PN AAK trajectory and waveform")

else:
    waveform_class = 'FastSchwarzschildEccentricFlux'
    trajectory_class = "SchwarzEccFlux"

    # Parameters for Schwarzschild Model
    p0 = 9.7  
    e0 = 0.2 
    Y0 = 0.0  # Not used 
    a = 0.0   # Not used 
    Phi_theta0 = 0.0  # Not used 

    # keyword arguments for summation generator (InterpolatedModeSum)

    print("Eccentric Schwarzschild trajectory and waveform")


# Generate trajectory module - adapative
inspiral_kwargs_adapt = {
    "DENSE_STEPPING": 0,  # Set to 1 if we want a sparsely sampled trajectory
    "max_init_len": int(1e3), # all of the trajectories will be well under len = 1000
    "err" : 1e-10,
    "integrate_backwards": False
}

traj = EMRIInspiral(func = trajectory_class)

# Generate forwards integration
t_adapt, p_adapt, e_adapt, Y_adapt, Phi_phi_adapt, Phi_r_adapt, Phi_theta_adapt = traj(M, mu, a, p0, e0, Y0, 
                                             Phi_phi0=Phi_phi0, Phi_theta0=Phi_theta0, Phi_r0=Phi_r0, T=1.0, **inspiral_kwargs_adapt)

# Generate trajectory module - fixed
inspiral_kwargs_fixed = {
    "DENSE_STEPPING": 1,  # Set to 1 if we want a sparsely sampled trajectory
    "dT": 1e3,
    #"max_init_len": int(1e7), # all of the trajectories will be well under len = 1000
    "err" : 1e-10,
    "integrate_backwards": False
}

t_fixed, p_fixed, e_fixed, Y_fixed, Phi_phi_fixed, Phi_r_fixed, Phi_theta_fixed = traj(M, mu, a, p0, e0, Y0, 
                                             Phi_phi0=Phi_phi0, Phi_theta0=Phi_theta0, Phi_r0=Phi_r0, T=1.0, **inspiral_kwargs_fixed)


print("Finished trajectories")

# plt.plot(t_adapt, p_adapt, c = 'blue', alpha = 0.5, label = 'adaptive')
# plt.plot(t_fixed, p_fixed, c = 'red', alpha = 1, linestyle = 'dashed', label = 'fixed')
# plt.xlabel(r'time [days]', fontsize = 16)
# plt.ylabel(r'$p$', fontsize = 16)
# plt.title('Comparison - adaptive and fixed time steps')
# plt.legend()
# plt.show()
# ======================== WAVEFORM GENERATION =============================

# Set extrinsic parameters for waveform generation
qS = 0.2
phiS = 0.2
qK = 0.8
phiK = 0.8
dist = 1.0
dt = 10.0

# Generate waveform using adaptive step

wave_generator_adaptive = GenerateEMRIWaveform(waveform_class, inspiral_kwargs = inspiral_kwargs_adapt, 
                                                                amplitude_kwargs = amplitude_kwargs,
                                                                Ylm_kwargs = Ylm_kwargs,
                                                                sum_kwargs = sum_kwargs)

waveform_adaptive_time_step = wave_generator_adaptive(M, mu, a, p0, e0, Y0, dist, qS, phiS, qK, phiK, 
                          Phi_phi0=Phi_phi0, Phi_theta0=Phi_theta0, Phi_r0=Phi_r0, dt=dt, T=1.0) 
print("Finished waveform using adapative step")

# Extract plus polarised waveform
h_p_adaptive_time_step = waveform_adaptive_time_step.real

# Generate waveform using forwards integration 
print("Building waveform")
wave_generator_fixed_time_step = GenerateEMRIWaveform(waveform_class, inspiral_kwargs = inspiral_kwargs_fixed, 
                                                                amplitude_kwargs = amplitude_kwargs,
                                                                Ylm_kwargs = Ylm_kwargs,
                                                                sum_kwargs = sum_kwargs)

waveform_fixed_time_step = wave_generator_fixed_time_step(M, mu, a, p0, e0, Y0, dist, qS, phiS, qK, phiK,
                          Phi_phi0=Phi_phi0, Phi_theta0=Phi_theta0, Phi_r0=Phi_r0, dt=dt, T=1.0)

# quit()
print("Finished forwards integration")
# Extract plus polarised waveform
h_p_fixed_time_step = waveform_fixed_time_step.real

n_t_fixed = len(h_p_fixed_time_step)
# h_p_adaptive_time_step = h_p_adaptive_time_step[0:n_t_fixed]

# adaptive step size will always get closer to separatrix, so truncate
# Output non-noise_weighted mismatch
print("Mismatch between two waveforms is", get_mismatch(h_p_adaptive_time_step,h_p_fixed_time_step))

# Extract relevant times [seconds]

t = np.arange(0,n_t_fixed*dt,dt)
# Plot the two waveforms against eachother
plt.plot(t/24/60/60,h_p_adaptive_time_step, c = 'blue', alpha = 1, linestyle = 'dashed', label = 'adaptive time step')
plt.plot(t/24/60/60,h_p_fixed_time_step, c = 'red', alpha = 0.5, label = 'fixed time step')
plt.xlabel(r'Time $t$ [days]')
plt.ylabel(r'$h_{p}$')
plt.title(r'Comparison')
plt.legend()
plt.show()
plt.clf()

