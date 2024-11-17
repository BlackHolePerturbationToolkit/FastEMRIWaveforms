import sys
import os

import matplotlib.pyplot as plt
import numpy as np

# import cupy as cp

from few.trajectory.inspiral import EMRIInspiral
from few.waveform import (
    GenerateEMRIWaveform,
)
from few.utils.constants import *

from few.utils.utility import (
    get_separatrix,
)
from few.trajectory.ode import *
from scipy.interpolate import interp1d

use_gpu = False

# initial parameters
M = 1e6
mu = 1e1
a = 0.1
p_f = 8.8 # Final semi-latus rectum
e_f = 0.1  # Final eccentricity
iota0_f = 0.9  # Final inclination
Y_f = np.cos(iota0_f)

Phi_phi0 = 1.0
Phi_theta0 = 2.0
Phi_r0 = 3.0

dt = 10.0
T = 1.0

p_sep = get_separatrix(a, e_f, 1.0)
print("Separatrix is", p_sep)

trajectory_class = PN5
# trajectory_class = "pn5_nofrequencies"
# trajectory_class = "SchwarzEccFlux"
# trajectory_class = "KerrEccentricEquatorial"
# trajectory_class = "KerrEccentricEquatorial_nofrequencies"
# trajectory_class = "KerrEccentricEquatorial_ELQ"
# trajectory_class = "KerrEccentricEquatorial_ELQ_nofrequencies"

if trajectory_class is KerrEccEqFlux:
    Y_f = 1.0
elif trajectory_class is SchwarzEccFlux:
    Y_f = 1.0
    a = 0.0

inspiral_kwargs_back = {
    "DENSE_STEPPING": 0,  # Set to 1 if we want a sparsely sampled trajectory
    "max_init_len": int(1e3),  # All of the trajectories will be well under len = 1000
    "err": 1e-12,  # Set error tolerance on integrator -- RK8
    "integrate_backwards": True,  # Integrate trajectories backwards
}

inspiral_kwargs_forward = {
    "DENSE_STEPPING": 0,  # Set to 1 if we want a sparsely sampled trajectory
    "max_init_len": int(1e3),  # all of the trajectories will be well under len = 1000
    "err": 1e-12,  # Set error tolerance on integrator -- RK8
    "integrate_backwards": False,  # Integrate trajectories forwards
}


# Set up trajectory module for backwards integration
traj_backwards = EMRIInspiral(func=trajectory_class)

import time

start = time.time()
# Generate backwards integration
t_back, p_back, e_back, Y_back, Phi_phi_back, Phi_r_back, Phi_theta_back = (
    traj_backwards(
        M,
        mu,
        a,
        p_f,
        e_f,
        Y_f,
        Phi_phi0=Phi_phi0,
        Phi_theta0=Phi_theta0,
        Phi_r0=Phi_r0,
        dt=dt,
        T=T,
        **inspiral_kwargs_back
    )
)
print("Integrating backwards took", time.time() - start, "seconds")
t_new = np.linspace(t_back.min(), t_back.max(), 5000)
phases = traj_backwards.inspiral_generator.eval_integrator_spline(t_new)

phases_at_old_points = traj_backwards.inspiral_generator.eval_integrator_spline(t_back)[:,3]

print("Phase differences:" , phases_at_old_points - Phi_phi_back)
assert np.allclose(phases_at_old_points, Phi_phi_back)
# plt.plot(t_new, phases)
# plt.show()
# breakpoint()
# Set up trajectory module for forwards integration
traj_forwards = EMRIInspiral(func=trajectory_class)

# Start trajectory at final point of backwards integration
initial_p = p_back[-1]
initial_e = e_back[-1]
initial_Y = Y_back[-1]

# Report results
print("Initial parameters would be:")
print("p = ", initial_p)
print("e = ", initial_e)
print("Y = ", initial_Y)
print("New separatrix is p_sep = ", get_separatrix(a, initial_e, 1.0))
print("")

# Generate forwards integration
start = time.time()
(
    t_forward,
    p_forward,
    e_forward,
    Y_forward,
    Phi_phi_forward,
    Phi_r_forward,
    Phi_theta_forward,
) = traj_forwards(
    M,
    mu,
    a,
    initial_p,
    initial_e,
    initial_Y,
    Phi_phi0=Phi_phi0,
    Phi_theta0=Phi_theta0,
    Phi_r0=Phi_r0,
    T=T,  ## CHECK THIS !!!!!!!!!!!
    **inspiral_kwargs_forward
)
print("Integrating forwards took", time.time() - start, "seconds")
print("Finished trajectories")

p_back_interp = interp1d((max(t_back) - t_back)[::-1], p_back[::-1], kind="cubic")
p_check_forward = p_back_interp(t_forward[t_forward < t_back.max()])
# p_check_forward should be near identical to p_forward up to mass-ratio corrections.

# TODO: turn this into a test!!!!
assert np.all(np.abs(p_check_forward - p_check_forward) < 1e-3)

# ======================== WAVEFORM GENERATION =============================
# Set extrinsic parameters for waveform generation

qS = 0.2
phiS = 0.2
qK = 0.8
phiK = 0.8

dist = 1.0
mich = False

inspiral_kwargs_waveform_backwards = {
    "DENSE_STEPPING": 0,  # Set to 1 if we want a sparsely sampled trajectory
    "max_init_len": int(1e3),  # all of the trajectories will be well under len = 1000
    "err": 1e-12,
    "integrate_backwards": True,  # Integrate backwards
    "func": trajectory_class,  # Use predetermined trajectory class abvoe
}
amplitude_kwargs_kerr = {
    "specific_spins": [0.80, 0.90, 0.95]
}  # For the sake of speed, only include set spin parameters

# keyword arguments for summation generator (InterpolatedModeSum)
sum_kwargs = {
    "use_gpu": False,  # GPU is availabel for this type of summation
    "pad_output": False,
}
model_name = "Pn5AAKWaveform"

wave_generator_backwards = GenerateEMRIWaveform(
    model_name,
    inspiral_kwargs=inspiral_kwargs_waveform_backwards,
    # amplitude_kwargs=amplitude_kwargs_kerr,
    sum_kwargs=sum_kwargs,
    use_gpu=False,
)

waveform_back = wave_generator_backwards(
    M,
    mu,
    a,
    p_f,
    e_f,
    Y_f,
    dist,
    qS,
    phiS,
    qK,
    phiK,
    Phi_phi0=Phi_phi0,
    Phi_theta0=Phi_theta0,
    Phi_r0=Phi_r0,
    dt=dt,
    T=T,
)

print("Finished backwards integration")

# Extract plus polarised waveform
waveform_back = waveform_back

h_p_back_int = np.asarray(waveform_back.real)

inspiral_kwargs_waveform_forwards = {
    "DENSE_STEPPING": 0,  # Set to 1 if we want a sparsely sampled trajectory
    "max_init_len": int(1e3),  # all of the trajectories will be well under len = 1000
    "err": 1e-12,
    "integrate_backwards": False,
    "func": trajectory_class,
}
# Generate waveform using forwards integration
sum_kwargs = {
    "use_gpu": use_gpu,  # GPU is availabel for this type of summation
    "pad_output": False,
}
wave_generator_forwards = GenerateEMRIWaveform(
    model_name,
    inspiral_kwargs=inspiral_kwargs_waveform_forwards,
    # amplitude_kwargs=amplitude_kwargs_kerr,
    # sum_kwargs=sum_kwargs,
    use_gpu=use_gpu,
)

T_in_forward = waveform_back.shape[0] * dt / YRSID_SI

waveform_forward = wave_generator_forwards(
    M,
    mu,
    a,
    float(initial_p),
    float(initial_e),
    float(initial_Y),
    dist,
    qS,
    phiS,
    qK,
    phiK,
    Phi_phi0=Phi_phi0,
    Phi_theta0=Phi_theta0,
    Phi_r0=Phi_r0,
    dt=dt,
    T=T,
)

waveform_forward = waveform_forward


print("Finished forwards integration")
# Extract plus polarised waveform
h_p_forward_int = waveform_forward.real  # cp.asnumpy(waveform_forward.real)

# Extract relevant times [seconds]
n_t = len(h_p_forward_int)
t = np.arange(0, n_t * dt, dt)

# Compute mismatches between waveforms

h_p_forward_fft = np.fft.rfft(
    # waveform_forward.real
    waveform_forward.real
)  # Roll array because numerical integrator not saving last point
h_p_back_reversed_fft = np.fft.rfft(
    waveform_back.real[::-1]
)  # Reverse backwards integrated waveform


def inner_prod(sig1_f, sig2_f):
    """
    Filthy overlap. No noise weighting. Just RAAAW signal.
    """
    ab = np.sum(sig1_f * sig2_f.conj())
    aa = np.sum(np.abs(sig1_f) ** 2)
    bb = np.sum(np.abs(sig2_f) ** 2)

    return np.real(ab / np.sqrt(aa * bb))


overlap = inner_prod(h_p_forward_fft, h_p_back_reversed_fft)
mismatch = 1 - overlap
print("Mismatch is", 1 - overlap)

# Plot the two waveforms against eachother
# plot_direc = "/home/ad/burkeol/work/Current_Projects/FastEMRIWaveforms_backwards/waveform_plots"
plt.plot(t,h_p_back_int[::-1], c = 'blue', alpha = 1, linestyle = 'dashed', label = 'integrated backwards')
plt.plot(t,h_p_forward_int, c = 'red', alpha = 0.5, label = 'integrated forwards')
plt.plot(t,h_p_back_int[::-1] - h_p_forward_int, c = 'green', alpha = 0.5, label = 'diff')
breakpoint()
title_str = r'Kerr: $(M, \mu, a, p_f, e_f, T) = (1e6, {}, {}, {}, {}, {} \, \text{{year}})$'.format(mu,a, np.round(p_f,3), e_f, 1)
plt.title(title_str, fontsize = 12)
plt.xlabel(r'Time $t$')
plt.ylabel(r'$h_{p}$')
# plt.xlim([365-0.01,365])
plt.xlim([t[-1] - 500, t[-1] + 5])
# plt.xlim(0, 500)
plt.legend()
plt.grid()
plt.show()
# plt.savefig(plot_direc + "/waveform_back_forward_Kerr.pdf", bbox_inches="tight")
# plt.clf()
