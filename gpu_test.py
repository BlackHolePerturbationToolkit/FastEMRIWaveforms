import sys
import os
import time
import numpy as np
import matplotlib.pyplot as plt

from few.trajectory.inspiral import EMRIInspiral
from few.waveform import GenerateEMRIWaveform, FastSchwarzschildEccentricFlux
from few.utils.constants import *
from few.utils.utility import get_separatrix

# initial parameters
M = 1e6
mu = 1e1
a = 0.
p_f = 8.8  # Final semi-latus rectum
e_f = 0.1  # Final eccentricity
iota0_f = 0.9  # Final inclination
Y_f = np.cos(iota0_f)

Phi_phi0 = 1.0
Phi_theta0 = 2.0
Phi_r0 = 3.0

dt = 1.0
T = 0.01

p_sep = get_separatrix(a, e_f, 1.0)
print("Separatrix is", p_sep)

trajectory_class = "SchwarzEccFlux"

if "Kerr" in trajectory_class:
    Y_f = 1.0
elif "Schwarz" in trajectory_class:
    Y_f = 1.0
    a = 0.0

inspiral_kwargs_forward = {
    "DENSE_STEPPING": 0,  # Set to 1 if we want a sparsely sampled trajectory
    "max_init_len": int(1e3),  # all of the trajectories will be well under len = 1000
    "err": 1e-12,  # Set error tolerance on integrator -- RK8
    "integrate_backwards": False,  # Integrate trajectories forwards
}

# Set extrinsic parameters for waveform generation
qS = 0.2
phiS = 0.2
qK = 0.8
phiK = 0.8
dist = 1.0

inspiral_kwargs_waveform_forwards = {
    "DENSE_STEPPING": 0,  # Set to 1 if we want a sparsely sampled trajectory
    "max_init_len": int(1e3),  # all of the trajectories will be well under len = 1000
    "err": 1e-12,
    "integrate_backwards": False,
    "func": trajectory_class,
}

sum_kwargs = {
    "pad_output": True,
}
model_name = FastSchwarzschildEccentricFlux # "FastSchwarzschildEccentricFlux"

# Function to generate waveform
def generate_waveform(use_gpu):
    wave_generator_forwards = GenerateEMRIWaveform(
        model_name,
        use_gpu=use_gpu,
        sum_kwargs=sum_kwargs,
    )

    waveform_forward = wave_generator_forwards(
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

    return waveform_forward

# Measure time for CPU
start_time = time.time()
waveform_cpu = generate_waveform(use_gpu=False)
cpu_time = time.time() - start_time
print(f"Waveform generation using CPU took {cpu_time:.2f} seconds")

# Measure time for GPU
start_time = time.time()
waveform_gpu = generate_waveform(use_gpu=True)
gpu_time = time.time() - start_time
print(f"Waveform generation using GPU took {gpu_time:.2f} seconds")

# Plot the waveforms
plt.figure()
plt.plot(np.real(waveform_cpu), label='CPU')
plt.plot(np.real(waveform_gpu), label='GPU')
plt.legend()
plt.title('Waveform Comparison')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.show()