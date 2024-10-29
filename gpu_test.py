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
T = 0.1

# Set extrinsic parameters for waveform generation
qS = 0.2
phiS = 0.2
qK = 0.8
phiK = 0.8
dist = 1.0

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

    return wave_generator_forwards

# Measure time for CPU
gen_cpu = generate_waveform(use_gpu=False)
gen_cpu(M, mu, a, p_f, e_f, Y_f, dist, qS, phiS, qK, phiK, Phi_phi0, Phi_theta0, Phi_r0, dt=dt, T=T)
start_time = time.time()
waveform_cpu = gen_cpu(M, mu, a, p_f, e_f, Y_f, dist, qS, phiS, qK, phiK, Phi_phi0, Phi_theta0, Phi_r0, dt=dt, T=T)
cpu_time = time.time() - start_time
print(f"Waveform generation using CPU took {cpu_time:.2f} seconds")

# Measure time for GPU
gen_gpu = generate_waveform(use_gpu=True)
gen_gpu(M, mu, a, p_f, e_f, Y_f, dist, qS, phiS, qK, phiK, Phi_phi0, Phi_theta0, Phi_r0, dt=dt, T=T)
start_time = time.time()
waveform_gpu = gen_gpu(M, mu, a, p_f, e_f, Y_f, dist, qS, phiS, qK, phiK, Phi_phi0, Phi_theta0, Phi_r0, dt=dt, T=T)
gpu_time = time.time() - start_time
print(f"Waveform generation using GPU took {gpu_time:.2f} seconds")

# Plot the waveforms
plt.figure()
plt.plot(np.real(waveform_cpu)[-1000:], label='CPU')
plt.plot(np.real(waveform_gpu.get())[-1000:], '--',label='GPU')
plt.legend()
plt.title('Waveform Comparison')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.savefig('waveform_comparison.png')