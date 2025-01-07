import os
import time
import numpy as np
import matplotlib.pyplot as plt

from few.waveform import GenerateEMRIWaveform, FastSchwarzschildEccentricFlux, FastKerrEccentricEquatorialFlux
from few.utils.constants import *

# specify gpu device
os.environ["CUDA_VISIBLE_DEVICES"] = "7"

# initial parameters
M = 1e6
mu = 5e1
a = 0.0
p_0 = 8.8  # Final semi-latus rectum
e_0 = 0.1  # Final eccentricity
Y_0 = 1.0

Phi_phi0 = 1.0
Phi_theta0 = 2.0
Phi_r0 = 3.0

dt = 5.0
T = 1.0

# Set extrinsic parameters for waveform generation
qS = 0.2
phiS = 0.2
qK = 0.8
phiK = 0.8
dist = 1.0



for model_name in ["FastKerrEccentricEquatorialFlux"]:#"FastSchwarzschildEccentricFlux",
    print(f"Testing model: {model_name}, waveform duration: {T} years, dt: {dt} seconds")
    # print parameters
    print(f"M = {M}, mu = {mu}, a = {a}, p_0 = {p_0}, e_0 = {e_0}, Y_0 = {Y_0}")
    # Measure time for CPU
    gen_cpu = GenerateEMRIWaveform(model_name,use_gpu=False,sum_kwargs={"use_gpu": False})
    gen_cpu(M, mu, a, p_0, e_0, Y_0, dist, qS, phiS, qK, phiK, Phi_phi0, Phi_theta0, Phi_r0, dt=dt, T=T)
    start_time = time.time()
    waveform_cpu = gen_cpu(M, mu, a, p_0, e_0, Y_0, dist, qS, phiS, qK, phiK, Phi_phi0, Phi_theta0, Phi_r0, dt=dt, T=T)
    cpu_time = time.time() - start_time
    print(f"Waveform generation using CPU took {cpu_time:.2f} seconds")

    # Measure time for GPU
    gen_gpu = GenerateEMRIWaveform(model_name,use_gpu=True,sum_kwargs={"use_gpu": True})
    gen_gpu(M, mu, a, p_0, e_0, Y_0, dist, qS, phiS, qK, phiK, Phi_phi0, Phi_theta0, Phi_r0, dt=dt, T=T)
    start_time = time.time()
    waveform_gpu = gen_gpu(M, mu, a, p_0, e_0, Y_0, dist, qS, phiS, qK, phiK, Phi_phi0, Phi_theta0, Phi_r0, dt=dt, T=T)
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
    plt.savefig(model_name+'_waveform_comparison.png')
