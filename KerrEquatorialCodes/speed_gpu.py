import os
import time
import numpy as np
import matplotlib.pyplot as plt
import corner

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

dt = 2.0
T = 4.0

# Set extrinsic parameters for waveform generation
qS = 0.2
phiS = 0.2
qK = 0.8
phiK = 0.8
dist = 1.0


model_name = "FastKerrEccentricEquatorialFlux"
# Measure time for GPU
gen_gpu = GenerateEMRIWaveform(model_name,use_gpu=True,sum_kwargs={"use_gpu": True})
gen_gpu(M, mu, a, p_0, e_0, Y_0, dist, qS, phiS, qK, phiK, Phi_phi0, Phi_theta0, Phi_r0, dt=dt, T=T)
# create random points in parameter space to check speed
num = 1000
log10M = np.random.uniform(5,7,size=num)
log10eta = np.random.uniform(-6,-3,size=num)
a = np.random.uniform(0.0,0.9,size=num)
p_0 = np.random.uniform(8.8,15.0,size=num)
e_0 = np.random.uniform(0.0,0.6,size=num)
Y_0 = np.random.choice([1.0,-1.0],size=num)
speed = np.zeros(num)

for i in range(num):
    print('---------------------------------------')
    print("i=",i)
    M = 10**log10M[i]
    mu = 10**log10eta[i] * M
    # print(M, mu, a[i], p_0[i], e_0[i], Y_0[i])
    try:
        start_time = time.time()
        waveform_gpu = gen_gpu(M, mu, a[i], p_0[i], e_0[i], 1.0, dist, qS, phiS, qK, phiK, Phi_phi0, Phi_theta0, Phi_r0, dt=dt, T=T, eps=1e-3)
        gpu_time = time.time() - start_time
    except:
        print("Error")
        gpu_time = 0.0
    # print(f"Waveform generation using GPU took {gpu_time:.2f} seconds")
    speed[i] = gpu_time

# select only successful runs
mask = speed > 0.0
fig = corner.corner(np.array([log10M[mask],log10eta[mask],a[mask],p_0[mask],e_0[mask],speed[mask]]).T,labels=[r"$\log_{10} M$",r"$\log_{10} \eta$",r"$a$",r"$p_0$",r"$e_0$", "Speed (s)"])
plt.savefig(f"speed_gpu_T{T}yr_dt{dt}.png")
# fig = corner.corner(np.array([log10M,log10eta,a,p_0,e_0,speed]).T,labels=[r"$\log_{10} M$",r"$\log_{10} \eta$",r"$a$",r"$p_0$",r"$e_0$", "Speed (s)"])
# print parameters of unsuccessful runs
for i in range(num):
    if speed[i] == 0.0:
        print("i=",i)
        print(10**log10M[i], 10**log10eta[i], a[i], p_0[i], e_0[i], Y_0[i])
