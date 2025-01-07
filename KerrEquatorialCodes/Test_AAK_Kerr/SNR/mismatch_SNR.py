import matplotlib.pyplot as plt
from few.waveform import GenerateEMRIWaveform

from lisatools.sensitivity import get_sensitivity
from few.trajectory.inspiral import EMRIInspiral
from few.waveform import SchwarzschildEccentricWaveformBase
from few.summation.interpolatedmodesum import InterpolatedModeSum
from few.amplitude.romannet import RomanAmplitude
from few.amplitude.interp2dcubicspline import Interp2DAmplitude
from few.utils.ylm import GetYlms
from few.utils.utility import get_p_at_t

from few.utils.modeselector import ModeSelector
from few.utils.utility import get_p_at_t
from few.utils.utility import get_separatrix

import cupy as cp
import numpy as np

use_gpu = True

if use_gpu:
    xp = cp
else:
    xp = np

def zero_pad(data):
    N = len(data)
    pow_2 = xp.ceil(np.log2(N))
    return xp.pad(data,(0,int((2**pow_2)-N)),'constant')

def FFT(data):
    data_pad = zero_pad(data)
    data_f = xp.fft.rfft(data_pad)[1:]
    return data_f

def inner_prod(sig1_f,sig2_f,N_t,delta_t,PSD):
    prefac = 4*delta_t / N_t
    sig2_f_conj = xp.conjugate(sig2_f)
    return prefac * xp.real(xp.sum((sig1_f * sig2_f_conj)/PSD))

def overlap_f(sig1_f,sig2_f,N_t,delta_t,PSD):
    aa = inner_prod(sig1_f,sig1_f,N_t,delta_t,PSD)
    bb = inner_prod(sig2_f,sig2_f,N_t,delta_t,PSD)
    ab = inner_prod(sig1_f,sig2_f,N_t,delta_t,PSD)
    return  (ab/(cp.sqrt(bb*aa)))

few_kerr = GenerateEMRIWaveform("KerrEccentricEquatorialFlux", use_gpu=True)
few_schw = GenerateEMRIWaveform("FastSchwarzschildEccentricFluxBicubic", use_gpu=True)

delta_t = 10.
T = 2.

# ============ Set parameters ================

M = 1e6; mu = 1e1; a=0.0; e0=0.03;Y0 = 1.0; dist=1.0;
theta_S=np.pi/2;phi_S = np.pi/4; theta_K = np.pi/6; phi_K = 3*np.pi/4;
Phi_phi0 = 1.0; Phi_theta0 = 0.0; Phi_r0 = 3.0

# ==================== Check out trajectory =================

kerr_traj = EMRIInspiral(func="KerrEccentricEquatorial", use_gpu = True)
schw_traj = EMRIInspiral(func="SchwarzEccFlux", use_gpu = True)

p_new = get_p_at_t(
    schw_traj,
    T,
    [M,mu,a,e0,Y0],
    index_of_p=3,
    index_of_a=2,
    index_of_e=4,
    index_of_x=5,
    traj_kwargs={},
    xtol=2e-12,
    rtol=8.881784197001252e-16,
    bounds=None,
)

print("We need to set p_new = {} to get a waveform of length {} years".format(p_new,T))
print("setting... p0 = ",p_new)
p0 = p_new+0.01
t_kerr, p_kerr, e_kerr, x_kerr, Phi_phi_kerr, Phi_theta_kerr, Phi_r_kerr = kerr_traj(M, mu, 0.0, p0, e0, 1.0)
t_schw, p_schw, e_schw, x_schw, Phi_phi_schw, Phi_theta_schw, Phi_r_schw = schw_traj(M, mu, 0.0, p0, e0, 1.0)

breakpoint()
print("Final point of semi-latus rectum for Kerr is",p_kerr[-1])
print("Final point of semi-latus rectum for Schw is",p_schw[-1])

pars = np.array([
    M,  # M
    mu,  # mu
    a,  # a
    p0,  # p
    e0,  # e
    Y0,   # Y0
    dist,   # Distance
    theta_S,   # Sky position
    phi_S,
    theta_K,   # Orbital angular momentum (spin vector)
    phi_K,
    Phi_phi0,  # Phi_phi0
    Phi_theta0, # Phi_theta0
    Phi_r0 # Phi_r0
])

wf_kerr = few_kerr(
    *pars,
    dt=delta_t,
    T=T,
    # mode_selection = [(2,2,0),]
)
wf_kerr_p = wf_kerr.real
print(few_kerr.waveform_generator.num_modes_kept)

wf_schw = few_schw(
    *pars,
    dt=delta_t,
    T=T,
    # mode_selection = [(2,2,0),]
)

wf_schw_p = wf_schw.real

print(few_schw.waveform_generator.num_modes_kept)

# ================ Compute things in the frequency domain ================
kerr_fft = FFT(wf_kerr_p)
schw_fft = FFT(wf_schw_p)

N_t = 2**xp.ceil(np.log2(len(wf_kerr_p)))
freq = xp.fft.rfftfreq(N_t,delta_t)[1:]
PSD = get_sensitivity(freq,sens_fn="cornish_lisa_psd")


SNR2_kerr = inner_prod(kerr_fft,kerr_fft,N_t,delta_t,PSD)
SNR2_schw = inner_prod(schw_fft,schw_fft,N_t,delta_t,PSD)

print("SNR accumulated for Kerr inspiral is", SNR2_kerr**(1/2))
print("SNR accumulated for Schw inspiral is", SNR2_schw**(1/2))

#===================== Now compute mismatches between waveforms =============

mismatch_schw_kerr = overlap_f(kerr_fft,schw_fft,N_t,delta_t,PSD)
print("Overlap between kerr built wave and vanilla few is {} over {} years".format(mismatch_schw_kerr,T))



print("Now plotting!")

t_kerr = np.linspace(0,few_kerr.waveform_generator.end_time, len(wf_kerr))
t_schw = np.linspace(0,few_schw.waveform_generator.end_time, len(wf_schw))
waves = [wf_kerr, wf_schw]
times = [t_kerr, t_schw]
labs = ["KerrEccEquatorial", "FastSchwarzschild"]

fig, ax = plt.subplots(nrows=2, dpi=150)
for tm,wv, lb in zip(times, waves, labs):
    ax[0].plot(tm[::100], wv.get().real[::100],label=lb, lw=0.5, alpha=0.5)
    ax[1].plot(tm[::100], wv.get().imag[::100],label=lb, lw=0.5, alpha=0.5)
ax[0].legend()
ax[0].set_title(f"p0, e0, Y0 = {pars[3]:.3f},{pars[4]:.3f},{pars[5]:.3f}")
ax[1].set_title("Above = real, below = imag")
plt.tight_layout()
plt.savefig("kwave.png")
plt.close()

fig, ax = plt.subplots(nrows=2, dpi=150)
for tm,wv, lb in zip(times, waves, labs):
    ax[0].plot(tm[:1000], wv.get().real[:1000],label=lb)
    ax[1].plot(tm[:1000], wv.get().imag[:1000],label=lb)
ax[0].legend()
ax[0].set_title(f"p0, e0, Y0 = {pars[3]:.3f},{pars[4]:.3f},{pars[5]:.3f}")
ax[1].set_title("Above = real, below = imag")
ax[0].set_xlim(0,2000)
ax[1].set_xlim(0,2000)
plt.tight_layout()
plt.savefig("kwave_close.png")
plt.close()

fig, ax = plt.subplots(nrows=2, dpi=150)
for tm,wv, lb in zip(times, waves, labs):
    ax[0].plot(tm[-5000:], wv.get().real[-5000:],label=lb)
    ax[1].plot(tm[-5000:], wv.get().imag[-5000:],label=lb)
ax[0].legend()
ax[0].set_title(f"p0, e0, Y0 = {pars[3]:.3f},{pars[4]:.3f},{pars[5]:.3f}")
ax[1].set_title("Above = real, below = imag")
ax[0].set_xlim(tm[-1000],tm[-1])
ax[1].set_xlim(tm[-1000],tm[-1])
plt.tight_layout()
plt.savefig("kwave_close_end.png")
plt.close()
