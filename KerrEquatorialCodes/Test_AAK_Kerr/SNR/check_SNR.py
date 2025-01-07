import cupy as cp
import numpy as np
import os
import sys
sys.path.append("../")

# from scipy.signal import tukey       # I'm always pro windowing.

from fastlisaresponse import ResponseWrapper             # Response

# Import relevant EMRI packages
from few.waveform import GenerateEMRIWaveform, AAKWaveformBase, KerrEquatorialEccentric,KerrEquatorialEccentricWaveformBase
from few.trajectory.inspiral import EMRIInspiral
from few.summation.directmodesum import DirectModeSum
from few.summation.interpolatedmodesum import InterpolatedModeSum
from few.summation.aakwave import AAKSummation

from few.amplitude.ampinterp2d import AmpInterpKerrEqEcc
from few.utils.modeselector import ModeSelector, NeuralModeSelector
from few.utils.utility import get_separatrix, Y_to_xI, get_p_at_t

# Import features from eryn
from eryn.ensemble import EnsembleSampler
from eryn.moves import StretchMove
from eryn.prior import ProbDistContainer, uniform_dist
from eryn.backends import HDFBackend

YRSID_SI = 31558149.763545603

np.random.seed(1234)

def sensitivity_LWA(f):
    """
    LISA sensitivity function in the long-wavelength approximation (https://arxiv.org/pdf/1803.01944.pdf).

    args:
        f (float): LISA-band frequency of the signal

    Returns:
        The output sensitivity strain Sn(f)
    """

    #Defining supporting functions
    L = 2.5e9 #m
    fstar = 19.09e-3 #Hz

    P_OMS = (1.5e-11**2)*(1+(2e-3/f)**4) #Hz-1
    P_acc = (3e-15**2)*(1+(0.4e-3/f)**2)*(1+(f/8e-3)**4) #Hz-1

    #S_c changes depending on signal duration (Equation 14 in 1803.01944)
    #for 1 year
    alpha = 0.171
    beta = 292
    kappa = 1020
    gamma = 1680
    fk = 0.00215
    #log10_Sc = (np.log10(9)-45) -7/3*np.log10(f) -(f*alpha + beta*f*np.sin(kappa*f))*np.log10(np.e) + np.log10(1 + np.tanh(gamma*(fk-f))) #Hz-1

    A=9e-45
    Sc = A*f**(-7/3)*np.exp(-f**alpha+beta*f*np.sin(kappa*f))*(1+np.tanh(gamma*(fk-f)))
    sensitivity_LWA = (10/(3*L**2))*(P_OMS+4*(P_acc)/((2*np.pi*f)**4))*(1 + 6*f**2/(10*fstar**2))+Sc
    return sensitivity_LWA
def zero_pad(data):
    r"""
    Inputs: data stream of length N
    Returns: zero_padded data stream of new length 2^{J} for J \in \mathbb{N}
    """
    N = len(data)
    pow_2 = xp.ceil(np.log2(N))
    return xp.pad(data,(0,int((2**pow_2)-N)),'constant')

def inner_prod(sig1_f,sig2_f,N_t,delta_t,PSD):
    """
    Compute stationary noise-weighted inner product
    Inputs: sig1_f and sig2_f are signals in frequency domain
            N_t length of padded signal in time domain
            delta_t sampling interval
            PSD Power spectral density

    Returns: Noise weighted inner product
    """
    prefac = 4*delta_t / N_t
    sig2_f_conj = xp.conjugate(sig2_f)
    return prefac * xp.real(xp.sum((sig1_f * sig2_f_conj)/PSD))
def SNR_function(sig1_t, dt, N_channels = 2):
    N_t = len(sig1_t[0])

    sig1_f = [xp.fft.rfft(zero_pad(sig1_t[i])) for i in range(N_channels)]
    N_t = len(zero_pad(sig1_t[0]))

    freq_np = xp.asnumpy(xp.fft.rfftfreq(N_t, dt))

    freq_np[0] = freq_np[1]

    PSD = 2 * [xp.asarray(sensitivity_LWA(freq_np))]

    SNR2 = xp.asarray([inner_prod(sig1_f[i], sig1_f[i], N_t, dt,PSD[i]) for i in range(N_channels)])

    SNR = xp.sum(SNR2)**(1/2)

    return SNR
##======================Likelihood and Posterior (change this)=====================

def llike(params):
    """
    Inputs: Parameters to sample over
    Outputs: log-whittle likelihood
    """
    # Intrinsic Parameters
    M_val = float(params[0])
    mu_val = float(params[1])

    a_val =  float(params[2])
    p0_val = float(params[3])
    e0_val = float(params[4])
    xI0_val = 1.0

    # Luminosity distance
    D_val = float(params[5])

    # Angular Parameters
    qS_val = float(params[6])
    phiS_val = float(params[7])
    qK_val = float(params[8])
    phiK_val = float(params[9])

    # Angular parameters
    Phi_phi0_val = float(params[10])
    Phi_theta0_val = Phi_theta0
    Phi_r0_val = float(params[11])

    # Secondary charge
    gamma_val = float(params[12])

    if a_val < 0:
        a_val *= -1.0
        xI0_val *= -1.0

    # Propose new waveform model
    waveform_prop = Waveform_model(M_val, mu_val, a_val, p0_val, e0_val,
                                  xI0_val, D_val, qS_val, phiS_val, qK_val, phiK_val,
                                    Phi_phi0_val, Phi_theta0_val, Phi_r0_val, gamma_val,
                                    mich=True, dt=delta_t, T=T)  # EMRI waveform across A, E and T.


    # Taper and then zero pad.
    EMRI_w_pad_prop = [zero_pad(waveform_prop[i]) for i in range(N_channels)]

    # Compute in frequency domain
    EMRI_fft_prop = [xp.fft.rfft(item) for item in EMRI_w_pad_prop]

    # Compute (d - h| d- h)
    diff_f = [data_f[k] - EMRI_fft_prop[k] for k in range(N_channels)]
    inn_prod = xp.asarray([inner_prod(diff_f[k],diff_f[k],N_t,delta_t,PSD[k]) for k in range(N_channels)])

    # Return log-likelihood value as numpy val.
    llike_val_np = xp.asnumpy(-0.5 * (xp.sum(inn_prod)))
    return (llike_val_np)

M = 1e6; mu = 10; a = 0.9; p0 = 8.54; e0 = 0.3; x_I0 = 1.0;
dist = 1.0; qS = 0.7; phiS = 0.7; qK = 0.7; phiK = 0.7;
Phi_phi0 = 2.0; Phi_theta0 = 3.0; Phi_r0 = 4.0

delta_t = 10.0;  # Sampling interval [seconds]
T = 2.0     # Evolution time [years]

use_gpu = True
xp = cp
mich = True
# define trajectory
func = "KerrEccentricEquatorial"
insp_kwargs = {
    "err": 1e-10,
    "DENSE_STEPPING": 0,
    "use_rk4": False,
    "func": func,
    }
# keyword arguments for summation generator (AAKSummation)
sum_kwargs = {
    "use_gpu": use_gpu,  # GPU is availabel for this type of summation
    "pad_output": True,
}

## ===================== CHECK TRAJECTORY ====================
#
traj = EMRIInspiral(func=func, inspiral_kwargs = insp_kwargs)  # Set up trajectory module, pn5 AAK

# Compute trajectory
if a < 0:
    a *= -1.0
    x_I0 *= -1.0


t_traj, *out_GR = traj(M, mu, a, p0, e0, x_I0, 0.0,
                                             Phi_phi0=Phi_phi0, Phi_theta0=Phi_theta0, Phi_r0=Phi_r0, T=T)

print("Final value in semi-latus rectum", out_GR[0][-1])

traj_args_GR = [M, mu, a, out_GR[1][0], x_I0]
index_of_p = 3
# Check to see what value of semi-latus rectum is required to build inspiral lasting T years.
p_new = 30
# p_new = get_p_at_t(
#     traj,
#     T,
#     traj_args_GR,
#     index_of_p=3,
#     index_of_a=2,
#     index_of_e=4,
#     index_of_x=5,
#     xtol=2e-12,
#     rtol=8.881784197001252e-16,
#     bounds=[25, 28],
# )

print("We require initial semi-latus rectum of ",p_new, "for inspiral lasting", T, "years")
print("Your chosen semi-latus rectum is", p0)
if p0 < p_new:
    print("Careful, the smaller body is plunging. Expect instabilities.")
else:
    print("Body is not plunging.")
print("Final point in semilatus rectum achieved is", out_GR[0][-1])
print("Separatrix : ", get_separatrix(a, out_GR[1][-1], x_I0))

import time

inspiral_kwargs = {
        "DENSE_STEPPING": 0,
        "max_init_len": int(1e4),
        "err": 1e-10,  # To be set within the class
        "use_rk4": False,
        "integrate_phases":True,
        'func': 'KerrEccentricEquatorial'
    }
# keyword arguments for summation generator (AAKSummation)
sum_kwargs = {
    "use_gpu": True,  # GPU is availabel for this type of summation
    "pad_output": True,
}


amplitude_kwargs = {
    "specific_spins":[0.8, 0.9, 0.95],
    "use_gpu": True
    }
Waveform_model_AAK = GenerateEMRIWaveform(
AAKWaveformBase, # Define the base waveform
EMRIInspiral, # Define the trajectory
AAKSummation, # Define the interpolation for the amplitudes
inspiral_kwargs=inspiral_kwargs,
sum_kwargs=sum_kwargs,
use_gpu=use_gpu,
return_list=True,
frame="detector"
)

Waveform_model_Kerr = GenerateEMRIWaveform(
KerrEquatorialEccentricWaveformBase, # Define the base waveform
EMRIInspiral, # Define the trajectory
AmpInterpKerrEqEcc, # Define the interpolation for the amplitudes
InterpolatedModeSum, # Define the type of summation
ModeSelector, # Define the type of mode selection
inspiral_kwargs=inspiral_kwargs,
sum_kwargs=sum_kwargs,
amplitude_kwargs=amplitude_kwargs,
use_gpu=use_gpu,
return_list=True,
frame='detector'
)

## ============= USE THE LONG WAVELENGTH APPROXIMATION, VOMIT ================ ##
# nmodes =
specific_modes = [(2,2,n) for n in range(-2,2)]
params = [M,mu,a,p0,e0,1.0, dist, qS, phiS, qK, phiK, Phi_phi0, Phi_theta0, Phi_r0]

waveform_AAK = Waveform_model_AAK(*params, T = T, dt = delta_t, mich = True)  # Generate h_plus and h_cross
waveform_Kerr = Waveform_model_Kerr(*params, T = T, dt = delta_t, mich = True)  # Generate h_plus and h_cross

N_t = len(zero_pad(waveform_AAK[0]))

freq_bin_np = np.fft.rfftfreq(N_t, delta_t)
freq_bin_np[0] = freq_bin_np[1]

PSD = 2*[cp.asarray(sensitivity_LWA(freq_bin_np))]

N_channels = 2

waveform_AAK_fft = [xp.fft.rfft(zero_pad(waveform_AAK[i])) for i in range(N_channels)]
waveform_Kerr_fft = [xp.fft.rfft(zero_pad(waveform_Kerr[i])) for i in range(N_channels)]

SNR_Kerr = SNR_function(waveform_Kerr, delta_t, N_channels = 2)
SNR_AAK = SNR_function(waveform_AAK, delta_t, N_channels = 2)

print("Truth waveform, final SNR for Kerr = ",SNR_Kerr)
print("Truth waveform, final SNR for AAK = ",SNR_AAK)

os.chdir('/home/ad/burkeol/work/Kerr_Systematics/test_few/diagnostics/data_file/eccentricity')
e0_vec = np.arange(0.1,0.5,0.02)

SNR_Kerr_vec=[]
SNR_AAK_vec=[]

import matplotlib.pyplot as plt
for eccentricity in e0_vec:
    params = [M,mu,a,p0,eccentricity,1.0, dist, qS, phiS, qK, phiK, Phi_phi0, Phi_theta0, Phi_r0]

    waveform_Kerr = Waveform_model_Kerr(*params, mich = True, dt = delta_t, T = T)
    waveform_AAK = Waveform_model_AAK(*params, mich = True, dt = delta_t, T = T)

    SNR_Kerr = SNR_function(waveform_Kerr, delta_t, N_channels = 2)
    SNR_AAK = SNR_function(waveform_AAK, delta_t, N_channels = 2)

    SNR_Kerr_vec.append(xp.asnumpy(SNR_Kerr))
    SNR_AAK_vec.append(xp.asnumpy(SNR_AAK))


np.save("e0_vec.npy", e0_vec)
np.save("SNR_Kerr_vec.npy", SNR_Kerr_vec)
np.save("SNR_AAK_vec.npy", SNR_AAK_vec)

plt.plot(e0_vec,SNR_Kerr_vec, label = 'Kerr amplitudes')
plt.plot(e0_vec,SNR_AAK_vec, label = 'AAK amplitudes')
plt.grid()
plt.xlabel(r'Eccentricity $e_{0}$')
plt.ylabel(r'SNR')
plt.title("(M,mu,a,p0, T) = (1e6, 10, 0.9, 8.58, 2 years)")
plt.legend()
plt.savefig("Eccentricity_Plot_strong_field.pdf",bbox_inches = 'tight')
plt.clf()

p0 = 14.0 # Weak field

SNR_Kerr_vec=[]
SNR_AAK_vec=[]
for eccentricity in e0_vec:
    params = [M,mu,a,p0,eccentricity,1.0, dist, qS, phiS, qK, phiK, Phi_phi0, Phi_theta0, Phi_r0]

    waveform_Kerr = Waveform_model_Kerr(*params, mich = True, dt = delta_t, T = T)
    waveform_AAK = Waveform_model_AAK(*params, mich = True, dt = delta_t, T = T)

    SNR_Kerr = SNR_function(waveform_Kerr, delta_t, N_channels = 2)
    SNR_AAK = SNR_function(waveform_AAK, delta_t, N_channels = 2)

    SNR_Kerr_vec.append(xp.asnumpy(SNR_Kerr))
    SNR_AAK_vec.append(xp.asnumpy(SNR_AAK))

plt.plot(e0_vec,SNR_Kerr_vec, label = 'Kerr amplitudes')
plt.plot(e0_vec,SNR_AAK_vec, label = 'AAK amplitudes')
plt.grid()
plt.xlabel(r'Eccentricity $e_{0}$')
plt.ylabel(r'SNR')
plt.title("(M,mu,a,p0, T) = (1e6, 10, 0.9, 14.0, 2 years)")
plt.legend()
plt.savefig("Eccentricity_Plot_weak_field.pdf",bbox_inches = 'tight')

# Compute simple mismatches

breakpoint()
aa = xp.asarray([inner_prod(waveform_AAK_fft[i],waveform_AAK_fft[i], N_t, delta_t, PSD[i]) for i in range(N_channels)])
bb = xp.asarray([inner_prod(waveform_Kerr_fft[i],waveform_Kerr_fft[i], N_t, delta_t, PSD[i]) for i in range(N_channels)])
ab = xp.asarray([inner_prod(waveform_AAK_fft[i],waveform_Kerr_fft[i], N_t, delta_t, PSD[i]) for i in range(N_channels)])

mismatch = 1 - 0.5 * xp.sum(ab/(np.sqrt(aa*bb)))
quit()
##=====================Noise Setting: Currently 0=====================

# Compute Variance and build noise realisation
variance_noise = [N_t * PSD[k] / (4*delta_t) for k in range(N_channels)]
noise_f_real = [xp.random.normal(0,np.sqrt(variance_noise[k])) for k in range(N_channels)]
noise_f_imag = [xp.random.normal(0,np.sqrt(variance_noise[k])) for k in range(N_channels)]

# Compute noise in frequency domain
noise_f = xp.asarray([noise_f_real[k] + 1j * noise_f_imag[k] for k in range(N_channels)])

# Dealing with positive transform, so first and last values are real.
# todo: fix
#noise_f_AET[0] = np.sqrt(2)*np.real(noise_f_AET)
#noise_f_AET[-1] = np.sqrt(2)*np.real(noise_f_AET)

data_f = EMRI_truth_fft + 0*noise_f   # define the data

##===========================MCMC Settings============================

iterations = 30000  # The number of steps to run of each walker
burnin = 0 # I always set burnin when I analyse my samples
nwalkers = 50  #50 #members of the ensemble, like number of chains

ntemps = 1             # Number of temperatures used for parallel tempering scheme.
                       # Each group of walkers (equal to nwalkers) is assigned a temperature from T = 1, ... , ntemps.

tempering_kwargs=dict(ntemps=ntemps)  # Sampler requires the number of temperatures as a dictionary

d = 1 # A parameter that can be used to dictate how close we want to start to the true parameters
# Useful check: If d = 0 and noise_f = 0, llike(*params)!!

# We start the sampler exceptionally close to the true parameters and let it run. This is reasonable
# if and only if we are quantifying how well we can measure parameters. We are not performing a search.

if x_I0 < 0:
    a *= -1.0
    x_I0 *= -1.0
# Intrinsic Parameters

start_M = M*(1. + d * 1e-7 * np.random.randn(nwalkers,1))
start_mu = mu*(1. + d * 1e-7 * np.random.randn(nwalkers,1))

start_a = a*(1. + d * 1e-7 * np.random.randn(nwalkers,1))

start_p0 = p0*(1. + d * 1e-7 * np.random.randn(nwalkers, 1))
start_e0 = e0*(1. + d * 1e-7 * np.random.randn(nwalkers, 1))

# Luminosity distance
start_D = params[6]*(1 + d * 1e-6 * np.random.randn(nwalkers,1))

# Angular parameters
start_qS = qS*(1. + d * 1e-6 * np.random.randn(nwalkers,1))
start_phiS = phiS*(1. + d * 1e-6 * np.random.randn(nwalkers,1))
start_qK = qK*(1. + d * 1e-6 * np.random.randn(nwalkers,1))
start_phiK = phiK*(1. + d * 1e-6 * np.random.randn(nwalkers,1))

# Initial phases
start_Phi_Phi0 = Phi_phi0*(1. + d * 1e-5 * np.random.randn(nwalkers, 1))
start_Phi_theta0 = Phi_theta0*(1. + d * 1e-5 * np.random.randn(nwalkers, 1))
start_Phi_r0 = Phi_r0*(1. + d * 1e-5 * np.random.randn(nwalkers, 1))

# Charge
start_gamma = gamma + d * 1e-7 * np.random.randn(nwalkers,1)

start = np.hstack((start_M,start_mu, start_a, start_p0, start_e0, start_D,
start_qS, start_phiS, start_qK, start_phiK,start_Phi_Phi0, start_Phi_r0, start_gamma))

# start = np.hstack((start_M, start_a, start_p0, start_e0, start_D,
# start_qS, start_phiS, start_qK, start_phiK,start_Phi_Phi0, start_Phi_r0, start_Charge))
if ntemps > 1:
    # If we decide to use parallel tempering, we fall into this if statement. We assign each *group* of walkers
    # an associated temperature. We take the original starting values and "stack" them on top of each other.
    start = np.tile(start,(ntemps,1,1))

if np.size(start.shape) == 1:
    start = start.reshape(start.shape[-1], 1)
    ndim = 1
else:
    ndim = start.shape[-1]

# ================= SET UP PRIORS ========================

n = 25 # size of prior

Delta_theta_intrinsic = [100, 1e-3, 1e-4, 1e-4, 1e-4]  # M, mu, a, p0, e0 Y0
Delta_theta_D = dist/np.sqrt(np.sum(SNR_truth))

priors_in = {
    # Intrinsic parameters
    0: uniform_dist(M - n*Delta_theta_intrinsic[0], M + n*Delta_theta_intrinsic[0]), # Primary Mass M
    1: uniform_dist(mu - n*Delta_theta_intrinsic[1], mu + n*Delta_theta_intrinsic[1]), # Secondary Mass mu
    2: uniform_dist(a - n*Delta_theta_intrinsic[2], a + n*Delta_theta_intrinsic[2]), # Spin parameter a
    3: uniform_dist(p0 - n*Delta_theta_intrinsic[3], p0 + n*Delta_theta_intrinsic[3]), # semi-latus rectum p0
    4: uniform_dist(e0 - n*Delta_theta_intrinsic[4], e0 + n*Delta_theta_intrinsic[4]), # eccentricity e0
    5: uniform_dist(params[6] - n*Delta_theta_D, params[6] + n* Delta_theta_D), # distance D
    # Extrinsic parameters -- Angular parameters
    6: uniform_dist(0, np.pi), # Polar angle (sky position)
    7: uniform_dist(0, 2*np.pi), # Azimuthal angle (sky position)
    8: uniform_dist(0, np.pi),  # Polar angle (spin vec)
    9: uniform_dist(0, 2*np.pi), # Azimuthal angle (spin vec)
    # Initial phases
    10: uniform_dist(0, 2*np.pi), # Phi_phi0
    11: uniform_dist(0, 2*np.pi), # Phi_r00
    # 12: uniform_dist(-1, 1) # Charge
    12: uniform_dist(-1, 1) # Gamma
}


priors = ProbDistContainer(priors_in, use_cupy = False)   # Set up priors so they can be used with the sampler.

# =================== SET UP PROPOSAL ==================

moves_stretch = StretchMove(a=2.0, use_gpu=True)

# Quick checks
if ntemps > 1:
    print("Value of starting log-likelihood points", llike(start[0][0]))
    if np.isinf(sum(priors.logpdf(xp.asarray(start[0])))):
        print("You are outside the prior range, you fucked up")
        quit()
else:
    print("Value of starting log-likelihood points", llike(start[0]))
os.chdir('/work/scratch/data/burkeol/tgr_mcmc_results/Constrain_charge')

# fp = "kerr_traj_AAK_amp_M_1e5_mu_5_a_0p9_p0_23p2_e0_0p4_dist_SNR_50_Charge_0_gamma_directly_sample_prior_neg1_1.h5"
fp = "kerr_traj_AAK_amp_M_1e5_mu_5_a_0p9_p0_23p2_e0_0p4_dist_SNR_50_Charge_zero_gamma_directly_sample_prior_neg1_1.h5"

backend = HDFBackend(fp)

ensemble = EnsembleSampler(
                            nwalkers,
                            ndim,
                            llike,
                            priors,
                            backend = backend,                 # Store samples to a .h5 file
                            tempering_kwargs=tempering_kwargs,  # Allow tempering!
                            moves = moves_stretch
                            )
Reset_Backend = True # NOTE: CAREFUL HERE. ONLY TO USE IF WE RESTART RUNS!!!!
if Reset_Backend:
    os.remove(fp) # Manually get rid of backend
    backend = HDFBackend(fp) # Set up new backend
    ensemble = EnsembleSampler(
                            nwalkers,
                            ndim,
                            llike,
                            priors,
                            backend = backend,                 # Store samples to a .h5 file
                            tempering_kwargs=tempering_kwargs,  # Allow tempering!
                            moves = moves_stretch
                            )
else:
    start = backend.get_last_sample() # Start from last sample
out = ensemble.run_mcmc(start, iterations, progress=True)  # Run the sampler

##===========================MCMC Settings (change this)============================
