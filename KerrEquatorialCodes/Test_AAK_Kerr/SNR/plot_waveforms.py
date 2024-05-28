import cupy as cp
import numpy as np
import os 
import sys
sys.path.append("../")

# from scipy.signal import tukey       # I'm always pro windowing.  

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
    """
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

M = 1e6; mu = 10; a = 0.3; p0 = 13.0; e0 = 0.05; x_I0 = 1.0;
dist = 1.0; 

qS = np.pi/3 ; phiS = np.pi; qK = np.pi/4; phiK = 0.9; 

Phi_phi0 = 2.0; Phi_theta0 = 0.0; Phi_r0 = 2.0

delta_t = 2.0;  # Sampling interval [seconds]
T = 2.0     # Evolution time [years]

use_gpu = True
xp = cp

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
    "specific_spins":[0.3],
    "use_gpu": True
    }
Waveform_model_AAK = GenerateEMRIWaveform(
AAKWaveformBase, # Define the base waveform
EMRIInspiral, # Define the trajectory
KerrAAKSummation, # Define the summation for the amplitudes
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
p0 = 14.0
params = [M,mu,a,p0,e0,1.0, dist, qS, phiS, qK, phiK, Phi_phi0, Phi_theta0, Phi_r0] 

specific_modes = [(2,2,n) for n in range(-2,2)]

waveform_AAK = Waveform_model_AAK(*params, T = T, dt = delta_t, mich = False)  
waveform_Kerr = Waveform_model_Kerr(*params, T = T, dt = delta_t, mich = False, mode_selection=specific_modes) 

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

aa = xp.asarray([inner_prod(waveform_AAK_fft[i],waveform_AAK_fft[i], N_t, delta_t, PSD[i]) for i in range(N_channels)])
bb = xp.asarray([inner_prod(waveform_Kerr_fft[i],waveform_Kerr_fft[i], N_t, delta_t, PSD[i]) for i in range(N_channels)])
ab = xp.asarray([inner_prod(waveform_AAK_fft[i],waveform_Kerr_fft[i], N_t, delta_t, PSD[i]) for i in range(N_channels)])

overlap = 0.5*xp.sum(ab/(np.sqrt(aa*bb)))
mismatch = 1 - overlap

t = np.arange(0,N_t*delta_t, delta_t)
waveform_AAK_I_np = xp.asnumpy(waveform_AAK[0])
waveform_Kerr_I_np = xp.asnumpy(waveform_Kerr[0])

waveform_AAK_II_np = xp.asnumpy(waveform_AAK[1])
waveform_Kerr_II_np = xp.asnumpy(waveform_Kerr[1])

import matplotlib.pyplot as plt

os.chdir('waveform_plots/')
plt.plot(t[0:5000], waveform_AAK_I_np[0:5000], label = "AAK -- SNR = {}".format(np.round(SNR_AAK,5)))
plt.plot(t[0:5000],waveform_Kerr_I_np[0:5000], label = "Kerr -- SNR = {}".format(np.round(SNR_Kerr,5)))
plt.xlabel(r'Time [seconds]')
plt.ylabel(r'Waveform strain (channel I)')
plt.title("(M, mu, a, p0, e0) = (1e6, 10, 0.3, 14.0, 0.05)")
plt.legend(fontsize = 16)
plt.savefig("waveform_plots/waveform_plot_start_plus.pdf",bbox_inches='tight')
plt.clf()

plt.plot(t[0:5000], waveform_AAK_II_np[0:5000], label = "AAK -- SNR = {}".format(np.round(SNR_AAK,5)))
plt.plot(t[0:5000],waveform_Kerr_II_np[0:5000], label = "Kerr -- SNR = {}".format(np.round(SNR_Kerr,5)))
plt.xlabel(r'Time [seconds]')
plt.ylabel(r'Waveform strain (channel II)')
plt.title("(M, mu, a, p0, e0) = (1e6, 10, 0.3, 14.0, 0.05)")
plt.legend(fontsize = 16)
plt.savefig("waveform_plots/waveform_plot_start_cross.pdf",bbox_inches='tight')
plt.clf()

quit()