#%%
from few.waveform import (
    FastSchwarzschildEccentricFlux,
    RunSchwarzEccFluxInspiral,
    SlowSchwarzschildEccentricFlux,
)
from few.utils.utility import *
from few.utils.utility import get_mu_at_t
from few.utils.constants import *
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

def zero_pad(data):
    
    N = len(data)
    pow_2 = np.ceil(np.log2(N))
    res = np.pad(data,(1,0),'constant')
    N = len(res)
    pow_2 = np.ceil(np.log2(N))
    return np.pad(res,(0,int((2**pow_2)-N)),'constant')



use_gpu = False

sum_kwargs = dict(pad_output=True)

few_base = FastSchwarzschildEccentricFlux(sum_kwargs=sum_kwargs, use_gpu=use_gpu)
# p0=12;e0=0.3 -> mismatch =  0.000778239562794103

M = 1e6
mu = 40.0
p0 = 10.0
e0 = 0.7
theta = np.pi / 3
phi = np.pi / 5  # mismatch =  0.0014153820095872405

M = 1e6
mu = 50
p0 = 11.0
e0 = 0.4
theta = np.pi / 3
phi = np.pi / 5  # mismatch =  0.02257016504821807

# traj_args = [M, p0, e0]
# traj_kwargs = {}
# index_of_mu = 1

# t_out = 1.
# run trajectory
# mu = get_mu_at_t(traj_module, t_out, traj_args)
# print(mu)
# mism 0.0024275870323797744
dt = 15
T = 1.0


l = 2  # 2
m = 1  # 1
n = -4  # -4

modes = [(2, 1, -4,)]
#%% TIME DOMAIN
wave_22 = few_base(
    M,
    mu,
    p0,
    e0,
    theta,
    phi,
    T=T,
    dt=dt,
    #eps=1e-2,
    mode_selection=modes,
    include_minus_m=True,
)  # ,eps=1e-2)# , batch_size=int(1e2),mode_selection=[(l,m,n)])#,include_minus_m=True) #

signal_td = wave_22 # zero_pad(wave_22)
N = len(signal_td)

# Window
rect_td = zero_pad(signal.tukey(len(wave_22),alpha=0.5)) #zero_pad(np.ones_like(wave_22))/N
rect_fft = np.fft.fftshift(np.fft.fft(rect_td))  # * signal.tukey(len(wave_22))

freq_fft = np.fft.fftshift(np.fft.fftfreq(N, dt))
fft_wave = np.flip( np.fft.fftshift(np.fft.fft(signal_td )) * dt)  # * signal.tukey(len(wave_22))


######################################################
#%%
# FD
sum_kwargs = dict(pad_output=True, output_type="fd")

wave = FastSchwarzschildEccentricFlux(sum_kwargs=sum_kwargs, use_gpu=use_gpu)

fd_h = wave(
    M,
    mu,
    p0,
    e0,
    theta,
    phi,
    T=T, #(N-1)*dt/YRSID_SI,
    dt=dt,
    mode_selection=modes,
    #eps=1e-2,
    include_minus_m=True,
)  # ,eps=1e-2)# , mode_selection=[(l,m,n)],include_minus_m=True) #

f = np.arange(-1 / (2 * dt), +1 / (2 * dt), 1 / (len(fd_h) * dt))

# plt.plot(f, fd_h.real)
# plt.show()
# breakpoint()

#%% mismatch

print("nans in waveform", np.sum(np.isnan(fd_h)))

fd_h_correct = fd_h # signal.fftconvolve(fd_h, rect_fft/N, mode='same')  # -np.roll(
# np.flip(np.real(fd_h)) + 1j * np.flip(np.imag(fd_h)), 1
# )  # np.sin(dt*len(wave_22)*freq_fft/4/np.pi)/np.sin(dt*freq_fft/4/np.pi)#*np.exp(-1j* (len(wave_22)-1)/2 )
index_nonzero = [np.abs(fd_h_correct) != complex(0.0)][0]

# index_nonzero = np.arange(len(fd_h_correct))

def innprod(a,b):
    return np.real(np.dot(np.conj(a),b))

den = np.sqrt(
    np.real(np.dot(np.conj(fft_wave[index_nonzero]), fft_wave[index_nonzero]))
    * np.real(np.dot(np.conj(fd_h_correct[index_nonzero]), fd_h_correct[index_nonzero]))
)
print("den", den, "index", np.sum(index_nonzero))
print(
    "full mismatch = ",
    1
    - np.real(np.dot(np.conj(fd_h_correct[index_nonzero]), -fft_wave[index_nonzero]))
    / den,
)

den = np.sqrt(
    np.real(np.dot(np.conj(fft_wave[index_nonzero]), fft_wave[index_nonzero]))
    * np.real(np.dot(np.conj(fd_h_correct[index_nonzero]), fd_h_correct[index_nonzero]))
)
print("den", den, "index", np.sum(index_nonzero))
print(
    "mismatch = ",
    1
    - np.real(
        np.dot(np.abs(fd_h_correct[index_nonzero]), np.abs(fft_wave[index_nonzero]))
    )
    / den,
)

df = wave.create_waveform.frequency[1] - wave.create_waveform.frequency[0]

# figure
plt.figure()
plt.ylabel(r"Re $\tilde{h}(f)$")
plt.xlabel("f [Hz]")
# TD model
plt.plot(freq_fft, -np.real(fft_wave), label="fft TD waveform")
# FD model
plt.plot(
    wave.create_waveform.frequency,
    np.real(fd_h_correct),
    # "--",
    alpha=0.9,
    label="FD domain waveform",
)
plt.legend(loc="right")
plt.show()


# %%

# figure
plt.figure()
plt.ylabel(r"Imag $\tilde{h}(f)$")
plt.xlabel("f [Hz]")
# TD model
plt.plot(freq_fft, -np.imag(fft_wave), label="fft TD waveform")
# FD model
plt.plot(
    wave.create_waveform.frequency,
    np.imag(fd_h_correct),
    "--",
    alpha=0.9,
    label="FD domain waveform",
)
plt.legend()
plt.show()

# figure
plt.figure()
plt.ylabel(r"Abs $\tilde{h}(f)$")
plt.xlabel("f [Hz]")
# TD model
plt.plot(freq_fft, np.abs(fft_wave), label="fft TD waveform")
# FD model
plt.plot(
    wave.create_waveform.frequency,
    np.abs(fd_h_correct),
    "--",
    alpha=0.9,
    label="FD domain waveform",
)
plt.legend()
plt.show()

#breakpoint()
