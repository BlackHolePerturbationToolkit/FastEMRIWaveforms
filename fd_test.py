#%%
from few.waveform import FastSchwarzschildEccentricFlux, RunSchwarzEccFluxInspiral, SlowSchwarzschildEccentricFlux
from few.utils.utility import *
from few.utils.utility import get_mu_at_t
from few.utils.constants import *
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

use_gpu = False

sum_kwargs = dict(pad_output=True)

few_base = FastSchwarzschildEccentricFlux(
    sum_kwargs=sum_kwargs,
)

# initial condition
M = 1e6; mu = 40.; p0 = 10.; e0 = 0.7; theta= np.pi/3; phi = np.pi/5 # mismatch =  0.0014153820095872405
#M = 1e6; mu = 50; p0 = 11.48; e0 = 0.7; theta= np.pi/3; phi = np.pi/5 # mismatch =  0.02257016504821807

# you can get mu to plung in one year
traj_args = [M, p0, e0]
traj_kwargs = {}
index_of_mu = 1
t_out = 1.
# run trajectory
traj_module = RunSchwarzEccFluxInspiral()
mu = get_mu_at_t(traj_module, t_out, traj_args)
print(mu)
# mism 0.0024275870323797744
dt=10.
T=1.#/2.5


l = 2 #2
m = 1 #1
n = 3 #-4

#%% TIME DOMAIN
wave_22 = few_base(M, mu, p0, e0, theta, phi,T=T,dt=dt,eps=5e-2)#,mode_selection=[(l,m,n)])mode_selection=[(l,m,n)],include_minus_m=True)  , batch_size=int(1e2),mode_selection=[(l,m,n)])#,include_minus_m=True) #
freq_fft = np.fft.fftshift(np.fft.fftfreq(len(wave_22),dt))
fft_wave = np.fft.fftshift(np.fft.fft(np.real(wave_22)-1j* np.imag(wave_22) )*dt)

sum_kwargs = dict(pad_output=True, output_type="fd")

wave = FastSchwarzschildEccentricFlux(sum_kwargs=sum_kwargs)

fd_h = wave(M,mu,p0,e0,theta,phi,T=T,dt=dt,eps=5e-2)#,mode_selection=[(l,m,n)])#,eps=1e-2)#,include_minus_m=True) #,eps=1e-2)# , mode_selection=[(l,m,n)],include_minus_m=True) #

f = np.arange(-1/(2*dt),+1/(2*dt),1/(len(fd_h)*dt))

real_fd = np.real(fd_h + np.flip(fd_h))/2 + 1j* np.imag(fd_h - np.flip(fd_h))/2
imag_fd = -np.imag(fd_h + np.flip(fd_h))/2 - 1j* np.real(fd_h - np.flip(fd_h))/2

#%% mismatch

print("nans in waveform", np.sum(np.isnan(fd_h)))

fd_h_correct = fd_h #-np.roll( np.flip(np.real(fd_h)) + 1j* np.flip(np.imag(fd_h)), 1)#np.sin(dt*len(wave_22)*freq_fft/4/np.pi)/np.sin(dt*freq_fft/4/np.pi)#*np.exp(-1j* (len(wave_22)-1)/2 )

def innprod(a,b):
    return np.real(np.dot(np.conj(a),b))

# check nan
den = np.sqrt(innprod(fft_wave,fft_wave) * innprod(fd_h_correct,fd_h_correct))
print("mismatch = " ,1-innprod(fft_wave,fft_wave)/den)
print("overlap = " ,innprod(fft_wave,fft_wave)/den)


# figure
plt.figure()
plt.ylabel(r' $|\tilde{h}(f)|$')
plt.xlabel('f [Hz]')
# TD model
plt.semilogx(freq_fft, np.abs(fft_wave), label='fft TD waveform')
# FD model
plt.semilogx(freq_fft, np.abs(fd_h_correct),'--',alpha=0.9,label='FD domain waveform' )
plt.legend(loc='best')
plt.show()


# %%


# figure
plt.figure()
plt.ylabel(r'Imag $\tilde{h}(f)$')
plt.xlabel('f [Hz]')
# TD model
plt.semilogx(freq_fft, np.imag(fft_wave), label='fft TD waveform')
# FD model
plt.semilogx(freq_fft, np.imag(fd_h_correct),'--',alpha=0.9,label='FD domain waveform' )
plt.legend()
plt.show()

# %%
