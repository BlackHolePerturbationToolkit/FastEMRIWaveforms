#%%
from few.waveform import FastSchwarzschildEccentricFlux
from few.utils.constants import *
import numpy as np
import matplotlib.pyplot as plt
def shift5(arr, num, fill_value=np.nan):
    result = np.empty_like(arr)
    if num > 0:
        result[:num] = fill_value
        result[num:] = arr[:-num]
    elif num < 0:
        result[num:] = fill_value
        result[:num] = arr[-num:]
    else:
        result[:] = arr
    return result

sum_kwargs = dict(pad_output=True)

few_base = FastSchwarzschildEccentricFlux(
    sum_kwargs=sum_kwargs,
)

M = 1e6
mu = 50
p0 = 12
e0 = 0.3
theta= np.pi/3
phi = np.pi/5
dt=10
T=.9



#%% TIME DOMAIN
wave_22 = few_base(M, mu, p0, e0, theta, phi,T=T,dt=dt, eps=1e-2)
freq_fft = np.fft.fftfreq(len(wave_22),dt)
fft_wave = np.fft.fft(wave_22)*dt



sum_kwargs = dict(pad_output=True, output_type="fd",f=fft_wave)

wave = FastSchwarzschildEccentricFlux(sum_kwargs=sum_kwargs)

fd_h = wave(M,mu,p0,e0,theta,phi,T=T,dt=dt,eps=1e-2)

f = np.arange(-1/(2*dt),+1/(2*dt),1/(len(fd_h)*dt))
#%% mismatch
index_nonzero = [fd_h != 0][0]
fd_h_correct = -np.roll( np.flip(np.real(fd_h)) + 1j* np.flip(np.imag(fd_h)), 1)
den = np.sqrt(np.real(np.dot(np.conj(fft_wave[index_nonzero]),fft_wave[index_nonzero])) * np.real(np.dot(np.conj(fd_h_correct[index_nonzero]),fd_h_correct[index_nonzero])) )
print("mismatch = " ,1-np.real(np.dot(np.conj(fd_h_correct[index_nonzero]) , fft_wave[index_nonzero] ) )/den)

# np.dot(np.real(fft_wave),-np.real(fd_h)) + np.dot(np.imag(fft_wave),-np.imag(fd_h))

# figure
plt.figure()
plt.ylabel(r'Re $\tilde{h}(f)$')
plt.xlabel('f [Hz]')
# TD model
plt.plot(freq_fft, np.real(fft_wave), label='fft TD waveform')
# FD model
plt.plot(freq_fft, np.real(fd_h_correct),'--',alpha=0.9,label='FD domain waveform' )
plt.legend()
plt.show()


# %%


# figure
plt.figure()
plt.ylabel(r'Imag $\tilde{h}(f)$')
plt.xlabel('f [Hz]')
# TD model
plt.plot(freq_fft, np.imag(fft_wave), label='fft TD waveform')
# FD model
plt.plot(freq_fft, np.imag(fd_h_correct),'--',alpha=0.9,label='FD domain waveform' )
plt.legend()
plt.show()
# %%
