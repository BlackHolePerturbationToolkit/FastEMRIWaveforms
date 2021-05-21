#%%
from few.waveform import FastSchwarzschildEccentricFlux, RunSchwarzEccFluxInspiral, SlowSchwarzschildEccentricFlux
from few.utils.utility import *
from few.utils.constants import *
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal


sum_kwargs = dict(pad_output=True)


use_gpu = False

# keyword arguments for inspiral generator (RunSchwarzEccFluxInspiral)
inspiral_kwargs={
        "DENSE_STEPPING": 1,  # we want a sparsely sampled trajectory
        "max_init_len": int(1e7),  # dense stepping trajectories
    }

# keyword arguments for inspiral generator (RomanAmplitude)
amplitude_kwargs = {
    "max_init_len": int(1e4),  # this must be >= batch_size
}

# keyword arguments for Ylm generator (GetYlms)
Ylm_kwargs = {
    "assume_positive_m": False  # if we assume positive m, it will generate negative m for all m>0
}

# keyword arguments for summation generator (InterpolatedModeSum)
sum_kwargs = {
    "use_gpu": use_gpu  # GPU is available for this type of summation
}

slow = SlowSchwarzschildEccentricFlux(
    inspiral_kwargs=inspiral_kwargs,
    amplitude_kwargs=amplitude_kwargs,
    Ylm_kwargs=Ylm_kwargs,
    sum_kwargs=sum_kwargs,
    use_gpu=use_gpu,
)


few_base = FastSchwarzschildEccentricFlux(
    sum_kwargs=sum_kwargs,
)
# p0=12;e0=0.3 -> mismatch =  0.000778239562794103

M = 1e6; mu = 40.; p0 = 10.; e0 = 0.7; theta= np.pi/3; phi = np.pi/5 # mismatch =  0.0014153820095872405

#M = 1e6; mu = 50; p0 = 11.48; e0 = 0.7; theta= np.pi/3; phi = np.pi/5 # mismatch =  0.02257016504821807

traj_module = RunSchwarzEccFluxInspiral()

#traj_args = [M, p0, e0]
#traj_kwargs = {}
#index_of_mu = 1

t_out = 1.
# run trajectory
#mu = get_mu_at_t(traj_module, t_out, traj_args)
print(mu)
# mism 0.0024275870323797744
dt=50
T=1/2.5


l=2 #2
m=0 #1
n=-2 #-4

#%% TIME DOMAIN
wave_22 = slow(M, mu, p0, e0, theta, phi,T=T,dt=dt, batch_size=int(1e3),mode_selection=[(l,m,n)])#,include_minus_m=True) #,eps=1e-2)# 
freq_fft = np.fft.fftfreq(len(wave_22),dt)
fft_wave = np.fft.fft(wave_22 )*dt #* signal.tukey(len(wave_22))

rect_fft = np.fft.fft(np.ones_like(wave_22)) #* signal.tukey(len(wave_22))


sum_kwargs = dict(pad_output=True, output_type="fd")

wave = FastSchwarzschildEccentricFlux(sum_kwargs=sum_kwargs)

fd_h = wave(M,mu,p0,e0,theta,phi,T=T,dt=dt,mode_selection=[(l,m,n)],include_minus_m=True) #,eps=1e-2)# , mode_selection=[(l,m,n)],include_minus_m=True) #

f = np.arange(-1/(2*dt),+1/(2*dt),1/(len(fd_h)*dt))


#%% mismatch
print("nans in waveform", np.sum(np.isnan(fd_h)))

fd_h_correct = -np.roll( np.flip(np.real(fd_h)) + 1j* np.flip(np.imag(fd_h)), 1)#np.sin(dt*len(wave_22)*freq_fft/4/np.pi)/np.sin(dt*freq_fft/4/np.pi)#*np.exp(-1j* (len(wave_22)-1)/2 )
index_nonzero = [np.abs(fd_h_correct) !=complex(0.0)][0]

# check nan

den = np.sqrt(np.real(np.dot(np.conj(fft_wave[index_nonzero]),fft_wave[index_nonzero])) * np.real(np.dot(np.conj(fd_h_correct[index_nonzero]),fd_h_correct[index_nonzero])) )
print('den',den,'index',np.sum(index_nonzero))
print("mismatch = " ,1-np.real(np.dot(np.conj(fd_h_correct[index_nonzero]) , fft_wave[index_nonzero] ) )/den)

den = np.sqrt(np.real(np.dot(np.conj(fft_wave[index_nonzero]),fft_wave[index_nonzero])) * np.real(np.dot(np.conj(fd_h_correct[index_nonzero]),fd_h_correct[index_nonzero])) )
print('den',den,'index',np.sum(index_nonzero))
print("mismatch = " ,1-np.real(np.dot(np.abs(fd_h_correct[index_nonzero]) , np.abs(fft_wave[index_nonzero]) ) )/den)

#%%
# np.dot(np.real(fft_wave),-np.real(fd_h)) + np.dot(np.imag(fft_wave),-np.imag(fd_h))
"""
# figure
plt.figure()
plt.ylabel(r'Re $\tilde{h}(f)$')
plt.xlabel('f [Hz]')
# TD model
plt.plot(freq_fft, np.real(fft_wave), label='fft TD waveform')
# FD model
plt.plot(freq_fft, np.real(fd_h_correct),'--',alpha=0.9,label='FD domain waveform' )
plt.legend(loc='right')
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
"""
# %%
"""
# figure
plt.figure()
plt.ylabel(r'Ang $\tilde{h}(f)$')
plt.xlabel('f [Hz]')
# TD model
plt.plot(freq_fft, np.angle(fft_wave), label='fft TD waveform')
# FD model
plt.plot(freq_fft, np.angle(fd_h_correct),'--',alpha=0.9,label='FD domain waveform' )
plt.legend()
plt.show()

"""
#%%
f_re_im_abs_log = np.loadtxt('a0.0_p0_10_e0_0.7.FDv_2_0_0_2')
fact = 3.7#4.101694915254237
#545*mu/(5852) #mu #(mu * MRSUN_SI)#/(1 * Gpc) 
frequency = f_re_im_abs_log[:,0]/(M*MTSUN_SI)
re_h = f_re_im_abs_log[:,1] * fact
im_h = f_re_im_abs_log[:,2] * fact
abs_h = f_re_im_abs_log[:,3] * fact

#(t/M)    Re(H_{lmkn})    -Im(H_{lmkn})      \Phi_{mkn}

t_re_im_phi = np.loadtxt('a0.0_p0_10_e0_0.7.TDv_2_0_0_2')
scott_t = t_re_im_phi[:,0]*(M*MTSUN_SI)
scott_freq = np.fft.fftfreq(len(scott_t),scott_t[1]-scott_t[0])
scott_td = np.fft.fft( (t_re_im_phi[:,1] + 1j*t_re_im_phi[:,2])*np.exp(1j*t_re_im_phi[:,3])) *(scott_t[1]-scott_t[0])
"""
# figure
plt.figure()
plt.ylabel(r' $|\tilde{h}(f)|$')
plt.xlabel('f [Hz]')
# TD model
plt.plot(scott_freq, np.abs(scott_td), label='Scott fft TD waveform')
# FD model
plt.plot(frequency, abs_h,'-',alpha=0.9,label='Scott FD domain waveform' )

plt.legend()
plt.show()
"""
#%%
"""
# figure
plt.figure()
plt.ylabel(r' $|\tilde{h}(f)|$')
plt.xlabel('f [Hz]')
# TD model
plt.plot(freq_fft, np.abs(fft_wave), label='fft TD waveform')
# FD model
plt.plot(freq_fft, np.abs(fd_h_correct),'--',alpha=0.9,label='FD domain waveform' )
plt.plot(frequency, abs_h,'-',alpha=0.9,label='Scott FD domain waveform' )

plt.legend()
plt.show()

"""
# figure
plt.figure()
plt.ylabel(r' $h_{+}(t)$')
plt.xlabel('t [s]')
# TD model
plt.plot(scott_t, (t_re_im_phi[:,1] + 1j*t_re_im_phi[:,2])*np.exp(1j*t_re_im_phi[:,3]), label='Scott TD waveform')
# FD model
plt.plot(np.arange(0,len(wave_22)*dt,dt) , wave_22,'--',alpha=0.9,label='few TD waveform' )

plt.legend()
plt.show()
# %%
