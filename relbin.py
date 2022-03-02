import unittest
import numpy as np
import warnings

from few.trajectory.inspiral import EMRIInspiral
from few.amplitude.romannet import RomanAmplitude
from few.amplitude.interp2dcubicspline import Interp2DAmplitude
from few.waveform import FastSchwarzschildEccentricFlux, SlowSchwarzschildEccentricFlux, FastSchwarzschildEccentricFluxHarmonics
from few.utils.utility import get_overlap, get_mismatch
from few.utils.ylm import GetYlms
from few.utils.modeselector import ModeSelector
from few.summation.interpolatedmodesum import CubicSplineInterpolant

try:
    import cupy as xp

    gpu_available = True

except (ModuleNotFoundError, ImportError) as e:
    import numpy as xp

    warnings.warn(
        "CuPy is not installed or a gpu is not available. If trying to run on a gpu, please install CuPy."
    )
    gpu_available = False

# process FD
def transform_to_fft_hp_hcross(wave):
    fd_sig = -xp.flip(wave)

    ind =int(( len(fd_sig) - 1 ) / 2 + 1)

    fft_sig_r = xp.real(fd_sig + xp.flip(fd_sig) )/2.0 + 1j * xp.imag(fd_sig - xp.flip(fd_sig))/2.0
    fft_sig_i = -xp.imag(fd_sig + xp.flip(fd_sig) )/2.0 + 1j * xp.real(fd_sig - xp.flip(fd_sig))/2.0
    return [fft_sig_r[ind:], fft_sig_i[ind:]]


###############################################################################
# keyword arguments for inspiral generator (RunSchwarzEccFluxInspiral)
inspiral_kwargs = {
    "DENSE_STEPPING": 0,  # we want a sparsely sampled trajectory
    "max_init_len": int(
        1e3
    ),  # all of the trajectories will be well under len = 1000
}

# keyword arguments for inspiral generator (RomanAmplitude)
amplitude_kwargs = {
    "max_init_len": int(
        1e3
    )  # all of the trajectories will be well under len = 1000
}

# keyword arguments for Ylm generator (GetYlms)
Ylm_kwargs = {
    "assume_positive_m": False  # if we assume positive m, it will generate negative m for all m>0
}

# keyword arguments for summation generator (InterpolatedModeSum)
sum_kwargs = dict(pad_output=True, output_type="fd")

# list of waveforms
fast = FastSchwarzschildEccentricFluxHarmonics(
    inspiral_kwargs=inspiral_kwargs,
    amplitude_kwargs=amplitude_kwargs,
    Ylm_kwargs=Ylm_kwargs,
    sum_kwargs=sum_kwargs,
    use_gpu=gpu_available,
)

# setup datastream
slow = FastSchwarzschildEccentricFlux(
    inspiral_kwargs=inspiral_kwargs,
    amplitude_kwargs=amplitude_kwargs,
    Ylm_kwargs=Ylm_kwargs,
    sum_kwargs=sum_kwargs,
    use_gpu=gpu_available,
)

# parameters
T = 1.0  # years
dt = 9.0  # seconds
M = 1e6
mu = 5e1
p0 = 10.0
e0 = 0.7
theta = np.pi / 3  # polar viewing angle
phi = np.pi / 4  # azimuthal viewing angle
dist = 1.0  # distance


# FD f array

N = 2854577#len(f_in)
f_in = xp.array(np.linspace(-1 / (2 * dt), +1 / (2 * dt), num= N ))
freq = f_in[int(( N - 1 ) / 2 + 1):]
kwargs = dict(f_arr=f_in)

# full datastream
datastream = slow(
    M, mu, p0, e0, theta, phi, dist, T=T, dt=dt, **kwargs, mode_selection=[(2,2,0)]#, (2,2,2)]
)
# crosse and plus
d_p, d_c = transform_to_fft_hp_hcross(datastream)


# ------------- pre computation ------------- #
# list of reference waveforms
list_ref_waves = fast(
    M, mu, p0, e0, theta, phi, dist, T=T, dt=dt, **kwargs, mode_selection=[(2,2,0)]#, (2,2,2)]
    )
ref_wave_p_c = [transform_to_fft_hp_hcross(ll) for ll in list_ref_waves]
# A_vector
A_vec_p = [xp.array([xp.dot(xp.conj(d_p), wave_mode[0]*fpow)  for fpow in [xp.ones_like(freq), freq, freq**2]]) for wave_mode in ref_wave_p_c]
A_vec_c = [xp.array([xp.dot(xp.conj(d_c), wave_mode[1]*fpow)  for fpow in [xp.ones_like(freq), freq, freq**2]]) for wave_mode in ref_wave_p_c]
# get frequency bin of each mode
mask_mode = [(ll[0]!=complex(0.0)) for ll in ref_wave_p_c]
f_bin = [freq[mm] for mm in mask_mode]
check = [ref_wave_p_c[0][0][mm] for mm in mask_mode]
# frequency to evaluate
bin_number = 1
f_to_eval = [xp.array([fb[i] for i in xp.linspace(int((len(fb)-2)*0.2), int((len(fb)-2)*0.8), num=bin_number*3, dtype=int)]) for fb in f_bin]
feval = xp.sort(xp.array(f_to_eval)).flatten()
f_sym = xp.sort(xp.hstack((feval,0,-feval))).flatten()
# re-compute list of reference waveforms at specific frequencies
# this step is not necessary because we could find the correspondent index
ind_f = xp.array([xp.where(freq==f_to_eval[0][i])[0] for i in range(bin_number*3)]).flatten()
ref_waves = [[ref_w[0][ind_f],ref_w[1][ind_f]]  for ref_w in ref_wave_p_c]

import matplotlib.pyplot as plt
f_min = np.min(f_to_eval[0].get())
f_max = np.max(f_to_eval[0].get())
plt.figure(); plt.semilogx(freq.get(), xp.real(ref_wave_p_c[0][0]).get(), alpha=0.5); plt.axvline(f_to_eval[0][0].get()); plt.axvline(f_to_eval[0][1].get());plt.axvline(f_to_eval[0][2].get()); plt.xlim([f_min*0.99, f_max*1.01]); plt.savefig('test_rel') 
# ------------- online computation ------------- #
# ind_minus_f = xp.array([xp.where(f_in==-f_to_eval[0][i])[0] for i in range(bin_number*3)]).flatten()
# ind_plus_f = xp.array([xp.where(f_in==f_to_eval[0][i])[0] for i in range(bin_number*3)]).flatten()
# ind_final = xp.append(ind_minus_f, ind_plus_f)
# kw = dict(f_arr = f_in[ind_final] )
kw = dict(f_arr = f_in )

list_h = fast(
    M*(1+1e-8), mu, p0, e0, theta, phi, dist, T=T, dt=dt, **kw, mode_selection=[ (2,2,0)]
    )
h_wave_p_c = [transform_to_fft_hp_hcross(ll) for ll in list_h]
h_waves = [[ref_w[0][ind_f],ref_w[1][ind_f]]  for ref_w in h_wave_p_c]


Mat_F = [xp.array([[xp.ones_like(ff), ff, ff**2] for ff in fev]) for fev in f_to_eval]
# ratio of the waveforms
ratio_p = [num[0]/den[0] for num,den in zip(h_waves, ref_waves)]
ratio_c = [num[1]/den[1] for num,den in zip(h_waves, ref_waves)]
# breakpoint()
r_p = [xp.linalg.solve(Matrix, B) for Matrix,B in zip(Mat_F,ratio_p) ]
r_c = [xp.linalg.solve(Matrix, B) for Matrix,B in zip(Mat_F,ratio_c) ]
# approximate d h
d_h_app = 4*xp.real(xp.dot(A_vec_p[0],r_p[0])) #+ 4*xp.real(xp.dot(xp.array(A_vec_c[0]),r_c[0]))
print('d h',d_h_app)
d_h_true = 4*xp.real(xp.dot(xp.conj(h_wave_p_c[0][0]),ref_wave_p_c[0][0])) #+ 4*xp.real(xp.dot(h_wave_p_c[0][1],ref_wave_p_c[0][1]))
h_h = 4*xp.real(xp.dot(xp.conj(ref_wave_p_c[0][0]),ref_wave_p_c[0][0])) #+ 4*xp.real(xp.dot(ref_wave_p_c[0][1],ref_wave_p_c[0][1]))
print('d h true',d_h_true)
breakpoint()




