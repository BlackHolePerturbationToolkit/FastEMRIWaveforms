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
from few.utils.utility import get_fundamental_frequencies
from few.utils.constants import *

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
T = 0.2  # years
dt = 9.0  # seconds
M = 1e6
mu = 5e1
p0 = 10.0
e0 = 0.7
theta = np.pi / 3  # polar viewing angle
phi = np.pi / 4  # azimuthal viewing angle
dist = 1.0  # distance

mod_sel = [ (2,2,1), (3,3,1)]#, (2,2,0)]#, (3,2,0), (3,2,2)]
# FD f array

N = 2854577#len(f_in)
f_in = xp.array(np.linspace(-1 / (2 * dt), +1 / (2 * dt), num= N ))
freq = f_in[int(( N - 1 ) / 2 + 1):]
kwargs = dict(f_arr=f_in)

# full datastream
datastream = slow(
    M, mu, p0, e0, theta, phi, dist, T=T, dt=dt, **kwargs, mode_selection=mod_sel,
)
# crosse and plus
d_p, d_c = transform_to_fft_hp_hcross(datastream)
t, p, e, x, Phi_phi, Phi_theta, Phi_r = slow.inspiral_generator(M,mu, 0.0, p0, e0, 1.0, T=T, dt=dt)
Omega_phi, Omega_theta, Omega_r = get_fundamental_frequencies(
            0.0, p, e, np.zeros_like(e)
        )
f_phi, f_r = (
            xp.asarray(Omega_phi / (2 * np.pi * M * MTSUN_SI)),
            xp.asarray(Omega_r / (2 * np.pi * M * MTSUN_SI)),
        )
f_range = xp.array([mm[1]*f_phi + mm[2]*f_r for mm in mod_sel])
# ------------- pre computation ------------- #
# list of reference waveforms
list_ref_waves = [slow(
    M, mu, p0, e0, theta, phi, dist, T=T, dt=dt, **kwargs, mode_selection=mm
    ) for mm in mod_sel]
ref_wave_mode_pol = xp.array([transform_to_fft_hp_hcross(ll) for ll in list_ref_waves])

def A_vector(d, h_ref):
    return xp.array([xp.dot(xp.conj(d), h_ref*fpow)  for fpow in [xp.ones_like(freq), freq, freq**2]])

# A_vector
A_vec = xp.array([[A_vector(d_p, wave_mode[0]), A_vector(d_c, wave_mode[1])] for wave_mode in ref_wave_mode_pol])

# get frequency bin of each mode
f_bin = [freq[((xp.min(fr)<freq)*(freq<xp.max(fr)))] for fr in f_range]
print([f_bin[i] for  i in range(len(mod_sel))])
# frequency to evaluate
bin_number = 1
f_to_eval = [xp.array([fb[int(len(fb)*0.2)], fb[int(len(fb)/2)], fb[int(len(fb)*0.8)]]) for fb in f_bin]
# print("f eval", f_to_eval)

# feval = xp.sort(xp.array(f_to_eval)).flatten()
# f_sym = xp.sort(xp.hstack((feval,0,-feval))).flatten()
# re-compute list of reference waveforms at specific frequencies
# this step is not necessary because we could find the correspondent index
ind_f = [xp.array([xp.where(freq==ff[i])[0] for i in range(bin_number*3)]).flatten() for ff in f_to_eval]
print("ind_f", ind_f)
# import matplotlib.pyplot as plt
# f_min = np.min(f_to_eval[0].get())
# f_max = np.max(f_to_eval[0].get())
# plt.figure(); plt.semilogx(freq.get(), xp.real(ref_wave_p_c[0][0]).get(), alpha=0.5); plt.axvline(f_to_eval[0][0].get()); plt.axvline(f_to_eval[0][1].get());plt.axvline(f_to_eval[0][2].get()); plt.xlim([f_min*0.99, f_max*1.01]); plt.savefig('test_rel') 
# ------------- online computation ------------- #
# ind_minus_f = xp.array([xp.where(f_in==-f_to_eval[0][i])[0] for i in range(bin_number*3)]).flatten()
# ind_plus_f = xp.array([xp.where(f_in==f_to_eval[0][i])[0] for i in range(bin_number*3)]).flatten()
# ind_final = xp.append(ind_minus_f, ind_plus_f)
# kw = dict(f_arr = f_in[ind_final] )
kw = dict(f_arr = f_in )
import time
st = time.time()
list_h = [slow(
    M*(1+1e-7), mu, p0, e0, theta, phi, dist, T=T, dt=dt, **kw, mode_selection=mm
    ) for mm in mod_sel]

h_wave_mode_pol = xp.array([transform_to_fft_hp_hcross(ll) for ll in list_h])

# get the ratio
bb = xp.array([h_wave_mode_pol[i,:,ind_f[i]]/ref_wave_mode_pol[i,:,ind_f[i]] for i in range(len(mod_sel))])
print("bb",bb)
# construct matrix for 2nd order poly
Mat_F = [xp.array([[xp.ones_like(ff), ff, ff**2] for ff in fev]) for fev in f_to_eval]
print("mat",Mat_F)
# solve system of eq
r_vec = xp.array([[xp.linalg.solve(Mat_F[mod_numb], bb[mod_numb,:, pol]) for pol in range(2)] for mod_numb in range(len(mod_sel))])

print("r_vec",r_vec)
# r_c = [xp.linalg.solve(Matrix, B) for Matrix,B in zip(Mat_F,ratio_c) ]
# approximate d h
# breakpoint()
# xp.array([[ xp.dot(A_vec[mod,pol,:], r_vec[mod,pol,:]) for pol in range(2)] for mod in range(len(mod_sel))])
# sum along the modes and then take the real part and sum along polarizations
d_h_app =xp.real(xp.sum(A_vec*r_vec))
print(time.time()- st)
print('d h',d_h_app)
check_h = slow(
    M*(1+1e-8), mu, p0, e0, theta, phi, dist, T=T, dt=dt, **kw, mode_selection=mod_sel
    )
h_p, h_c = transform_to_fft_hp_hcross(check_h)
d_h_true = xp.real(xp.dot(xp.conj(d_p),h_p)) + xp.real(xp.dot(xp.conj(d_c),h_c))
# h_h = 4*xp.real(xp.dot(xp.conj(ref_wave_p_c[0][0]),ref_wave_p_c[0][0])) #+ 4*xp.real(xp.dot(ref_wave_p_c[0][1],ref_wave_p_c[0][1]))
print('d h true',d_h_true)
for i in range(len(mod_sel)):
    print(mod_sel[i])
    print(A_vec[i,0,:],r_vec[i,0,:])
    print(xp.real(xp.dot(A_vec[i,0,:],r_vec[i,0,:])), xp.real(xp.dot(xp.conj(ref_wave_mode_pol[i,0,:]),h_wave_mode_pol[i,0,:]))  )
    print(xp.real(xp.dot(A_vec[i,1,:],r_vec[i,1,:])), xp.real(xp.dot(xp.conj(ref_wave_mode_pol[i,1,:]),h_wave_mode_pol[i,1,:]))  )

breakpoint()




