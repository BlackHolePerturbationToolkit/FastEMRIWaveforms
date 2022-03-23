dev = 7
import os
os.system(f"CUDA_VISIBLE_DEVICES={dev}")
os.environ["CUDA_VISIBLE_DEVICES"] = f"{dev}"
os.system("echo $CUDA_VISIBLE_DEVICES")

os.system("export OMP_NUM_THREADS=1")
os.environ["OMP_NUM_THREADS"] = "1"

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
T = 1.0  # years
dt = 9.0  # seconds
M = 1e6
mu = 5e1
p0 = 10.0
e0 = 0.7
theta = np.pi / 3  # polar viewing angle
phi = np.pi / 4  # azimuthal viewing angle
dist = 1.0  # distance

mod_sel = [ (2,2,1), (3,2,0), (2,2,0), (3,2,2), (2,2,-1)]#, (3,2,1)]
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
list_ref_waves = fast(
    M, mu, p0, e0, theta, phi, dist, T=T, dt=dt, **kwargs, mode_selection=mod_sel
)
ref_wave_mode_pol = xp.array([transform_to_fft_hp_hcross(ll) for ll in list_ref_waves])

def A_vector(d, h_ref):
    return xp.array([xp.dot(xp.conj(d), h_ref*fpow)  for fpow in [xp.ones_like(freq), freq, freq**2]])

def B_vector(d, h_ref):
    return xp.array([xp.dot(xp.conj(d), h_ref*fpow)  for fpow in [xp.ones_like(freq), freq, freq**2, freq**3, freq**4]])

# A_vector
A_vec = xp.array([[A_vector(d_p, wave_mode[0]), A_vector(d_c, wave_mode[1])] for wave_mode in ref_wave_mode_pol])

# B matrix
B_matrix_p = xp.array([[B_vector(ref_wave_mode_pol[m1,pol,:], ref_wave_mode_pol[m2,pol,:]) for m1 in range(len(mod_sel)) for m2 in range(len(mod_sel)) if m1<=m2] for pol in range(2) ])
# tri_up_list = [xp.triu(B_matrix_p[:,i]) for i in range(n_voices)]
# final_B = xp.array([tri_up + tri_up.T - xp.diag(xp.diag(tri_up)) for tri_up in tri_up_list])
# tri = xp.zeros((n_voices,n_voices))
# tri[np.triu_indices(n_voices)] = B_matrix_p[:,0]

# get frequency bin of each mode
f_bin = [freq[((xp.min(fr)<freq)*(freq<xp.max(fr)))] for fr in f_range]
# print([f_bin[i] for  i in range(len(mod_sel))])
# frequency to evaluate
bin_number = 1
f_to_eval = [xp.array([fb[int(len(fb)*0.2)], fb[int(len(fb)/2)], fb[int(len(fb)*0.8)]]) for fb in f_bin]
# print("f eval", f_to_eval)

# feval = xp.unique(xp.array(f_to_eval).flatten())
# f_sym = xp.sort(xp.hstack((f_in[0],feval,0,-feval,f_in[-1])).flatten())
# re-compute list of reference waveforms at specific frequencies
# this step is not necessary because we could find the correspondent index
ind_f = [xp.array([xp.where(freq==ff[i])[0] for i in range(bin_number*3)]).flatten() for ff in f_to_eval]
# print("ind_f", ind_f)
# ------------- online computation ------------- #
# ind_minus_f = xp.array([xp.where(f_in==-f_to_eval[0][i])[0] for i in range(bin_number*3)]).flatten()
# ind_plus_f = xp.array([xp.where(f_in==f_to_eval[0][i])[0] for i in range(bin_number*3)]).flatten()
# ind_final = xp.append(ind_minus_f, ind_plus_f)
# kw = dict(f_arr = f_in[ind_final] )
# breakpoint()
kw = dict(f_arr = f_in )
# kw = dict(f_arr = f_sym )

list_h = fast(
    M*(1+5e-6), mu, p0, e0, theta, phi, dist, T=T, dt=dt, **kw, mode_selection=mod_sel
    )

h_wave_mode_pol = xp.array([transform_to_fft_hp_hcross(ll) for ll in list_h])

# get the ratio
# to insert the approximation we need to change only this ratio
bb = xp.array([h_wave_mode_pol[i,:,ind_f[i]]/ref_wave_mode_pol[i,:,ind_f[i]] for i in range(len(mod_sel))])
# print("bb",bb)
# construct matrix for 2nd order poly
Mat_F = [xp.array([[xp.ones_like(ff), ff, ff**2] for ff in fev]) for fev in f_to_eval]
# print("mat",Mat_F)
# solve system of eq
r_vec = xp.array([[xp.linalg.solve(Mat_F[mod_numb], bb[mod_numb,:, pol]) for pol in range(2)] for mod_numb in range(len(mod_sel))])

r_mat = xp.array([[
    [
        xp.conj(r_vec[v1, pol, 0])*r_vec[v2, pol, 0],
        xp.conj(r_vec[v1, pol, 0])*r_vec[v2, pol, 1] + xp.conj(r_vec[v1, pol, 1])*r_vec[v2, pol, 0],
        xp.conj(r_vec[v1, pol, 0])*r_vec[v2, pol, 2] + xp.conj(r_vec[v1, pol, 1])*r_vec[v2, pol, 1] + xp.conj(r_vec[v1, pol, 2])*r_vec[v2, pol, 0],
        xp.conj(r_vec[v1, pol, 1])*r_vec[v2, pol, 2] + xp.conj(r_vec[v1, pol, 2])*r_vec[v2, pol, 1],
        xp.conj(r_vec[v1, pol, 2])*r_vec[v2, pol, 2]
    ]  for v1 in range(len(mod_sel)) for v2 in range(len(mod_sel)) if v1<=v2] for pol in range(2)])


print("r_vec",r_vec)
# r_c = [xp.linalg.solve(Matrix, B) for Matrix,B in zip(Mat_F,ratio_c) ]
# approximate d h
# breakpoint()
# xp.array([[ xp.dot(A_vec[mod,pol,:], r_vec[mod,pol,:]) for pol in range(2)] for mod in range(len(mod_sel))])
# sum along the modes and then take the real part and sum along polarizations
d_h_app =xp.real(xp.sum(A_vec*r_vec))
h_h_app = xp.sum(xp.array([[xp.real(xp.dot(B_matrix_p[pol,:,mm],r_mat[pol,:,mm])) for mm in range(len(mod_sel))] for pol in range(2)]))

# CHECK #

# check each polarizartion
tot_dh = 0.0
tot_hh_p = xp.array([xp.dot(xp.conj(h_wave_mode_pol[v1,0,:]),h_wave_mode_pol[v2,0,:]) for v1 in range(len(mod_sel)) for v2 in range(len(mod_sel)) if v1<=v2])
tot_hh_c = xp.array([xp.dot(xp.conj(h_wave_mode_pol[v1,1,:]),h_wave_mode_pol[v2,1,:]) for v1 in range(len(mod_sel)) for v2 in range(len(mod_sel)) if v1<=v2])
tot_hh = xp.real(xp.sum(tot_hh_p) + xp.sum(tot_hh_c))
for i in range(len(mod_sel)):
    print(mod_sel[i])
    print(xp.real(xp.dot(A_vec[i,0,:],r_vec[i,0,:])),'=', xp.real(xp.dot(xp.conj(d_p),h_wave_mode_pol[i,0,:]))  )
    print(xp.real(xp.dot(A_vec[i,1,:],r_vec[i,1,:])),'=', xp.real(xp.dot(xp.conj(d_c),h_wave_mode_pol[i,1,:]))  )
    tot_dh = tot_dh + xp.real(xp.dot(xp.conj(d_p),h_wave_mode_pol[i,0,:])) + xp.real(xp.dot(xp.conj(d_c),h_wave_mode_pol[i,1,:]))


print('d h',d_h_app)
print("d h true", tot_dh)
print(" rel diff", (d_h_app - tot_dh)/tot_dh)


print('h h',h_h_app)
print("h h true", tot_hh)
print(" rel diff", (h_h_app - tot_hh)/tot_hh)
# breakpoint()