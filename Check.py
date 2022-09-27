import warnings
import time
import numpy as np
import matplotlib.pyplot as plt
# try to import cupy
try:
    import cupy as xp

except (ImportError, ModuleNotFoundError) as e:
    import numpy as xp

# Cython imports
from pyinterp_cpu import interpolate_arrays_wrap as interpolate_arrays_wrap_cpu
from pyinterp_cpu import get_waveform_wrap as get_waveform_wrap_cpu

# Python imports
from few.summation.interpolatedmodesum import InterpolatedModeSum
from few.utils.ylm import GetYlms
from few.utils.baseclasses import SummationBase, SchwarzschildEccentric
from few.utils.citations import *
from few.utils.utility import get_fundamental_frequencies
from few.utils.constants import *
from few.summation.interpolatedmodesum import CubicSplineInterpolant
from few.trajectory.inspiral import EMRIInspiral
from few.waveform import FastSchwarzschildEccentricFlux, SlowSchwarzschildEccentricFlux, GenerateEMRIWaveform
from few.amplitude.romannet import RomanAmplitude
from few.amplitude.interp2dcubicspline import Interp2DAmplitude
from few.utils.modeselector import ModeSelector
from few.infomat import InfoMatrixFastSchwarzschildEccentricFlux



# Attempt Cython imports of GPU functions
try:
    from pyinterp import interpolate_arrays_wrap, get_waveform_wrap

except (ImportError, ModuleNotFoundError) as e:
    pass

# for special functions
from scipy import special
from scipy.interpolate import CubicSpline
import multiprocessing as mp

from lisatools.diagnostic import *

gpu_available = False

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
sum_kwargs = {"pad_output":True}

gen_wave = FastSchwarzschildEccentricFlux(inspiral_kwargs=inspiral_kwargs,
    amplitude_kwargs=amplitude_kwargs,
    Ylm_kwargs=Ylm_kwargs,
    sum_kwargs=sum_kwargs,
    use_gpu=gpu_available,)

fast = InfoMatrixFastSchwarzschildEccentricFlux(
    inspiral_kwargs=inspiral_kwargs,
    amplitude_kwargs=amplitude_kwargs,
    Ylm_kwargs=Ylm_kwargs,
    sum_kwargs=sum_kwargs,
    use_gpu=gpu_available,
    # normalize_amps=False
)

# parameters
T = 0.5  # years
dt = 15.0  # seconds
M = 1e6
mu = 3e1
p0 = 10.0
e0 = 0.657
theta = np.pi / 3  # polar viewing angle
phi = np.pi / 4  # azimuthal viewing angle
dist = 1.0  # distance
batch_size = int(1e4)

deriv_inds = [0]
wv_kw = dict(T=T, dt=dt, mode_selection=[(2,2,0),(2,2,-1),(2,2,-2),(2,2,-4), (2,2,2),(4,2,0)], dist=0.3)
fast_wave = fast(M, mu, p0, e0, theta, phi, delta_deriv=[5e-2], deriv_inds=deriv_inds, **wv_kw)

from lisatools.diagnostic import *

inner_product_kwargs = dict(dt=dt, PSD="cornish_lisa_psd")

par = np.array([M, mu, p0, e0, theta, phi])
fish, dh = fisher(gen_wave, par, 1e-2, deriv_inds=deriv_inds, return_derivs=True, waveform_kwargs=wv_kw, inner_product_kwargs=inner_product_kwargs)

h = gen_wave(*par, **wv_kw)
print("snr", inner_product(h, h, **inner_product_kwargs)**(1/2))

print("overlap deriv",inner_product(fast_wave[0], dh[0], **inner_product_kwargs, normalize=True))
Gamma = inner_product(fast_wave[0].real, fast_wave[0].real, **inner_product_kwargs) + \
    inner_product(fast_wave[0].imag, fast_wave[0].imag, **inner_product_kwargs)
print(Gamma, fish)

# plt.plot(dh[0], '-', label='fish'); plt.plot(fast_wave[0], '--', label='app'); plt.legend(); plt.show()

# # numerical
# newpar = par.copy()
# newpar[0] += fish[0][0]**(-0.5)
# h_plus_dh_num = gen_wave(*newpar, **wv_kw)
# app = h + dh[0]*fish[0][0]**(-0.5)
# check2 = inner_product(app.real, h_plus_dh_num.real, **inner_product_kwargs, normalize=True)
# print("check", check2)

# # analytical
# newpar = par.copy()
# newpar[0] += Gamma**(-0.5)
# h_plus_dh_an = gen_wave(*newpar, **wv_kw)
# print(len(h), len(fast_wave[0]))

# app = h + fast_wave[0] * Gamma**(-0.5)
# check2 = inner_product(app.real, h_plus_dh_an.real, **inner_product_kwargs, normalize=True)
# print("check", check2)


###################################################

deriv_inds = [0, 1]
dim = len(deriv_inds)
fast_wave = fast(M, mu, p0, e0, theta, phi, delta_deriv=[1e-1, 1e-2], deriv_inds=deriv_inds, **wv_kw)

# cross corr
Gamma = np.array([inner_product(fast_wave[i].real, fast_wave[j].real, **inner_product_kwargs) + \
    inner_product(fast_wave[i].imag, fast_wave[j].imag, **inner_product_kwargs) for i in range(dim) for j in range(dim) if i>=j])
# Gamma = np.array([inner_product(fast_wave[i].real, fast_wave[j].real, **inner_product_kwargs) for i in range(dim) for j in range(dim) if i>=j])

size_X = dim
X = np.zeros((size_X,size_X))
X[np.triu_indices(X.shape[0], k = 0)] = Gamma
X = X + X.T - np.diag(np.diag(X))

X_my = X.copy()

# newpar = par.copy()
# newpar[0] += X[0,0]**(-0.5)
# newpar[1] += -2.0* X[0,1]/( X[0,0]**(0.5) * X[1,1])
# h_plus_dh = gen_wave(*newpar, **wv_kw)
# delta_h = h - h_plus_dh
# print(inner_product(delta_h, delta_h, **inner_product_kwargs))

deriv_inds = [0, 1]
fish, dh = fisher(gen_wave, par, 1e-2, deriv_inds=deriv_inds, return_derivs=True, waveform_kwargs=wv_kw, inner_product_kwargs=inner_product_kwargs)

X_num = fish.copy()

# X = fish
# newpar = par.copy()
# newpar[0] += X[0,0]**(-0.5)
# newpar[1] += -2.0* X[0,1]/( X[0,0]**(0.5) * X[1,1])
# h_plus_dh = gen_wave(*newpar, **wv_kw)
# delta_h = h - h_plus_dh
# print(inner_product(delta_h, delta_h, **inner_product_kwargs))

######################################################################################################

def loglike(pp):
    htmp = gen_wave(*pp, **wv_kw)
    d_m_h = h - htmp
    return inner_product(d_m_h, d_m_h, **inner_product_kwargs)

diff_ll = []
diff_ll_num = []
ll_vec = []
for i in range(100):
    x0 = np.random.multivariate_normal(par[:2], np.linalg.inv(X_my)*1000.0)
    newpar = par.copy()
    newpar[:2] = x0
    delta_par = par[:2] - x0
    check_sigma = 1/2 * np.dot(delta_par.T, np.dot(X_my,delta_par))
    check_sigma_num = 1/2 * np.dot(delta_par.T, np.dot(X_num,delta_par))

    ll = loglike(newpar)
    
    print(ll , check_sigma, check_sigma_num, "\t", 1-check_sigma/ll )
    
    ll_vec.append(ll)
    diff_ll.append(check_sigma - ll)
    diff_ll_num.append(check_sigma_num - ll)

plt.figure(); plt.plot(ll_vec, diff_ll, '.'); plt.plot(ll_vec, diff_ll_num, '.'); plt.show()
