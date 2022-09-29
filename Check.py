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
from scipy import special, signal
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
dt = 10.0  # seconds
M = 1e6
mu = 3e1
p0 = 11.0
e0 = 0.321
theta = np.pi / 3  # polar viewing angle
phi = np.pi / 4  # azimuthal viewing angle
dist = 1.0  # distance
batch_size = int(1e4)
par = np.array([M, mu, p0, e0, theta, phi])
wv_kw = dict(T=T, dt=dt, mode_selection=[(2,2,i) for i in range(-10,11)], dist=dist)

from lisatools.diagnostic import *

inner_product_kwargs = dict(dt=dt, PSD="cornish_lisa_psd")

h = gen_wave(*par, **wv_kw)
window =1.0#signal.tukey(len(h),0.01)

print("snr", inner_product(h*window, h*window, **inner_product_kwargs)**(1/2))
print("h end", h[-10:])
###################################################

deriv_inds = [0, 1, 3, 4]
list_ind = [0, 1, 2, 3]
delta_deriv = [5.0, 1e-2, 1e-2, 1e-2]

dim = len(deriv_inds)
fast_wave = fast(M, mu, p0, e0, theta, phi, delta_deriv=delta_deriv, deriv_inds=deriv_inds, **wv_kw)

# cross corr
X = np.array([[inner_product(fast_wave[i].real*window, fast_wave[j].real*window, **inner_product_kwargs) + \
    inner_product(fast_wave[i].imag*window, fast_wave[j].imag*window, **inner_product_kwargs) for i in range(dim)] for j in range(dim)])

X_my = X.copy()
print(X_my)

print("zero check",[inner_product(h,fast_wave[i], **inner_product_kwargs) for i in range(4)] )
######################################################################################################

def loglike(pp):
    htmp = gen_wave(*pp, **wv_kw)
    d_m_h = h - htmp
    d_m_h = [d_m_h.real, d_m_h.imag]
    return inner_product(d_m_h, d_m_h, **inner_product_kwargs)

diff_ll = []
ll_vec = []

from scipy import linalg
inv_X = np.linalg.inv(X_my)
print(np.linalg.cond(X_my) )

w,v = np.linalg.eig(X_my)
print(v,w)
for i in range(dim):
    v_tilde = v[:,i] * w[i]**(-0.5)
    check_sigma = np.dot(v_tilde.T, np.dot(X_my,v_tilde))
    newpar = par.copy()
    newpar[list_ind] = par[list_ind] + v_tilde
    print(v_tilde)
    ll = loglike(newpar)
    print(ll , check_sigma)

for i in range(100):
    delta_par = np.random.multivariate_normal(np.zeros(dim) , inv_X*100.0)

    newpar = par.copy()
    newpar[list_ind] = par[list_ind] + delta_par
    check_sigma = np.dot(delta_par.T, np.dot(X_my,delta_par))
    
    ll = loglike(newpar)
    
    print(ll , check_sigma, "\t", 1-check_sigma/ll )
    
    ll_vec.append(ll)
    diff_ll.append(check_sigma - ll)

plt.figure(); plt.plot(ll_vec, diff_ll, '.'); plt.ylabel(r'$\Delta \ln p$'); plt.xlabel(r'$ \ln p$');  plt.show()
