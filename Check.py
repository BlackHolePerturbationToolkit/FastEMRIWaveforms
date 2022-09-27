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

gen_wave = FastSchwarzschildEccentricFlux()#GenerateEMRIWaveform("FastSchwarzschildEccentricFlux", return_list=False)

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
sum_kwargs = {}

fast = InfoMatrixFastSchwarzschildEccentricFlux(
    inspiral_kwargs=inspiral_kwargs,
    amplitude_kwargs=amplitude_kwargs,
    Ylm_kwargs=Ylm_kwargs,
    sum_kwargs=sum_kwargs,
    use_gpu=gpu_available,
    normalize_amps=False
)

# parameters
T = 0.5  # years
dt = 15.0  # seconds
M = 1e6
mu = 5e1
p0 = 10.0
e0 = 0.65
theta = np.pi / 3  # polar viewing angle
phi = np.pi / 4  # azimuthal viewing angle
dist = 1.0  # distance
batch_size = int(1e4)

wv_kw = dict(T=T, dt=dt, mode_selection=[(2,2,0)])
fast_wave = fast(M, mu, p0, e0, theta, phi, delta_deriv=[1e-2], deriv_inds=[0], **wv_kw)

from lisatools.diagnostic import *

inner_product_kwargs = dict(dt=dt, PSD="cornish_lisa_psd")

par = np.array([M, mu, p0, e0, theta, phi])
fish, dh = fisher(gen_wave, par, 1e-2, deriv_inds=[0], return_derivs=True, waveform_kwargs=wv_kw, inner_product_kwargs=inner_product_kwargs)


print(inner_product(fast_wave[0], dh[0], **inner_product_kwargs, normalize=True))
Gamma = inner_product(fast_wave[0], fast_wave[0], **inner_product_kwargs)
print(Gamma, fish)
fish = inner_product(dh[0], dh[0], **inner_product_kwargs)
print(Gamma, fish)

plt.plot(dh[0], '-', label='fish'); plt.plot(fast_wave[0], '--', label='app'); plt.legend(); plt.show()
