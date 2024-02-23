import unittest
import numpy as np
import warnings

from few.trajectory.inspiral import EMRIInspiral
from few.utils.ylm import GetYlms
from few.utils.modeselector import ModeSelector
from few.summation.interpolatedmodesum import CubicSplineInterpolant
from few.amplitude.romannet import RomanAmplitude
try:
    import cupy as xp

    gpu_available = True

except (ModuleNotFoundError, ImportError) as e:
    import numpy as xp

    warnings.warn(
        "CuPy is not installed or a gpu is not available. If trying to run on a gpu, please install CuPy."
    )
    gpu_available = False


class ModeSelectorTest(unittest.TestCase):
        
    # first, lets get amplitudes for a trajectory
    traj = EMRIInspiral(func="SchwarzEccFlux")
    ylm_gen = GetYlms(assume_positive_m=True, use_gpu=False)

    # parameters
    M = 1e5
    mu = 1e1
    p0 = 10.0
    e0 = 0.3
    theta = np.pi/3.
    phi = np.pi/2.

    t, p, e, x, Phi_phi, Phi_theta, Phi_r = traj(M, mu, 0.0, p0, e0, 1.0,T=10.0)

    # get amplitudes along trajectory
    amp = RomanAmplitude()

    teuk_modes = amp(p, e)

    # get ylms
    ylms = ylm_gen(amp.unique_l, amp.unique_m, theta, phi).copy()[amp.inverse_lm]

    # select modes

    mode_selector = ModeSelector(amp.m0mask, use_gpu=False)

    eps = 1e-2  # tolerance on mode contribution to total power

    modeinds = [amp.l_arr, amp.m_arr, amp.n_arr]
    (teuk_modes_in, ylms_in, ls, ms, ns) = mode_selector(teuk_modes, ylms, modeinds, eps=eps)

    print("We reduced the mode content from {} modes to {} modes.".format(teuk_modes.shape[1], teuk_modes_in.shape[1]))
    ms_orig = ms
    ns_orig = ns
    
    # produce sensitivity function

    noise = np.genfromtxt("examples/files/LPA.txt", names=True)
    f, PSD = (
        np.asarray(noise["f"], dtype=np.float64),
        np.asarray(noise["ASD"], dtype=np.float64) ** 2,
    )

    sens_fn = CubicSplineInterpolant(f, PSD, use_gpu=False)

    # select modes with noise weighting

    # provide sensitivity function kwarg
    mode_selector_noise_weighted = ModeSelector(amp.m0mask, sensitivity_fn=sens_fn, use_gpu=False)


    # Schwarzschild
    a = 0.0
    Y = np.zeros_like(p)  # equatorial / cos iota
    fund_freq_args = (M, a, p , e, Y, t)

    modeinds = [amp.l_arr, amp.m_arr, amp.n_arr]
    (teuk_modes_in, ylms_in, ls, ms, ns) = mode_selector_noise_weighted(teuk_modes, ylms, modeinds,
                                                                        fund_freq_args=fund_freq_args, eps=eps)

    print("We reduced the mode content from {} modes to {} modes when using noise-weighting.".format(teuk_modes.shape[1], teuk_modes_in.shape[1]))
    # import matplotlib.pyplot as plt
    # plt.figure(); plt.title(f'Mode selection comparison \n M={M:.1e},mu={mu:.1e},e0={e0},p0={p0},eps={eps:.2e}'); 
    # plt.plot(ms,ns,'o',label=f'new select, N={len(ms)}', ms=10); plt.plot(ms_orig,ns_orig,'P',label=f'old select, N={len(ms_orig)}', ms=5); plt.legend(); plt.ylabel('n'); plt.xlabel('m'); plt.show()
    