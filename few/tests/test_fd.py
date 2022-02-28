import unittest
import numpy as np
import warnings

from few.trajectory.inspiral import EMRIInspiral
from few.amplitude.romannet import RomanAmplitude
from few.amplitude.interp2dcubicspline import Interp2DAmplitude
from few.waveform import FastSchwarzschildEccentricFlux, SlowSchwarzschildEccentricFlux
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


class WaveformTest(unittest.TestCase):
    def test_fast_and_slow(self):

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

        fast = FastSchwarzschildEccentricFlux(
            inspiral_kwargs=inspiral_kwargs,
            amplitude_kwargs=amplitude_kwargs,
            Ylm_kwargs=Ylm_kwargs,
            sum_kwargs=sum_kwargs,
            use_gpu=gpu_available,
        )

        # setup td
        sum_kwargs = dict(pad_output=True)

        slow = FastSchwarzschildEccentricFlux(
            inspiral_kwargs=inspiral_kwargs,
            amplitude_kwargs=amplitude_kwargs,
            Ylm_kwargs=Ylm_kwargs,
            sum_kwargs=sum_kwargs,
            use_gpu=gpu_available,
        )

        # parameters
        T = 1.0  # years
        dt = 11.0  # seconds
        M = 1e6
        mu = 1e1
        p0 = 8.0
        e0 = 0.2
        theta = np.pi / 3  # polar viewing angle
        phi = np.pi / 4  # azimuthal viewing angle
        dist = 1.0  # distance
        batch_size = int(1e4)

        slow_wave = slow(
            M, mu, p0, e0, theta, phi, dist, T=T, dt=dt, #mode_selection=[(2,2,0)]
        )

        # N = int(T*365*3600*24/dt)+1
        f_in = xp.array(np.linspace(-1 / (2 * dt), +1 / (2 * dt), num= len(slow_wave) ))
        N = len(f_in)
        kwargs = dict(f_arr=f_in)

        fast_wave = fast(
            M, mu, p0, e0, theta, phi, dist, T=T, dt=dt, **kwargs, #mode_selection=[(2,2,0)]
            )

        # process FD
        fd_sig = -xp.flip(fast_wave)

        ind =int(( len(fd_sig) - 1 ) / 2 + 1)

        fft_sig_r = xp.real(fd_sig + xp.flip(fd_sig) )/2.0 + 1j * xp.imag(fd_sig - xp.flip(fd_sig))/2.0
        fft_sig_i = -xp.imag(fd_sig + xp.flip(fd_sig) )/2.0 + 1j * xp.real(fd_sig - xp.flip(fd_sig))/2.0

        # take fft of TD
        freq = f_in[int(( N - 1 ) / 2 + 1):].get()
        h_td = xp.asarray(slow_wave)

        h_td_real = xp.real(h_td)
        h_td_imag = -xp.imag(h_td)
        time_series_1_fft = xp.fft.fftshift(xp.fft.fft(h_td_real))[int(( N - 1 ) / 2 + 1):] * dt #- 1j * xp.fft.fftshift(xp.fft.fft(h_td_imag))[int(( N - 1 ) / 2 + 1):] * dt
        time_series_2_fft = fft_sig_r[ind:]
        
        # make sure they have equal length
        self.assertAlmostEqual(len(time_series_1_fft), len(time_series_2_fft))

        # overlap
        ac = xp.dot(time_series_1_fft.conj(), time_series_2_fft) / xp.sqrt(
            xp.dot(time_series_1_fft.conj(), time_series_1_fft)
            * xp.dot(time_series_2_fft.conj(), time_series_2_fft)
        )
        
        if gpu_available:
            result = ac.item().real

        # import matplotlib.pyplot as plt
        # plt.figure(); plt.semilogx(freq, xp.real(time_series_1_fft).get(), alpha=0.5); plt.plot(freq, xp.real(time_series_2_fft).get(), '--', alpha=0.5); plt.xlim([3e-3,3.01e-3]); plt.savefig('test_real') 
        # plt.figure(); plt.semilogx(freq, xp.imag(time_series_1_fft).get(), alpha=0.5); plt.plot(freq, xp.imag(time_series_2_fft).get(), '--', alpha=0.5); plt.xlim([3e-3,3.01e-3]);plt.savefig('test_imag') 

        self.assertLess(1-result, 1e-2)
