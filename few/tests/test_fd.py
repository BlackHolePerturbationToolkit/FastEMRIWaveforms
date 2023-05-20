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
from few.waveform import GenerateEMRIWaveform


try:
    import cupy as xp

    gpu_available = True

except (ModuleNotFoundError, ImportError) as e:
    import numpy as xp

    warnings.warn(
        "CuPy is not installed or a gpu is not available. If trying to run on a gpu, please install CuPy."
    )
    gpu_available = False


few_gen = GenerateEMRIWaveform(
    "FastSchwarzschildEccentricFlux", 
    sum_kwargs=dict(pad_output=True, output_type="fd", odd_len=True),
    use_gpu=gpu_available,
    return_list=False,
)

few_gen_list = GenerateEMRIWaveform(
    "FastSchwarzschildEccentricFlux", 
    sum_kwargs=dict(pad_output=True, output_type="fd", odd_len=True),
    use_gpu=gpu_available,
    return_list=True,
)

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
        dt = 10.0  # seconds
        M = 1000000.0
        mu = 50.0
        p0 = 12.510272236947417
        e0 = 0.4
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
            M, mu, p0, e0, theta, phi, dist, T=T, dt=dt, eps=1e-3, **kwargs
            )

        # process FD
        freq = fast.create_waveform.frequency
        mask = (freq>=0.0)

        # take fft of TD
        h_td = xp.asarray(slow_wave)
        h_td_real = xp.real(h_td)
        h_td_imag = -xp.imag(h_td)
        time_series_1_fft = xp.fft.fftshift(xp.fft.fft(h_td_real))[mask]
        
        # mask only positive frequencies
        time_series_2_fft = fast_wave[0,mask]
        
        # make sure they have equal length
        self.assertAlmostEqual(len(time_series_1_fft), len(time_series_2_fft))

        # overlap
        ac = xp.dot(time_series_1_fft.conj(), time_series_2_fft) / xp.sqrt(
            xp.dot(time_series_1_fft.conj(), time_series_1_fft)
            * xp.dot(time_series_2_fft.conj(), time_series_2_fft)
        )

        injection_in = np.array([1.16350676e+06, 1.27978018e+02, 0.00000000e+00, 1.39313102e+01, 5.87723275e-01, 0.00000000e+00, 1.00000085e+00, 2.38505897e+00, 5.95081752e+00, 2.50714543e+00, 2.82403338e+00, 4.33433216e+00, 0.00000000e+00, 2.73356978e+00])
        data_channels_fd = few_gen(*injection_in)
        sig_fd = few_gen_list(*injection_in)
        print("check 1 == ", xp.dot(xp.conj(sig_fd[0] - 1j * sig_fd[1]),data_channels_fd)/xp.dot(xp.conj(data_channels_fd),data_channels_fd) )

        # problematic point
        # 3697957.511659888 861.3377098262883 14.418959668893407 0.6707784770461537
        prob_point = xp.array([909080.3243424094, 39.53732872443626, 0.0, 13.902109123486886, 0.5590977383700271, 1.0, 57.88963690750407, 2.7464152838466274, 3.2109893163133503, 0.20280877216654694, 1.2513852793041993, 2.4942857598445087, 0.0, 3.003630047126699])
        prob_point = xp.array([1864440.3414742905, 10.690959453789679, 0.0, 12.510272236947417, 0.5495976916153483, 1.0, 57.88963690750407, 2.7464152838466274, 3.2109893163133503, 0.20280877216654694, 1.2513852793041993, 2.4942857598445087, 0.0, 3.003630047126699])
        if gpu_available:
            few_gen(*prob_point.get(),T=4.0,eps=1e-5,dt=3.0,f_arr=freq)
            print("works, freq=",freq)
            few_gen(*prob_point.get(),T=4.0,eps=1e-5,dt=3.0)
            print("works")
            few_gen(*prob_point.get(),T=4.0,eps=1e-5,dt=3.0)
            print("nope")
        else:
            few_gen(*prob_point,T=1.0,eps=1e-3,dt=6.0,f_arr=freq)
            print("works, freq=",freq)
            few_gen(*prob_point,T=4.0,eps=1e-3,dt=6.0)
            print("works")
            few_gen(*prob_point,T=1.0,eps=1e-3,dt=6.0)
            print("nope")

        # if gpu_available:
        result = ac.item().real
        
        # import matplotlib.pyplot as plt
        # plt.figure(); plt.semilogx(freq, xp.real(time_series_1_fft).get(), alpha=0.5); plt.plot(freq, xp.real(time_series_2_fft).get(), '--', alpha=0.5); plt.xlim([3e-3,3.01e-3]); plt.savefig('test_real') 
        # plt.figure(); plt.semilogx(freq, xp.imag(time_series_1_fft).get(), alpha=0.5); plt.plot(freq, xp.imag(time_series_2_fft).get(), '--', alpha=0.5); plt.xlim([3e-3,3.01e-3]);plt.savefig('test_imag') 
        print("mismatch", 1-result)
        self.assertLess(1-result, 1e-2)
