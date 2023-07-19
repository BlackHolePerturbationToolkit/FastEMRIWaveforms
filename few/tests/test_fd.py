import unittest
import pickle
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
    def test_pickle(self):
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

        check_pickle = pickle.dumps(fast)
        extracted_gen = pickle.loads(check_pickle)

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

        N = int(T * YRSID_SI / dt)

        fast_wave = extracted_gen(
            M, mu, p0, e0, theta, phi, dist=dist, T=T, dt=dt, eps=1e-3
        )

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
            M,
            mu,
            p0,
            e0,
            theta,
            phi,
            dist,
            T=T,
            dt=dt,  # mode_selection=[(2,2,0)]
        )

        # make sure frequencies will be equivalent
        f_in = xp.array(np.linspace(-1 / (2 * dt), +1 / (2 * dt), num=len(slow_wave)))
        N = len(f_in)
        kwargs = dict(f_arr=f_in)

        fast_wave = fast(
            M, mu, p0, e0, theta, phi, dist=dist, T=T, dt=dt, eps=1e-3, **kwargs
        )

        # process FD
        freq = fast.create_waveform.frequency
        mask = freq >= 0.0

        # take fft of TD
        h_td = xp.asarray(slow_wave)
        h_td_real = xp.real(h_td)
        h_td_imag = -xp.imag(h_td)
        time_series_1_fft = xp.fft.fftshift(xp.fft.fft(h_td_real))[mask]

        # mask only positive frequencies
        time_series_2_fft = fast_wave[0, mask]

        # make sure they have equal length
        self.assertAlmostEqual(len(time_series_1_fft), len(time_series_2_fft))

        # overlap
        ac = xp.dot(time_series_1_fft.conj(), time_series_2_fft) / xp.sqrt(
            xp.dot(time_series_1_fft.conj(), time_series_1_fft)
            * xp.dot(time_series_2_fft.conj(), time_series_2_fft)
        )

        injection_test = np.array(
            [
                1864440.3414742905,
                10.690959453789679,
                0.0,
                12.510272236947417,
                0.5495976916153483,
                1.0,
                57.88963690750407,
                2.7464152838466274,
                3.2109893163133503,
                0.20280877216654694,
                1.2513852793041993,
                2.4942857598445087,
                0.0,
                3.003630047126699,
            ]
        )

        # test some different configurations
        few_gen_list(*injection_test, T=1.0, eps=1e-3, dt=6.0)
        few_gen_list(*injection_test, T=1.0, eps=1e-3, dt=6.0, f_arr=freq)
        few_gen_list(*injection_test, T=4.0, eps=1e-3, dt=6.0)
        few_gen_list(*injection_test, T=1.0, eps=1e-3, dt=3.0)
        few_gen_list(*injection_test, T=1.0, eps=1e-2, dt=6.0)

        result = ac.item().real
        self.assertLess(1 - result, 1e-2)
