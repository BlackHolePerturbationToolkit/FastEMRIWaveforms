import unittest
import pickle
import numpy as np

from few.waveform import FastSchwarzschildEccentricFlux, FastKerrEccentricEquatorialFlux
from few.waveform import GenerateEMRIWaveform
from few.utils.constants import YRSID_SI

from few.utils.globals import get_logger, get_first_backend

few_logger = get_logger()

best_backend = get_first_backend(FastSchwarzschildEccentricFlux.supported_backends())
xp = best_backend.xp
few_logger.warning("FD Test is running with backend {}".format(best_backend.name))


class WaveformTest(unittest.TestCase):
    def test_pickle(self):
        # keyword arguments for inspiral generator (RunSchwarzEccFluxInspiral)
        inspiral_kwargs = {
            "DENSE_STEPPING": 0,  # we want a sparsely sampled trajectory
            "buffer_length": int(
                1e3
            ),  # all of the trajectories will be well under len = 1000
        }

        # keyword arguments for inspiral generator (RomanAmplitude)
        amplitude_kwargs = {
            "buffer_length": int(
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
            force_backend=best_backend,
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

        _N = int(T * YRSID_SI / dt)

        _fast_wave = extracted_gen(
            M, mu, p0, e0, theta, phi, dist=dist, T=T, dt=dt, eps=1e-3
        )

    def test_fast_and_slow_schwarzschild(self):
        # keyword arguments for summation generator (InterpolatedModeSum)
        sum_kwargs = dict(pad_output=True, output_type="fd")

        generator = FastSchwarzschildEccentricFlux
        fast = generator(
            sum_kwargs=sum_kwargs,
            force_backend=best_backend,
        )

        # setup td
        sum_kwargs = dict(pad_output=True)

        slow = generator(
            sum_kwargs=sum_kwargs,
            force_backend=best_backend,
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

        slow_wave = slow(M,mu,p0,e0,theta,phi,dist,T=T,dt=dt,)

        # make sure frequencies will be equivalent
        f_in = xp.array(np.linspace(-1 / (2 * dt), +1 / (2 * dt), num=len(slow_wave)))
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
        _h_td_imag = -xp.imag(h_td)
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

        result = ac.item().real
        self.assertLess(1 - result, 1e-2)

    def test_fast_and_slow_kerr(self):
        # keyword arguments for summation generator (InterpolatedModeSum)
        sum_kwargs = dict(pad_output=True, output_type="fd")

        generator = FastKerrEccentricEquatorialFlux
        fast = generator(
            sum_kwargs=sum_kwargs,
            force_backend=best_backend,
        )

        # setup td
        sum_kwargs = dict(pad_output=True)

        slow = generator(
            sum_kwargs=sum_kwargs,
            force_backend=best_backend,
        )

        # parameters
        T = 1.0  # years
        dt = 10.0  # seconds
        M = 1000000.0
        mu = 50.0
        a = 0.5
        p0 = 12.510272236947417
        e0 = 0.4
        theta = np.pi / 3  # polar viewing angle
        phi = np.pi / 4  # azimuthal viewing angle
        dist = 1.0  # distance

        slow_wave = slow(M,mu,a,p0,e0,1.0,theta,phi,dist,T=T,dt=dt,)

        # make sure frequencies will be equivalent
        f_in = xp.array(np.linspace(-1 / (2 * dt), +1 / (2 * dt), num=len(slow_wave)))
        kwargs = dict(f_arr=f_in)

        fast_wave = fast(M, mu, a,  p0, e0, 1.0, theta, phi, dist=dist, T=T, dt=dt, eps=1e-3, **kwargs)

        # process FD
        freq = fast.create_waveform.frequency
        mask = freq >= 0.0

        # take fft of TD
        h_td = xp.asarray(slow_wave)
        h_td_real = xp.real(h_td)
        _h_td_imag = -xp.imag(h_td)
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

        result = ac.item().real
        self.assertLess(1 - result, 1e-2)
