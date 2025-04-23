import gc
import pickle

import numpy as np

from few.tests.base import FewBackendTest, tagged_test
from few.utils.constants import YRSID_SI
from few.waveform import FastKerrEccentricEquatorialFlux, FastSchwarzschildEccentricFlux


class WaveformTest(FewBackendTest):
    @classmethod
    def name(self) -> str:
        return "FD"

    @classmethod
    def parallel_class(self):
        return FastSchwarzschildEccentricFlux

    @tagged_test(slow=True)
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
            "include_minus_m": False  # if we include positive m, it will generate negative m for all m>0
        }

        # keyword arguments for summation generator (InterpolatedModeSum)
        sum_kwargs = dict(pad_output=True, output_type="fd")

        fast = FastSchwarzschildEccentricFlux(
            inspiral_kwargs=inspiral_kwargs,
            amplitude_kwargs=amplitude_kwargs,
            Ylm_kwargs=Ylm_kwargs,
            sum_kwargs=sum_kwargs,
            force_backend=self.backend,
        )

        check_pickle = pickle.dumps(fast)
        extracted_gen = pickle.loads(check_pickle)

        # parameters
        T = 1.0  # years
        dt = 10.0  # seconds
        m1 = 1000000.0
        m2 = 50.0
        p0 = 12.510272236947417
        e0 = 0.4
        theta = np.pi / 3  # polar viewing angle
        phi = np.pi / 4  # azimuthal viewing angle
        dist = 1.0  # distance

        _N = int(T * YRSID_SI / dt)

        _fast_wave = extracted_gen(
            m1, m2, p0, e0, theta, phi, dist=dist, T=T, dt=dt, eps=1e-3
        )

    @tagged_test(slow=True)
    def test_fast_and_slow_schwarzschild(self):
        # parameters
        T = 1.0  # years
        dt = 10.0  # seconds
        m1 = 1000000.0
        m2 = 50.0
        p0 = 12.510272236947417
        e0 = 0.4
        theta = np.pi / 3  # polar viewing angle
        phi = np.pi / 4  # azimuthal viewing angle
        dist = 1.0  # distance

        # setup td
        generator = FastSchwarzschildEccentricFlux
        slow = generator(
            sum_kwargs=dict(pad_output=True),
            force_backend=self.backend,
        )
        slow_wave = slow(
            m1,
            m2,
            p0,
            e0,
            theta,
            phi,
            dist,
            T=T,
            dt=dt,
        )
        del slow  # Free up memory
        gc.collect()

        # make sure frequencies will be equivalent
        xp = self.backend.xp
        f_in = xp.array(np.linspace(-1 / (2 * dt), +1 / (2 * dt), num=len(slow_wave)))
        kwargs = dict(f_arr=f_in)

        # Build Fast
        fast = generator(
            sum_kwargs=dict(pad_output=True, output_type="fd"),
            force_backend=self.backend,
        )
        fast_wave = fast(
            m1, m2, p0, e0, theta, phi, dist=dist, T=T, dt=dt, eps=1e-3, **kwargs
        )
        # process FD
        freq = fast.create_waveform.frequency
        del fast  # Free up memory

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

    @tagged_test(slow=True)
    def test_fast_and_slow_kerr(self):
        # parameters
        T = 1.0  # years
        dt = 10.0  # seconds
        m1 = 1000000.0
        m2 = 50.0
        a = 0.5
        p0 = 12.510272236947417
        e0 = 0.4
        theta = np.pi / 3  # polar viewing angle
        phi = np.pi / 4  # azimuthal viewing angle
        dist = 1.0  # distance

        generator = FastKerrEccentricEquatorialFlux

        # setup td
        slow = generator(
            sum_kwargs=dict(pad_output=True),
            force_backend=self.backend,
        )

        slow_wave = slow(
            m1,
            m2,
            a,
            p0,
            e0,
            1.0,
            theta,
            phi,
            dist,
            T=T,
            dt=dt,
        )
        del slow  # Free up memory
        gc.collect()  # and recover it

        # make sure frequencies will be equivalent
        xp = self.backend.xp
        f_in = xp.array(np.linspace(-1 / (2 * dt), +1 / (2 * dt), num=len(slow_wave)))
        kwargs = dict(f_arr=f_in)

        fast = generator(
            sum_kwargs=dict(pad_output=True, output_type="fd"),
            force_backend=self.backend,
        )
        fast_wave = fast(
            m1,
            m2,
            a,
            p0,
            e0,
            1.0,
            theta,
            phi,
            dist=dist,
            T=T,
            dt=dt,
            eps=1e-3,
            **kwargs,
        )

        # process FD
        freq = fast.create_waveform.frequency
        del fast  # Free up memory

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
