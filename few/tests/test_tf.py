import unittest
import numpy as np
import warnings

from few.waveform import FastSchwarzschildEccentricFlux, SlowSchwarzschildEccentricFlux
from few.utils.utility import get_overlap, get_mismatch

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
        sum_kwargs = {"output_type": "tf"}

        fast_tf = FastSchwarzschildEccentricFlux(
            inspiral_kwargs=inspiral_kwargs,
            amplitude_kwargs=amplitude_kwargs,
            Ylm_kwargs=Ylm_kwargs,
            sum_kwargs=sum_kwargs,
            use_gpu=gpu_available,
        )

        fast_td = FastSchwarzschildEccentricFlux(
            inspiral_kwargs=inspiral_kwargs,
            amplitude_kwargs=amplitude_kwargs,
            Ylm_kwargs=Ylm_kwargs,
            sum_kwargs={},
            use_gpu=gpu_available,
        )

        # parameters
        T = 0.1  # years
        dt = 10.0  # seconds
        M = 1e6
        mu = 1e1
        p0 = 8.0
        e0 = 0.2
        theta = np.pi / 3  # polar viewing angle
        phi = np.pi / 4  # azimuthal viewing angle
        dist = 1.0  # distance
        t_window = 21600.0

        wave_tf = fast_tf(
            M, mu, p0, e0, theta, phi, dist, T=T, dt=dt, t_window=t_window
        )

        wave_td = fast_td(
            M, mu, p0, e0, theta, phi, dist, T=T, dt=dt, t_window=t_window
        )

        hp = wave_td.real
        wind = 0
        hp_f = xp.fft.rfft(
            hp[
                wind
                * fast_tf.create_waveform.num_per_window : (wind + 1)
                * fast_tf.create_waveform.num_per_window
            ]
        )

        from scipy.signal import stft

        check = stft(
            hp,
            fs=1 / dt,
            window="boxcar",
            nperseg=fast_tf.create_waveform.num_per_window * 2,
            noverlap=None,
            nfft=None,
            detrend=False,
            return_onesided=True,
            boundary="zeros",
            padded=False,
            axis=-1,
        )

        import matplotlib.pyplot as plt

        fig, (ax0, ax1, ax2) = plt.subplots(1, 3)

        ax0.loglog(np.abs(hp_f))
        ax0.loglog(np.abs(wave_tf[0][wind]), "--")

        ax1.semilogx(hp_f.real)
        ax1.semilogx(wave_tf[0][wind].real, "--")

        ax2.semilogx(hp_f.imag)
        ax2.semilogx(wave_tf[0][wind].imag, "--")

        check_fft = np.zeros_like(wave_tf[0])
        for wind in range(wave_tf.shape[1]):
            hp_f = xp.fft.rfft(
                hp[
                    wind
                    * fast_tf.create_waveform.num_per_window : (wind + 1)
                    * fast_tf.create_waveform.num_per_window
                ]
            )
            check_fft[wind] = hp_f

            overlap = np.dot(hp_f.conj(), wave_tf[0][wind]) / np.sqrt(
                np.dot(hp_f.conj(), hp_f)
                * np.dot(wave_tf[0][wind].conj(), wave_tf[0][wind])
            )
            print(wind, overlap)

        check_fft = check_fft.flatten()
        tf_check = wave_tf[0].flatten()
        overlap = np.dot(check_fft.conj(), tf_check) / np.sqrt(
            np.dot(check_fft.conj(), check_fft) * np.dot(tf_check.conj(), tf_check)
        )
        print("all", overlap)
        # plt.show()
        breakpoint()
        # mm = get_mismatch(slow_wave, fast_wave, use_gpu=gpu_available)

        # self.assertLess(mm, 1e-4)
