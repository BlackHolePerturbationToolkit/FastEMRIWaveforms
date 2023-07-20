import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

try:
    import cupy as cp
    from cupyx.scipy.signal import convolve as cupy_convolve
except:
    from scipy.signal import convolve as numpy_convolve
    import numpy as np


def get_convolution(a, b):
    # determine if using gpu or cpu based on input arrays
    try:
        if isinstance(a, cp.ndarray) or isinstance(b, cp.ndarray):
            if isinstance(a, cp.ndarray) and isinstance(b, cp.ndarray):
                use_gpu = True
            else:
                raise ValueError(
                    "One array is cupy and one array is numpy. Need to be the same."
                )

    # if cupy did not import
    except NameError:
        use_gpu = False

    xp = cp if use_gpu else np
    convolve = cupy_convolve if use_gpu else numpy_convolve

    # convolve two signals
    return convolve(xp.hstack((a[1:], a)), b, mode="valid") / len(b)


def get_fft_td_windowed(signal, window, dt):
    fft_td_wave_p = xp.fft.fftshift(xp.fft.fft(signal[0] * window)) * dt
    fft_td_wave_c = xp.fft.fftshift(xp.fft.fft(signal[1] * window)) * dt
    return [fft_td_wave_p, fft_td_wave_c]


def get_fd_windowed(signal, window, window_in_fd=False):
    if window is None:
        transf_fd_0 = signal[0]
        transf_fd_1 = signal[1]
    else:
        # # fft convolution
        # transf_fd_0 = xp.fft.fftshift(xp.fft.fft(xp.fft.ifft( xp.fft.ifftshift( signal[0] ) ) * window))
        # transf_fd_1 = xp.fft.fftshift(xp.fft.fft(xp.fft.ifft( xp.fft.ifftshift( signal[1] ) ) * window))

        # standard convolution
        if window_in_fd:
            fft_window = window.copy()
        else:
            fft_window = xp.fft.fft(window)
        transf_fd_0 = get_convolution(xp.conj(fft_window), signal[0])
        transf_fd_1 = get_convolution(xp.conj(fft_window), signal[1])

        # # test check convolution
        # sum_0 = xp.sum(xp.abs(transf_fd_0)**2)
        # yo = get_convolution( xp.conj(fft_window) , signal[0] )
        # sum_yo = xp.sum(xp.abs(yo)**2)
        # xp.dot(xp.conj(yo) , transf_fd_0 ) /xp.sqrt(sum_0 * sum_yo)

    return [transf_fd_0, transf_fd_1]


class GetFDWaveformFromFD:
    def __init__(
        self,
        waveform_generator,
        positive_frequency_mask,
        dt,
        non_zero_mask=None,
        window=None,
        window_in_fd=False,
    ):
        self.waveform_generator = waveform_generator
        self.positive_frequency_mask = positive_frequency_mask
        self.non_zero_mask = non_zero_mask
        self.window = window
        self.window_in_fd = window_in_fd

    def __call__(self, *args, **kwargs):
        data_channels_td = self.waveform_generator(*args, **kwargs)
        list_p_c = get_fd_windowed(
            data_channels_td, self.window, window_in_fd=self.window_in_fd
        )
        ch1 = list_p_c[0][self.positive_frequency_mask]
        ch2 = list_p_c[1][self.positive_frequency_mask]
        if self.non_zero_mask is not None:
            ch1[~self.non_zero_mask] = complex(0.0)
            ch2[~self.non_zero_mask] = complex(0.0)
        return [ch1, ch2]


# conversion
class GetFDWaveformFromTD:
    def __init__(
        self,
        waveform_generator,
        positive_frequency_mask,
        dt,
        non_zero_mask=None,
        window=None,
    ):
        self.waveform_generator = waveform_generator
        self.positive_frequency_mask = positive_frequency_mask
        self.dt = dt
        self.non_zero_mask = non_zero_mask
        if window is None:
            self.window = np.ones_like(self.positive_frequency_mask)
        else:
            self.window = window

    def __call__(self, *args, **kwargs):
        data_channels_td = self.waveform_generator(*args, **kwargs)
        list_p_c = get_fft_td_windowed(data_channels_td, self.window, self.dt)
        fft_td_wave_p = list_p_c[0][self.positive_frequency_mask]
        fft_td_wave_c = list_p_c[1][self.positive_frequency_mask]
        if self.non_zero_mask is not None:
            fft_td_wave_p[~self.non_zero_mask] = complex(0.0)
            fft_td_wave_c[~self.non_zero_mask] = complex(0.0)
        return [fft_td_wave_p, fft_td_wave_c]
