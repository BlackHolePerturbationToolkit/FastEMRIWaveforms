# Interpolated summation of modes in python for the FastEMRIWaveforms Package

# Copyright (C) 2023 Michael L. Katz, Lorenzo Speri
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import numpy as np

try:
    import cupy as cp
    from cupyx.scipy.signal import convolve as cupy_convolve
except:
    from scipy.signal import convolve as numpy_convolve
    import numpy as np

from typing import Optional

def get_convolution(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Calculate the convolution of two arrays.

    arguments:
        a: First array to convolve.
        b: First array to convolve.

    returns:
        convolution of the two arrays.

    """
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


def get_fft_td_windowed(signal: list, window: np.ndarray, dt: float) -> list[np.ndarray]:
    """
    Calculate the Fast Fourier Transform of a windowed time domain signal.

    arguments:
        signal: A length-2 list containing the signals plus and cross polarizations.
        window: Array to be applied in time domain to each signal.
        dt: Time sampling interval of the signal and window.

    returns:
        Fast Fourier Transform of the windowed time domain signals.

    """
    try:
        if isinstance(signal[0], cp.ndarray) or isinstance(window, cp.ndarray):
            if isinstance(signal[0], cp.ndarray) and isinstance(window, cp.ndarray):
                use_gpu = True
            else:
                raise ValueError(
                    "One array is cupy and one array is numpy. Need to be the same."
                )

    # if cupy did not import
    except NameError:
        use_gpu = False

    xp = cp if use_gpu else np

    fft_td_wave_p = xp.fft.fftshift(xp.fft.fft(signal[0] * window)) * dt
    fft_td_wave_c = xp.fft.fftshift(xp.fft.fft(signal[1] * window)) * dt
    return [fft_td_wave_p, fft_td_wave_c]


def get_fd_windowed(signal: list, window: Optional[bool]=None, window_in_fd: bool=False) -> list[np.ndarray]:
    """
    Calculate the convolution of a frequency domain signal with a window in time domain.

    arguments:
        signal: A length-2 list containing the signals plus and cross polarizations in frequency domain.
        window: Array of the time domain window. If ``None``, do not apply window.
            This is added for flexibility. Default is ``None``.
        window_in_fd: If ``True``, ``window`` is given in the frequency domain.
            If ``False``, window is given in the time domain. Default is ``False``.

    returns:
        convolution of a frequency domain signal with a time domain window.

    """
    try:
        if isinstance(signal[0], cp.ndarray) or isinstance(window, cp.ndarray):
            if isinstance(signal[0], cp.ndarray) and isinstance(window, cp.ndarray):
                use_gpu = True
            else:
                raise ValueError(
                    "One array is cupy and one array is numpy. Need to be the same."
                )

    # if cupy did not import
    except NameError:
        use_gpu = False

    xp = cp if use_gpu else np

    # apply no window
    if window is None:
        transf_fd_0 = signal[0]
        transf_fd_1 = signal[1]
    else:
        # standard convolution
        if window_in_fd:
            fft_window = window.copy()
        else:
            fft_window = xp.fft.fft(window)
        transf_fd_0 = get_convolution(xp.conj(fft_window), signal[0])
        transf_fd_1 = get_convolution(xp.conj(fft_window), signal[1])

    return [transf_fd_0, transf_fd_1]


class GetFDWaveformFromFD:
    """Generic frequency domain class

    This class allows to obtain the frequency domain signal given the frequency domain waveform class
    from the FEW package.

    Args:
        waveform_generator: FEW waveform class.
        positive_frequency_mask: boolean array to indicate where the frequencies are positive.
        dt: time sampling interval of the signal and window.
        non_zero_mask: boolean array to indicate where the waveform needs to be set to zero.
        window: Array of the time domain window. If ``None``, do not apply window.
            This is added for flexibility. Default is ``None``.
        window_in_fd: If ``True``, ``window`` is given in the frequency domain.
            If ``False``, window is given in the time domain. Default is ``False``.

    """

    def __init__(
        self,
        waveform_generator: object,
        positive_frequency_mask: np.ndarray,
        dt: float,
        non_zero_mask: Optional[np.ndarray]=None,
        window: Optional[np.ndarray]=None,
        window_in_fd: Optional[bool]=False,
    ):
        self.waveform_generator = waveform_generator
        self.positive_frequency_mask = positive_frequency_mask
        self.non_zero_mask = non_zero_mask
        self.window = window
        self.window_in_fd = window_in_fd

    def __call__(self, *args, **kwargs) -> list[np.ndarray]:
        """Run the waveform generator.

        args:
            *args: Arguments passed to waveform generator.
            **kwargs: Keyword arguments passed to waveform generator.

        returns:
            FD Waveform as [h+, hx]

        """
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
    """Generic time domain class

    This class allows to obtain the frequency domain signal given the time domain waveform class
    from the FEW package.

    Args:
        waveform_generator: FEW waveform class.
        positive_frequency_mask: boolean array to indicate where the frequencies are positive.
        dt: time sampling interval of the signal and window.
        non_zero_mask: boolean array to indicate where the waveform needs to be set to zero.
        window: Array of the time domain window. If ``None``, do not apply window.
            This is added for flexibility. Default is ``None``.

    """

    def __init__(
        self,
        waveform_generator: object,
        positive_frequency_mask: np.ndarray,
        dt: float,
        non_zero_mask: Optional[np.ndarray]=None,
        window: Optional[np.ndarray]=None,
    ):
        self.waveform_generator = waveform_generator
        self.positive_frequency_mask = positive_frequency_mask
        self.dt = dt
        self.non_zero_mask = non_zero_mask
        if window is None:
            self.window = np.ones_like(self.positive_frequency_mask)
        else:
            self.window = window

    def __call__(self, *args, **kwargs) -> list[np.ndarray]:
        """Run the waveform generator.

        args:
            *args: Arguments passed to waveform generator.
            **kwargs: Keyword arguments passed to waveform generator.

        returns:
            FD Waveform as [h+, hx] (fft from TD)

        """
        data_channels_td = self.waveform_generator(*args, **kwargs)
        list_p_c = get_fft_td_windowed(data_channels_td, self.window, self.dt)
        fft_td_wave_p = list_p_c[0][self.positive_frequency_mask]
        fft_td_wave_c = list_p_c[1][self.positive_frequency_mask]
        if self.non_zero_mask is not None:
            fft_td_wave_p[~self.non_zero_mask] = complex(0.0)
            fft_td_wave_c[~self.non_zero_mask] = complex(0.0)
        return [fft_td_wave_p, fft_td_wave_c]
