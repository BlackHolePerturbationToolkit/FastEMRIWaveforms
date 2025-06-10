# Interpolated summation of modes in python for the FastEMRIWaveforms Package

from typing import Optional

import numpy as np


def get_convolution(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Calculate the convolution of two arrays.

    arguments:
        a: First array to convolve.
        b: First array to convolve.

    returns:
        convolution of the two arrays.

    """

    # Check a and b are the same type
    if type(a) is not type(b):
        raise ValueError(
            "One array is cupy and one array is numpy. Need to be the same."
        )

    # Handle both np and cp cases
    if isinstance(a, np.ndarray):
        xp = np
        from scipy.signal import convolve
    else:
        from few import has_backend

        assert has_backend("cuda")
        import cupy as xp
        from cupyx.scipy.signal import convolve

    return convolve(xp.hstack((a[1:], a)), b, mode="valid") / len(b)


def get_fft_td_windowed(
    signal: list, window: np.ndarray, dt: float
) -> list[np.ndarray]:
    """
    Calculate the Fast Fourier Transform of a windowed time domain signal.

    arguments:
        signal: A length-2 list containing the signals plus and cross polarizations.
        window: Array to be applied in time domain to each signal.
        dt: Time sampling interval of the signal and window.

    returns:
        Fast Fourier Transform of the windowed time domain signals.

    """
    # Check a and b are the same type
    if type(signal[0]) is not type(window):
        raise ValueError(
            "One array is cupy and one array is numpy. Need to be the same."
        )

    # Handle the case where arrays are cupy.ndarray
    if isinstance(window, np.ndarray):
        xp = np
    else:
        from few import has_backend

        assert has_backend("cuda")
        import cupy as xp

    # Compute the FFT of the windowed signals
    fft_td_wave_p = xp.fft.fftshift(xp.fft.fft(signal[0] * window)) * dt
    fft_td_wave_c = xp.fft.fftshift(xp.fft.fft(signal[1] * window)) * dt
    return [fft_td_wave_p, fft_td_wave_c]


def get_fd_windowed(
    signal: list, window: Optional[bool] = None, window_in_fd: bool = False
) -> list[np.ndarray]:
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
    if type(signal[0]) is not type(window):
        raise ValueError(
            "One array is cupy and one array is numpy. Need to be the same."
        )

    if isinstance(window, np.ndarray):
        xp = np
    else:
        from few import has_backend

        assert has_backend("cuda")
        import cupy as xp

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
        non_zero_mask: Optional[np.ndarray] = None,
        window: Optional[np.ndarray] = None,
        window_in_fd: Optional[bool] = False,
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
        non_zero_mask: Optional[np.ndarray] = None,
        window: Optional[np.ndarray] = None,
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
