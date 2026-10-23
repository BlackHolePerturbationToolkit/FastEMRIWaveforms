# Interpolated summation of modes in python for the FastEMRIWaveforms Package

"""Frequency-domain helpers for FEW waveforms (windowing, FFTs, convolutions).

- :class:`GetFDWaveformFromTD`: wraps a time-domain waveform generator and returns the
  windowed FFT of its two polarizations on a frequency mask.
- :class:`GetFDWaveformFromFD`: wraps a frequency-domain generator (e.g. with
  :class:`~few.summation.fdinterp.FDInterpolatedModeSum`) and applies a time-domain window
  as a frequency-domain circular convolution.
- The underlying functions :func:`get_fft_td_windowed`, :func:`get_fd_windowed` and
  :func:`get_convolution`.

Not used inside FEW; see the frequency-domain tutorial notebooks.

Warning:
    The cupy branches import ``has_backend`` from ``few``, which FEW does not define, so
    cupy inputs raise ``ImportError``; and :func:`get_fd_windowed` (hence
    :class:`GetFDWaveformFromFD`) raises ``ValueError`` for ``window=None``, because its
    type check compares the window's type with the signal's before the ``None`` branch.
"""

from typing import Optional

import numpy as np


def get_convolution(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Return the circular convolution of ``a`` and ``b`` divided by ``len(b)``.

    Computed as a "valid" linear convolution of ``a`` extended by its own periodic copy
    (``scipy.signal.convolve`` / ``cupyx.scipy.signal.convolve``); for ``len(a) == len(b)
    == N`` the output has length ``N`` and element ``k`` is
    ``sum_j a[(k - j) mod N] b[j] / N``.

    Args:
        a: First array (numpy or cupy).
        b: Second array, same array type as ``a``.

    Returns:
        np.ndarray: The normalised circular convolution (cupy for cupy inputs), length
        ``2 len(a) - len(b)``.

    Raises:
        ValueError: If ``a`` and ``b`` are of different array types.
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
    """Return the FFT of two windowed time-domain signals, ``fftshift``-ed and scaled by ``dt``.

    Args:
        signal: ``[h_plus, h_cross]`` time series of a common length.
        window: Time-domain window, same length and array type as the signals.
        dt: Sampling interval in s.

    Returns:
        list[np.ndarray]: ``[fftshift(fft(h_plus * window)) * dt, fftshift(fft(h_cross *
        window)) * dt]`` (frequency order of ``np.fft.fftshift``).

    Raises:
        ValueError: If ``signal[0]`` and ``window`` are of different array types.
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
    """Apply a window to two frequency-domain signals by circular convolution.

    Each polarization is convolved with ``conj(FFT(window))`` (or ``conj(window)`` with
    ``window_in_fd``) through :func:`get_convolution`.

    Args:
        signal: ``[h_plus, h_cross]`` frequency-domain signals.
        window: Window, same length and array type as the signals: in the time domain, or
            in the frequency domain with ``window_in_fd``. ``None`` (default) is meant to
            skip the windowing, but the type check fails first (``ValueError``).
        window_in_fd: ``window`` is already in the frequency domain. Default ``False``.

    Returns:
        list[np.ndarray]: The two windowed signals.

    Raises:
        ValueError: If ``signal[0]`` and ``window`` are of different types (including
            ``window=None``).
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
    """Windowed frequency-domain waveform from a frequency-domain FEW generator.

    Calls the generator, windows its two polarizations with :func:`get_fd_windowed` and
    keeps the frequencies of ``positive_frequency_mask``.

    Args:
        waveform_generator: Callable returning ``[h_plus, h_cross]`` in the frequency domain
            (e.g. a :class:`~few.waveform.waveform.GenerateEMRIWaveform` built with
            ``sum_kwargs=dict(output_type="fd")`` and ``return_list=True``, as in the
            frequency-domain tutorial).
        positive_frequency_mask: Boolean mask of the frequencies to return.
        dt: Sampling interval in s; stored nowhere and not used.
        non_zero_mask: Boolean mask over the returned frequencies; entries where it is
            ``False`` are set to zero. Default ``None`` (no zeroing).
        window: Window for :func:`get_fd_windowed`. Default ``None``, which fails at call
            time (see :func:`get_fd_windowed`).
        window_in_fd: ``window`` is given in the frequency domain. Default ``False``.

    Example:
        .. code-block:: python

            fd_gen = GetFDWaveformFromFD(few_gen, positive_frequency_mask, dt, window=window)
            h_plus, h_cross = fd_gen(*params, **waveform_kwargs)
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
        """Run the waveform generator and return the windowed, masked polarizations.

        Args:
            *args: Forwarded to the waveform generator.
            **kwargs: Forwarded to the waveform generator.

        Returns:
            list[np.ndarray]: ``[h_plus, h_cross]`` on the positive-frequency mask.
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
    """Frequency-domain waveform from a time-domain FEW generator (windowed FFT).

    Calls the generator, takes :func:`get_fft_td_windowed` of its two polarizations and
    keeps the frequencies of ``positive_frequency_mask``.

    Args:
        waveform_generator: Callable returning ``[h_plus, h_cross]`` time series (e.g. a
            :class:`~few.waveform.waveform.GenerateEMRIWaveform` with ``return_list=True``).
        positive_frequency_mask: Boolean mask, over the ``fftshift``-ed FFT frequencies, of
            the frequencies to return.
        dt: Sampling interval in s (scales the FFT).
        non_zero_mask: Boolean mask over the returned frequencies; entries where it is
            ``False`` are set to zero. Default ``None`` (no zeroing).
        window: Time-domain window, same length and array type as the signals. Default
            ``None`` uses ``np.ones_like(positive_frequency_mask)`` (no windowing; numpy,
            so cupy signals then fail the type check).

    Example:
        .. code-block:: python

            td_fd_gen = GetFDWaveformFromTD(td_gen, positive_frequency_mask, dt, window=window)
            h_plus, h_cross = td_fd_gen(*params, **waveform_kwargs)
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
        """Run the waveform generator and return the masked FFTs of its polarizations.

        Args:
            *args: Forwarded to the waveform generator.
            **kwargs: Forwarded to the waveform generator.

        Returns:
            list[np.ndarray]: ``[h_plus, h_cross]`` (windowed FFT times ``dt``) on the mask.
        """
        data_channels_td = self.waveform_generator(*args, **kwargs)
        list_p_c = get_fft_td_windowed(data_channels_td, self.window, self.dt)
        fft_td_wave_p = list_p_c[0][self.positive_frequency_mask]
        fft_td_wave_c = list_p_c[1][self.positive_frequency_mask]
        if self.non_zero_mask is not None:
            fft_td_wave_p[~self.non_zero_mask] = complex(0.0)
            fft_td_wave_c[~self.non_zero_mask] = complex(0.0)
        return [fft_td_wave_p, fft_td_wave_c]
