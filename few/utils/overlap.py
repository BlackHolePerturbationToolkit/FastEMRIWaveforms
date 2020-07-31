import numpy as np
import numpy

try:
    import cupy as cp

except (ImportError, ModuleNotFoundError) as e:
    import numpy as np


def get_overlap(time_series_1, time_series_2, use_gpu=False):
    """Calculate the overlap.

    Takes two time series and finds which one is shorter in length. It then
    shortens the longer time series if necessary. Then it performs a
    normalized correlation calulation on the two time series to give the
    overlap. The overlap of :math:`a(t)` and
    :math:`b(t)`, :math:`\gamma_{a,b}`, is given by,

    .. math:: \gamma_{a,b} = <a,b>/(<a,a><b,b>)^{(1/2)},

    where :math:`<a,b>` is the inner product of the two time series.

    args:
        time_series_1 (1D complex128 xp.ndarray): Strain time series 1.
        time_series_2 (1D complex128 xp.ndarray): Strain time series 2.
        use_gpu (bool, optional): If True use cupy. If False, use numpy. Default
            is False.

    """
    if use_gpu:
        xp = cp

        if isinstance(time_series_1, np.ndarray):
            time_series_1 = xp.asarray(time_series_1)
        if isinstance(time_series_2, np.ndarray):
            time_series_2 = xp.asarray(time_series_2)

    else:
        xp = np

        try:
            if isinstance(time_series_1, cp.ndarray):
                time_series_1 = xp.asarray(time_series_1)

        except NameError:
            pass

        try:
            if isinstance(time_series_2, cp.ndarray):
                time_series_2 = xp.asarray(time_series_2)

        except NameError:
            pass

    min_len = int(np.min([len(time_series_1), len(time_series_2)]))
    time_series_1_fft = xp.fft.fft(time_series_1[:min_len])
    time_series_2_fft = xp.fft.fft(time_series_2[:min_len])
    ac = xp.dot(time_series_1_fft.conj(), time_series_2_fft) / xp.sqrt(
        xp.dot(time_series_1_fft.conj(), time_series_1_fft)
        * xp.dot(time_series_2_fft.conj(), time_series_2_fft)
    )

    if use_gpu:
        return ac.item().real
    return ac.real


def get_mismatch(time_series_1, time_series_2, use_gpu=False):
    """Calculate the mismatch.

    The mismatch is 1 - overlap. Therefore, see documentation for
    :func:`few.utils.overlap.overlap` for information on the overlap
    calculation.

    args:
        time_series_1 (1D complex128 xp.ndarray): Strain time series 1.
        time_series_2 (1D complex128 xp.ndarray): Strain time series 2.
        use_gpu (bool, optional): If True use cupy. If False, use numpy. Default
            is False.

    """
    overlap = get_overlap(time_series_1, time_series_2, use_gpu=use_gpu)
    return 1.0 - overlap
