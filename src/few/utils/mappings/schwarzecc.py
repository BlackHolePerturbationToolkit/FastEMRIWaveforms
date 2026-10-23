"""Grid coordinate of the Schwarzschild eccentric amplitude data."""

import numpy as np


def schwarzecc_p_to_y(p, e, use_gpu=False):
    r"""Convert the semi-latus rectum :math:`p` to the Schwarzschild grid coordinate :math:`y`.

    :math:`y = \ln(p - 2.1 - 2e) = \ln(p - p_{\rm sep} + 3.9)` with the Schwarzschild
    separatrix :math:`p_{\rm sep} = 6 + 2e`. Used by
    :class:`~few.amplitude.romannet.RomanAmplitude` and
    :class:`~few.amplitude.ampinterp2d.AmpInterpSchwarzEcc`.

    Args:
        p (float or xp.ndarray): Semi-latus rectum in units of :math:`M`.
        e (float or xp.ndarray): Eccentricity, broadcastable with ``p``.
        use_gpu (bool, optional): Convert to cupy and evaluate there. Default ``False``.

    Returns:
        float or xp.ndarray: :math:`y` (NaN where :math:`p \le 2.1 + 2e`); cupy with
        ``use_gpu=True``.

    Example:
        >>> from few.utils.mappings.schwarzecc import schwarzecc_p_to_y
        >>> float(schwarzecc_p_to_y(3.1, 0.0))
        0.0
    """
    if use_gpu:
        import cupy as cp

        e_cp = cp.asarray(e)
        p_cp = cp.asarray(p)
        return cp.log(-(21 / 10) - 2 * e_cp + p_cp)

    else:
        return np.log(-(21 / 10) - 2 * e + p)
