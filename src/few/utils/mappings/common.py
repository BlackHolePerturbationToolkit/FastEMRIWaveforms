"""Mass-parameter conversions of a binary (not used inside the FEW package code)."""

from typing import Union

import numpy as np


def m1m2_to_muM(
    m1: Union[float, np.ndarray], m2: Union[float, np.ndarray]
) -> tuple[Union[float, np.ndarray], Union[float, np.ndarray]]:
    r"""Convert the component masses to the reduced mass and the total mass.

    Args:
        m1: Mass of the first body (any unit).
        m2: Mass of the second body, same unit.

    Returns:
        tuple: ``(mu, M)`` with :math:`\mu = m_1 m_2/(m_1 + m_2)` and :math:`M = m_1 + m_2`,
        in the unit of the inputs (elementwise for arrays).

    Example:
        >>> from few.utils.mappings.common import m1m2_to_muM
        >>> m1m2_to_muM(3.0, 1.0)
        (0.75, 4.0)
    """

    mu = m1 * m2 / (m1 + m2)
    M = m1 + m2
    return mu, M


def muM_to_m1m2(
    mu: Union[float, np.ndarray], M: Union[float, np.ndarray]
) -> tuple[Union[float, np.ndarray], Union[float, np.ndarray]]:
    r"""Intended inverse of :func:`m1m2_to_muM`: component masses with ``m1 >= m2``.

    Args:
        mu: Reduced mass.
        M: Total mass, same unit.

    Returns:
        tuple: ``(m1, m2) = ((M + s)/2, (M - s)/2)`` with :math:`s = \sqrt{M^2 - 4\mu^2}`.

    Warning:
        The component masses solve :math:`m^2 - M m + \mu M = 0`, i.e.
        :math:`s = \sqrt{M^2 - 4\mu M}`; the code uses :math:`M^2 - 4\mu^2`, so the result
        is wrong for any physical input (e.g. ``mu = 0.5, M = 2`` gives ``(1.87, 0.13)``
        instead of ``(1, 1)``). The mass-invariance cell of ``examples/waveform.ipynb``
        uses this function.
    """

    sqrdet = np.sqrt(M**2 - 4 * mu**2)
    m1 = (M + sqrdet) / 2
    m2 = (M - sqrdet) / 2
    return m1, m2
