from typing import Union

import numpy as np


def m1m2_to_muM(
    m1: Union[float, np.ndarray], m2: Union[float, np.ndarray]
) -> tuple[Union[float, np.ndarray], Union[float, np.ndarray]]:
    """
    Convert the individual masses of a binary system to the reduced mass and total mass.
    Args:
        m1: Mass of the first body.
        m2: Mass of the second body.

    Returns:
        mu: Reduced mass.
        M: Total mass.
    """

    mu = m1 * m2 / (m1 + m2)
    M = m1 + m2
    return mu, M


def muM_to_m1m2(
    mu: Union[float, np.ndarray], M: Union[float, np.ndarray]
) -> tuple[Union[float, np.ndarray], Union[float, np.ndarray]]:
    """
    Convert the reduced mass and total mass of a binary system to the individual masses, assuming m1 >= m2.
    Args:
        mu: Reduced mass.
        M: Total mass.

    Returns:
        m1: Mass of the first body.
        m2: Mass of the second body.
    """

    sqrdet = np.sqrt(M**2 - 4 * mu**2)
    m1 = (M + sqrdet) / 2
    m2 = (M - sqrdet) / 2
    return m1, m2
