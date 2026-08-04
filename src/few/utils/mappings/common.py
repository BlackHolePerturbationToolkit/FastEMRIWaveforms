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

    sqrdet = np.sqrt(M**2 - 4 * M * mu)
    m1 = (M + sqrdet) / 2
    m2 = (M - sqrdet) / 2
    return m1, m2

def chi_to_chit(
    chi1: Union[float, np.ndarray], chi2: Union[float, np.ndarray], nu: Union[float, np.ndarray]
) -> tuple[Union[float, np.ndarray], Union[float, np.ndarray]]:
    """
    Convert the individual dimensionless spins chi_i=S_i/m_i^2 to tilde{chi}_i=S_i/(m_i M).
    Args:
        chi1: dimensionless kerr parameter of the first body.
        chi2: dimensionless kerr parameter of the second body.
        nu: symmetric mass-ratio.

    Returns:
        chit1: (m1/M)*chi1.
        chit2: (m2/M)*chi2.
    """

    chit1 = chi1/2.*(1.+np.sqrt(1.-4.*nu))
    chit2 = chi2/2.*(1.-np.sqrt(1.-4.*nu))
    return chit1, chit2

def chit_to_chi(
    chit1: Union[float, np.ndarray], chit2: Union[float, np.ndarray], nu: Union[float, np.ndarray]
) -> tuple[Union[float, np.ndarray], Union[float, np.ndarray]]:
    """
    Convert the individual rescaled spins tilde{chi}_i=S_i/(m_i M) back to the standard dimensionless spins chi_i=S_i/m_i^2.
    Args:
        chit1: rescaled primary spin (m1/M)*chi1.
        chit2: rescaled secondary spin (m2/M)*chi2.
        nu: symmetric mass-ratio.

    Returns:
        chi1: dimensionless kerr parameter of the first body.
        chi2: dimensionless kerr parameter of the second body.
    """

    chi1 = 2.*chit1/(1.+np.sqrt(1.-4.*nu))
    chi2 = 2.*chit2/(1.-np.sqrt(1.-4.*nu))
    return chi1, chi2
