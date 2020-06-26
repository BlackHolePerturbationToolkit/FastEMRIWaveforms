try:
    import cupy as xp
except ImportError:
    import numpy as xp

import numpy as np

from pySpinWeightedSpherHarm import get_spin_weighted_spher_harm_wrap


class GetYlms:
    def __init__(self, num_teuk_modes, assume_positive_m=True):
        self.num_teuk_modes = num_teuk_modes
        self.assume_positive_m = assume_positive_m

        self.buffer = xp.zeros((2 * self.num_teuk_modes,), dtype=xp.complex128)

    # These are the spin-weighted spherical harmonics with s=2
    def __call__(self, l_in, m_in, theta, phi):

        if self.assume_positive_m:
            l = xp.zeros(2 * l_in.shape[0], dtype=int)
            m = xp.zeros(2 * l_in.shape[0], dtype=int)

            l[: l_in.shape[0]] = l_in
            l[l_in.shape[0] :] = l_in

            m[: l_in.shape[0]] = m_in
            m[l_in.shape[0] :] = -m_in

        else:
            l = l_in
            m = m_in

        return xp.asarray(
            get_spin_weighted_spher_harm_wrap(
                l.get().astype(np.int32), m.get().astype(np.int32), theta, phi
            )
        )
