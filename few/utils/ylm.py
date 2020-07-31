try:
    import cupy as xp
except (ImportError, ModuleNotFoundError) as e:
    import numpy as xp

import numpy as np

from pySpinWeightedSpherHarm import get_spin_weighted_spher_harm_wrap


class GetYlms:
    """(-2) Spin-weighted Spherical Harmonics

    The class generates (-2) spin-weighted spherical hackarmonics,
    :math:`Y_{lm}(\Theta,\phi)`. **Important Note**: this class also applies
    the parity operator (:math:`-1^l`) to modes with :math:`m<0`.

    args:
        assume_positive_m (bool, optional): Set true if only providing :math:`m\geq0`,
            it will return twice the number of requested modes with the seconds
            half as modes with :math:`m<0`. **Warning**: It will also duplicate
            the :math:`m=0` modes. Default is False.
        use_gpu (bool, optional): If True, allocate arrays for GPU.
            Default is False.

    """

    def __init__(self, assume_positive_m=False, use_gpu=False):
        self.assume_positive_m = assume_positive_m

        if use_gpu:
            self.xp = xp

        else:
            self.xp = np

    def attributes_GetYlms(self):
        """
        attributes:
            xp (obj): cupy or numpy based on GPU usage.
        """
        pass

    # These are the spin-weighted spherical harmonics with s=2
    def __call__(self, l_in, m_in, theta, phi):
        """Call method for Ylms.

        This returns ylms based on requested :math:`(l,m)` values and viewing
        angles.

        args:
            l_in (1D int xp.ndarray): :math:`l` values requested.
            m_in (1D int xp.ndarray): :math:`m` values requested.
            theta (double): Polar viewing angle.
            phi (double): Azimuthal viewing angle.

        Returns:
            1D complex128 xp.ndarray: Ylm values.

        """

        if self.assume_positive_m:
            l = self.xp.zeros(2 * l_in.shape[0], dtype=int)
            m = self.xp.zeros(2 * l_in.shape[0], dtype=int)

            l[: l_in.shape[0]] = l_in
            l[l_in.shape[0] :] = l_in

            m[: l_in.shape[0]] = m_in
            m[l_in.shape[0] :] = -m_in

        else:
            l = l_in
            m = m_in

        try:
            l = l.get()
            m = m.get()

        except AttributeError:
            pass

        return self.xp.asarray(
            get_spin_weighted_spher_harm_wrap(
                l.astype(np.int32), m.astype(np.int32), theta, phi
            )
        )
