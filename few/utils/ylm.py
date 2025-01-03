# Function for ylm generation for FastEMRIWaveforms Packages

# Copyright (C) 2020 Michael L. Katz, Alvin J.K. Chua, Niels Warburton, Scott A. Hughes
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

# test import of cupy
try:
    import cupy as cp
except (ImportError, ModuleNotFoundError) as e:
    pass

import numpy as np

# import Cython wrapped C++ function
from ..cutils.pySpinWeightedSpherHarm import get_spin_weighted_spher_harm_wrap

# base classes
from few.utils.baseclasses import ParallelModuleBase


class GetYlms(ParallelModuleBase):
    """(-2) Spin-weighted Spherical Harmonics

    The class generates (-2) spin-weighted spherical hackarmonics,
    :math:`Y_{lm}(\Theta,\phi)`. **Important Note**: this class also applies
    the parity operator (:math:`-1^l`) to modes with :math:`m<0`.

    args:
        assume_positive_m (bool, optional): Set true if only providing :math:`m\geq0`,
            it will return twice the number of requested modes with the seconds
            half as modes with :math:`m<0`. **Warning**: It will also duplicate
            the :math:`m=0` modes. Default is False.
        **kwargs (dict, optional): Keyword arguments for the base classes:
            :class:`few.utils.baseclasses.ParallelModuleBase`.
            Default is {}.

    """

    def __init__(self, assume_positive_m=False, **kwargs):
        ParallelModuleBase.__init__(self, **kwargs)
        # see args in docstring
        self.assume_positive_m = assume_positive_m

    @property
    def gpu_capability(self):
        """Confirms GPU capability"""
        return True

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

        if self.use_gpu:
            xp = cp
        else:
            xp = np

        # if assuming positive m, repeat entries for negative m
        # this will duplicate m = 0
        if self.assume_positive_m:
            l = xp.zeros(2 * l_in.shape[0], dtype=int)
            m = xp.zeros(2 * l_in.shape[0], dtype=int)

            l[: l_in.shape[0]] = l_in
            l[l_in.shape[0] :] = l_in

            m[: l_in.shape[0]] = m_in
            m[l_in.shape[0] :] = -m_in

        # if not, just l_in, m_in
        else:
            l = l_in
            m = m_in

        # the function only works with CPU allocated arrays
        # if l and m are cupy arrays, turn into numpy arrays
        try:
            l = l.get()
            m = m.get()

        except AttributeError:
            pass

        # get ylm arrays and cast back to cupy if using cupy/GPUs
        return xp.asarray(
            get_spin_weighted_spher_harm_wrap(
                l.astype(np.int32), m.astype(np.int32), theta, phi
            )
        )
