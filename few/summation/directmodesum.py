# Direct summation of modes in python for the FastEMRIWaveforms Package

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

import numpy as np

# check for cupy
try:
    import cupy as xp

except (ImportError, ModuleNotFoundError) as e:
    import numpy as xp

# Necessary base classes
from few.utils.baseclasses import SummationBase, SchwarzschildEccentric
from few.utils.citations import *


class DirectModeSum(SummationBase, SchwarzschildEccentric):
    """Create waveform by direct summation.

    This class sums the amplitude and phase information as received.

    args:
        *args (list, placeholder): Added for flexibility.
        **kwargs (dict, placeholder):  Added for flexibility.

    """

    def __init__(self, *args, use_gpu=False, **kwargs):

        SchwarzschildEccentric.__init__(self, *args, **kwargs)
        SummationBase.__init__(self, *args, **kwargs)

        if use_gpu:
            self.xp = xp
        else:
            self.xp = np

    @property
    def citation(self):
        """Return citations for this class"""
        return few_citation + few_software_citation

    def sum(self, t, teuk_modes, ylms, Phi_phi, Phi_r, m_arr, n_arr, *args, **kwargs):
        """Direct summation function.

        This function directly sums the amplitude and phase information, as well
        as the spin-weighted spherical harmonic values.

        args:
            t (1D double np.ndarray): Array of t values.
            teuk_modes (2D double np.array): Array of complex amplitudes.
                Shape: (len(t), num_teuk_modes).
            ylms (1D complex128 xp.ndarray): Array of ylm values for each mode,
                including m<0. Shape is (num of m==0,) + (num of m>0,)
                + (num of m<0). Number of m<0 and m>0 is the same, but they are
                ordered as (m==0 first then) m>0 then m<0.
            Phi_phi (1D double np.ndarray): Array of azimuthal phase values
                (:math:`\Phi_\phi`).
            Phi_r (1D double np.ndarray): Array of radial phase values
                 (:math:`\Phi_r`).
            m_arr (1D int np.ndarray): :math:`m` values associated with each mode.
            n_arr (1D int np.ndarray): :math:`n` values associated with each mode.
            *args (list, placeholder): Added for future flexibility.
            **kwargs (dict, placeholder): Added for future flexibility.

        """

        # numpy -> cupy if requested
        # it will never go the other way
        teuk_modes = self.xp.asarray(teuk_modes)
        ylms = self.xp.asarray(ylms)
        Phi_phi = self.xp.asarray(Phi_phi)
        Phi_r = self.xp.asarray(Phi_r)
        m_arr = self.xp.asarray(m_arr)
        n_arr = self.xp.asarray(n_arr)

        # waveform with M >= 0
        w1 = self.xp.sum(
            ylms[self.xp.newaxis, : teuk_modes.shape[1]]
            * teuk_modes
            * self.xp.exp(
                -1j
                * (
                    m_arr[self.xp.newaxis, :] * Phi_phi[:, self.xp.newaxis]
                    + n_arr[self.xp.newaxis, :] * Phi_r[:, self.xp.newaxis]
                )
            ),
            axis=1,
        )

        inds = self.xp.where(m_arr > 0)[0]

        # waveform sum where m < 0
        w2 = self.xp.sum(
            (m_arr[self.xp.newaxis, inds] > 0)
            * ylms[self.xp.newaxis, teuk_modes.shape[1] :][:, inds]
            * self.xp.conj(teuk_modes[:, inds])
            * self.xp.exp(
                -1j
                * (
                    -m_arr[self.xp.newaxis, inds] * Phi_phi[:, self.xp.newaxis]
                    - n_arr[self.xp.newaxis, inds] * Phi_r[:, self.xp.newaxis]
                )
            ),
            axis=1,
        )

        # they can be directly summed
        # the base class function __call__ will return the waveform
        self.waveform = w1 + w2
