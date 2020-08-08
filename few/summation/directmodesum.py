import numpy as np

from few.utils.baseclasses import SummationBase, SchwarzschildEccentric
from few.utils.citations import *

try:
    import cupy as xp

except (ImportError, ModuleNotFoundError) as e:
    import numpy as xp


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
        return few_citation

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

        teuk_modes = self.xp.asarray(teuk_modes)
        ylms = self.xp.asarray(ylms)
        Phi_phi = self.xp.asarray(Phi_phi)
        Phi_r = self.xp.asarray(Phi_r)
        m_arr = self.xp.asarray(m_arr)
        n_arr = self.xp.asarray(n_arr)

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

        self.waveform = w1 + w2
