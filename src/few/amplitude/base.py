from typing import Optional, Union

import numpy as np

from ..utils.baseclasses import ParallelModuleBase


class AmplitudeBase(ParallelModuleBase):
    """Base class used for amplitude modules.

    This class provides a common flexible interface to various amplitude
    implementations. Specific arguments to each amplitude module can be found
    with each associated module discussed below.
    """

    @classmethod
    def get_amplitudes(self, *args, **kwargs):
        """Amplitude Generator

        @classmethod that requires a child class to have a get_amplitudes method.

        raises:
            NotImplementedError: The child class does not have this method.

        """
        raise NotImplementedError

    def __call__(
        self, 
        a: Union[float, np.ndarray],
        p: Union[float, np.ndarray],
        e: Union[float, np.ndarray],
        xI: Union[float, np.ndarray],
        *args, 
        specific_modes: Optional[Union[list, np.ndarray]] = None, 
        **kwargs
    ) -> Union[dict, np.ndarray]:
        """Common call for Teukolsky amplitudes

        This function takes the inputs the trajectory in :math:`(p,e)` as arrays
        and returns the complex amplitude of all modes to adiabatic order at
        each step of the trajectory.

        Args:
            a: Spin parameter of the black hole.
            p: Semilatus rectum of the orbit.
            e: Eccentricity of the orbit.
            xI: Cosine of the inclination angle of the orbit.
            *args: Added to create future flexibility when calling different
                amplitude modules. Transfers directly into get_amplitudes function.
            specific_modes: Either indices or mode index tuples of modes to be generated (optional; defaults to all modes).
            **kwargs (dict, placeholder): Added to create flexibility when calling different
                amplitude modules.

        Returns:
            If specific_modes is a list of tuples, returns a dictionary of complex mode amplitudes.
            Else, returns an array of complex mode amplitudes.

        """

        a = self.xp.atleast_1d(a)
        p = self.xp.atleast_1d(p)
        e = self.xp.atleast_1d(e)
        xI = self.xp.atleast_1d(xI)

        lengths = [len(arr) for arr in (a, p, e, xI)]
        non_one_lengths = {
            l for l in lengths if l > 1
        }  # Collect lengths greater than 1

        assert len(non_one_lengths) <= 1, (
            f"Arrays must be length one or, if larger, have the same length. Found lengths: {lengths}"
        )

        # symmetry of flipping the sign of the spin to keep xI positive
        # CAREFUL for generic - sign flip during the trajectory? Or if the user asks many amplitudes not from trajectory.
        if self.xp.all(xI < 0.0):
            m_mode_sign = -1
        else:
            m_mode_sign = 1

        if specific_modes is not None: # if the user has specified modes
            self.num_modes_eval = len(specific_modes)
            if isinstance(specific_modes, self.xp.ndarray):
                mode_indexes = specific_modes.copy()
                # Identify requested negative mkn modes, the conjugate relation must be applied at the end
                conj_mode_mask = mode_indexes >= self.num_teuk_modes
                mode_indexes[conj_mode_mask] = self.negative_mode_indexes[mode_indexes[conj_mode_mask] - self.num_m_1_up]

            elif isinstance(specific_modes, list):
                specific_modes_arr = self.xp.asarray(specific_modes)
                mode_indexes = self.special_index_map_arr[
                    specific_modes_arr[:, 0],
                    m_mode_sign * specific_modes_arr[:, 1],  # may need fix for array mode index input?
                    specific_modes_arr[:, 2],
                    specific_modes_arr[:, 3],
                ] # find locations of the modes in the special index map array
                if self.xp.any(mode_indexes == -1):
                    failed_mode = specific_modes_arr[
                        self.xp.where(mode_indexes == -1)[0][0]
                    ]
                    raise ValueError(
                        f"Could not find mode index ({failed_mode[0]},{failed_mode[1]},{failed_mode[2]},{failed_mode[3]})."
                    )
            else:
                mode_indexes = specific_modes
        else: # if the user has not specified modes
            if m_mode_sign < 0: # check to see whether xI is negative
                mode_indexes = self.negative_mode_indexes # if so, use negative m-modes. Note this is defined in SphericalHarmonic base class
                conj_mode_mask = self.xp.ones_like(mode_indexes, dtype=bool) # Identify requested negative mkn modes

            else:
                mode_indexes = self.mode_indexes
                conj_mode_mask = self.xp.zeros_like(mode_indexes, dtype=bool) # Identify requested negative mkn modes
            self.num_modes_eval = self.num_teuk_modes

        
        teuk_modes = self.get_amplitudes(a, p, e, xI, *args, specific_modes=mode_indexes, **kwargs)

        if not isinstance(specific_modes, list):
            # apply xI flip symmetry
            if m_mode_sign < 0:
                # this requires a sign flip of the m mode because the default is to return only m > 0 modes
                teuk_modes = self.xp.conj(teuk_modes)
            if self.xp.any(conj_mode_mask):
                # apply +/- m symmetry
                teuk_modes[:, conj_mode_mask] = (
                    (-1) ** (self.l_arr + self.k_arr)[mode_indexes[conj_mode_mask]]
                    * self.xp.conj(teuk_modes[:, conj_mode_mask])
                )
            
            return teuk_modes

        else:
            temp = {}
            for i, lmkn in enumerate(specific_modes):
                temp[lmkn] = teuk_modes[:, i]
                l, m, k, n = lmkn

                # apply xI flip symmetry
                if m_mode_sign < 0:
                    temp[lmkn] = (-1) ** l * temp[lmkn]

                # apply +/- m symmetry
                if m_mode_sign * m < 0:
                    temp[lmkn] = (-1) ** (l + k) * self.xp.conj(temp[lmkn])

            return temp


