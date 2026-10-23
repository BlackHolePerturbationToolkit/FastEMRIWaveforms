"""Common call interface of the FEW amplitude modules (:class:`AmplitudeBase`)."""

from typing import Optional, Union

import numpy as np

from ..utils.baseclasses import FewBackendConsumer


class AmplitudeBase(FewBackendConsumer):
    """Base class of the amplitude modules.

    Subclasses implement :meth:`get_amplitudes`; :meth:`__call__` provides the common call
    (input broadcasting, mode-index lookup and the mode symmetries). Concrete modules also
    derive from a model class of :mod:`few.utils.baseclasses`
    (:class:`~few.utils.baseclasses.KerrEccentricEquatorial`,
    :class:`~few.utils.baseclasses.SchwarzschildEccentric`, ...), whose mode bookkeeping
    (``num_teuk_modes``, ``mode_indexes``, ``negative_mode_indexes``,
    ``special_index_map_arr``, ``l_arr``, ``k_arr``, ``num_m_1_up``) :meth:`__call__` uses.
    The stock modules are listed in :mod:`few.amplitude`.
    """

    @classmethod
    def get_amplitudes(self, *args, **kwargs):
        """Compute the amplitudes (implemented by the subclasses).

        Called by :meth:`__call__` with the 1-D arrays ``a, p, e, xI``, the extra positional
        and keyword arguments, and ``specific_modes`` set to an integer index array.

        Raises:
            NotImplementedError: Always, in the base class.
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
        r"""Evaluate the complex Teukolsky amplitudes along a trajectory.

        The inputs are made at least 1-D (each must have length 1 or a common length ``N``).
        If every ``xI`` is negative, the amplitudes are built from the :math:`x_I > 0`
        convention by the inclination-flip symmetry; requested :math:`m < 0` modes are served
        from their :math:`m > 0` partners by
        :math:`A_{l,-m,-k,-n} = (-1)^{l+k} A^*_{lmkn}`. Called by
        :class:`~few.utils.modeselector.ModeSelector` with the trajectory arrays.

        Args:
            a: Dimensionless spin of the primary (scalar or array).
            p: Semi-latus rectum in units of :math:`M`, shape ``(N,)``.
            e: Eccentricity, shape ``(N,)``.
            xI: Cosine of the inclination :math:`x_I`, scalar or shape ``(N,)``.
            *args: Forwarded to :meth:`get_amplitudes`.
            specific_modes: Modes to evaluate. ``None`` (default): all ``num_teuk_modes``
                :math:`m \ge 0` modes. An integer array: mode indices into the model's
                ``l_arr`` / ``m_arr`` ordering (indices ``>= num_teuk_modes`` are :math:`m < 0`
                modes). A list of ``(l, m, k, n)`` tuples: those modes, returned as a dict.
            **kwargs: Forwarded to :meth:`get_amplitudes`.

        Returns:
            xp.ndarray or dict: Complex amplitudes on the backend. Array of shape
            ``(N, number of modes)`` for ``specific_modes`` ``None`` or an index array; for a
            list of tuples, a dict mapping each ``(l, m, k, n)`` tuple to its ``(N,)`` array.

        Raises:
            AssertionError: If the input lengths are inconsistent.
            ValueError: If a requested ``(l, m, k, n)`` mode does not exist in the model.
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


