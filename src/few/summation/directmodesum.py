"""Direct mode summation over a dense trajectory (:class:`DirectModeSum`)."""

# Direct summation of modes in python for the FastEMRIWaveforms Package

from .base import SummationBase


class DirectModeSum(SummationBase):
    r"""Create a waveform by direct summation at the trajectory points (no interpolation).

    This class sums the amplitude and phase information as received:

    .. math::

        h = \sum_{\rm modes} Y_{lm} A_{lmkn} e^{-i \Phi_{mkn}}
            + \sum_{m > 0} (-1)^l Y_{l,-m} A^*_{lmkn} e^{+i \Phi_{mkn}},
        \qquad \Phi_{mkn} = m \Phi_\phi + k \Phi_\theta + n \Phi_r.

    The output has one sample per trajectory point, so the trajectory must already be dense
    (fixed steps of ``dt``, as in :class:`~few.waveform.waveform.SlowSchwarzschildEccentricFlux`).

    Args:
        *args: Forwarded to :class:`~few.summation.base.SummationBase` (positionally
            ``output_type``, ``pad_output``, ``odd_len``, ``force_backend``).
        **kwargs: Forwarded to :class:`~few.summation.base.SummationBase`.
    """

    def __init__(self, *args, **kwargs):
        SummationBase.__init__(self, *args, **kwargs)

    @classmethod
    def supported_backends(cls):
        """Backend preference order: ``["cpu", "cuda13x", "cuda12x", "cuda11x"]``."""
        return cls.CPU_RECOMMENDED_WITH_GPU_SUPPORT()

    def sum(
        self,
        t,
        teuk_modes,
        ylms,
        phase_interp_t,
        phases_in,
        l_arr,
        m_arr,
        k_arr,
        n_arr,
        *args,
        dt=10.0,
        integrate_backwards=False,
        **kwargs,
    ):
        r"""Directly sum the modes at the trajectory points; sets :attr:`waveform`.

        This function directly sums the amplitude and phase information, as well
        as the spin-weighted spherical harmonic values. Inputs are moved to the backend.

        Args:
            t (xp.ndarray): Trajectory times in seconds, shape ``(N,)``.
            teuk_modes (xp.ndarray): Complex amplitudes, shape ``(N, K)``.
            ylms (xp.ndarray): Complex spherical harmonics, shape ``(2 K,)``: :math:`Y_{lm}`
                of the ``K`` modes, then :math:`Y_{l,-m}` of the same modes.
            phase_interp_t (xp.ndarray): Not used (``None`` is replaced by a copy of ``t``).
            phases_in (xp.ndarray): Phases ``(Phi_phi, Phi_theta, Phi_r)`` in rad at the
                trajectory points, shape ``(3, N)``.
            l_arr (xp.ndarray): :math:`\ell` of each mode, shape ``(K,)``.
            m_arr (xp.ndarray): :math:`m` of each mode, shape ``(K,)``.
            k_arr (xp.ndarray): :math:`k` of each mode, shape ``(K,)``.
            n_arr (xp.ndarray): :math:`n` of each mode, shape ``(K,)``.
            *args: Accepted and ignored.
            dt (float): Not used. Default ``10.0``.
            integrate_backwards (bool): Must be ``False``. Default ``False``.
            **kwargs: Accepted and ignored.

        Raises:
            NotImplementedError: If ``integrate_backwards`` is ``True``.
        """

        Phi_phi, Phi_theta, Phi_r = phases_in

        if phase_interp_t is None:
            phase_interp_t = t.copy()

        # numpy -> cupy if requested
        # it will never go the other way
        teuk_modes = self.xp.asarray(teuk_modes)
        ylms = self.xp.asarray(ylms)
        Phi_phi = self.xp.asarray(Phi_phi)
        Phi_theta = self.xp.asarray(Phi_theta)
        Phi_r = self.xp.asarray(Phi_r)
        l_arr = self.xp.asarray(l_arr)
        m_arr = self.xp.asarray(m_arr)
        k_arr = self.xp.asarray(k_arr)
        n_arr = self.xp.asarray(n_arr)

        if integrate_backwards:
            # For consistency with forward integration, we slightly shift the knots so that they line up at t=0
            raise NotImplementedError  # TODO: spline the below quantities to shift them onto phase_interp_t
            # offset = h_t[-1] - int(h_t[-1] / dt) * dt
            # h_t = h_t - offset
            # phase_interp_t = phase_interp_t - offset

        # waveform with M >= 0
        w1 = self.xp.sum(
            ylms[self.xp.newaxis, : teuk_modes.shape[1]]
            * teuk_modes
            * self.xp.exp(
                -1j
                * (
                    m_arr[self.xp.newaxis, :] * Phi_phi[:, self.xp.newaxis]
                    + k_arr[self.xp.newaxis, :] * Phi_theta[:, self.xp.newaxis]
                    + n_arr[self.xp.newaxis, :] * Phi_r[:, self.xp.newaxis]
                )
            ),
            axis=1,
        )

        inds = self.xp.where(m_arr > 0)[0]

        # waveform sum where m < 0
        w2 = self.xp.sum(
            (m_arr[self.xp.newaxis, inds] > 0)
            * (-1) ** l_arr[self.xp.newaxis, inds]
            * ylms[self.xp.newaxis, teuk_modes.shape[1] :][:, inds]
            * self.xp.conj(teuk_modes[:, inds])
            * self.xp.exp(
                -1j
                * (
                    -m_arr[self.xp.newaxis, inds] * Phi_phi[:, self.xp.newaxis]
                    - k_arr[self.xp.newaxis, inds] * Phi_theta[:, self.xp.newaxis]
                    - n_arr[self.xp.newaxis, inds] * Phi_r[:, self.xp.newaxis]
                )
            ),
            axis=1,
        )

        # they can be directly summed
        # the base class function __call__ will return the waveform
        self.waveform = w1 + w2
