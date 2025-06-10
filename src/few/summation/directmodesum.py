# Direct summation of modes in python for the FastEMRIWaveforms Package

from .base import SummationBase


class DirectModeSum(SummationBase):
    """Create waveform by direct summation.

    This class sums the amplitude and phase information as received.

    args:
        *args (list, placeholder): Added for flexibility.
        **kwargs (dict, placeholder):  Added for flexibility.

    """

    def __init__(self, *args, **kwargs):
        SummationBase.__init__(self, *args, **kwargs)

    @classmethod
    def supported_backends(cls):
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
        n_arr,
        *args,
        dt=10.0,
        integrate_backwards=False,
        **kwargs,
    ):
        r"""Direct summation function.

        This function directly sums the amplitude and phase information, as well
        as the spin-weighted spherical harmonic values.

        args:
            t (1D double np.ndarray): Array of t values.
            teuk_modes (2D double np.array): Array of complex amplitudes.
                Shape: (len(t), num_teuk_modes).
            ylms (1D complex128 self.xp.ndarray): Array of ylm values for each mode,
                including m<0. Shape is (num of m==0,) + (num of m>0,)
                + (num of m<0). Number of m<0 and m>0 is the same, but they are
                ordered as (m==0 first then) m>0 then m<0.
            Phi_phi (1D double np.ndarray): Array of azimuthal phase values
                (:math:`\Phi_\phi`).
            Phi_r (1D double np.ndarray): Array of radial phase values
                 (:math:`\Phi_r`).
            l_arr (1D int np.ndarray): :math:`\ell` values associated with each mode.
            m_arr (1D int np.ndarray): :math:`m` values associated with each mode.
            n_arr (1D int np.ndarray): :math:`n` values associated with each mode.
            *args (list, placeholder): Added for future flexibility.
            **kwargs (dict, placeholder): Added for future flexibility.

        """

        Phi_phi, Phi_theta, Phi_r = phases_in

        if phase_interp_t is None:
            phase_interp_t = t.copy()

        # numpy -> cupy if requested
        # it will never go the other way
        teuk_modes = self.xp.asarray(teuk_modes)
        ylms = self.xp.asarray(ylms)
        Phi_phi = self.xp.asarray(Phi_phi)
        Phi_r = self.xp.asarray(Phi_r)
        l_arr = self.xp.asarray(l_arr)
        m_arr = self.xp.asarray(m_arr)
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
                    - n_arr[self.xp.newaxis, inds] * Phi_r[:, self.xp.newaxis]
                )
            ),
            axis=1,
        )

        # they can be directly summed
        # the base class function __call__ will return the waveform
        self.waveform = w1 + w2
