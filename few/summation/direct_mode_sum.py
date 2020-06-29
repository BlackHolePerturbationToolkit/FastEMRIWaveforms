import numpy as xp


class DirectModeSum:
    def __init__(self, pad_output=False):
        self.num_phases = 2
        self.pad_output = pad_output

    def _sum(
        self,
        teuk_modes,
        Phi_phi,
        Phi_r,
        m_arr,
        n_arr,
        init_len,
        num_pts,
        num_teuk_modes,
        ylms,
        dt,
    ):

        w1 = xp.sum(
            ylms[xp.newaxis, : teuk_modes.shape[1]]
            * teuk_modes
            * xp.exp(
                -1j
                * (
                    m_arr[xp.newaxis, :] * Phi_phi[:, xp.newaxis]
                    + n_arr[xp.newaxis, :] * Phi_r[:, xp.newaxis]
                )
            ),
            axis=1,
        )

        inds = xp.where(m_arr > 0)[0]

        w2 = xp.sum(
            (m_arr[xp.newaxis, inds] > 0)
            * ylms[xp.newaxis, teuk_modes.shape[1] :][:, inds]
            * xp.conj(teuk_modes[:, inds])
            * xp.exp(
                -1j
                * (
                    -m_arr[xp.newaxis, inds] * Phi_phi[:, xp.newaxis]
                    - n_arr[xp.newaxis, inds] * Phi_r[:, xp.newaxis]
                )
            ),
            axis=1,
        )

        self.waveform = w1 + w2

        """
        i = 0
        for (tmodes, m, n, ylm, minus_m_ylm) in zip(
            teuk_modes.T, m_arr, n_arr, ylms[: len(m_arr)], ylms[len(m_arr) :]
        ):

            phi = m * Phi_phi + n * Phi_r
            self.waveform += ylm * tmodes * xp.exp(-1j * phi)

            if m > 0:
                phi = -m * Phi_phi - n * Phi_r
                self.waveform += minus_m_ylm * xp.conj(tmodes) * xp.exp(-1j * phi)
            i += 1
        print("modes", i)
        """

    def __call__(
        self, t, p, e, Phi_phi, Phi_r, teuk_modes, m_arr, n_arr, ylms, dt, T, **kwargs
    ):

        # TODO: fix this for padding (?)
        num_pts = len(p)
        num_pts_pad = 0

        # TODO: make sure num points adjusts for zero padding
        self.num_pts, self.num_pts_pad = num_pts, num_pts_pad
        self.dt = dt
        init_len = len(t)
        num_teuk_modes = teuk_modes.shape[1]

        self.waveform = xp.zeros(
            (self.num_pts + self.num_pts_pad,), dtype=xp.complex128
        )

        self._sum(
            teuk_modes,
            Phi_phi,
            Phi_r,
            m_arr,
            n_arr,
            init_len,
            num_pts,
            num_teuk_modes,
            ylms,
            dt,
        )

        return self.waveform
