import numpy as np

try:
    import cupy as xp

except (ImportError, ModuleNotFoundError) as e:
    import numpy as xp


class ModeSelector:
    """Filter teukolsky amplitudes based on power contribution.

    This module takes teukolsky modes, combines them with their associated ylms,
    and determines the power contribution from each mode. It then filters the
    modes bases on the fractional accuracy on the total power (eps) parameter.

    The mode filtering is a major contributing factor to the speed of these
    waveforms as it removes large numbers of useles modes from the final
    summation calculation.

    Be careful as this is built based on the construction that input mode arrays
    will in order of :math:`m=0`, :math:`m>0`, and then :math:`m<0`.

    args:
        m0mask (1D bool xp.ndarray): This mask highlights which modes have
            :math:`m=0`. Value is False if :math:`m=0`, True if not.
            This only includes :math:`m\geq0`.
        use_gpu (bool, optional): If True, allocate arrays for usage on a GPU.
            Default is False.

    """

    def __init__(self, m0mask, use_gpu=False):

        if use_gpu:
            self.xp = xp

        else:
            self.xp = np

        self.m0mask = m0mask
        self.num_m_zero_up = len(m0mask)
        self.num_m_1_up = len(np.arange(len(m0mask))[m0mask])
        self.num_m0 = len(np.arange(len(m0mask))[~m0mask])

    def attributes_ModeSelector(self):
        """
        attributes:
            xp: cupy or numpy depending on GPU usage.
            num_m_zero_up (int): Number of modes with :math:`m\geq0`.
            num_m_1_up (int): Number of modes with :math:`m\geq1`.
            num_m0 (int): Number of modes with :math:`m=0`.

        """
        pass

    def __call__(self, teuk_modes, ylms, modeinds, eps=1e-5):
        """Call to sort and filer teukolsky modes.

        This is the call function that takes the teukolsky modes, ylms,
        mode indices and fractional accuracy of the total power and returns
        filtered teukolsky modes and ylms.

        args:
            teuk_modes (2D complex128 xp.ndarray): Complex teukolsky amplitudes
                from the amplitude modules.
                Shape: (number of trajectory points, number of modes).
            ylms (1D complex128 xp.ndarray): Array of ylm values for each mode,
                including m<0. Shape is (num of m==0,) + (num of m>0,)
                + (num of m<0). Number of m<0 and m>0 is the same, but they are
                ordered as (m==0 first then) m>0 then m<0.
            modeinds (list of int xp.ndarrays): List containing the mode index arrays. If in an
                equatorial model, need :math:`(l,m,n)` arrays. If generic,
                :math:`(l,m,k,n)` arrays. e.g. [l_arr, m_arr, n_arr].
            eps (double, optional): Fractional accuracy of the total power used
                to determine the contributing modes. Lowering this value will
                calculate more modes slower the waveform down, but generally
                improving accuracy. Increasing this value removes modes from
                consideration and can have a considerable affect on the speed of
                the waveform, albeit at the cost of some accuracy (usually an
                acceptable loss). Default that gives good mismatch qualities is
                1e-5.

        """
        power = (
            self.xp.abs(
                self.xp.concatenate(
                    [teuk_modes, self.xp.conj(teuk_modes[:, self.m0mask])], axis=1
                )
                * ylms
            )
            ** 2
        )

        inds_sort = self.xp.argsort(power, axis=1)[:, ::-1]
        power = self.xp.sort(power, axis=1)[:, ::-1]
        cumsum = self.xp.cumsum(power, axis=1)

        inds_keep = self.xp.full(cumsum.shape, True)

        inds_keep[:, 1:] = cumsum[:, :-1] < cumsum[:, -1][:, self.xp.newaxis] * (
            1 - eps
        )

        temp = inds_sort[inds_keep]

        temp = temp * (temp < self.num_m_zero_up) + (temp - self.num_m_1_up) * (
            temp >= self.num_m_zero_up
        )

        keep_modes = self.xp.unique(temp)

        # set ylms
        temp2 = keep_modes * (keep_modes < self.num_m0) + (
            keep_modes + self.num_m_1_up
        ) * (keep_modes >= self.num_m0)

        ylmkeep = self.xp.concatenate([keep_modes, temp2])

        out1 = (teuk_modes[:, keep_modes], ylms[ylmkeep])

        out2 = tuple([arr[keep_modes] for arr in modeinds])

        return out1 + out2
