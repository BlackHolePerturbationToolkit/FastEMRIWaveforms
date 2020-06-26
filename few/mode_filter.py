import numpy as np

try:
    import cupy as xp

except ImportError:
    import numpy as xp


class ModeFilter:
    def __init__(self, m0mask, num_m_zero_up, num_m_1_up, num_m0):
        self.m0mask = m0mask
        self.num_m_zero_up, self.num_m_1_up = num_m_zero_up, num_m_1_up
        self.num_m0 = num_m0

    def __call__(self, eps, teuk_modes, ylms, l_arr, m_arr, n_arr):
        power = (
            xp.abs(
                xp.concatenate(
                    [teuk_modes, xp.conj(teuk_modes[:, self.m0mask])], axis=1
                )
                * ylms
            )
            ** 2
        )

        inds_sort = xp.argsort(power, axis=1)[:, ::-1]
        power = xp.sort(power, axis=1)[:, ::-1]
        cumsum = xp.cumsum(power, axis=1)

        inds_keep = xp.full(cumsum.shape, True)

        inds_keep[:, 1:] = cumsum[:, :-1] < cumsum[:, -1][:, xp.newaxis] * (1 - eps)

        temp = inds_sort[inds_keep]

        temp = temp * (temp < self.num_m_zero_up) + (temp - self.num_m_1_up) * (
            temp >= self.num_m_zero_up
        )

        keep_modes = xp.unique(temp)

        # set ylms
        temp2 = keep_modes * (keep_modes < self.num_m0) + (
            keep_modes + self.num_m_1_up
        ) * (keep_modes >= self.num_m0)

        ylmkeep = xp.concatenate([keep_modes, temp2])

        return (
            teuk_modes[:, keep_modes],
            ylms[ylmkeep],
            l_arr[keep_modes],
            m_arr[keep_modes],
            n_arr[keep_modes],
        )
