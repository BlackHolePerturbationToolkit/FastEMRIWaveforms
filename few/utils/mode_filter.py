import numpy as np

try:
    import cupy as xp

except (ImportError, ModuleNotFoundError) as e:
    import numpy as xp

import numpy as np


class ModeFilter:
    def __init__(self, m0mask, num_m_zero_up, num_m_1_up, num_m0, use_gpu=False):

        if use_gpu:
            self.xp = xp

        else:
            self.xp = np

        self.m0mask = m0mask
        self.num_m_zero_up, self.num_m_1_up = num_m_zero_up, num_m_1_up
        self.num_m0 = num_m0

    def __call__(self, eps, teuk_modes, ylms, l_arr, m_arr, n_arr):
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

        return (
            teuk_modes[:, keep_modes],
            ylms[ylmkeep],
            l_arr[keep_modes],
            m_arr[keep_modes],
            n_arr[keep_modes],
        )
