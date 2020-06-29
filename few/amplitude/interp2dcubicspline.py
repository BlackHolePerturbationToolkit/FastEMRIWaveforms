import numpy as np
import os
import h5py

from few.utils.baseclasses import SchwarzschildEccentric

from pyInterp2DAmplitude import Interp2DAmplitude_wrap, pyAmplitudeCarrier


class Interp2DAmplitude(SchwarzschildEccentric):
    def __init__(self, num_teuk_modes=3843, lmax=10, nmax=30):

        self.amplitude_carrier = pyAmplitudeCarrier(lmax, nmax)
        self.num_modes = num_teuk_modes

    def __call__(self, p, e, l_arr, m_arr, n_arr):

        input_len = len(p)
        teuk_modes = Interp2DAmplitude_wrap(
            p,
            e,
            l_arr.astype(np.int32),
            m_arr.astype(np.int32),
            n_arr.astype(np.int32),
            input_len,
            self.num_modes,
            self.amplitude_carrier,
        )
        return teuk_modes
