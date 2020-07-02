import numpy as np
import os
import h5py

from few.utils.baseclasses import SchwarzschildEccentric

from pyInterp2DAmplitude import Interp2DAmplitude_wrap, pyAmplitudeCarrier

import os

dir_path = os.path.dirname(os.path.realpath(__file__))


class Interp2DAmplitude(SchwarzschildEccentric):
    """Calculate Teukolsky amplitudes by 2D Cubic Spline interpolation.

    Please see the documentations for
    :class:`few.utils.baseclasses.SchwarzschildEccentric`
    for overall aspects of these models.

    Each mode is setup with a 2D cubic spline interpolant. When the user
    inputs :math:`(p,e)`, the interpolatant determines the corresponding
    amplitudes for each mode in the model.

    args:
        **kwargs (dict, optional): Keyword arguments for the base class:
            :class:`few.utils.baseclasses.SchwarzschildEccentric`. Default is
            {}.

    """

    def __init__(self, **kwargs):

        SchwarzschildEccentric.__init__(self, **kwargs)

        few_dir = dir_path + "/../../"
        self.amplitude_carrier = pyAmplitudeCarrier(self.lmax, self.nmax, few_dir)

    def __call__(self, p, e, l_arr, m_arr, n_arr, *args, **kwargs):
        """Calculate Teukolsky amplitudes for Schwarzschild eccentric.

        This function takes the inputs the trajectory in :math:`(p,e)` as arrays
        and returns the complex amplitude of all modes to adiabatic order at
        each step of the trajectory.

        args:
            p (1D double numpy.ndarray): Array containing the trajectory for values of
                the semi-latus rectum.
            e (1D double numpy.ndarray): Array containing the trajectory for values of
                the eccentricity.
            l_arr (1D int numpy.ndarray): :math:`l` values to evaluate.
            m_arr (1D int numpy.ndarray): :math:`m` values to evaluate.
            n_arr (1D int numpy.ndarray): :math:`ns` values to evaluate.
            *args (tuple, placeholder): Added to create flexibility when calling different
                amplitude modules. It is not used.
            **kwargs (dict, placeholder): Added to create flexibility when calling different
                amplitude modules. It is not used.


        """

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
