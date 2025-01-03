# Schwarzschild Eccentric amplitude module for Fast EMRI Waveforms
# performed with bicubic splines

# Copyright (C) 2020 Michael L. Katz, Alvin J.K. Chua, Niels Warburton, Scott A. Hughes
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.


import os

import h5py
import numpy as np

# Cython/C++ imports
from ..cutils.pyInterp2DAmplitude import pyAmplitudeGenerator

# Python imports
from few.utils.baseclasses import SchwarzschildEccentric, AmplitudeBase
from few.utils.utility import check_for_file_download
from few.utils.citations import *

# get path to file
dir_path = os.path.dirname(os.path.realpath(__file__))


class Interp2DAmplitude(AmplitudeBase, SchwarzschildEccentric):
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
        AmplitudeBase.__init__(self, **kwargs)

        # check if you have the necessary file
        # it will download from download.bhptoolkit.org if the user does not have it.
        few_dir = dir_path + "/../../"

        fp = "Teuk_amps_a0.0_lmax_10_nmax_30_new.h5"
        check_for_file_download(fp, few_dir)

        self.amplitude_generator = pyAmplitudeGenerator(self.lmax, self.nmax, few_dir)

    def attributes_Interp2DAmplitude(self):
        """
        attributes:
            amplitude_generator (obj): C++ class that performs the bicubic
                interpolation. It stores all of the splines during initialization
                steps.

        """
        pass

    @property
    def citation(self):
        """Return citations for this class"""
        return larger_few_citation + few_citation + few_software_citation

    def get_amplitudes(self, p, e, *args, specific_modes=None, **kwargs):
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
            specific_modes (list, optional): List of tuples for (l, m, n) values
                desired modes. Default is None.
            **kwargs (dict, placeholder): Added to create flexibility when calling different
                amplitude modules. It is not used.

        returns:
            2D array (double): If specific_modes is None, Teukolsky modes in shape (number of trajectory points, number of modes)
            dict: Dictionary with requested modes.


        """

        input_len = len(p)

        # set the l,m,n arrays
        # if all modes, return modes from the model
        if specific_modes is None:
            l_arr, m_arr, n_arr = (
                self.l_arr[self.m_zero_up_mask],
                self.m_arr[self.m_zero_up_mask],
                self.n_arr[self.m_zero_up_mask],
            )
            try:  # move to CPU if needed before feeding in
                l_arr, m_arr, n_arr = l_arr.get(), m_arr.get(), n_arr.get()
            except AttributeError:
                pass

        # prepare arrays if specific modes are requested
        else:
            l_arr = np.zeros(len(specific_modes), dtype=int)
            m_arr = np.zeros(len(specific_modes), dtype=int)
            n_arr = np.zeros(len(specific_modes), dtype=int)

            # to deal with weird m structure
            inds_revert = []
            for i, (l, m, n) in enumerate(specific_modes):
                l_arr[i] = l
                m_arr[i] = np.abs(m)
                n_arr[i] = n

                if m < 0:
                    inds_revert.append(i)

            inds_revert = np.asarray(inds_revert)

        # interface to C++
        teuk_modes = self.amplitude_generator(
            p,
            e,
            l_arr.astype(np.int32),
            m_arr.astype(np.int32),
            n_arr.astype(np.int32),
            input_len,
            len(l_arr),
        )

        # determine return quantities

        # return array of all modes
        if specific_modes is None:
            return teuk_modes

        # dict containing requested modes
        else:
            temp = {}
            for i, lmn in enumerate(specific_modes):
                temp[lmn] = teuk_modes[:, i]
                l, m, n = lmn

                # apply +/- m symmetry
                if m < 0:
                    temp[lmn] = np.conj(temp[lmn])

            return temp
