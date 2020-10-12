# Schwarzschild Eccentric amplitude module for Fast EMRI Waveforms
# performed with bicubic splines
# TODO: fix this
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

# try to import cupy
try:
    import cupy as xp

except (ImportError, ModuleNotFoundError) as e:
    import numpy as xp

# Cython/C++ imports
from pycpuAAK import pyWaveform as pyWaveform_cpu

# Attempt Cython imports of GPU functions
try:
    from pygpuAAK import pyWaveform as pyWaveform_gpu

except (ImportError, ModuleNotFoundError) as e:
    pass

# Python imports
from few.utils.baseclasses import Pn5AAK, SummationBase
from few.utils.citations import *
from few.utils.utility import get_fundamental_frequencies
from few.utils.constants import *
from pyParameterMap import pyParMap
from few.summation.interpolatedmodesum import CubicSplineInterpolant

# get path to file
dir_path = os.path.dirname(os.path.realpath(__file__))


class AAKSumation(SummationBase, Pn5AAK):
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

        Pn5AAK.__init__(self, **kwargs)
        SummationBase.__init__(self, **kwargs)

        if self.use_gpu:
            self.waveform_generator = pyWaveform_gpu

        else:
            self.waveform_generator = pyWaveform_cpu

    def attributes_AmplitudeAAK(self):
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
        return (
            few_citation + AAK_citation_1 + AAK_citation_2 + AK_citation + NK_citation
        )

    def sum(
        self,
        tvec,
        M,
        a,
        p,
        e,
        Y,
        Phi_phi,
        Phi_theta,
        Phi_r,
        mu,
        qS,
        phiS,
        qK,
        phiK,
        dist,
        nmodes,
        *args,
        mich=False,
        dt=10.0,
        **kwargs
    ):
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

        Msec = M * MTSUN_SI

        iota = np.arccos(Y)

        # these are dimensionless and in radians
        OmegaPhi, OmegaTheta, OmegaR = get_fundamental_frequencies(a, p, e, Y)

        OmegaPhi, OmegaTheta, OmegaR = (
            OmegaPhi / Msec,
            OmegaTheta / Msec,
            OmegaR / Msec,
        )
        # convert to rotations per sec as in AAK parameter mapping
        OmegaPhi_mapping, OmegaTheta_mapping, OmegaR_mapping = (
            2 * np.pi * OmegaPhi,
            2 * np.pi * OmegaTheta,
            2 * np.pi * OmegaR,
        )

        v_map, M_map, S_map = pyParMap(
            OmegaPhi_mapping, OmegaTheta_mapping, OmegaR_mapping, p, e, iota, M, a
        )

        # get OmegaPhi with the mapped spin values rather than constant
        # TODO: check if this should be done dimensionalized with mapped masses as well
        OmegaPhi_spin_mapped = get_fundamental_frequencies(S_map, p, e, Y)[0] / (
            M_map * MTSUN_SI
        )

        Phivec = Phi_r
        gimvec = Phi_theta - Phi_r
        alpvec = Phi_phi - Phi_theta

        # TODO: check if this should be in radians or revs
        # TODO: check nuvec relation
        # TODO: check if these should be related to evolving/mapped spin values
        nuvec = OmegaR / (2 * PI)
        gimdotvec = OmegaTheta - OmegaR

        # TODO: no evolution on iota in AAK, therefore lam constant

        # lam is iota0
        lam = iota[0]

        tvec_temp = self.xp.asarray(tvec)
        init_len = len(tvec)

        ninterps = 7
        y_all = self.xp.zeros((ninterps, init_len))

        # do not need p anymore since we are inputing OmegaPhiMapped
        # y_all[0] = self.xp.asarray(p)

        y_all[0] = self.xp.asarray(e)
        y_all[1] = self.xp.asarray(Phivec)
        y_all[2] = self.xp.asarray(gimvec)
        y_all[3] = self.xp.asarray(alpvec)
        y_all[4] = self.xp.asarray(nuvec)
        y_all[5] = self.xp.asarray(gimdotvec)
        y_all[6] = self.xp.asarray(OmegaPhi_spin_mapped)

        self.spline = CubicSplineInterpolant(tvec_temp, y_all, use_gpu=self.use_gpu)

        self.waveform_generator(
            self.waveform,
            self.spline.interp_array,
            M,
            mu,
            lam,
            qS,
            phiS,
            qK,
            phiK,
            dist,
            nmodes,
            mich,
            init_len,
            self.num_pts,
            dt,
            tvec,
        )

        return
