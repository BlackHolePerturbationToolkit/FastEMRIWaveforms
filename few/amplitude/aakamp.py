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

# Cython/C++ imports


# Python imports
from few.utils.baseclasses import Pn5AAK, AmplitudeBase
from few.utils.citations import *
from few.utils.utility import get_fundamental_frequencies
from few.utils.constants import *
from pyParameterMap import pyParMap, pyWaveform

# get path to file
dir_path = os.path.dirname(os.path.realpath(__file__))


class AmplitudeAAK(Pn5AAK, AmplitudeBase):
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
        AmplitudeBase.__init__(self, **kwargs)

        # self.amplitude_generator = pyAmplitudeGenerator(self.lmax, self.nmax, few_dir)

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
        return few_citation + Pn5_citation

    def get_amplitudes(self, M, a, p, e, Y, *args, mich=False, **kwargs):
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

        (tvec, Phi_phi, Phi_theta, Phi_r, mu, qS, phiS, qK, phiK, dist, nmodes) = args

        Phivec = Phi_r
        gimvec = Phi_theta - Phi_r
        alpvec = Phi_phi - Phi_theta

        # TODO: check if this should be in radians or revs
        # TODO: check nuvec relation
        # TODO: check if these should be related to evolving/mapped spin values
        nuvec = OmegaR / (2 * PI)
        gimdotvec = OmegaTheta - OmegaR

        # TODO: no evolution on iota in AAK, therefore lam constant

        hI = self.xp.zeros(len(tvec))
        hII = self.xp.zeros(len(tvec))

        # lam is iota0
        lam = iota[0]

        pvec, evec = (self.xp.asarray(p), self.xp.asarray(e))
        Phivec = self.xp.asarray(Phivec)
        gimvec = self.xp.asarray(gimvec)
        alpvec = self.xp.asarray(alpvec)
        nuvec = self.xp.asarray(nuvec)
        gimdotvec = self.xp.asarray(gimdotvec)
        tvec = self.xp.asarray(tvec)
        length = len(tvec)
        OmegaPhi_spin_mapped = self.xp.asarray(OmegaPhi_spin_mapped)

        pyWaveform(
            hI,
            hII,
            tvec,
            evec,
            pvec,  # vvec
            gimvec,
            Phivec,
            alpvec,
            nuvec,
            gimdotvec,
            OmegaPhi_spin_mapped,
            M,
            mu,
            lam,
            qS,
            phiS,
            qK,
            phiK,
            dist,
            length,
            nmodes,
            mich,
        )

        return (hI, hII)

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
        """
