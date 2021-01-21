# AAK summation module for Fast EMRI Waveforms
#
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
from pyParameterMap import pyParMap

# Attempt Cython imports of GPU functions
try:
    from pygpuAAK import pyWaveform as pyWaveform_gpu

except (ImportError, ModuleNotFoundError) as e:
    pass

# Python imports
from few.utils.baseclasses import Pn5AAK, SummationBase, GPUModuleBase
from few.utils.citations import *
from few.utils.utility import get_fundamental_frequencies
from few.utils.constants import *
from few.summation.interpolatedmodesum import CubicSplineInterpolant

# get path to file
dir_path = os.path.dirname(os.path.realpath(__file__))


class AAKSummation(SummationBase, Pn5AAK, GPUModuleBase):
    """Calculate an AAK waveform from an input trajectory.

    Please see the documentations for
    :class:`few.waveform.Pn5AAKWaveform`
    for overall aspects of this model.

    Given an input trajectory and other parameters, this module maps that
    trajectory to the Analytic Kludge basis as performed for the Augmented
    Analytic Kludge model. Please see
    `the AAK paper <https://arxiv.org/abs/1510.06245`_ for more information.

    args:
        **kwargs (dict, optional): Keyword arguments for the base class:
            :class:`few.utils.baseclasses.SchwarzschildEccentric`. Default is
            {}.

    """

    def __init__(self, **kwargs):

        GPUModuleBase.__init__(self, **kwargs)
        Pn5AAK.__init__(self, **kwargs)
        SummationBase.__init__(self, **kwargs)

        if self.use_gpu:
            self.waveform_generator = pyWaveform_gpu

        else:
            self.waveform_generator = pyWaveform_cpu

    def attributes_AmplitudeAAK(self):
        """
        attributes:
            waveform_generator (obj): C++ class that performs the AAK calculation
                to create the waveform.
            spline (obj): Cubic spline class that holds the coefficients for
                all splines of the arrays necessary for the AAK calculation.

        """
        pass

    @property
    def citation(self):
        """Return citations for this class"""
        return (
            few_citation
            + few_software_citation
            + AAK_citation_1
            + AAK_citation_2
            + AK_citation
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
            tvec (1D double numpy.ndarray): Array containing the time values
                associated with the sparse trajectory.
            M (double): Mass of massive black hole in solar masses.
            a (double): Dimensionless spin of massive black hole.
            p (1D double numpy.ndarray): Array containing the trajectory for values of
                the semi-latus rectum.
            e (1D double numpy.ndarray): Array containing the trajectory for values of
                the eccentricity.
            Y (1D double numpy.ndarray): Array containing the trajectory for values of
                the cosine of the inclination.
            Phi_phi (1D double numpy.ndarray): Array containing the trajectory for
                :math:`\Phi_\phi`.
            Phi_theta (1D double numpy.ndarray): Array containing the trajectory for
                :math:`\Phi_\theta`.
            Phi_r (1D double numpy.ndarray): Array containing the trajectory for
                :math:`\Phi_r`.
            mu (double): Mass of compact object in solar masses.
            qS (double): Sky location polar angle in ecliptic
                coordinates.
            phiS (double): Sky location azimuthal angle in
                ecliptic coordinates.
            qK (double): Initial BH spin polar angle in ecliptic
                coordinates.
            phiK (double): Initial BH spin azimuthal angle in
                ecliptic coordinates.
            dist (double): Luminosity distance in Gpc.
            nmodes (int): Number of modes to analyze. This is determined by
                the eccentricity.
            *args (tuple, placeholder): Added to create flexibility when calling different
                amplitude modules. It is not used.
            mich (bool, optional): If True, produce waveform with
                long-wavelength response approximation (hI, hII). Please
                note this is not TDI. If False, return hplus and hcross.
                Default is False.
            dt (double, optional): Time between samples in seconds
                (inverse of sampling frequency). Default is 10.0.
            **kwargs (dict, placeholder): Added to create flexibility when calling different
                amplitude modules. It is not used.

        """

        # mass in seconds
        Msec = M * MTSUN_SI

        # get inclination for mapping
        iota = np.arccos(Y)

        # these are dimensionless and in radians
        OmegaPhi, OmegaTheta, OmegaR = get_fundamental_frequencies(a, p, e, Y)

        # dimensionalize the frequencies
        OmegaPhi, OmegaTheta, OmegaR = (
            OmegaPhi / Msec,
            OmegaTheta / Msec,
            OmegaR / Msec,
        )

        # convert phases to AK basis
        Phivec = Phi_r
        gimvec = Phi_theta - Phi_r
        alpvec = Phi_phi - Phi_theta

        nuvec = OmegaR / (2 * PI)
        gimdotvec = OmegaTheta - OmegaR

        # lam in the code is iota
        lam = iota

        # make sure same sky frame as few
        # qK = np.pi / 2.0 - qK

        # convert to gpu if desired
        tvec_temp = self.xp.asarray(tvec)
        init_len = len(tvec)

        # setup interpolation
        ninterps = 8
        y_all = self.xp.zeros((ninterps, init_len))

        # do not need p anymore since we are inputing OmegaPhi

        # fill y_all with all arrays that need interpolation
        y_all[0] = self.xp.asarray(e)
        y_all[1] = self.xp.asarray(Phivec)
        y_all[2] = self.xp.asarray(gimvec)
        y_all[3] = self.xp.asarray(alpvec)
        y_all[4] = self.xp.asarray(nuvec)
        y_all[5] = self.xp.asarray(gimdotvec)
        y_all[6] = self.xp.asarray(OmegaPhi)
        y_all[7] = self.xp.asarray(lam)

        # get all cubic splines
        self.spline = CubicSplineInterpolant(tvec_temp, y_all, use_gpu=self.use_gpu)

        # generator the waveform
        self.waveform_generator(
            self.waveform,
            self.spline.interp_array,
            M,
            a,
            mu,
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
