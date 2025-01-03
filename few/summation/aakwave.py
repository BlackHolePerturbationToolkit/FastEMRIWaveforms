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
import warnings

import h5py
import numpy as np

# try to import cupy
try:
    import cupy as cp

except (ImportError, ModuleNotFoundError) as e:
    import numpy as np

# Cython/C++ imports
from ..cutils.pycpuAAK import pyWaveform as pyWaveform_cpu
from ..cutils.pyParameterMap import pyParMap

# Attempt Cython imports of GPU functions
try:
    from ..cutils.pygpuAAK import pyWaveform as pyWaveform_gpu

except (ImportError, ModuleNotFoundError) as e:
    pass

# Python imports
from few.utils.baseclasses import Pn5AAK, SummationBase, ParallelModuleBase
from few.utils.citations import *
from few.utils.utility import get_fundamental_frequencies, Y_to_xI
from few.utils.constants import *
from few.summation.interpolatedmodesum import CubicSplineInterpolant

# get path to file
dir_path = os.path.dirname(os.path.realpath(__file__))


class AAKSummation(SummationBase, Pn5AAK, ParallelModuleBase):
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
        ParallelModuleBase.__init__(self, **kwargs)
        Pn5AAK.__init__(self, **kwargs)
        SummationBase.__init__(self, **kwargs)

        if self.use_gpu:
            self.waveform_generator = pyWaveform_gpu

        else:
            self.waveform_generator = pyWaveform_cpu

    @property
    def gpu_capability(self):
        return True

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
            larger_few_citation
            + few_citation
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
        dist,
        Phi_phi,
        Phi_theta,
        Phi_r,
        mu,
        qS,
        phiS,
        qK,
        phiK,
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

        **Please note:** the 5PN trajectory and AAK waveform take the parameter
        :math:`Y\equiv\cos{\iota}=L/\sqrt{L^2 + Q}` rather than :math:`x_I` as is accepted
        for relativistic waveforms and in the generic waveform interface discussed above.
        The generic waveform interface directly converts :math:`x_I` to :math:`Y`.

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
                :math:`\cos{\iota}`. **Note**: This value is different from :math:`x_I`
                used in the relativistic waveforms.
            dist (double): Luminosity distance in Gpc.
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

        xp = cp if self.use_gpu else np

        # mass in seconds
        Msec = M * MTSUN_SI

        # get inclination for mapping
        iota = np.arccos(Y)

        # convert Y to x_I for fund freqs
        xI = Y_to_xI(a, p, e, Y)

        # these are dimensionless and in radians
        OmegaPhi, OmegaTheta, OmegaR = get_fundamental_frequencies(a, p, e, xI)

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

        fill_val = 1e-6
        if np.any(
            (lam > np.pi - fill_val)
            | (lam < fill_val)
            | (np.abs(lam - (np.pi / 2.0)) < fill_val)
        ):
            warnings.warn(
                "Inclination trajectory includes values within 1e-6 of the poles. We shift these values automatically away from poles by 1e-6."
            )
            inds_fix_up = lam > np.pi - fill_val
            lam[inds_fix_up] = np.pi - fill_val

            inds_fix_up = lam < fill_val
            lam[inds_fix_up] = fill_val

            inds_fix = (np.abs(lam - (np.pi / 2.0)) < fill_val) & (lam > np.pi / 2.0)
            lam[inds_fix] = np.pi / 2.0 + fill_val

            inds_fix = (np.abs(lam - (np.pi / 2.0)) < fill_val) & (lam < np.pi / 2.0)
            lam[inds_fix] = np.pi / 2.0 - fill_val

        if qK < fill_val or qK > np.pi - fill_val:
            warnings.warn(
                "qK is within 1e-6 of the poles. We shift this value automatically away from poles by 1e-6."
            )
            if qK < fill_val:
                qK = fill_val
            else:
                qK = np.pi - fill_val

        if qS < fill_val or qS > np.pi - fill_val:
            warnings.warn(
                "qS is within 1e-6 of the poles. We shift this value automatically away from poles by 1e-6."
            )
            if qS < fill_val:
                qS = fill_val
            else:
                qS = np.pi - fill_val

        # convert to gpu if desired
        tvec_temp = xp.asarray(tvec)
        init_len = len(tvec)

        # setup interpolation
        ninterps = 8
        y_all = xp.zeros((ninterps, init_len))

        # do not need p anymore since we are inputing OmegaPhi

        # fill y_all with all arrays that need interpolation
        y_all[0] = xp.asarray(e)
        y_all[1] = xp.asarray(Phivec)
        y_all[2] = xp.asarray(gimvec)
        y_all[3] = xp.asarray(alpvec)
        y_all[4] = xp.asarray(nuvec)
        y_all[5] = xp.asarray(gimdotvec)
        y_all[6] = xp.asarray(OmegaPhi)
        y_all[7] = xp.asarray(lam)

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
