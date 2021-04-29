# Interpolated summation of modes in python for the FastEMRIWaveforms Package

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

import warnings

import numpy as np

# try to import cupy
try:
    import cupy as xp

except (ImportError, ModuleNotFoundError) as e:
    import numpy as xp

# Cython imports
from pyinterp_cpu import interpolate_arrays_wrap as interpolate_arrays_wrap_cpu
from pyinterp_cpu import get_waveform_wrap as get_waveform_wrap_cpu

# Python imports
from few.utils.baseclasses import SummationBase, SchwarzschildEccentric, GPUModuleBase
from few.utils.citations import *
from few.utils.utility import get_fundamental_frequencies
from few.utils.constants import *

# Attempt Cython imports of GPU functions
try:
    from pyinterp import interpolate_arrays_wrap, get_waveform_wrap

except (ImportError, ModuleNotFoundError) as e:
    pass


class CubicSplineInterpolant(GPUModuleBase):
    """GPU-accelerated Multiple Cubic Splines

    This class produces multiple cubic splines on a GPU. It has a CPU option
    as well. The cubic splines are produced with "not-a-knot" boundary
    conditions.

    This class can be run out of Python similar to
    `scipy.interpolate.CubicSpline <https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.CubicSpline.html#scipy-interpolate-cubicspline>`_.
    However, the most efficient way to use this method is in a customized
    cuda kernel. See the
    `source code for the interpolated summation<https://github.com/BlackHolePerturbationToolkit/FastEMRIWaveforms/blob/master/src/interpolate.cu>`_
    in cuda for an example of this.

    This class can be run on GPUs and CPUs.

    args:
        t (1D double xp.ndarray): t values as input for the spline.
        y_all (1D or 2D double xp.ndarray): y values for the spline.
            Shape: (length,) or (ninterps, length).
        use_gpu (bool, optional): If True, prepare arrays for a GPU. Default is
            False.

    """

    def __init__(self, t, y_all, use_gpu=False):

        GPUModuleBase.__init__(self, use_gpu=use_gpu)

        if use_gpu:
            self.interpolate_arrays = interpolate_arrays_wrap

        else:
            self.interpolate_arrays = interpolate_arrays_wrap_cpu

        y_all = self.xp.atleast_2d(y_all)

        # hardcoded,only cubic spline is available
        self.degree = 3

        # get quantities related to how many interpolations
        ninterps, length = y_all.shape

        # store this for reshaping flattened arrays
        self.reshape_shape = (self.degree + 1, ninterps, length)

        # interp array is (y, c1, c2, c3)
        interp_array = self.xp.zeros(self.reshape_shape)

        # fill y
        interp_array[0] = y_all

        interp_array = interp_array.flatten()

        # arrays to store banded matrix and solution
        B = self.xp.zeros((ninterps * length,))
        upper_diag = self.xp.zeros_like(B)
        diag = self.xp.zeros_like(B)
        lower_diag = self.xp.zeros_like(B)

        self.t = t.astype(xp.float64)
        # perform interpolation
        self.interpolate_arrays(
            self.t, interp_array, ninterps, length, B, upper_diag, diag, lower_diag,
        )

        # set up storage of necessary arrays

        self.interp_array = self.xp.transpose(
            interp_array.reshape(self.reshape_shape), [0, 2, 1]
        ).flatten()

        # update reshape_shape
        self.reshape_shape = (self.degree + 1, length, ninterps)

    def attributes_CubicSplineInterpolate(self):
        """
        attributes:
            interpolate_arrays (func): CPU or GPU function for mode interpolation.
            interp_array (1D double xp.ndarray): Array containing all spline
                coefficients. It is flattened after fitting from shape
                (4, length, ninterps). The 4 is the 4 spline coefficients.

        """

    @property
    def gpu_capability(self):
        """Confirms GPU capability"""
        return True

    @property
    def citation(self):
        """Return the citation for this class"""
        return larger_few_citation + few_citation + few_software_citation

    @property
    def y(self):
        """y values associated with the spline"""
        return self.interp_array.reshape(self.reshape_shape)[0].T

    @property
    def c1(self):
        """constants for the linear term"""
        return self.interp_array.reshape(self.reshape_shape)[1].T

    @property
    def c2(self):
        """constants for the quadratic term"""
        return self.interp_array.reshape(self.reshape_shape)[2].T

    @property
    def c3(self):
        """constants for the cubic term"""
        return self.interp_array.reshape(self.reshape_shape)[3].T

    def __call__(self, tnew):
        """Evaluation function for the spline

        Put in an array of new t values at which all interpolants will be
        evaluated. If t values are outside of the original t values, edge values
        are used to fill the new array at these points.

        args:
            tnew (1D double xp.ndarray): Array of new t values. All of these new
                t values must be within the bounds of the input t values,
                including the beginning t and **excluding** the ending t.

        raises:
            ValueError: a new t value is not in the bounds of the input t array.

        """

        # find were in the old t array the new t values split
        inds = self.xp.searchsorted(self.t, tnew, side="right") - 1

        # get values outside the edges
        inds_bad_left = tnew < self.t[0]
        inds_bad_right = tnew > self.t[-1]

        if np.any(inds < 0) or np.any(inds >= len(self.t)):
            warnings.warn(
                "New t array outside bounds of input t array. These points are filled with edge values."
            )

        x = tnew - self.t[inds]
        x2 = x * x
        x3 = x2 * x

        out = (
            self.y[:, inds]
            + self.c1[:, inds] * x
            + self.c2[:, inds] * x2
            + self.c3[:, inds] * x3
        )

        # fix bad values
        if self.xp.any(inds_bad_left):
            out[:, inds_bad_left] = self.y[:, 0]

        if self.xp.any(inds_bad_right):
            out[:, inds_bad_right] = self.y[:, -1]
        return out.squeeze()

    def deriv(self, tnew, order=1):
        """derivative of the spline
        """
        if order == 0:
            return self(tnew)
        elif order > 0 and order <= 3:
            return getattr(self, "_d" + str(order))(tnew)
        else:
            raise ValueError("order parameter must be 0, 1, 2 or 3.")

    def _d1(self, tnew):
        inds = self.xp.searchsorted(self.t, tnew)

        x = tnew - self.t[inds]
        x2 = x * x

        out = (
            self.c1[:, inds] + 2.0 * self.c2[:, inds] * x + 3.0 * self.c3[:, inds] * x2
        )
        return out

    def _d2(self, tnew):
        inds = self.xp.searchsorted(self.t, tnew)

        x = tnew - self.t[inds]

        out = 2.0 * self.c2[:, inds] + 6.0 * self.c3[:, inds] * x
        return out

    def _d3(self, tnew):
        inds = self.xp.searchsorted(self.t, tnew)
        out = 6.0 * self.c3[:, inds]
        return out


class InterpolatedModeSum(SummationBase, SchwarzschildEccentric, GPUModuleBase):
    """Create waveform by interpolating sparse trajectory.

    It interpolates all of the modes of interest and phases at sparse
    trajectories. Within the summation phase, the values are calculated using
    the interpolants and summed.

    This class can be run on GPUs and CPUs.

    """

    def __init__(self, *args, **kwargs):

        GPUModuleBase.__init__(self, *args, **kwargs)
        SchwarzschildEccentric.__init__(self, *args, **kwargs)
        SummationBase.__init__(self, *args, **kwargs)

        self.kwargs = kwargs

        if self.use_gpu:
            self.get_waveform = get_waveform_wrap

        else:
            self.get_waveform = get_waveform_wrap_cpu

    def attributes_InterpolatedModeSum(self):
        """
        attributes:
            get_waveform (func): CPU or GPU function for waveform creation.

        """

    @property
    def gpu_capability(self):
        """Confirms GPU capability"""
        return True

    @property
    def citation(self):
        return larger_few_citation + few_citation + few_software_citation

    def sum(
        self,
        t,
        teuk_modes,
        ylms,
        Phi_phi,
        Phi_r,
        m_arr,
        n_arr,
        *args,
        dt=10.0,
        **kwargs,
    ):
        """Interpolated summation function.

        This function interpolates the amplitude and phase information, and
        creates the final waveform with the combination of ylm values for each
        mode.

        args:
            t (1D double xp.ndarray): Array of t values.
            teuk_modes (2D double xp.array): Array of complex amplitudes.
                Shape: (len(t), num_teuk_modes).
            ylms (1D complex128 xp.ndarray): Array of ylm values for each mode,
                including m<0. Shape is (num of m==0,) + (num of m>0,)
                + (num of m<0). Number of m<0 and m>0 is the same, but they are
                ordered as (m==0 first then) m>0 then m<0.
            Phi_phi (1D double xp.ndarray): Array of azimuthal phase values
                (:math:`\Phi_\phi`).
            Phi_r (1D double xp.ndarray): Array of radial phase values
                 (:math:`\Phi_r`).
            m_arr (1D int xp.ndarray): :math:`m` values associated with each mode.
            n_arr (1D int xp.ndarray): :math:`n` values associated with each mode.
            *args (list, placeholder): Added for future flexibility.
            dt (double, optional): Time spacing between observations (inverse of sampling
                rate). Default is 10.0.
            **kwargs (dict, placeholder): Added for future flexibility.

        """

        init_len = len(t)
        num_teuk_modes = teuk_modes.shape[1]
        num_pts = self.num_pts

        length = init_len
        ninterps = self.ndim + 2 * num_teuk_modes  # 2 for re and im
        y_all = self.xp.zeros((ninterps, length))

        y_all[:num_teuk_modes] = teuk_modes.T.real
        y_all[num_teuk_modes : 2 * num_teuk_modes] = teuk_modes.T.imag

        y_all[-2] = Phi_phi
        y_all[-1] = Phi_r

        spline = CubicSplineInterpolant(t, y_all, use_gpu=self.use_gpu)

        try:
            h_t = t.get()
        except:
            h_t = t

        # the base class function __call__ will return the waveform
        self.get_waveform(
            self.waveform,
            spline.interp_array,
            m_arr,
            n_arr,
            init_len,
            num_pts,
            num_teuk_modes,
            ylms,
            dt,
            h_t,
        )
