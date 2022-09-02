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
from pyinterp_cpu import get_waveform_generic_wrap as get_waveform_tf_generic_wrap_cpu

# Python imports
from ..utils.baseclasses import (
    SummationBase,
    SchwarzschildEccentric,
    ParallelModuleBase,
    GenericWaveform
)
from ..utils.citations import *
from ..utils.utility import get_fundamental_frequencies
from ..utils.constants import *
from .interpolatedmodesum import CubicSplineInterpolant

# Attempt Cython imports of GPU functions
try:
    from pyinterp import get_waveform_tf_generic_wrap

except (ImportError, ModuleNotFoundError) as e:
    pass


class InterpolatedModeSumGenericTF(SummationBase, GenericWaveform, ParallelModuleBase):
    """Create waveform by interpolating sparse trajectory.

    It interpolates all of the modes of interest and phases at sparse
    trajectories. Within the summation phase, the values are calculated using
    the interpolants and summed.

    This class can be run on GPUs and CPUs.

    """

    def __init__(self, *args, **kwargs):

        ParallelModuleBase.__init__(self, *args, **kwargs)
        SummationBase.__init__(self, *args, **kwargs)
        GenericWaveform.__init__(self, *args, **kwargs)

        self.kwargs = kwargs

        if self.use_gpu:
            self.get_waveform = get_waveform_tf_generic_wrap

        else:
            self.get_waveform = get_waveform_tf_generic_wrap_cpu

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
        Phi_phi,
        Phi_theta,
        Phi_r,
        m_arr,
        k_arr,
        n_arr,
        M,
        a,
        p,
        e,
        x,
        *args,
        dt=10.0,
        inds_left_right=-1,
        separate_modes=False,
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
        data_length = self.num_pts

        length = init_len
        ninterps = 2 * self.ndim + 4 * num_teuk_modes  # 4 for re and im of combinations of Slm and Zlmkn
        y_all = self.xp.zeros((ninterps, length))

        # R modes
        y_all[: num_teuk_modes] = teuk_modes[:, :, 0].real.T
        y_all[num_teuk_modes: 2 * num_teuk_modes] = teuk_modes[:, :, 0].imag.T
        # L modes
        y_all[2 * num_teuk_modes: 3 * num_teuk_modes] = teuk_modes[:, :, 1].real.T
        y_all[3 * num_teuk_modes: 4 * num_teuk_modes] = teuk_modes[:, :, 1].imag.T

        y_all[-6] = Phi_phi
        y_all[-5] = Phi_theta
        y_all[-4] = Phi_r

        try:
            p, e, x = p.get(), e.get(), x.get()
        except AttributeError:
            pass
        breakpoint()
        Omega_phi, Omega_theta, Omega_r = get_fundamental_frequencies(
            a, p, e, x
        )

        f_phi, f_theta, f_r = (
            self.xp.asarray(Omega_phi / (2 * self.xp.pi * M * MTSUN_SI)),
            self.xp.asarray(Omega_phi / (2 * self.xp.pi * M * MTSUN_SI)),
            self.xp.asarray(Omega_r / (2 * self.xp.pi * M * MTSUN_SI)),
        )

        y_all[-3] = f_phi
        y_all[-2] = f_theta
        y_all[-1] = f_r

        spline = CubicSplineInterpolant(t, y_all, use_gpu=self.use_gpu)

        # TODO: change this
        
        t_new = self.t_new
        start_t = t_new[0].item()
        if inds_left_right == -1:
            inds_left_right = self.num_per_window

        interval_inds = self.xp.searchsorted(t, t_new, side="right").astype(self.xp.int32) - 1

        waveform = self.xp.zeros_like(self.waveform).flatten()
        include_L = True
        breakpoint()
        get_waveform_tf_generic_wrap(waveform,
            spline.interp_array,
            m_arr.astype(self.xp.int32),
            k_arr.astype(self.xp.int32),
            n_arr.astype(self.xp.int32), num_teuk_modes,
            dt, start_t, t, init_len, data_length, interval_inds, separate_modes,
            self.num_windows, self.num_per_window, inds_left_right, self.num_per_window, include_L)

        
        
        # reshape array if separating modes
        if separate_modes:
            raise NotImplementedError
            self.waveform = self.waveform.reshape(2, num_teuk_modes, -1)
