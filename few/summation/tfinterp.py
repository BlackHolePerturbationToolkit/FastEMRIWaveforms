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

# Attempt Cython imports of GPU functions
try:
    from pyinterp import interpolate_arrays_wrap, get_waveform_wrap

except (ImportError, ModuleNotFoundError) as e:
    pass


def DirichletKernel(f, T, dt):
    num = np.sin(np.pi * f * T)
    denom = np.sin(Pi * f * dt)

    out = np.ones_like(f)
    inds = denom != 0
    out[inds] = num[inds] / denom[inds]
    return np.exp(-1j * np.pi * f * (T - dt)) * out


def get_DFT(A, n, dt, f, f0=0.0, phi0=0.0):
    T = n * dt
    return (
        A
        * (
            DirichletKernel(f - f0, T, dt) * np.exp(-1j * phi0)
            + DirichletKernel(f + f0, T, dt) * np.exp(1j * phi0)
        )
        / 2
    )


class TFInterpolatedModeSum(SummationBase, SchwarzschildEccentric, GPUModuleBase):
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
        M,
        p,
        e,
        *args,
        dt=10.0,
        num_left_right=-1,
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
        ninterps = (2 * self.ndim) + 2 * num_teuk_modes  # 2 for re and im
        y_all = self.xp.zeros((ninterps, length))

        y_all[:num_teuk_modes] = teuk_modes.T.real
        y_all[num_teuk_modes : 2 * num_teuk_modes] = teuk_modes.T.imag

        y_all[-2] = Phi_phi
        y_all[-1] = Phi_r

        try:
            p, e = p.get(), e.get()
        except AttributeError:
            pass

        Omega_phi, Omega_theta, Omega_r = get_fundamental_frequencies(
            0.0, p, e, np.zeros_like(e)
        )

        f_phi, f_r = (
            Omega_phi / (2 * np.pi * M * MTSUN_SI),
            Omega_r / (2 * np.pi * M * MTSUN_SI),
        )

        y_all[-4] = f_phi
        y_all[-3] = f_r

        spline = CubicSplineInterpolant(t, y_all, use_gpu=self.use_gpu)

        try:
            h_t = t.get()
        except:
            h_t = t

        self.t_new

        breakpoint()
        new_y_all = spline(self.t_new)

        Amps = (
            new_y_all[:num_teuk_modes]
            + 1j * new_y_all[num_teuk_modes : 2 * num_teuk_modes]
        )

        f_phi_new = new_y_all[-4]
        f_r_new = new_y_all[-3]
        Phi_phi_new = new_yall[-2]
        Phi_r_new = new_yall[-1]

        # TODO: (IMPORTANT) Need to think about complex numbers
        # for initial testing
        stft = self.xp.zeros(
            (self.num_windows_for_waveform, self.num_frequencies),
            dtype=self.xp.complex128,
        )
        for i, (Amp_i, f_phi_i, f_r_i, Phi_phi_i, Phi_r_i) in enumerate(
            zip(Amps, f_phi_new, f_r_new, Phi_phi_new, Phi_r_new)
        ):
            for j, (m, n) in enumerate(zip(m_arr, n_arr)):
                freq_mode = m * f_phi_i + n * f_r_i
                amp = Amp_i[j]
                phi0 = m * Phi_phi_i + n * Phi_r_i

                max_bin = int(freq_mode / df)

                if num_left_right < 0:
                    temp_freqs = self.bin_frequencies
                    inds = self.xp.arange(len(bin_frequencies))

                else:
                    inds = np.arange(
                        max_bin - num_left_right, max_bin + num_left_right + 1
                    )
                    inds = inds[(inds >= 0) & (inds < self.num_frequencies - 1)]
                    temp_freqs = self.bin_frequencies[inds]

                temp_DFT = get_DFT(
                    amp, self.num_per_window, dt, temp_freqs, f0=freq_mode, phi0=phi0
                )
                stft[i, inds] += temp_DFT

        self.waveform[
            : self.num_windows_for_waveform * len(bin_frequencies)
        ] = stft.flatten()
