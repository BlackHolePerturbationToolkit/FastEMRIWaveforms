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
from pyinterp_cpu import get_waveform_fd_wrap as get_waveform_fd_wrap_cpu
from pyinterp_cpu import interp_time_for_fd as interp_time_for_fd_cpu

# Python imports
from few.utils.baseclasses import (
    SummationBase,
    SchwarzschildEccentric,
    ParallelModuleBase,
)
from few.utils.citations import *
from few.utils.utility import get_fundamental_frequencies
from few.utils.constants import *
from few.summation.interpolatedmodesum import CubicSplineInterpolant

# Attempt Cython imports of GPU functions
try:
    from pyinterp import (
        interpolate_arrays_wrap,
        get_waveform_wrap,
        get_waveform_fd_wrap,
        interp_time_for_fd,
    )

except (ImportError, ModuleNotFoundError) as e:
    pass

# for special functions
from scipy import special
from scipy.interpolate import CubicSpline
import multiprocessing as mp

# added to return element or none in list
def find_element_in_list(element, list_element):
    try:
        index_element = list_element.index(element)
        return index_element
    except ValueError:
        return None


class FDInterpolatedModeSum(SummationBase, SchwarzschildEccentric, ParallelModuleBase):
    """Create waveform by interpolating sparse trajectory in the frequency domain.

    It interpolates all of the modes of interest and phases at sparse
    trajectories. Within the summation phase, the values are calculated using
    the interpolants and summed.

    This class can be run on GPUs and CPUs.
    """

    def __init__(self, *args, **kwargs):

        ParallelModuleBase.__init__(self, *args, **kwargs)
        SchwarzschildEccentric.__init__(self, *args, **kwargs)
        SummationBase.__init__(self, *args, **kwargs)

        self.kwargs = kwargs

        # eventually the name will change to adapt for the gpu implementantion
        if self.use_gpu:
            self.get_waveform = get_waveform_wrap
            self.get_waveform_fd = get_waveform_fd_wrap
            self.interp_time = interp_time_for_fd

        else:
            self.get_waveform = get_waveform_wrap_cpu
            self.get_waveform_fd = get_waveform_fd_wrap_cpu
            self.interp_time = interp_time_for_fd_cpu

    def attributes_FDInterpolatedModeSum(self):
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
        # TODO: check fftshift

        init_len = len(t)  # length of sparse traj
        num_teuk_modes = teuk_modes.shape[1]
        num_pts = self.num_pts  # from the base clase, adjust in the baseclass

        length = init_len
        # number of quantities to be interp in time domain
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
            self.xp.asarray(Omega_phi / (2 * np.pi * M * MTSUN_SI)),
            self.xp.asarray(Omega_r / (2 * np.pi * M * MTSUN_SI)),
        )

        y_all[-4] = f_phi
        y_all[-3] = f_r

        spline = CubicSplineInterpolant(t, y_all, use_gpu=self.use_gpu)

        # this should be the same
        # plt.plot(t,Omega_phi/(M * MTSUN_SI)); plt.plot(t,spline.deriv(t)[-2],'--' ); plt.show()

        # define a frequency vector

        self.frequency = self.xp.fft.fftshift(
            self.xp.fft.fftfreq(self.num_pts + self.num_pts_pad, dt)
        )

        seg_freqs = m_arr[:, None] * f_phi[None, :] + n_arr[:, None] * f_r[None, :]

        changes = self.xp.concatenate(
            [seg_freqs[:, :-1, None], seg_freqs[:, 1:, None]], axis=-1
        )

        changes_copy = changes.copy()
        changes[:, :, 0] = changes_copy[:, :, 1] * (
            changes_copy[:, :, 0] > changes_copy[:, :, 1]
        ) + changes_copy[:, :, 0] * (changes_copy[:, :, 0] < changes_copy[:, :, 1])

        changes[:, :, 1] = changes_copy[:, :, 0] * (
            changes_copy[:, :, 0] > changes_copy[:, :, 1]
        ) + changes_copy[:, :, 1] * (changes_copy[:, :, 0] < changes_copy[:, :, 1])

        reshape_shape = changes.shape

        df = self.frequency[1] - self.frequency[0]

        inds = np.zeros_like(changes).astype(np.int32)

        inds[:, :, 0] = self.xp.ceil(
            (changes[:, :, 0] - self.frequency[0]) / df
        ).astype(np.int32)
        inds[:, :, 1] = self.xp.floor(
            (changes[:, :, 1] - self.frequency[0]) / df
        ).astype(np.int32)

        # inds[:, :, 0] += 1
        # inds[:, :, 1] -= 1`

        seg_inds = self.xp.zeros_like(seg_freqs, dtype=int)

        freq_mode_start = (
            m_arr[:, None] * spline.c1[-4, None, :]
            + n_arr[:, None] * spline.c1[-3, None, :]
        )

        inds0, inds1 = self.xp.where(
            (self.xp.abs(self.xp.diff(self.xp.sign(freq_mode_start))) == 2.0)
        )

        inds_turnover = self.xp.full(num_teuk_modes, False)
        inds_turnover[inds0] = True
        turnover_seg = self.xp.empty(num_teuk_modes, dtype=np.int32)

        turnover_seg[inds_turnover] = inds1
        turnover_seg[~inds_turnover] = seg_freqs.shape[1]

        turnover_slice = (self.xp.arange(len(turnover_seg)), turnover_seg)

        # find t_star analytically

        bad = turnover_seg >= init_len

        # still ~1 ms

        turnover_seg[bad] = init_len - 1
        c1 = (
            m_arr[:] * spline.c1[-4, turnover_seg]
            + n_arr[:] * spline.c1[-3, turnover_seg]
        )
        c2 = (
            m_arr[:] * spline.c2[-4, turnover_seg]
            + n_arr[:] * spline.c2[-3, turnover_seg]
        )
        c3 = (
            m_arr[:] * spline.c3[-4, turnover_seg]
            + n_arr[:] * spline.c3[-3, turnover_seg]
        )
        slope0 = m_arr[:] * spline.c1[-4, 0] + n_arr[:] * spline.c1[-3, 0]

        ratio = c2 / (3 * c3)
        second_ratio = c1 / (3 * c3)

        # shape is (nmodes, init_len)
        t_star = t[turnover_seg] - ratio + self.xp.sqrt(ratio ** 2 - second_ratio)

        t_star[bad] = t[-1]

        # TODO: don't evaluate everything
        inds_eval = self.xp.searchsorted(t, t_star).astype(self.xp.int32)

        num_modes_here = int(1 / 2 * (ninterps - 4))
        spline_out = self.xp.zeros((int(2 * num_modes_here),))

        run = ~bad

        self.interp_time(
            spline_out,
            t,
            t_star,
            turnover_seg,
            spline.interp_array,
            ninterps,
            length,
            run,
        )

        spline_out = spline_out.reshape(2, num_modes_here)

        spline_out[0, bad] = f_phi[-1]
        spline_out[1, bad] = f_r[-1]

        # 34
        if spline_out.ndim == 1:
            spline_out = spline_out[:, None]

        Fstar = m_arr * spline_out[0] + n_arr * spline_out[1]

        inds_pass_through_zero = self.xp.any(
            self.xp.diff(self.xp.sign(seg_freqs), axis=1) != 0.0, axis=1
        )

        min_freq = seg_freqs.min(axis=1) * (Fstar > seg_freqs.min(axis=1)) + Fstar * (
            Fstar <= seg_freqs.min(axis=1)
        )
        shift_freq = 2 * min_freq
        shift_freq[~inds_pass_through_zero] = 0.0

        fix_negative = (seg_freqs - shift_freq[:, None])[:, 0] < 0.0
        seg_freqs_for_special = self.xp.abs(seg_freqs - shift_freq[:, None])
        Fstar_for_special = self.xp.abs(Fstar - shift_freq)

        temp1 = Fstar_for_special[:, None] + abs(
            seg_freqs_for_special - Fstar_for_special[:, None]
        )
        temp2 = Fstar_for_special[:, None] - abs(
            seg_freqs_for_special - Fstar_for_special[:, None]
        )

        slope0[fix_negative] = slope0[fix_negative] * -1

        special_f_arrs = (
            seg_freqs_for_special
            * ((t[None, :] <= t_star[:, None]) & (slope0[:, None] >= 0.0))
            + (temp1) * ((t[None, :] > t_star[:, None]) & (slope0[:, None] >= 0.0))
            + (
                seg_freqs_for_special[:, 0][:, None]
                + (seg_freqs_for_special[:, 0][:, None] - seg_freqs_for_special)
            )
            * ((t[None, :] <= t_star[:, None]) & (slope0[:, None] < 0.0))
            + (
                seg_freqs_for_special[:, 0][:, None]
                + (seg_freqs_for_special[:, 0][:, None] - temp2)
            )
            * ((t[None, :] > t_star[:, None]) & (slope0[:, None] < 0.0))
        )

        # TODO: this spline sets the turnover slightly off
        special_f_spline = CubicSplineInterpolant(
            special_f_arrs,
            self.xp.tile(t, (special_f_arrs.shape[0], 1)),
            use_gpu=self.use_gpu,
        )

        # fix non-turnover setup
        inds_fix = self.xp.arange(len(turnover_seg))[~bad]

        fixed_inds = ((Fstar[inds_fix] - self.frequency[0]) / df).astype(int)

        fixed_inds = fixed_inds * (slope0[inds_fix] > 0.0) + (fixed_inds + 1) * (
            slope0[inds_fix] < 0.0
        )

        which_to_switch = self.xp.ones_like(fixed_inds) * (
            slope0[inds_fix] > 0.0
        ) + self.xp.zeros_like(fixed_inds) * (slope0[inds_fix] < 0.0)

        inds[(inds_fix, turnover_seg[~bad], which_to_switch)] = fixed_inds

        # final waveform
        h = self.xp.zeros_like(self.frequency, dtype=complex)

        max_points = self.xp.diff(inds, axis=1).max().item()

        inds0_in = inds[:, :, 0].flatten().copy()
        inds1_in = inds[:, :, 1].flatten().copy()

        zero_index = int(len(self.frequency) / 2)

        # 85
        initial_freqs = seg_freqs_for_special[:, 0].copy()
        self.get_waveform_fd(
            h,
            spline.interp_array,
            special_f_spline.interp_array,
            special_f_arrs.T.flatten().copy(),
            m_arr,
            n_arr,
            num_teuk_modes,
            ylms,
            t,
            inds0_in,
            inds1_in,
            init_len,
            self.frequency[0].item(),
            turnover_seg,
            Fstar,
            max_points,
            df,
            self.frequency,
            zero_index,
            shift_freq,
            slope0,
            initial_freqs,
        )

        self.waveform = h

        # 95
