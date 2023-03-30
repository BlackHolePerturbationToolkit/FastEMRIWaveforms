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
from pyinterp_cpu import find_segments_fd as find_segments_fd_cpu
from pyinterp_cpu import find_segments_fd as find_segments_fd_cpu
from pyinterp_cpu import get_waveform_generic_fd_wrap as get_waveform_generic_fd_wrap_cpu

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
        find_segments_fd,
        get_waveform_generic_fd_wrap
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


def searchsorted2d_vec(a,b, batch_size=-1, xp=None, **kwargs):

    if xp is None:
        xp = np

    if a.ndim == 1:
        if b.ndim > 1:
            reshape = b.shape
            b = b.flatten()
        else:
            reshape = None
        out = xp.searchsorted(a, b, **kwargs)

        if reshape is not None:
            out = out.reshape(reshape)
        return out

    elif a.ndim > 2 or b.ndim > 2:
        raise ValueError("Input arrays must not be more than 2 dimensions.")

    if b.ndim == 1:
        b = xp.expand_dims(b, (0,))

    if batch_size < 0:
            inds_split_all = [xp.arange(a.shape[0])]
    else:
        split_inds = []
        i = 0
        while i < a.shape[0]:
            i += batch_size
            if i >= a.shape[0]:
                break
            split_inds.append(i)

        inds_split_all = xp.split(xp.arange(a.shape[0]), split_inds)

    # select tqdm if user wants to see progress
    iterator = enumerate(inds_split_all)
    #iterator = tqdm(iterator, desc="time batch") if show_progress else iterator

    out = xp.zeros((a.shape[0], b.shape[1]))
    for i, inds_in in iterator:
        # get subsections of the arrays for each batch
        a_temp = a[inds_in]
        if b.shape[0] > 1:
            b_temp = b[inds_in]
        else:
            b_temp = b

        m,n = a_temp.shape
        max_num = xp.maximum(a_temp.max() - a_temp.min(), b_temp.max() - b_temp.min()) + 1
        r = max_num*xp.arange(a_temp.shape[0])[:,None]
        p = xp.searchsorted( (a_temp+r).ravel(), (b_temp+r).ravel(), **kwargs).reshape(m,-1)
        out[inds_in] = p - n*(xp.arange(m)[:,None])
    
    return out


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
            self.get_waveform_fd = get_waveform_generic_fd_wrap
            self.interp_time = interp_time_for_fd
            self.find_segments = find_segments_fd

        else:
            self.get_waveform = get_waveform_wrap_cpu
            self.get_waveform_fd = get_waveform_generic_fd_wrap_cpu
            self.interp_time = interp_time_for_fd_cpu
            self.find_segments = find_segments_fd_cpu

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
        include_minus_m=True,
        separate_modes=False,
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

        y_all_freqs = self.xp.asarray([f_phi, f_r])
        freqs_spline = CubicSplineInterpolant(t, y_all_freqs, use_gpu=self.use_gpu)
        spline = CubicSplineInterpolant(t, y_all, use_gpu=self.use_gpu)
        t_new = self.xp.linspace(t.min(), t.max(), 5000)
        new_freqs = freqs_spline(t_new)
        seg_freqs = m_arr[:, None] * f_phi[None, :] + n_arr[:, None] * f_r[None, :] 
        
        argsort = self.xp.argsort(self.xp.abs(self.xp.asarray([seg_freqs[:, :-1], seg_freqs[:, 1:]]).transpose(1, 2, 0)), axis=-1)
        tmp_freqs_segs = self.xp.sort(self.xp.abs(self.xp.asarray([seg_freqs[:, :-1], seg_freqs[:, 1:]]).transpose(1, 2, 0)), axis=-1)
        tmp_freqs_base_sorted_segs = self.xp.sort(self.xp.asarray([seg_freqs[:, :-1], seg_freqs[:, 1:]]).transpose(1, 2, 0), axis=-1)
        
        new_freqs_per_mode = m_arr[:, None] * new_freqs[0][None, :] + n_arr[:, None] * new_freqs[1][None, :]

        # fix for turnover
        t_of_max_freq = t_new[new_freqs_per_mode.argmax(axis=-1)]
        t_of_min_freq = t_new[new_freqs_per_mode.argmin(axis=-1)]
        check_turnover_list = [
            self.xp.where((t_of_max_freq > 0.0) & (t_of_max_freq < t[-1]))[0],
            self.xp.where((t_of_min_freq > 0.0) & (t_of_min_freq < t[-1]))[0]
        ]

        fix_turnover_seg_ind_list = []
        if len(check_turnover_list[0]) > 0:
            fix_turnover_seg_ind_list.append(
                self.xp.searchsorted(t, t_of_max_freq[check_turnover_list[0]]) - 1,
            )
        if len(check_turnover_list[1]) > 0:
            fix_turnover_seg_ind_list.append(
                self.xp.searchsorted(t, t_of_min_freq[check_turnover_list[1]]) - 1
            )

        try:
            fix_turnover_seg_ind = self.xp.concatenate(fix_turnover_seg_ind_list)
            check_turnover = self.xp.concatenate(check_turnover_list)

            a = (m_arr[check_turnover] * (freqs_spline.c3[(0, fix_turnover_seg_ind)])  + n_arr[check_turnover] * freqs_spline.c3[(1, fix_turnover_seg_ind)])
            b = (m_arr[check_turnover] * (freqs_spline.c2[(0, fix_turnover_seg_ind)])  + n_arr[check_turnover] * freqs_spline.c2[(1, fix_turnover_seg_ind)])
            c = (m_arr[check_turnover] * (freqs_spline.c1[(0, fix_turnover_seg_ind)])  + n_arr[check_turnover] * freqs_spline.c1[(1, fix_turnover_seg_ind)])
            d = (m_arr[check_turnover] * (freqs_spline.y[(0, fix_turnover_seg_ind)])  + n_arr[check_turnover] * freqs_spline.y[(1, fix_turnover_seg_ind)])
            
            inner_part = (2 * b) ** 2 - 4 * (3 * a) * c
            if self.xp.any(inner_part < -1e10):
                breakpoint()
            inner_part[(inner_part < 0.0)] = 0.0

            roots_upper_1 = (-(2 * b) + self.xp.sqrt(inner_part)) / (2 * (3 * a))
            roots_upper_2 = (-(2 * b) - self.xp.sqrt(inner_part)) / (2 * (3 * a))
            
            t_new_roots_upper_1 = t[fix_turnover_seg_ind] + roots_upper_1
            t_new_roots_upper_2 = t[fix_turnover_seg_ind] + roots_upper_2
            keep_root_1 = (t[fix_turnover_seg_ind] < t_new_roots_upper_1) & (t[fix_turnover_seg_ind + 1] > t_new_roots_upper_1)
            keep_root_2 = (t[fix_turnover_seg_ind] < t_new_roots_upper_2) & (t[fix_turnover_seg_ind + 1] > t_new_roots_upper_2)

            if self.xp.any(keep_root_1 & keep_root_2):
                breakpoint()
            elif not self.xp.any(keep_root_1 | keep_root_2):
                pass
            else:
                t_new_fix = t_new_roots_upper_1 * keep_root_1 + t_new_roots_upper_2 * keep_root_2
                beginning_of_seg = t[fix_turnover_seg_ind]
                x_fix = t_new_fix - beginning_of_seg
                max_or_min_f = a * x_fix ** 3 + b * x_fix ** 2 + c * x_fix + d
                tmp_segs_sorted_turnover = np.sort(np.concatenate([tmp_freqs_base_sorted_segs[check_turnover, fix_turnover_seg_ind], np.array([max_or_min_f])], axis=-1), axis=-1)

                tmp_freqs_base_sorted_segs[check_turnover, fix_turnover_seg_ind] = tmp_segs_sorted_turnover[:, np.array([0, 2])]
            
        except ValueError:
            pass
        #import matplotlib.pyplot as plt
        #plt.plot(new_freqs_per_mode[0])
        #plt.savefig("check0.png")
        try:
            raise NotImplementedError
            self.frequency = kwargs["f_arr"]
        except:
            self.frequency = self.xp.fft.fftshift(
                self.xp.fft.fftfreq(self.num_pts + self.num_pts_pad, dt)
            )

        ind_zero = self.xp.where(self.frequency == 0)[0][0]
        first_frequency =  self.frequency[0]
        df = self.frequency[1] - self.frequency[0]
        # inds_check = self.xp.searchsorted(self.frequency, seg_freqs.flatten(), side="right").reshape(seg_freqs.shape)
        inds_check = self.xp.abs((tmp_freqs_base_sorted_segs -  first_frequency)/ df).astype(int)
        start_inds = (inds_check[:, :, 0].copy() + 1).astype(int)
        end_inds = (inds_check[:, :, 1].copy()).astype(int)
        
        inds_fin = np.array([start_inds, end_inds]).transpose((1, 2, 0))

        # just for checking
        # inside the code it does not evaluate outside the bounds
        inds_fin[inds_fin > len(self.frequency) - 1] = len(self.frequency) - 1
        inds_fin[inds_fin < 0] = 0

        freq_check = self.frequency[inds_fin]

        run_seg = (np.diff(inds_fin, axis=-1) < 0)[:, :, 0]
        while self.xp.any(((tmp_freqs_base_sorted_segs[:, :, 1] < freq_check[:, :, 1]) | (tmp_freqs_base_sorted_segs[:, :, 0] > freq_check[:, :, 0])) & (run_seg)):
            breakpoint()
            fix_start = (((tmp_freqs_base_sorted_segs[:, :, 0] > freq_check[:, :, 0])) & (run_seg))
            start_inds[fix_start] += 1
            
            fix_end = (((tmp_freqs_base_sorted_segs[:, :, 1] < freq_check[:, :, 1])) & (run_seg))
            end_inds[fix_end] -= 1
            print("check")
        
        k_arr = self.xp.zeros_like(m_arr)
        data_length = len(self.frequency)

        spline_in = spline.interp_array.reshape(spline.reshape_shape).transpose((0, 2, 1)).flatten().copy()
        zero_index = self.xp.where(self.frequency == 0.0)[0][0]

        if separate_modes:
            include_minus_m = False

        self.get_waveform_fd(
            self.waveform,
            spline_in,
            m_arr,
            k_arr, 
            n_arr,
            num_teuk_modes,
            dt, 
            t, 
            init_len, 
            data_length,
            self.frequency, 
            start_inds.flatten().copy().astype(self.xp.int32), 
            end_inds.flatten().copy().astype(self.xp.int32), 
            init_len - 1,
            ylms,
            zero_index,
            include_minus_m,
            separate_modes
        )
           
        # x = t - 8.754992204872e+06
        # a = 1.120059270283e-21
        # b = 7.524898481384e-16
        # c = 4.191674114826e-101
        # d = -2.839495309885e-03

        # y_vals = a * x ** 3+ b * x ** 2 + c * x + d
        # import matplotlib.pyplot as plt
        # plt.plot(t, seg_freqs[0])
        # plt.plot(t, y_vals)
        # plt.axhline(-2.813821469256e-03)
        # plt.axvline(8.754992204872e+06 + 5.530640798130e+04)
        # plt.show()
        # breakpoint()
        return
        # TODO: check this missed_turn_overs
        """
            =x   =   {q + [q2 + (r-p2)3]1/2}1/3   +   {q - [q2 + (r-p2)3]1/2}1/3   +   p
        where

        p = -b/(3a),   q = p3 + (bc-3ad)/(6a2),   r = c/(3a)
        """

        breakpoint()
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
        self.f_modes = special_f_arrs
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
