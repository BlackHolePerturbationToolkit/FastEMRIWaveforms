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

# Python imports
from few.utils.baseclasses import SummationBase, SchwarzschildEccentric, GPUModuleBase
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


class FDInterpolatedModeSum(SummationBase, SchwarzschildEccentric, GPUModuleBase):
    """Create waveform by interpolating sparse trajectory in the frequency domain.

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

        # eventually the name will change to adapt for the gpu implementantion
        if self.use_gpu:
            self.get_waveform = get_waveform_wrap
            self.get_waveform_fd = get_waveform_fd_wrap

        else:
            self.get_waveform = get_waveform_wrap_cpu
            self.get_waveform_fd = get_waveform_fd_wrap_cpu

    def attributes_FDInterpolatedModeSum(self):
        """
        attributes:
            get_waveform (func): CPU or GPU function for waveform creation.

        """

    def waveform_spa_factors(self, fdot_spline, fddot_spline, extended_SPA=True):

        if extended_SPA == True:

            # Waveform Amplitudes
            arg = -1j * 2.0 * np.pi * fdot_spline ** 3 / (3.0 * fddot_spline ** 2)

            K_1over3 = special.kve(
                1.0 / 3.0, arg
            )  # special.kv(1./3.,arg)*np.exp(arg) #

            # to correct the special function nan
            if np.sum(np.isnan(special.kv(1 / 3, arg))) > 0:
                print("number of nans", np.sum(np.isnan(special.kv(1 / 3, arg))))
                # print(arg[np.isnan(special.kv(1./3.,arg))])

                X = 2 * np.pi * fdot_spline ** 3 / (3 * fddot_spline ** 2)
                # K_1over3[np.isnan(special.kv(1./3.,arg))] = (np.sqrt(np.pi/2) /(1j*np.sqrt(np.abs(X))) * np.exp(-1j*np.pi/4) )[np.isnan(special.kv(1./3.,arg))]
                # print('isnan',np.sum(np.isnan(arg)),np.sum(np.isnan(fdot_spline/np.abs(fddot_spline))))

            amp = (
                1j * fdot_spline / np.abs(fddot_spline) * K_1over3 * 2.0 / np.sqrt(3.0)
            )

            amp[np.isnan(amp)] = (
                np.exp(-1j * np.pi / 4 * np.sign(fdot_spline))
                / np.sqrt(np.abs(fdot_spline))
                * (
                    1
                    - 1j
                    * (5.0 / (48.0 * np.pi))
                    * fddot_spline ** 2
                    / (fdot_spline ** 3)
                )
            )[
                np.isnan(amp)
            ]  #

        else:
            amp = (
                np.exp(-np.pi / 4 * fnp.sign(fdot_spline))
                / np.sqrt(np.abs(fdot_spline))
                * (1 - 5j / (48.0 * np.pi) * fddot_spline ** 2 / (fdot_spline ** 3))
            )

        if np.sum(np.isnan(amp)) > 0:
            print("nan in amplitude")

        return amp

    def time_frequency_map(self, spline_f_mode, index_star):
        # turn over index
        index_star = index_star - 1

        t = spline_f_mode.t[-1]
        f_mn = spline_f_mode.y[-1]

        # find t_star analytically
        ratio = spline_f_mode.c2[0, index_star] / (3 * spline_f_mode.c3[0, index_star])
        second_ratio = spline_f_mode.c1[0, index_star] / (
            3 * spline_f_mode.c3[0, index_star]
        )
        t_star = t[index_star] - ratio + np.sqrt(ratio ** 2 - second_ratio)

        # new frequancy vector
        Fstar = spline_f_mode(np.array([t_star])).item()
        new_F = np.append(
            f_mn[: index_star + 1],
            Fstar + Fstar / np.abs(Fstar) * np.abs(f_mn[index_star + 1 :] - Fstar),
        )

        # new t_f
        sign_slope = spline_f_mode.c1[0, 0] / np.abs(spline_f_mode.c1[0, 0])

        # frequency split
        initial_frequency = f_mn[0]
        end_frequency = f_mn[-1]

        if initial_frequency > end_frequency:

            if sign_slope > 0:
                # alt imple
                modified2_t_f = CubicSpline((new_F), t)
                ind_1 = (self.frequency > end_frequency) * (self.frequency < Fstar)
                t_f_1 = modified2_t_f(
                    Fstar
                    + Fstar / np.abs(Fstar) * np.abs(self.frequency[ind_1] - Fstar)
                )
                ind_2 = (self.frequency > initial_frequency) * (self.frequency < Fstar)
                t_f_2 = modified2_t_f(self.frequency[ind_2])

            else:
                modified2_t_f = CubicSpline(-(new_F), t)  #
                ind_1 = (self.frequency < end_frequency) * (self.frequency > Fstar)
                ind_2 = (self.frequency < initial_frequency) * (self.frequency > Fstar)
                t_f_1 = modified2_t_f(
                    -(
                        Fstar
                        + Fstar / np.abs(Fstar) * np.abs(self.frequency[ind_1] - Fstar)
                    )
                )
                t_f_2 = modified2_t_f(-(self.frequency[ind_2]))

        if initial_frequency < end_frequency:

            if sign_slope > 0:

                modified2_t_f = CubicSpline((new_F), t)

                ind_1 = (self.frequency > initial_frequency) * (self.frequency < Fstar)
                ind_2 = (self.frequency > end_frequency) * (self.frequency < Fstar)
                t_f_2 = modified2_t_f(
                    (
                        Fstar
                        + Fstar / np.abs(Fstar) * np.abs(self.frequency[ind_2] - Fstar)
                    )
                )
                t_f_1 = modified2_t_f((self.frequency[ind_1]))

            else:
                modified2_t_f = CubicSpline(-(new_F), t)  #

                ind_1 = (self.frequency < end_frequency) * (self.frequency > Fstar)
                ind_2 = (self.frequency < initial_frequency) * (self.frequency > Fstar)
                t_f_1 = modified2_t_f(
                    -(
                        Fstar
                        + Fstar / np.abs(Fstar) * np.abs(self.frequency[ind_1] - Fstar)
                    )
                )
                t_f_2 = modified2_t_f(-(self.frequency[ind_2]))

        return ind_1, ind_2, t_f_1, t_f_2

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

        # TODO: check fftshift
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

        breakpoint()
        # inds[:, :, 0] += 1
        # inds[:, :, 1] -= 1

        seg_inds = self.xp.zeros_like(seg_freqs, dtype=int)
        inds_pos = self.xp.where(seg_freqs[:, 0] > 0.0)[0]
        inds_neg = self.xp.where(seg_freqs[:, 0] < 0.0)[0]

        freq_mode_start = (
            m_arr[:, None] * spline.c1[-4, None, :]
            + n_arr[:, None] * spline.c1[-3, None, :]
        )

        inds0, inds1 = self.xp.where(
            (self.xp.abs(self.xp.diff(self.xp.sign(freq_mode_start))) == 2.0)
        )

        inds_turnover = self.xp.full(num_teuk_modes, False)
        inds_turnover[inds0] = True
        turnover_seg = self.xp.empty(num_teuk_modes, dtype=int)

        turnover_seg[inds_turnover] = inds1
        turnover_seg[~inds_turnover] = seg_freqs.shape[1]

        turnover_slice = (self.xp.arange(len(turnover_seg)), turnover_seg)

        # find t_star analytically

        bad = turnover_seg >= init_len

        turnover_seg[bad] = 134
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
        spline_out = spline(t_star)

        if spline_out.ndim == 1:
            spline_out = spline_out[:, None]

        Fstar = m_arr * spline_out[-4] + n_arr * spline_out[-3]

        special_f_arrs = self.xp.abs(
            seg_freqs * (t[None, :] < t_star[:, None])
            + (
                Fstar[:, None]
                - Fstar[:, None] / abs(Fstar[:, None]) * abs(seg_freqs - Fstar[:, None])
            )
            * ((t[None, :] > t_star[:, None]) & (slope0[:, None] >= 0.0))
            + (
                Fstar[:, None]
                + Fstar[:, None] / abs(Fstar[:, None]) * abs(seg_freqs - Fstar[:, None])
            )
            * ((t[None, :] > t_star[:, None]) & (slope0[:, None] < 0.0))
        )

        # TODO: this spline sets the turnover slightly off
        special_f_spline = CubicSplineInterpolant(
            special_f_arrs,
            self.xp.tile(t, (special_f_arrs.shape[0], 1)),
            use_gpu=self.use_gpu,
        )

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

        self.get_waveform_fd(
            h,
            spline.interp_array,
            special_f_spline.interp_array,
            special_f_arrs.flatten().copy(),
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
        )

        breakpoint()

        """
        # indentify where there is a turnover
        for j, (m, n) in enumerate(zip(m_arr, n_arr)):
            print("percent", j / len(m_arr))

            # check turnover
            freq_mode_start = m * spline.c1[-4, 0] + n * spline.c1[-3, 0]
            index_star = find_element_in_list(
                True,
                list(
                    freq_mode_start * (m * spline.c1[-4, :] + n * spline.c1[-3, :])
                    < 0.0
                ),
            )

            # frequency mode
            f_mn = m * f_phi + n * f_r
            spline_f_mode = CubicSplineInterpolant(t, f_mn, use_gpu=self.use_gpu)

            second_spl = CubicSpline(t, f_mn)

            min_f_mn = np.min(np.abs(f_mn))
            max_f_mn = np.max(np.abs(f_mn))

            # fdot
            fdot_spline = second_spl.derivative(
                nu=1
            )  # # CubicSplineInterpolant(t, second_spl.derivative(nu=1), use_gpu=self.use_gpu) # CubicSpline(t, np.asarray(spline_f_mode.deriv(t)[-1])) # CubicSplineInterpolant(t, np.asarray(spline_f_mode.deriv(t)), use_gpu=self.use_gpu)

            # fddot
            fddot_spline = second_spl.derivative(
                nu=2
            )  # spline_f_mode._d2 #  #fdot_spline.derivative() # CubicSpline(t, fdot_spline.derivative(t)) #, use_gpu=self.use_gpu)

            if index_star is not None:
                print(m, n)
                print("there is a turn-over")

                # calculate frequency indeces and respective time of f
                ind_1, ind_2, t_f_1, t_f_2 = self.time_frequency_map(
                    spline_f_mode, index_star
                )

                h_contr = [
                    (spline(t_two)[j] + 1j * spline(t_two)[num_teuk_modes + j])
                    * ylms[j]
                    * self.waveform_spa_factors(fdot_spline(t_two), fddot_spline(t_two))
                    * np.exp(
                        1j
                        * (
                            2 * np.pi * f * t_two
                            - (m * spline(t_two)[-2] + n * spline(t_two)[-1])
                        )
                    )
                    for t_two, f in zip(
                        [t_f_1, t_f_2], [self.frequency[ind_1], self.frequency[ind_2]]
                    )
                ]

                h[ind_1] += h_contr[0]
                h[ind_2] += h_contr[1]

                # negative contribution
                if m != 0:
                    print("m different from zero")

                    h_contr = [
                        (spline(t_two)[j] - 1j * spline(t_two)[num_teuk_modes + j])
                        * ylms[j + num_teuk_modes]
                        * self.waveform_spa_factors(
                            -fdot_spline(t_two), -fddot_spline(t_two)
                        )
                        * np.exp(
                            1j
                            * (
                                2 * np.pi * f * t_two
                                + (m * spline(t_two)[-2] + n * spline(t_two)[-1])
                            )
                        )
                        for t_two, f in zip(
                            [np.flip(t_f_1), np.flip(t_f_2)],
                            [
                                self.frequency[np.flip(ind_1)],
                                self.frequency[np.flip(ind_2)],
                            ],
                        )
                    ]

                    h[np.flip(ind_1)] += h_contr[0]
                    h[np.flip(ind_2)] += h_contr[1]

            else:

                initial_frequency = f_mn[0]
                end_frequency = f_mn[-1]

                sign_slope = spline_f_mode.c1[0, 0] / np.abs(spline_f_mode.c1[0, 0])

                if sign_slope > 0:
                    t_of_f = CubicSplineInterpolant(f_mn, t, use_gpu=self.use_gpu)
                    index = (self.frequency > initial_frequency) * (
                        self.frequency < end_frequency
                    )
                    t_f = t_of_f(self.frequency[index])
                else:
                    t_of_f = CubicSplineInterpolant(-f_mn, t, use_gpu=self.use_gpu)
                    index = (self.frequency < initial_frequency) * (
                        self.frequency > end_frequency
                    )
                    t_f = t_of_f(-self.frequency[index])

                h_contr = (
                    (spline(t_f)[j] + 1j * spline(t_f)[num_teuk_modes + j])
                    * ylms[j]
                    * self.waveform_spa_factors(
                        fdot_spline(t_f), fddot_spline(t_f), extended_SPA=True
                    )
                    * np.exp(
                        1j
                        * (
                            2 * np.pi * self.frequency[index] * t_f
                            - (m * spline(t_f)[-2] + n * spline(t_f)[-1])
                        )
                    )
                )

                h[index] += h_contr

                if m != 0:

                    # negative contribution
                    t_f = np.flip(t_f)
                    index = np.flip(index)

                    h_neg = (
                        (spline(t_f)[j] - 1j * spline(t_f)[num_teuk_modes + j])
                        * ylms[j + num_teuk_modes]
                        * self.waveform_spa_factors(
                            -fdot_spline(t_f), -fddot_spline(t_f), extended_SPA=True
                        )
                        * np.exp(
                            1j
                            * (
                                2 * np.pi * self.frequency[index] * t_f
                                + (m * spline(t_f)[-2] + n * spline(t_f)[-1])
                            )
                        )
                    )

                    h[index] += h_neg
        """
        self.waveform = h
