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
    import cupy as cp

except (ImportError, ModuleNotFoundError) as e:
    import numpy as np

# Cython imports
from ..cutils.pyinterp_cpu import (
    get_waveform_generic_fd_wrap as get_waveform_generic_fd_wrap_cpu,
)

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
    from ..cutils.pyinterp import (
        get_waveform_generic_fd_wrap,
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


def searchsorted2d_vec(a, b, batch_size=-1, xp=None, **kwargs):
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
    # iterator = tqdm(iterator, desc="time batch") if show_progress else iterator

    out = xp.zeros((a.shape[0], b.shape[1]))
    for i, inds_in in iterator:
        # get subsections of the arrays for each batch
        a_temp = a[inds_in]
        if b.shape[0] > 1:
            b_temp = b[inds_in]
        else:
            b_temp = b

        m, n = a_temp.shape
        max_num = (
            xp.maximum(a_temp.max() - a_temp.min(), b_temp.max() - b_temp.min()) + 1
        )
        r = max_num * xp.arange(a_temp.shape[0])[:, None]
        p = xp.searchsorted(
            (a_temp + r).ravel(), (b_temp + r).ravel(), **kwargs
        ).reshape(m, -1)
        out[inds_in] = p - n * (xp.arange(m)[:, None])

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
            self.get_waveform_fd = get_waveform_generic_fd_wrap

        else:
            self.get_waveform_fd = get_waveform_generic_fd_wrap_cpu

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
        f_arr=None,
        mask_positive=False,
        **kwargs,
    ):
        """Interpolated summation function in Frequency Domain.

        This function interpolates the amplitude and phase information, and
        creates the final waveform with the combination of ylm values for each
        mode in the Frequency Domain.

        This turns the waveform array into a 2D array with shape (2, number of points).
        The 2 represents h+ and hx in that order.

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
            M (double): Total mass in solar masses.
            p (1D int xp.ndarray): Semi-latus rectum in units of M along trajectory.
            e (1D int xp.ndarray): Eccentricity value along trajectory.
            *args (list, placeholder): Added for flexibility.
            include_minus_m (bool, optional): Include the values for :math:`-m`. This is
                useful when looking at specific modes. Default is ``True``.
            separate_modes (bool, optional): Return each harmonic mode separated from each other.
                Default is ``False``.
            dt (double, optional): Time spacing between observations (inverse of sampling
                rate). Default is 10.0.
            f_arr (1D double xp.ndarray, optional): User-provided frequency array. For now,
                it must be evenly spaced and include both positive and negative frequencies.
                If ``None``, the frequency array is built from observation time and time step.
                Default is ``None``.
            mask_positive (bool, optional): Only return h+ and hx along positive frequencies.
                Default is False.
            **kwargs (dict, placeholder): Added for future flexibility.

        """

        if self.use_gpu:
            xp = cp
        else:
            xp = np

        init_len = len(t)  # length of sparse traj
        num_teuk_modes = teuk_modes.shape[1]
        num_pts = self.num_pts  # from the base clase, adjust in the baseclass

        length = init_len
        # number of quantities to be interp in time domain
        ninterps = (2 * self.ndim) + 2 * num_teuk_modes  # 2 for re and im
        y_all = xp.zeros((ninterps, length))

        # fill interpolant information in y
        # all interpolants
        y_all[:num_teuk_modes] = teuk_modes.T.real
        y_all[num_teuk_modes : 2 * num_teuk_modes] = teuk_modes.T.imag

        y_all[-2] = Phi_phi
        y_all[-1] = Phi_r

        try:
            p, e = p.get(), e.get()
        except AttributeError:
            pass

        # get fundamental frequencies across trajectory
        Omega_phi, Omega_theta, Omega_r = get_fundamental_frequencies(
            0.0, p, e, np.zeros_like(e)
        )

        # convert from dimensionless frequencies
        f_phi, f_r = (
            xp.asarray(Omega_phi / (2 * np.pi * M * MTSUN_SI)),
            xp.asarray(Omega_r / (2 * np.pi * M * MTSUN_SI)),
        )

        # add them for splines
        y_all[-4] = f_phi
        y_all[-3] = f_r

        # for a frequency-only spline because you
        # have to evaluate all splines when you evaluate the spline class
        y_all_freqs = xp.asarray([f_phi, f_r])

        # create two splines
        # spline: freqs, phases, and amplitudes for the summation kernel
        # freqs_spline: only the frequencies [f(t)]
        freqs_spline = CubicSplineInterpolant(t, y_all_freqs, use_gpu=self.use_gpu)
        spline = CubicSplineInterpolant(t, y_all, use_gpu=self.use_gpu)

        # t_new is a denser (but not completely dense) set
        # of time values to help estimate where/when turnovers occur
        # the denser it is the more likely you will not miss a turnover
        # but it is a tradeoff with speed (not exactly checked, just thought about)
        t_new = xp.linspace(t.min(), t.max(), 5000)

        # denser frequencies
        new_freqs = freqs_spline(t_new)

        # mode frequencies along the sparse trajectory
        seg_freqs = m_arr[:, None] * f_phi[None, :] + n_arr[:, None] * f_r[None, :]

        # properly orders each segment without absolute value
        tmp_freqs_base_sorted_segs = xp.sort(
            xp.asarray([seg_freqs[:, :-1], seg_freqs[:, 1:]]).transpose(1, 2, 0),
            axis=-1,
        )

        # denser mode frequencies
        new_freqs_per_mode = (
            m_arr[:, None] * new_freqs[0][None, :]
            + n_arr[:, None] * new_freqs[1][None, :]
        )

        # max/min frequency along dense array
        # helps determine where the turnover is because
        # it will always be above/below the max/min frequency
        # you get from the segment boundaries
        t_of_max_freq = t_new[new_freqs_per_mode.argmax(axis=-1)]
        t_of_min_freq = t_new[new_freqs_per_mode.argmin(axis=-1)]

        # checks what time the turnover happens
        # if it is at the initial time or end time (i.e. there is no turnover)
        # it is not included (hence >/< not <=/>=)
        check_turnover_list = [
            xp.where((t_of_max_freq > 0.0) & (t_of_max_freq < t[-1]))[0],
            xp.where((t_of_min_freq > 0.0) & (t_of_min_freq < t[-1]))[0],
        ]

        # figure out which segment the turnover occurs in
        # first list entry is turnovers at maximum
        # second list entry is turnovers at minimum
        fix_turnover_seg_ind_list = []
        if len(check_turnover_list[0]) > 0:
            fix_turnover_seg_ind_list.append(
                xp.searchsorted(t, t_of_max_freq[check_turnover_list[0]]) - 1,
            )
        if len(check_turnover_list[1]) > 0:
            fix_turnover_seg_ind_list.append(
                xp.searchsorted(t, t_of_min_freq[check_turnover_list[1]]) - 1
            )

        # determine the true maximum/minimum frequency and associated quantities
        # by solving for roots in the derivative of the cubic
        try:
            fix_turnover_seg_ind = xp.concatenate(fix_turnover_seg_ind_list)
            check_turnover = xp.concatenate(check_turnover_list)

            a = (
                m_arr[check_turnover] * (freqs_spline.c3[(0, fix_turnover_seg_ind)])
                + n_arr[check_turnover] * freqs_spline.c3[(1, fix_turnover_seg_ind)]
            )
            b = (
                m_arr[check_turnover] * (freqs_spline.c2[(0, fix_turnover_seg_ind)])
                + n_arr[check_turnover] * freqs_spline.c2[(1, fix_turnover_seg_ind)]
            )
            c = (
                m_arr[check_turnover] * (freqs_spline.c1[(0, fix_turnover_seg_ind)])
                + n_arr[check_turnover] * freqs_spline.c1[(1, fix_turnover_seg_ind)]
            )
            d = (
                m_arr[check_turnover] * (freqs_spline.y[(0, fix_turnover_seg_ind)])
                + n_arr[check_turnover] * freqs_spline.y[(1, fix_turnover_seg_ind)]
            )

            inner_part = (2 * b) ** 2 - 4 * (3 * a) * c
            if xp.any(inner_part < -1e10):
                breakpoint()
            inner_part[(inner_part < 0.0)] = 0.0

            roots_upper_1 = (-(2 * b) + xp.sqrt(inner_part)) / (2 * (3 * a))
            roots_upper_2 = (-(2 * b) - xp.sqrt(inner_part)) / (2 * (3 * a))

            # t roots of potential minimum/maximum
            t_new_roots_upper_1 = t[fix_turnover_seg_ind] + roots_upper_1
            t_new_roots_upper_2 = t[fix_turnover_seg_ind] + roots_upper_2

            # check to make sure root is in the segment of interest
            keep_root_1 = (t[fix_turnover_seg_ind] < t_new_roots_upper_1) & (
                t[fix_turnover_seg_ind + 1] > t_new_roots_upper_1
            )
            keep_root_2 = (t[fix_turnover_seg_ind] < t_new_roots_upper_2) & (
                t[fix_turnover_seg_ind + 1] > t_new_roots_upper_2
            )

            # should only be one root at maximum
            if xp.any(keep_root_1 & keep_root_2):
                breakpoint()

            # sometimes (I think?) neither root is kept
            elif not xp.any(keep_root_1 | keep_root_2):
                pass
            else:
                # new time of minimum / maximum frequencies
                t_new_fix = (
                    t_new_roots_upper_1 * keep_root_1
                    + t_new_roots_upper_2 * keep_root_2
                )

                # calculate the actual max/min frequencies
                beginning_of_seg = t[fix_turnover_seg_ind]
                x_fix = t_new_fix - beginning_of_seg
                max_or_min_f = a * x_fix**3 + b * x_fix**2 + c * x_fix + d

                # extends any segments with turnover to include all frequencies
                # up to and including the turnover frequency
                # this array holds the edge frequencies that occur in each
                # segment, so we need to make sure when the segment edges do not include
                # up to the turnover, they are added here
                tmp_segs_sorted_turnover = xp.sort(
                    xp.concatenate(
                        [
                            tmp_freqs_base_sorted_segs[
                                check_turnover, fix_turnover_seg_ind
                            ],
                            xp.array([max_or_min_f]),
                        ],
                        axis=-1,
                    ),
                    axis=-1,
                )

                tmp_freqs_base_sorted_segs[check_turnover, fix_turnover_seg_ind] = (
                    tmp_segs_sorted_turnover[:, np.array([0, 2])]
                )

        except ValueError:
            pass

        # get frequencies of interest
        if f_arr is not None:
            self.frequency = f_arr
            if len(self.frequency) == 0:
                raise ValueError("Input f_arr kwarg has zero length.")
        else:
            Len = self.num_pts + self.num_pts_pad
            # we do not use fft.fftfreqs here because cupy and numpy slightly differ
            self.frequency = xp.hstack(
                (xp.arange(-(Len // 2), 0), xp.arange(0, (Len - 1) // 2 + 1))
            ) / (Len * dt)

        # make sure there is one value of frequency at 0.0
        assert 1 == np.sum(self.frequency == 0.0)

        ind_zero = xp.where(self.frequency == 0)[0][0]

        # make sure ind_zero is where it is supposed to be
        assert ind_zero == int(len(self.frequency) / 2)

        # frequencies must be equally spaced for now
        # first frequency helps quickly determine which segments each
        # frequency in self.frequency falls into.
        first_frequency = self.frequency[0]
        df = self.frequency[1] - self.frequency[0]

        # figures out where in self.frequency each segment frequency falls
        inds_check = xp.abs((tmp_freqs_base_sorted_segs - first_frequency) / df).astype(
            int
        )

        # start frequency index of each segment
        start_inds = (inds_check[:, :, 0].copy() + 1).astype(int)

        # end frequency index of each segment
        end_inds = (inds_check[:, :, 1].copy()).astype(int)

        # final inds array
        inds_fin = xp.array([start_inds, end_inds]).transpose((1, 2, 0))

        # just for checking
        # inside the code it does not evaluate outside the bounds
        inds_fin[inds_fin > len(self.frequency) - 1] = len(self.frequency) - 1
        inds_fin[inds_fin < 0] = 0

        # frequencies to check boundaries
        freq_check = self.frequency[inds_fin]

        # place holder for k array
        k_arr = xp.zeros_like(m_arr)
        data_length = len(self.frequency)

        # prepare spline for GPU evaluation
        spline_in = (
            spline.interp_array.reshape(spline.reshape_shape)
            .transpose((0, 2, 1))
            .flatten()
            .copy()
        )

        # where is the zero index
        # for +/- m type of thing
        zero_index = xp.where(self.frequency == 0.0)[0][0].item()

        if separate_modes:
            include_minus_m = False

        # run GPU kernel
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
            start_inds.flatten().copy().astype(xp.int32),
            end_inds.flatten().copy().astype(xp.int32),
            init_len - 1,
            ylms,
            zero_index,
            include_minus_m,
            separate_modes,
        )

        # adjust from fourier transform across +/- frequencies
        # to h+/hx at positive frequencies
        fd_sig = -xp.flip(self.waveform)
        fft_sig_p = (
            xp.real(fd_sig + xp.flip(fd_sig)) / 2.0
            + 1j * xp.imag(fd_sig - xp.flip(fd_sig)) / 2.0
        )
        fft_sig_c = (
            -xp.imag(fd_sig + xp.flip(fd_sig)) / 2.0
            + 1j * xp.real(fd_sig - xp.flip(fd_sig)) / 2.0
        )

        # mask to only have positive frequency values
        if mask_positive:
            mask = self.frequency >= 0.0
            self.waveform = xp.vstack((fft_sig_p[mask], fft_sig_c[mask]))
        else:
            self.waveform = xp.vstack((fft_sig_p, fft_sig_c))

        return
