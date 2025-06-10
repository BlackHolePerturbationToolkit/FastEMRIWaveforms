# Interpolated summation of modes in python for the FastEMRIWaveforms Package

import numpy as np

from ..summation.interpolatedmodesum import CubicSplineInterpolant

# Cython imports
# Python imports
from ..utils.baseclasses import (
    BackendLike,
    SchwarzschildEccentric,
)
from ..utils.citations import REFERENCE
from ..utils.constants import MTSUN_SI
from ..utils.geodesic import get_fundamental_frequencies
from .base import SummationBase

# for special functions


# added to return element or none in list
def find_element_in_list(element, list_element):
    try:
        index_element = list_element.index(element)
        return index_element
    except ValueError:
        return None


def searchsorted2d_vec(self, a, b, batch_size=-1, xp=None, **kwargs):
    if xp is None:
        xp = np

    if a.ndim == 1:
        if b.ndim > 1:
            reshape = b.shape
            b = b.flatten()
        else:
            reshape = None
        out = self.xp.searchsorted(a, b, **kwargs)

        if reshape is not None:
            out = out.reshape(reshape)
        return out

    elif a.ndim > 2 or b.ndim > 2:
        raise ValueError("Input arrays must not be more than 2 dimensions.")

    if b.ndim == 1:
        b = self.xp.expand_dims(b, (0,))

    if batch_size < 0:
        inds_split_all = [self.xp.arange(a.shape[0])]
    else:
        split_inds = []
        i = 0
        while i < a.shape[0]:
            i += batch_size
            if i >= a.shape[0]:
                break
            split_inds.append(i)

        inds_split_all = self.xp.split(self.xp.arange(a.shape[0]), split_inds)

    # select tqdm if user wants to see progress
    iterator = enumerate(inds_split_all)
    # iterator = tqdm(iterator, desc="time batch") if show_progress else iterator

    out = self.xp.zeros((a.shape[0], b.shape[1]))
    for i, inds_in in iterator:
        # get subsections of the arrays for each batch
        a_temp = a[inds_in]
        if b.shape[0] > 1:
            b_temp = b[inds_in]
        else:
            b_temp = b

        m, n = a_temp.shape
        max_num = (
            self.xp.maximum(a_temp.max() - a_temp.min(), b_temp.max() - b_temp.min())
            + 1
        )
        r = max_num * self.xp.arange(a_temp.shape[0])[:, None]
        p = self.xp.searchsorted(
            (a_temp + r).ravel(), (b_temp + r).ravel(), **kwargs
        ).reshape(m, -1)
        out[inds_in] = p - n * (self.xp.arange(m)[:, None])

    return out


class FDInterpolatedModeSum(SummationBase, SchwarzschildEccentric):
    """Create waveform by interpolating sparse trajectory in the frequency domain.

    It interpolates all of the modes of interest and phases at sparse
    trajectories. Within the summation phase, the values are calculated using
    the interpolants and summed.

    This class can be run on GPUs and CPUs.
    """

    def __init__(self, /, force_backend: BackendLike = None, **kwargs):
        SummationBase.__init__(
            self,
            **{
                key: value
                for key, value in kwargs.items()
                if key in ["output_type", "pad_output", "odd_len"]
            },
            force_backend=force_backend,
        )
        SchwarzschildEccentric.__init__(
            self,
            **{
                key: value
                for key, value in kwargs.items()
                if key in ["lmax", "nmax", "ndim"]
            },
            force_backend=force_backend,
        )

        # self.kwargs = kwargs

    # eventually the name will change to adapt for the gpu implementantion
    @property
    def get_waveform_fd(self) -> callable:
        """GPU or CPU waveform generation."""
        return self.backend.get_waveform_generic_fd_wrap

    @classmethod
    def supported_backends(cls):
        return cls.GPU_RECOMMENDED()

    @classmethod
    def module_references(cls) -> list[REFERENCE]:
        """Return citations related to this module"""
        return [REFERENCE.FD] + super().module_references()

    def sum(
        self,
        t,
        teuk_modes,
        ylms,
        phase_interp_t,
        phase_interp_coeffs,
        l_arr,
        m_arr,
        n_arr,
        M,
        a,
        p,
        e,
        xI,
        *args,
        include_minus_m=True,
        separate_modes=False,
        dt=10.0,
        f_arr=None,
        mask_positive=False,
        integrate_backwards=False,
        **kwargs,
    ):
        r"""Interpolated summation function in Frequency Domain.

        This function interpolates the amplitude and phase information, and
        creates the final waveform with the combination of ylm values for each
        mode in the Frequency Domain.

        This turns the waveform array into a 2D array with shape (2, number of points).
        The 2 represents h+ and hx in that order.

        args:
            t (1D double self.xp.ndarray): Array of t values.
            teuk_modes (2D double self.xp.array): Array of complex amplitudes.
                Shape: (len(t), num_teuk_modes).
            ylms (1D complex128 self.xp.ndarray): Array of ylm values for each mode,
                including m<0. Shape is (num of m==0,) + (num of m>0,)
                + (num of m<0). Number of m<0 and m>0 is the same, but they are
                ordered as (m==0 first then) m>0 then m<0.
            Phi_phi (1D double self.xp.ndarray): Array of azimuthal phase values
                (:math:`\Phi_\phi`).
            Phi_r (1D double self.xp.ndarray): Array of radial phase values
                 (:math:`\Phi_r`).
            l_arr (1D int self.xp.ndarray): :math:`l` values associated with each mode.
            m_arr (1D int self.xp.ndarray): :math:`m` values associated with each mode.
            n_arr (1D int self.xp.ndarray): :math:`n` values associated with each mode.
            M (double): Total mass in solar masses.
            p (1D int self.xp.ndarray): Semi-latus rectum along trajectory.
            e (1D int self.xp.ndarray): Eccentricity value along trajectory.
            *args (list, placeholder): Added for flexibility.
            include_minus_m (bool, optional): Include the values for :math:`-m`. This is
                useful when looking at specific modes. Default is ``True``.
            separate_modes (bool, optional): Return each harmonic mode separated from each other.
                Default is ``False``.
            dt (double, optional): Time spacing between observations (inverse of sampling
                rate). Default is 10.0.
            f_arr (1D double self.xp.ndarray, optional): User-provided frequency array. For now,
                it must be evenly spaced and include both positive and negative frequencies.
                If ``None``, the frequency array is built from observation time and time step.
                Default is ``None``.
            mask_positive (bool, optional): Only return h+ and hx along positive frequencies.
                Default is False.
            **kwargs (dict, placeholder): Added for future flexibility.

        """

        init_len = len(t)  # length of sparse traj
        num_teuk_modes = teuk_modes.shape[1]
        _num_pts = self.num_pts  # from the base clase, adjust in the baseclass

        length = init_len
        # number of quantities to be interp in time domain
        ninterps = (self.ndim) + 2 * num_teuk_modes  # 2 for re and im
        y_all = self.xp.zeros((ninterps, length))

        # fill interpolant information in y
        # all interpolants
        y_all[:num_teuk_modes] = teuk_modes.T.real
        y_all[num_teuk_modes : 2 * num_teuk_modes] = teuk_modes.T.imag

        try:
            p, e, xI = p.get(), e.get(), xI.get()
        except AttributeError:
            pass

        # get fundamental frequencies across trajectory
        Omega_phi, Omega_theta, Omega_r = get_fundamental_frequencies(
            a,
            p,
            e,
            xI,
        )

        # convert from dimensionless frequencies
        f_phi, f_r = (
            abs(
                self.xp.asarray(Omega_phi / (2 * np.pi * M * MTSUN_SI))
            ),  # positive frequency to be consistent with amplitude generator for retrograde inspirals  # TODO get to the bottom of this!
            self.xp.asarray(Omega_r / (2 * np.pi * M * MTSUN_SI)),
        )

        # add them for splines
        y_all[-2] = f_phi
        y_all[-1] = f_r

        # for a frequency-only spline because you
        # have to evaluate all splines when you evaluate the spline class
        y_all_freqs = self.xp.asarray([f_phi, f_r])

        # create two splines
        # spline: freqs, phases, and amplitudes for the summation kernel
        # freqs_spline: only the frequencies [f(t)]
        freqs_spline = self.build_with_same_backend(
            CubicSplineInterpolant, args=[t, y_all_freqs]
        )
        spline = self.build_with_same_backend(CubicSplineInterpolant, args=[t, y_all])

        # t_new is a denser (but not completely dense) set
        # of time values to help estimate where/when turnovers occur
        # the denser it is the more likely you will not miss a turnover
        # but it is a tradeoff with speed (not exactly checked, just thought about)
        t_new = self.xp.linspace(t.min(), t.max(), 5000)

        # denser frequencies
        new_freqs = freqs_spline(t_new)

        # mode frequencies along the sparse trajectory
        seg_freqs = m_arr[:, None] * f_phi[None, :] + n_arr[:, None] * f_r[None, :]

        # properly orders each segment without absolute value
        tmp_freqs_base_sorted_segs = self.xp.sort(
            self.xp.asarray([seg_freqs[:, :-1], seg_freqs[:, 1:]]).transpose(1, 2, 0),
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
            self.xp.where((t_of_max_freq > 0.0) & (t_of_max_freq < t[-1]))[0],
            self.xp.where((t_of_min_freq > 0.0) & (t_of_min_freq < t[-1]))[0],
        ]

        # figure out which segment the turnover occurs in
        # first list entry is turnovers at maximum
        # second list entry is turnovers at minimum
        fix_turnover_seg_ind_list = []
        if len(check_turnover_list[0]) > 0:
            fix_turnover_seg_ind_list.append(
                self.xp.searchsorted(t, t_of_max_freq[check_turnover_list[0]]) - 1,
            )
        if len(check_turnover_list[1]) > 0:
            fix_turnover_seg_ind_list.append(
                self.xp.searchsorted(t, t_of_min_freq[check_turnover_list[1]]) - 1
            )

        # determine the true maximum/minimum frequency and associated quantities
        # by solving for roots in the derivative of the cubic
        try:
            fix_turnover_seg_ind = self.xp.concatenate(fix_turnover_seg_ind_list)
            check_turnover = self.xp.concatenate(check_turnover_list)

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
            if self.xp.any(inner_part < -1e10):
                breakpoint()
            inner_part[(inner_part < 0.0)] = 0.0

            roots_upper_1 = (-(2 * b) + self.xp.sqrt(inner_part)) / (2 * (3 * a))
            roots_upper_2 = (-(2 * b) - self.xp.sqrt(inner_part)) / (2 * (3 * a))

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
            if self.xp.any(keep_root_1 & keep_root_2):
                breakpoint()

            # sometimes (I think?) neither root is kept
            elif not self.xp.any(keep_root_1 | keep_root_2):
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
                tmp_segs_sorted_turnover = self.xp.sort(
                    self.xp.concatenate(
                        [
                            tmp_freqs_base_sorted_segs[
                                check_turnover, fix_turnover_seg_ind
                            ],
                            self.xp.array([max_or_min_f]),
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
            self.frequency = self.xp.hstack(
                (self.xp.arange(-(Len // 2), 0), self.xp.arange(0, (Len - 1) // 2 + 1))
            ) / (Len * dt)

        # make sure there is one value of frequency at 0.0
        assert 1 == np.sum(self.frequency == 0.0)

        ind_zero = self.xp.where(self.frequency == 0)[0][0]

        # make sure ind_zero is where it is supposed to be
        assert ind_zero == int(len(self.frequency) / 2)

        # frequencies must be equally spaced for now
        # first frequency helps quickly determine which segments each
        # frequency in self.frequency falls into.
        first_frequency = self.frequency[0]
        df = self.frequency[1] - self.frequency[0]

        # figures out where in self.frequency each segment frequency falls
        inds_check = self.xp.abs(
            (tmp_freqs_base_sorted_segs - first_frequency) / df
        ).astype(int)
        if f_arr is not None:
            mask_pos = tmp_freqs_base_sorted_segs > 0.0
            inds_check_pos = (
                self.xp.abs(
                    (tmp_freqs_base_sorted_segs - self.frequency[ind_zero + 1]) / df
                ).astype(int)
                + ind_zero
                + 1
            )
            inds_check[mask_pos] = inds_check_pos[mask_pos]

        # start frequency index of each segment
        start_inds = (inds_check[:, :, 0].copy() + 1).astype(int)

        # end frequency index of each segment
        end_inds = (inds_check[:, :, 1].copy()).astype(int)

        # final inds array
        inds_fin = self.xp.array([start_inds, end_inds]).transpose((1, 2, 0))

        # just for checking
        # inside the code it does not evaluate outside the bounds
        inds_fin[inds_fin > len(self.frequency) - 1] = len(self.frequency) - 1
        inds_fin[inds_fin < 0] = 0

        # frequencies to check boundaries
        _freq_check = self.frequency[inds_fin]

        # place holder for k array
        k_arr = self.xp.zeros_like(m_arr)
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
        zero_index = self.xp.where(self.frequency == 0.0)[0][0].item()

        if separate_modes:
            include_minus_m = False

        # block disabled since it uses h_t which is undefined
        # if integrate_backwards:
        #     # For consistency with forward integration, we slightly shift the knots so that they line up at t=0
        #     offset = h_t[-1] - int(h_t[-1] / dt) * dt
        #     h_t = h_t - offset
        #     phase_interp_t = phase_interp_t - offset

        phase_interp_t_in = self.xp.asarray(phase_interp_t)

        phase_interp_coeffs_in = self.xp.transpose(
            self.xp.asarray(phase_interp_coeffs), [2, 1, 0]
        ).flatten()

        # for ylm with negative m, need to multiply by (-1)**l as this is assumed to have happened by the kernel
        ylms = ylms.copy()
        ylms[num_teuk_modes:] *= (-1) ** l_arr

        # run GPU kernel
        self.get_waveform_fd(
            self.waveform,
            spline_in,
            phase_interp_t_in,
            phase_interp_coeffs_in,
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
            separate_modes,
        )

        # adjust from fourier transform across +/- frequencies
        # to h+/hx at positive frequencies
        fd_sig = -self.xp.flip(self.waveform)
        fft_sig_p = (
            self.xp.real(fd_sig + self.xp.flip(fd_sig)) / 2.0
            + 1j * self.xp.imag(fd_sig - self.xp.flip(fd_sig)) / 2.0
        )
        fft_sig_c = (
            -self.xp.imag(fd_sig + self.xp.flip(fd_sig)) / 2.0
            + 1j * self.xp.real(fd_sig - self.xp.flip(fd_sig)) / 2.0
        )

        # mask to only have positive frequency values
        if mask_positive:
            mask = self.frequency >= 0.0
            self.waveform = self.xp.vstack((fft_sig_p[mask], fft_sig_c[mask]))
        else:
            self.waveform = self.xp.vstack((fft_sig_p, fft_sig_c))

        return
