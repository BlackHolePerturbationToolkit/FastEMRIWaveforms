# Flux-based Schwarzschild Eccentric amplitude module for Fast EMRI Waveforms
# performs calculation with a Roman network

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

from copy import deepcopy
import os
import warnings

import numpy as np
import h5py
from scipy.interpolate import RectBivariateSpline
from tqdm import tqdm

# Cython/C++ imports

# Python imports
from ..utils.baseclasses import (
    SchwarzschildEccentric,
    ParallelModuleBase,
    KerrEccentricEquatorial,
)
from .base import AmplitudeBase
from ..utils.utility import check_for_file_download
from ..utils.citations import *
from ..utils.utility import p_to_y, kerr_p_to_u

# check for cupy and GPU version of pymatmul
try:
    # Cython/C++ imports
    from ..cutils.pyAmpInterp2D import interp2D as interp2D_gpu

    # Python imports
    import cupy as cp

except (ImportError, ModuleNotFoundError) as e:
    import numpy as np

from ..cutils.pyAmpInterp2D_cpu import interp2D as interp2D_cpu

from typing import Optional, Union

# get path to this file
dir_path = os.path.dirname(os.path.realpath(__file__))

# TODO: handle multiple waveform models
_DEFAULT_SPINS = [
    -0.99,
    -0.95,
    -0.9,
    -0.8,
    -0.7,
    -0.6,
    -0.5,
    -0.4,
    -0.3,
    -0.2,
    -0.1,
    0.0,
    0.1,
    0.2,
    0.3,
    0.4,
    0.5,
    0.6,
    0.7,
    0.8,
    0.9,
    0.95,
    0.99,
]

_DEFAULT_AMPLITUDE_FILENAMES = [
    f"KerrEqEccAmpCoeffs_a{spin:.3f}.h5" for spin in _DEFAULT_SPINS
]


class AmpInterp2D(AmplitudeBase, ParallelModuleBase):
    r"""Calculate Teukolsky amplitudes with a bicubic spline interpolation.

    This class is initialised by providing mode index arrays and a corresponding spline coefficients array.
    These coefficients can be computed for user-supplied data with the TODO METHOD method of this class.

    When called with arguments :math:`(a, p, e, xI)`, these parameters are transformed into a set of
    interpolation coordinates and the bicubic spline interpolant is evaluated at these coordinates for
    all sets of coefficients.

    This module is available for GPU and CPU.

    args:
        fp: The coefficients file name in `file_directory`.
        l_arr: Array of :math:`\ell` mode indices.
        m_arr: Array of :math:`m` mode indices.
        n_arr: Array of :math:`n` mode indices.
        file_directory: The path to the directory containing the coefficients file.
        **kwargs: Optional keyword arguments for the base class:
            :class:`few.utils.baseclasses.AmplitudeBase`,
            :class:`few.utils.baseclasses.ParallelModuleBase`.
    """

    def __init__(
            self,
            fp: str,
            l_arr: np.ndarray,
            m_arr: np.ndarray,
            n_arr: np.ndarray,
            file_directory:Optional[str]=None,
            **kwargs
        ):
        ParallelModuleBase.__init__(self, **kwargs)
        AmplitudeBase.__init__(self, **kwargs)

        self.fp = fp
        self.l_arr = l_arr
        self.m_arr = m_arr
        self.n_arr = n_arr

        if file_directory is None:
            file_directory = dir_path + "/../../few/files/"

        self.file_directory = file_directory

        # check if user has the necessary data
        # if not, the data will automatically download
        check_for_file_download(fp, self.file_directory)

        mystery_file = h5py.File(os.path.join(self.file_directory, fp))
        try:
            is_coeffs = mystery_file.attrs["is_coefficients"]
        except KeyError:
            is_coeffs = False

        if is_coeffs:
            coefficients = mystery_file
        else:
            print(fp, "is not a spline coefficients file. Attempting to convert...")
            spline_fp = _spline_coefficients_to_file(
                fp, self.l_arr, self.m_arr, self.n_arr, file_directory=self.file_directory
            )
            coefficients = h5py.File(os.path.join(self.file_directory, spline_fp))

        self.a_val_store = coefficients.attrs["signed_spin"]
        """float: The value of :math:`a` associated with this interpolant."""

        self.num_teuk_modes = coefficients.attrs["num_teuk_modes"]
        """int: Total number of mode amplitude grids this interpolant stores."""

        self.tck = [
            self.xp.asarray(coefficients["x1"]),
            self.xp.asarray(coefficients["x2"]),
            self.xp.asarray(coefficients["c"]),
        ]
        """list[np.ndarray]: Arrays holding all spline coefficient information."""

        self.len_indiv_c = coefficients.attrs["points_per_modegrid"]
        """int: Total number of coefficients per mode amplitude grid."""

    @property
    def interp2D(self) -> callable:
        """GPU or CPU interp2D"""
        interp2D = interp2D_cpu if not self.use_gpu else interp2D_gpu
        return interp2D

    @property
    def citation(self):
        """Return citations for this module"""
        return (
            romannet_citation
            + larger_few_citation
            + few_citation
            + few_software_citation
        )

    @property
    def gpu_capability(self):
        """Confirms GPU capability"""
        return True

    def __call__(self, a: Union[float,np.ndarray], p: Union[float,np.ndarray], e: Union[float,np.ndarray], xI: Union[float,np.ndarray], *args, specific_modes: Optional[Union[list, np.ndarray]]=None, **kwargs) -> np.ndarray:
        """
        Evaluate the spline or its derivatives at given positions.

        Args:
            a: Dimensionless spin parameter of MBH.
            p: Dimensionless semi-latus rectum.
            e: Eccentricity.
            xI: Cosine of orbital inclination. Only :math:`|x_I| = 1` is currently supported.
            specific_modes: Either indices or mode index tuples of modes to be generated (optional; defaults to all modes).
        Returns:
            Complex Teukolsky mode amplitudes at the requested points.
        """

        try:
            a_cpu, p_cpu, e_cpu, xI_cpu = (
                a.get().copy(),
                p.get().copy(),
                e.get().copy(),
                xI.get().copy(),
            )
        except AttributeError:
            a_cpu, p_cpu, e_cpu, xI_cpu = a.copy(), p.copy(), e.copy(), xI.copy()

        a = self.xp.asarray(a)
        p = self.xp.asarray(p)
        e = self.xp.asarray(e)
        xI = self.xp.asarray(xI)

        assert self.xp.all(a == self.a_val_store)
        a_cpu *= xI_cpu  # correct the sign of a now we've passed the check, for the reparameterisation
        # TODO: make this GPU accessible
        u = self.xp.asarray(kerr_p_to_u(a_cpu, p_cpu, e_cpu, xI_cpu, use_gpu=False))

        w = self.xp.sqrt(e)

        tw, tu, c = self.tck[:3]
        kw = ku = 3

        # standard Numpy broadcasting
        if w.shape != u.shape:
            w, u = np.broadcast_arrays(w, u)

        shape = w.shape
        w = w.ravel()
        u = u.ravel()

        if w.size == 0 or u.size == 0:
            return np.zeros(shape, dtype=self.tck[2].dtype)

        nw = tw.shape[0]
        nu = tu.shape[0]
        mw = w.shape[0]
        mu = u.shape[0]

        assert mw == mu

        # TODO: adjustable

        if specific_modes is None:
            mode_indexes = self.xp.arange(self.num_teuk_modes)

        else:
            if isinstance(specific_modes, self.xp.ndarray):
                mode_indexes = specific_modes
            elif isinstance(
                specific_modes, list
            ):  # the following is slow and kills efficiency
                mode_indexes = self.xp.zeros(len(specific_modes), dtype=self.xp.int32)
                for i, (l, m, n) in enumerate(specific_modes):
                    try:
                        mode_indexes[i] = np.where(
                            (self.l_arr == l)
                            & (self.m_arr == abs(m))
                            & (self.n_arr == n)
                        )[0]
                    except:
                        raise Exception(f"Could not find mode index ({l},{m},{n}).")
        # TODO: perform this in the kernel
        c_in = c[mode_indexes].flatten()

        num_indiv_c = 2 * len(mode_indexes)  # Re and Im
        len_indiv_c = self.len_indiv_c

        z = self.xp.zeros((num_indiv_c * mw))

        self.interp2D(
            z, tw, nw, tu, nu, c_in, kw, ku, w, mw, u, mu, num_indiv_c, len_indiv_c
        )

        # check = np.asarray([[spl.ev(e.get(), y.get()) for spl in spl1] for spl1 in self.spl2D.values()]).transpose(2, 1, 0)

        z = z.reshape(num_indiv_c // 2, 2, mw).transpose(2, 1, 0)

        z = z[:, 0] + 1j * z[:, 1]
        return z

    def __reduce__(self):
        return (
            self.__class__,
            (self.fp, self.l_arr, self.m_arr, self.n_arr, self.file_directory),
        )


class AmpInterpKerrEqEcc(AmplitudeBase, KerrEccentricEquatorial):
    """Calculate Teukolsky amplitudes in the Kerr eccentric equatorial regime with a bicubic spline + linear
    interpolation scheme.

    When called with arguments :math:`(a, p, e, xI)`, these parameters are transformed into a set of
    interpolation coordinates and the bicubic spline interpolant is evaluated at these coordinates for
    all sets of coefficients. To interpolate in the :math"`a` direction, the bicubic spline is evaluated at
    the adjacent grid points and a linear interpolation is performed.

    This module is available for GPU and CPU.

    args:
        fp: The coefficients file name in `file_directory`.
        file_directory: The path to the directory containing the coefficients file.
        **kwargs: Optional keyword arguments for the base classes:
            :class:`few.utils.baseclasses.AmplitudeBase`,
            :class:`few.utils.baseclasses.KerrEccentricEquatorial`.
    """
    def __init__(self, file_directory=None, filenames=None, **kwargs):
        KerrEccentricEquatorial.__init__(self, **kwargs)
        AmplitudeBase.__init__(self, **kwargs)

        if file_directory is None:
            self.file_directory = dir_path + "/../../few/files/"
        else:
            self.file_directory = file_directory

        if filenames is None:
            self.filenames = _DEFAULT_AMPLITUDE_FILENAMES
        else:
            self.filenames = filenames

        self.spin_information_holder_unsorted = [
            None for _ in range(len(self.filenames))
        ]
        for i, fp in enumerate(self.filenames):
            self.spin_information_holder_unsorted[i] = AmpInterp2D(
                fp,
                self.l_arr,
                self.m_arr,
                self.n_arr,
                file_directory=self.file_directory,
                use_gpu=self.use_gpu,
            )

        spin_values_unsorted = [
            sh.a_val_store for sh in self.spin_information_holder_unsorted
        ]
        rearrange_inds = np.argsort(spin_values_unsorted)

        self.spin_values = np.asarray(spin_values_unsorted)[rearrange_inds]
        self.spin_information_holder = [
            self.spin_information_holder_unsorted[i] for i in rearrange_inds
        ]

        pos_neg_n_swap_inds = []
        if self.use_gpu:
            for l, m, n in zip(
                self.l_arr_no_mask.get(),
                self.m_arr_no_mask.get(),
                self.n_arr_no_mask.get(),
            ):
                pos_neg_n_swap_inds.append(self.special_index_map[(l, m, -n)])
        else:
            for l, m, n in zip(
                self.l_arr_no_mask, self.m_arr_no_mask, self.n_arr_no_mask
            ):
                pos_neg_n_swap_inds.append(self.special_index_map[(l, m, -n)])

        self.pos_neg_n_swap_inds = self.xp.asarray(pos_neg_n_swap_inds)

    def get_amplitudes(self, a, p, e, xI, specific_modes=None) -> Union[dict, np.ndarray]:
        """
        Generate Teukolsky amplitudes for a given set of parameters.

        Args:
            a: Dimensionless spin parameter of MBH.
            p: Dimensionless semi-latus rectum.
            e: Eccentricity.
            xI: Cosine of orbital inclination. Only :math:`|x_I| = 1` is currently supported.
            specific_modes: Either indices or mode index tuples of modes to be generated (optional; defaults to all modes).
        Returns:
            If specific_modes is a list of tuples, returns a dictionary of complex mode amplitudes.
            Else, returns an array of complex mode amplitudes.
        """

        # prograde: spin pos, xI pos
        # retrograde: spin pos, xI neg - >  spin neg, xI pos
        assert isinstance(a, float)

        assert np.all(xI == 1.0) or np.all(
            xI == -1.0
        )  # either all prograde or all retrograde
        xI_in = np.ones_like(p) * xI

        signed_spin = a * xI_in[0].item()

        if signed_spin in self.spin_values:
            ind_1 = np.where(self.spin_values == signed_spin)[0][0]
            a_in = np.full_like(p, signed_spin)

            z = self.spin_information_holder[ind_1](
                a_in, p, e, xI_in, specific_modes=specific_modes
            )
            if xI_in[0] == -1 and signed_spin != 0.0:  # retrograde needs mode flip
                z = self.xp.conj(z[:, self.pos_neg_n_swap_inds])

        else:
            ind_above = np.where(self.spin_values > signed_spin)[0][0]
            ind_below = ind_above - 1
            assert ind_above < len(self.spin_values)
            assert ind_below >= 0

            a_above = np.full_like(p, self.spin_values[ind_above])
            a_above_single = a_above[0]
            assert np.all(a_above_single == a_above[0])

            a_below = np.full_like(p, self.spin_values[ind_below])
            a_below_single = a_below[0]
            assert np.all(a_below_single == a_below[0])

            # handle retrograde mode flip (n -> conj(-n))

            if a_below_single < 0:
                apply_conjugate_below = True
                if specific_modes is None:
                    specific_modes_below = self.pos_neg_n_swap_inds
                elif isinstance(specific_modes, self.xp.ndarray):
                    specific_modes_below = self.pos_neg_n_swap_inds[specific_modes]
                elif isinstance(specific_modes, list):
                    specific_modes_below = []
                    for l, m, n in specific_modes:
                        specific_modes_below.append((l, m, -n))
            else:
                apply_conjugate_below = False
                specific_modes_below = specific_modes

            if a_above_single < 0:
                apply_conjugate_above = True
                specific_modes_above = specific_modes_below
            else:
                apply_conjugate_above = False
                specific_modes_above = specific_modes

            if (
                apply_conjugate_above and apply_conjugate_below
            ):  # combine the flags to save a conj call if both retrograde
                apply_conjugate_total = True
                apply_conjugate_above = False
                apply_conjugate_below = False
            else:
                apply_conjugate_total = False

            z_above = self.spin_information_holder[ind_above](
                a_above, p, e, xI_in, specific_modes=specific_modes_above
            )
            z_below = self.spin_information_holder[ind_below](
                a_below, p, e, xI_in, specific_modes=specific_modes_below
            )
            if apply_conjugate_below:
                z_below = z_below.conj()
            if apply_conjugate_above:
                z_above = z_above.conj()
            z = ((z_above - z_below) / (a_above_single - a_below_single)) * (
                signed_spin - a_below_single
            ) + z_below
            if apply_conjugate_total:
                z = z.conj()

        if not isinstance(specific_modes, list):
            return z

        # dict containing requested modes
        else:
            temp = {}
            for i, lmn in enumerate(specific_modes):
                temp[lmn] = z[:, i]
                l, m, n = lmn

                # apply +/- m symmetry
                if m < 0:
                    temp[lmn] = np.conj(temp[lmn])

            return temp


class AmpInterpSchwarzEcc(AmplitudeBase, SchwarzschildEccentric):
    """Calculate Teukolsky amplitudes in the Schwarzschild eccentric regime with a bicubic spline interpolation.

    When called with arguments :math:`(a, p, e, xI)`, these parameters are transformed into a set of
    interpolation coordinates and the bicubic spline interpolant is evaluated at these coordinates for
    all sets of coefficients.

    This class is retained for legacy compatibility with the original Schwarzschild eccentric models. It is
    recommended to use `AmpInterpKerrEqEcc` instead of this class.

    This module is available for GPU and CPU.

    args:
        fp: The coefficients file name in `file_directory`.
        file_directory: The path to the directory containing the coefficients file.
        **kwargs: Optional keyword arguments for the base classes:
            :class:`few.utils.baseclasses.AmplitudeBase`,
            :class:`few.utils.baseclasses.SchwarzschildEccentric`.
    """

    def __init__(self, file_directory=None, filenames=None, **kwargs):
        SchwarzschildEccentric.__init__(self, **kwargs)
        AmplitudeBase.__init__(self, **kwargs)

        if file_directory is None:
            self.file_directory = dir_path + "/../../few/files/"
        else:
            self.file_directory = file_directory

        if filenames is None:
            self.filename = "Teuk_amps_a0.0_lmax_10_nmax_30_new.h5"
        else:
            if isinstance(filenames, list):
                assert len(filenames) == 1
            self.filename = filenames

        # check if user has the necessary data
        # if not, the data will automatically download
        check_for_file_download(self.filename, self.file_directory)

        data = {}
        with h5py.File(os.path.join(self.file_directory, self.filename), "r") as f:
            # load attributes in the right order for correct mode sorting later
            format_string1 = "l{}m{}"
            format_string2 = "n{}k0"
            savestring = "l{}m{}k0n{}"
            grid = f["grid"][:]
            for l, m, n in zip(self.l_arr, self.m_arr, self.n_arr):
                if m >= 0:
                    key1 = format_string1.format(l, m)
                    key2 = format_string2.format(n)
                    tmp = f[key1 + "/" + key2][:]
                    tmp2 = tmp[:, 0] + 1j * tmp[:, 1]
                    data[savestring.format(l, m, n)] = tmp2.T

            # create the coefficients file

            # adjust the grid
            p = grid.T[1].copy()
            e = grid.T[2].copy()
            u = np.round(p_to_y(p, e, use_gpu=False), 8)
            w = e.copy()

            grid_size = p.shape[0]

            unique_u = np.unique(u)
            unique_w = np.unique(w)
            num_u = len(unique_u)
            num_w = len(unique_w)

            data_copy = deepcopy(data)
            for mode, vals in data.items():
                data_copy[mode] = data_copy[mode].reshape(num_w, num_u)

            data = deepcopy(data_copy)

            data = {name: val[:, ::-1] for name, val in data.items()}

            spl2D = {
                name: [
                    RectBivariateSpline(unique_w, unique_u, val.real, kx=3, ky=3),
                    RectBivariateSpline(unique_w, unique_u, val.imag, kx=3, ky=3),
                ]
                for name, val in data.items()
            }

            mode_keys = list(data.keys())
            num_teuk_modes = len(mode_keys)

            first_key = list(spl2D.keys())[0]
            example_spl = spl2D[first_key][0]
            tck_last_entry = np.zeros((len(data), 2, grid_size))
            for i, mode in enumerate(mode_keys):
                tck_last_entry[i, 0] = spl2D[mode][0].tck[2]
                tck_last_entry[i, 1] = spl2D[mode][1].tck[2]

            self.tck = [
                self.xp.asarray(example_spl.tck[0]),
                self.xp.asarray(example_spl.tck[1]),
                self.xp.asarray(tck_last_entry.copy()),
            ]

        self.num_teuk_modes = num_teuk_modes

        self.len_indiv_c = tck_last_entry.shape[-1]

    @property
    def interp2D(self) -> callable:
        """GPU or CPU interp2D"""
        interp2D = interp2D_cpu if not self.use_gpu else interp2D_gpu
        return interp2D

    def get_amplitudes(self, a, p, e, xI, specific_modes=None) -> Union[dict,np.ndarray]:
        """
        Generate Teukolsky amplitudes for a given set of parameters.

        Args:
            a: Dimensionless spin parameter of MBH (must be equal to zero).
            p: Dimensionless semi-latus rectum.
            e: Eccentricity.
            xI: Cosine of orbital inclination. Only :math:`x_I = 1` is currently supported.
            specific_modes: Either indices or mode index tuples of modes to be generated (optional; defaults to all modes).
        Returns:
            If specific_modes is a list of tuples, returns a dictionary of complex mode amplitudes.
            Else, returns an array of complex mode amplitudes.
        """
        assert a == 0.0

        assert np.all(xI == 1.0)

        try:
            p_cpu, e_cpu = p.get().copy(), e.get().copy()
        except AttributeError:
            p_cpu, e_cpu = p.copy(), e.copy()

        p = self.xp.asarray(p)
        e = self.xp.asarray(e)

        # TODO: make this GPU accessible
        u = self.xp.asarray(p_to_y(p_cpu, e_cpu, use_gpu=False))

        w = e.copy()

        tw, tu, c = self.tck[:3]
        kw = ku = 3

        # standard Numpy broadcasting
        if w.shape != u.shape:
            w, u = np.broadcast_arrays(w, u)

        shape = w.shape
        w = w.ravel()
        u = u.ravel()

        if w.size == 0 or u.size == 0:
            return np.zeros(shape, dtype=self.tck[2].dtype)

        nw = tw.shape[0]
        nu = tu.shape[0]
        mw = w.shape[0]
        mu = u.shape[0]

        assert mw == mu

        # TODO: adjustable

        if specific_modes is None:
            mode_indexes = self.xp.arange(self.num_teuk_modes)

        else:
            if isinstance(specific_modes, self.xp.ndarray):
                mode_indexes = specific_modes
            elif isinstance(
                specific_modes, list
            ):  # the following is slow and kills efficiency
                mode_indexes = self.xp.zeros(len(specific_modes), dtype=self.xp.int32)
                for i, (l, m, n) in enumerate(specific_modes):
                    try:
                        mode_indexes[i] = np.where(
                            (self.l_arr == l)
                            & (self.m_arr == abs(m))
                            & (self.n_arr == n)
                        )[0]
                    except:
                        raise Exception(f"Could not find mode index ({l},{m},{n}).")

        # TODO: perform this in the kernel
        c_in = c[mode_indexes].flatten()

        num_indiv_c = 2 * len(mode_indexes)  # Re and Im
        len_indiv_c = self.len_indiv_c

        z = self.xp.zeros((num_indiv_c * mw))

        self.interp2D(
            z, tw, nw, tu, nu, c_in, kw, ku, w, mw, u, mu, num_indiv_c, len_indiv_c
        )

        z = z.reshape(num_indiv_c // 2, 2, mw).transpose(2, 1, 0)

        z = z[:, 0] + 1j * z[:, 1]

        if not isinstance(specific_modes, list):
            return z

        # dict containing requested modes
        else:
            temp = {}
            for i, lmn in enumerate(specific_modes):
                temp[lmn] = z[:, i]
                l, m, n = lmn

                # apply +/- m symmetry
                if m < 0:
                    temp[lmn] = np.conj(temp[lmn])

            return temp

    def __reduce__(self):
        return (self.__class__, (self.file_directory, self.filename))


def _spline_coefficients_to_file(fp, l_arr, m_arr, n_arr, file_directory=None):
    data = {}
    # get information about this specific model from the file
    with h5py.File(os.path.join(file_directory, fp), "r") as f:
        # load attributes in the right order for correct mode sorting later
        kerr_format_string = "l{}m{}k0n{}"
        grid = f["grid"][:]
        for l, m, n in zip(l_arr, m_arr, n_arr):
            if m >= 0:
                key1 = kerr_format_string.format(l, m, n)
                tmp = f[key1][:]
                tmp2 = tmp[:, 0] + 1j * tmp[:, 1]
                data[key1] = tmp2

    # create the coefficients file

    # adjust the grid
    a = grid.T[0].copy()
    p = grid.T[1].copy()
    e = grid.T[2].copy()
    xI = grid.T[3].copy()
    u = np.round(grid.T[4].copy(), 8)  # fix rounding errors in the files
    sep = grid.T[5].copy()
    w = grid.T[6].copy()

    assert np.all(a == a[0])
    assert np.all(xI == xI[0])

    # retrograde needs sign flip to be applied to a
    a *= xI
    a_val_store = a[0]

    out_fp = f"KerrEqEccAmpCoeffs_a{a_val_store:.3f}.h5"
    outfile = h5py.File(os.path.join(file_directory, out_fp), "w")
    outfile.attrs["signed_spin"] = a_val_store
    outfile.attrs["is_coefficients"] = True

    grid_size = p.shape[0]

    unique_u = np.unique(u)
    unique_w = np.unique(w)
    num_u = len(unique_u)
    num_w = len(unique_w)

    data_copy = deepcopy(data)
    for mode, vals in data.items():
        data_copy[mode] = data_copy[mode].reshape(num_w, num_u)

    data = deepcopy(data_copy)

    data = {name: val[:, ::-1] for name, val in data.items()}

    spl2D = {
        name: [
            RectBivariateSpline(unique_w, unique_u, val.real, kx=3, ky=3),
            RectBivariateSpline(unique_w, unique_u, val.imag, kx=3, ky=3),
        ]
        for name, val in data.items()
    }

    mode_keys = list(data.keys())
    num_teuk_modes = len(mode_keys)

    outfile.attrs["num_teuk_modes"] = num_teuk_modes

    first_key = list(spl2D.keys())[0]
    example_spl = spl2D[first_key][0]
    tck_last_entry = np.zeros((len(data), 2, grid_size))
    for i, mode in enumerate(mode_keys):
        tck_last_entry[i, 0] = spl2D[mode][0].tck[2]
        tck_last_entry[i, 1] = spl2D[mode][1].tck[2]

    degrees = example_spl.degrees

    len_indiv_c = tck_last_entry.shape[-1]

    outfile.attrs["spline_degree_x"] = degrees[0]
    outfile.attrs["spline_degree_y"] = degrees[1]
    outfile.attrs["points_per_modegrid"] = len_indiv_c

    outfile.create_dataset("x1", data=example_spl.tck[0])
    outfile.create_dataset("x2", data=example_spl.tck[1])
    outfile.create_dataset("c", data=tck_last_entry.copy())

    outfile.close()

    return out_fp


if __name__ == "__main__":
    # try and instantiate the amplitude class
    spin_values = np.r_[np.linspace(0.0, 0.9, 10), 0.95, 0.99]
    spin_values = np.r_[-np.flip(spin_values)[:-1], spin_values]

    base_path = "Teuk_amps_a{:.2f}_{}lmax_10_nmax_50_new_m+.h5"
    filepaths = []
    for spin in spin_values:
        part1 = abs(spin)
        if spin < 0:
            part2 = "r_"
        elif spin > 0:
            part2 = "p_"
        elif spin == 0:
            part2 = ""
        filepaths.append(base_path.format(part1, part2))

    # running this should auto-produce coefficients files
    AmpInterpKerrEqEcc(filenames=filepaths, file_directory="../../processed_amplitudes")

    amp = AmpInterpKerrEqEcc()
    print(amp(0.0, np.array([10.0]), np.array([0.3]), np.array([1.0])))
