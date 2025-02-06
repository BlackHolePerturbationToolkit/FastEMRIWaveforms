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
import pathlib

import numpy as np
import h5py
from scipy.interpolate import RectBivariateSpline

# Cython/C++ imports

# Python imports
from ..utils.baseclasses import (
    SchwarzschildEccentric,
    ParallelModuleBase,
    KerrEccentricEquatorial,
)
from .base import AmplitudeBase
from ..utils.citations import REFERENCE
from ..utils.mappings import a_of_z, kerrecceq_forward_map, kerrecceq_legacy_p_to_u, schwarzecc_p_to_y

from ..cutils.fast import interp2D as interp2D_gpu
from ..cutils.cpu import interp2D as interp2D_cpu

from typing import List, Optional, Union, Tuple

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
        w_knots: The knots in the w direction.
        u_knots: The knots in the u direction.
        coefficients: The Teukolsky mode amplitudes to be interpolated.
        l_arr: Array of :math:`\ell` mode indices.
        m_arr: Array of :math:`m` mode indices.
        n_arr: Array of :math:`n` mode indices.
        **kwargs: Optional keyword arguments for the base class:
            :class:`few.utils.baseclasses.AmplitudeBase`,
            :class:`few.utils.baseclasses.ParallelModuleBase`.
    """

    def __init__(
            self,
            w_knots: np.ndarray,
            u_knots: np.ndarray,
            coefficients: np.ndarray,
            l_arr: np.ndarray,
            m_arr: np.ndarray,
            n_arr: np.ndarray,
            **kwargs
        ):
        ParallelModuleBase.__init__(self, **kwargs)
        AmplitudeBase.__init__(self, **kwargs)

        self.l_arr = l_arr
        self.m_arr = m_arr
        self.n_arr = n_arr

        self.num_teuk_modes = coefficients.shape[0]
        """int: Total number of mode amplitude grids this interpolant stores."""
        
        self.knots = [
            self.xp.asarray(w_knots),
            self.xp.asarray(u_knots),
        ]
        """list[np.ndarray]: Arrays holding spline knots in each dimension."""

        self.coeff = coefficients
        """np.ndarray: Array holding all spline coefficient information."""

        # for mode_ind in range(self.num_teuk_modes):
        #     spl1 = RectBivariateSpline(w_knots, u_knots, coefficients[mode_ind,0], kx=3, ky=3)
        #     spl2 = RectBivariateSpline(w_knots, u_knots, coefficients[mode_ind,1], kx=3, ky=3)

        #     self.coeff[mode_ind,0] = spl1.tck[2].flatten()
        #     self.coeff[mode_ind,1] = spl2.tck[2].flatten()

        self.len_indiv_c = self.coeff.shape[2]
        """int: Total number of coefficients per mode amplitude grid."""

    @property
    def interp2D(self) -> callable:
        """GPU or CPU interp2D"""
        interp2D = interp2D_cpu if not self.use_gpu else interp2D_gpu
        return interp2D

    @classmethod
    def module_references(cls) -> list[REFERENCE]:
        """Return citations related to this module"""
        return [REFERENCE.ROMANNET] + super(AmpInterp2D, cls).module_references()

    @property
    def gpu_capability(self):
        """Confirms GPU capability"""
        return True

    def __call__(self, w: Union[float,np.ndarray], u: Union[float,np.ndarray], *args, mode_indexes: Optional[np.ndarray]=None, **kwargs) -> np.ndarray:
        """
        Evaluate the spline or its derivatives at given positions.

        Args:
            w: Eccentricity interpolation parameter.
            u: Dimensionless semi-latus rectum interpolation parameter.
            mode_indexes: Array indices of modes to be generated (optional; defaults to all modes).
        Returns:
            Complex Teukolsky mode amplitudes at the requested points.
        """

        w = self.xp.asarray(w)
        u = self.xp.asarray(u)

        tw, tu = self.knots
        c = self.coeff
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

        if mode_indexes is None:
            mode_indexes = self.xp.arange(self.num_teuk_modes)
        
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
        return z

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
        **kwargs: Optional keyword arguments for the base classes:
            :class:`few.utils.baseclasses.AmplitudeBase`,
            :class:`few.utils.baseclasses.KerrEccentricEquatorial`.
    """
    def __init__(self, filename: Optional[str] = None, downsample_Z=1, **kwargs):
        KerrEccentricEquatorial.__init__(self, **kwargs)
        AmplitudeBase.__init__(self, **kwargs)

        if filename is None:
            self.filename = "ZNAmps_l10_m10_n55.h5"
        else:
            self.filename = filename

        from few import get_file_manager

        file_path = get_file_manager().get_file(self.filename)

        with h5py.File(file_path, "r") as f:
            coeffsA = f["CoeffsRegionA"][()]
            w_knots = f['w_knots'][()]
            u_knots = f['u_knots'][()]
            z_knots = f["z_knots"][()]

            z_knots = z_knots[::downsample_Z]
            coeffsA = coeffsA[::downsample_Z]

            self.spin_information_holder_A = [
                None for _ in range(z_knots.size)
            ]

            for i in range(z_knots.size):
                self.spin_information_holder_A[i] = AmpInterp2D(
                    w_knots,
                    u_knots,
                    coeffsA[i],
                    self.l_arr,
                    self.m_arr,
                    self.n_arr,
                    use_gpu=self.use_gpu,
                )

            try:
                coeffsB = f["CoeffsRegionB"][()]
                coeffsB = coeffsB[::downsample_Z]
                self.spin_information_holder_B = [
                    None for _ in range(z_knots.size)
                ]

                for i in range(z_knots.size):
                    self.spin_information_holder_B[i] = AmpInterp2D(
                        w_knots,
                        u_knots,
                        coeffsB[i],
                        self.l_arr,
                        self.m_arr,
                        self.n_arr,
                        use_gpu=self.use_gpu,
                    )
            except KeyError:
                pass

            self.w_values = w_knots
            self.u_values = u_knots
            self.z_values = z_knots


    def evaluate_interpolant_at_index(self, index, region_A_mask, w, u, mode_indexes):
        z_out = self.xp.zeros((region_A_mask.size, self.num_modes_eval), dtype=self.xp.complex128)

        if self.xp.any(region_A_mask):
            z_out[region_A_mask, :] = self.spin_information_holder_A[index](
                    w[region_A_mask], u[region_A_mask], mode_indexes=mode_indexes
                )
        
        if self.xp.any(~region_A_mask):
            z_out[~region_A_mask, :] = self.spin_information_holder_B[index](
                    w[~region_A_mask], u[~region_A_mask], mode_indexes=mode_indexes
                )
        
        return z_out

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

        p = np.atleast_1d(p)
        e = np.atleast_1d(e)
        xI = np.atleast_1d(xI)

        assert np.all(xI == 1.0) or np.all(
            xI == -1.0
        )  # either all prograde or all retrograde
        xI_in = np.ones_like(p) * xI

        signed_spin = a * xI_in[0].item()
        a_in = np.full_like(p, signed_spin)
        xI_in = np.abs(xI_in)
        
        if specific_modes is not None:
            self.num_modes_eval = len(specific_modes)
            if isinstance(
                specific_modes, list
            ):
                specific_modes_arr = self.xp.asarray(specific_modes)
                mode_indexes = self.special_index_map_arr[specific_modes_arr[:,0], specific_modes_arr[:,1], specific_modes_arr[:,2]]
                if self.xp.any(mode_indexes == -1):
                        failed_mode = specific_modes_arr[self.xp.where(mode_indexes == -1)[0][0]]
                        raise ValueError(f"Could not find mode index ({failed_mode[0]},{failed_mode[1]},{failed_mode[2]}).")
            else:
                mode_indexes = specific_modes
        else:
            mode_indexes = self.xp.arange(self.num_teuk_modes)
            self.num_modes_eval = self.num_teuk_modes

        u, w, y, z, region_mask = kerrecceq_forward_map(a_in, p, e, xI_in, use_gpu=self.use_gpu, return_mask=True, kind="amplitude")
        z_check = z[0]

        for elem in [u, w, z]:
            if np.any((elem < 0)|(elem > 1)):
                raise ValueError("Amplitude interpolant accessed out-of-bounds.")

        if z_check in self.z_values:
            ind_1 = np.where(self.z_values == z_check)[0][0]

            z = self.evaluate_interpolant_at_index(
                ind_1, region_mask, w, u, mode_indexes=mode_indexes
            )

        else:
            ind_above = np.where(self.z_values > z_check)[0][0]
            ind_below = ind_above - 1
            assert ind_above < len(self.z_values)
            assert ind_below >= 0

            z_above = self.z_values[ind_above]
            Amp_above = self.evaluate_interpolant_at_index(
                ind_above, region_mask, w, u, mode_indexes
            )

            z_below = self.z_values[ind_below]
            Amp_below = self.evaluate_interpolant_at_index(
                ind_below, region_mask, w, u, mode_indexes
            )

            z = ((Amp_above - Amp_below) / (z_above - z_below)) * (
                z_check - z_below
            ) + Amp_below

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
                    temp[lmn] = self.xp.conj(temp[lmn])

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
        filenames: The coefficients file names.
        **kwargs: Optional keyword arguments for the base classes:
            :class:`few.utils.baseclasses.AmplitudeBase`,
            :class:`few.utils.baseclasses.SchwarzschildEccentric`.
    """

    def __init__(self, filenames: Optional[List[str]] = None, **kwargs):
        SchwarzschildEccentric.__init__(self, **kwargs)
        AmplitudeBase.__init__(self, **kwargs)

        if filenames is None:
            self.filename = "Teuk_amps_a0.0_lmax_10_nmax_30_new.h5"
        else:
            if isinstance(filenames, list):
                assert len(filenames) == 1
            self.filename = filenames

        # check if user has the necessary data
        # if not, the data will automatically download
        from few import get_file_manager

        file_path = get_file_manager().get_file(self.filename)

        data = {}
        with h5py.File(file_path, "r") as f:
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
            u = np.round(schwarzecc_p_to_y(p, e, use_gpu=False), 8)
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

    def get_amplitudes(
        self, a, p, e, xI, specific_modes=None
    ) -> Union[dict, np.ndarray]:
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

        p = self.xp.asarray(p)
        e = self.xp.asarray(e)

        u = self.xp.asarray(schwarzecc_p_to_y(p, e, use_gpu=self.use_gpu))
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
                    except:  # noqa: E722
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
        return (self.__class__, (self.filename,))


def _spline_coefficients_to_file(
    fp: os.PathLike, l_arr, m_arr, n_arr, output_directory: os.PathLike
) -> pathlib.Path:
    data = {}
    # get information about this specific model from the file
    with h5py.File(fp, "r") as f:
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
    # e = grid.T[2].copy()
    xI = grid.T[3].copy()
    u = np.round(grid.T[4].copy(), 8)  # fix rounding errors in the files
    # sep = grid.T[5].copy()
    w = grid.T[6].copy()

    assert np.all(a == a[0])
    assert np.all(xI == xI[0])

    # retrograde needs sign flip to be applied to a
    a *= xI
    a_val_store = a[0]

    out_fp = f"KerrEqEccAmpCoeffs_a{a_val_store:.3f}.h5"
    out_filepath = pathlib.Path(output_directory) / out_fp
    outfile = h5py.File(out_filepath, "w")
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

    return out_filepath


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
    AmpInterpKerrEqEcc(filenames=filepaths)

    amp = AmpInterpKerrEqEcc()
    print(amp(0.0, np.array([10.0]), np.array([0.3]), np.array([1.0])))
