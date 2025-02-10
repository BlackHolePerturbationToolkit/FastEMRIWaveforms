# Collection of base classes for FastEMRIWaveforms Packages

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

"""
The :code:`few.utils.baseclasses` module contains abstract base classes for the
various modules. When creating new modules, these classes should be used to maintain
a common interface and pass information related to each model.
"""


from abc import ABC
import numpy as np

from typing import Union, Optional

# try to import cupy
from ..cutils import fast as fast_backend
gpu_available = fast_backend.is_gpu

# Python imports
from ..utils.constants import *
from ..utils.citations import Citable, REFERENCE
from ..utils.mappings import kerrecceq_forward_map

from few.utils.globals import get_logger
few_logger = get_logger()

class ParallelModuleBase(Citable, ABC):
    """Base class for modules that can use GPUs.

    This class mainly handles setting GPU usage.

    args:
        use_gpu: If True, use GPU resources. Default is False.

    """

    def __init__(self, *args, use_gpu:bool=False, **kwargs):
        self.use_gpu = use_gpu

        # checks if gpu capability is available if requested
        self.sanity_check_gpu(use_gpu)

    @property
    def xp(self) -> object:
        """Cupy or Numpy"""
        xp = np if not self.use_gpu else fast_backend.xp
        return xp

    @classmethod
    @property
    def gpu_capability(self):
        """Indicator if the module has gpu capability"""
        raise NotImplementedError

    @classmethod
    def __call__(*args, **kwargs):
        """Method to call waveform model"""
        raise NotImplementedError

    def sanity_check_gpu(self, use_gpu: bool):
        """Check if this class has GPU capability

        If the user is requesting GPU usage, this will confirm the class has
        GPU capabilites.

        Args:
            use_gpu: If True, the user is requesting GPU usage.

        Raises:
            ValueError: The user is requesting GPU usage, but this class does
                not have that capability.

        """
        if (self.gpu_capability is False or gpu_available is False) and use_gpu is True:
            if self.gpu_capability is False:
                raise ValueError(
                    "The use_gpu kwarg is True, but this class does not have GPU capabilites."
                )
            else:
                raise ValueError("Either a GPU and/or CuPy is not available.")

    def adjust_gpu_usage(self, use_gpu: bool, kwargs: Union[list, dict]):
        """Adjust all inputs for gpu usage

        If user wants to use gpu, it will change all :code:`kwargs` in
        so that :code:`use_gpu=True`.

        args:
            use_gpu (bool): If True, use gpu resources.
            kwargs (list of dicts or dict): List of kwargs dictionaries or
                single dictionary for each constituent class in the
                waveform generator.

        """

        if use_gpu:
            if isinstance(kwargs, list):
                for i, kwargs_i in enumerate(kwargs):
                    kwargs[i]["use_gpu"] = use_gpu
            else:
                kwargs["use_gpu"] = use_gpu

        return kwargs

class SphericalHarmonic(ParallelModuleBase):
    r"""Base class for waveforms constructed in a spherical harmonic basis.

    This class creates shared traits between different implementations of the
    same model. Particularly, this class includes descriptive traits as well as
    the sanity check class method that should be used in all implementations of
    this model. This method can be overwritten if necessary. Here we describe
    the overall qualities of this base class.

    Currently, Eq. :eq:`emri_wave_eq` is reduced to the equatorial plane.
    Therefore, we only concerned with :math:`(l,m,n)` indices and
    the parameters :math:`(a, p,e)` because :math:`k=|\iota|=0`. Therefore, in this
    model we calculate :math:`A_{lmn}` and :math:`\Phi_{mn}=m\Phi_\phi+n\Phi_r`.
    As we assume the amplitudes have been remapped to a spherical harmonic basis,
    we use -2 spin-weighted spherical harmonics (:math:`(s=-2)Y_{l,m}`) in place
    of the more generic angular function from Eq. :eq:`emri_wave_eq`.

    args:
        use_gpu (bool, optional): If True, will allocate arrays on the GPU.
            Default is False.

    """
    def __init__(self, *args:Optional[list], use_gpu: bool=False, **kwargs:Optional[dict]):
        ParallelModuleBase.__init__(self, *args, use_gpu=use_gpu, **kwargs)

        # fill all lmn mode values
        md = []
        for l in  range(2, self.lmax+1):
            for m in range(0, l + 1):
                for n in range(-self.nmax, self.nmax+1):
                    md.append([l, m, n])

        # total number of modes in the model
        self.num_modes = len(md)
        self.num_teuk_modes = self.num_modes
        """int: Number of Teukolsky modes in the model."""

        # mask for m == 0
        m0mask = self.xp.array(
            [
                m == 0
                for l in range(2, self.lmax + 1)
                for m in range(0, l + 1)
                for n in range(-self.nmax, self.nmax + 1)
            ]
        )

        # sorts so that order is m=0, m<0, m>0
        self.m0sort = m0sort = self.xp.concatenate(
            [
                self.xp.arange(self.num_teuk_modes)[m0mask],
                self.xp.arange(self.num_teuk_modes)[~m0mask],
            ]
        )
        """1D np.ndarray: Sorted mode indices with m=0 first, then m<0, then m>0."""

        # sorts the mode indexes
        md = self.xp.asarray(md).T[:, m0sort].astype(self.xp.int32)

        # store l m and n values
        self.l_arr_no_mask = md[0]
        """1D np.ndarray: Array of l values for each mode before masking."""
        self.m_arr_no_mask = md[1]
        """1D np.ndarray: Array of m values for each mode before masking."""
        self.n_arr_no_mask = md[2]
        """1D np.ndarray: Array of n values for each mode before masking."""

        # adjust with .get method for cupy
        try:
            lmn_indices = {tuple(md_i): i for i, md_i in enumerate(md.T.get())}

        except AttributeError:
            lmn_indices = {tuple(md_i): i for i, md_i in enumerate(md.T)}

        self.lmn_indices = lmn_indices
        """dict: Dictionary of mode indices to mode number."""

        # store the mask as m != 0 is True
        self.m0mask = self.m_arr_no_mask != 0
        """1D np.ndarray: Mask for m != 0."""
        # number of m >= 0
        self.num_m_zero_up = len(self.m_arr_no_mask)
        """int: Number of modes with m >= 0."""
        # number of m == 0
        self.num_m0 = len(self.xp.arange(self.num_teuk_modes)[m0mask])
        """int: Number of modes with m == 0."""
        # number of m > 0
        self.num_m_1_up = self.num_m_zero_up - self.num_m0
        """int: Number of modes with m > 0."""
        # create final arrays to include -m modes
        self.l_arr = self.xp.concatenate([self.l_arr_no_mask, self.l_arr_no_mask[self.m0mask]])
        """1D np.ndarray: Array of l values for each mode."""
        self.m_arr = self.xp.concatenate([self.m_arr_no_mask, -self.m_arr_no_mask[self.m0mask]])
        """1D np.ndarray: Array of m values for each mode."""
        self.n_arr = self.xp.concatenate([self.n_arr_no_mask, self.n_arr_no_mask[self.m0mask]])
        """1D np.ndarray: Array of n values for each mode."""

        # mask for m >= 0
        self.m_zero_up_mask = self.m_arr >= 0
        """1D np.ndarray: Mask for m >= 0."""

        # find unique sets of (l,m)
        # create inverse array to build full (l,m,n) from unique l and m
        # also adjust for cupy
        try:
            temp, self.inverse_lm = np.unique(
                np.asarray([self.l_arr.get(), self.m_arr.get()]).T,
                axis=0,
                return_inverse=True,
            )

        except AttributeError:
            temp, self.inverse_lm = np.unique(
                np.asarray([self.l_arr, self.m_arr]).T, axis=0, return_inverse=True
            )

        # unique values of l and m
        unique_l, unique_m = self.xp.asarray(temp).T
        self.unique_l = unique_l
        """1D np.ndarray: Array of unique l values."""
        self.unique_m = unique_m
        """1D np.ndarray: Array of unique m values."""

        # number of unique values
        self.num_unique_lm = len(self.unique_l)
        """int: Number of unique (l,m) values."""

        # creates special maps to the modes
        self.index_map = {}
        """dict: Maps mode index to mode tuple."""
        self.special_index_map = {}  # maps the minus m values to positive m
        """dict: Maps mode index to mode tuple with m > 0."""
        self.index_map_arr = self.xp.zeros((self.lmax + 1, self.lmax * 2 + 1, self.nmax * 2 + 1), dtype=self.xp.int32) - 1
        """np.ndarray: Array mapping mode tuple to mode index - used for fast indexing. Returns -1 if mode does not exist."""
        self.special_index_map_arr = self.xp.zeros((self.lmax + 1, self.lmax * 2 + 1, self.nmax * 2 + 1), dtype=self.xp.int32) - 1
        """np.ndarray: Array mapping mode tuple to mode index with m > 0 - used for fast indexing. Returns -1 if mode does not exist."""
        for i, (l, m, n) in enumerate(zip(self.l_arr, self.m_arr, self.n_arr)):
            try:
                l = l.item()
                m = m.item()
                n = n.item()

            except AttributeError:
                pass

            # regular index to mode tuple
            self.index_map[(l, m, n)] = i
            self.index_map_arr[l, m, n] = i
            # special map that gives m < 0 indices as m > 0 indices
            sp_i = (
                i if i < self.num_modes else i - self.num_m_1_up
            )
            self.special_index_map[(l, m, n)] = sp_i
            self.special_index_map_arr[l, m, n] = sp_i

    def sanity_check_viewing_angles(self, theta: float, phi: float):
        """Sanity check on viewing angles.

        Make sure parameters are within allowable ranges.

        args:
            theta (double): Polar viewing angle.
            phi (double): Azimuthal viewing angle.

        Returns:
            tuple: (theta, phi). Phi is wrapped.

        Raises:
            ValueError: If any of the angular values are not allowed.

        """
        # if theta < 0.0 or theta > np.pi:
        #    raise ValueError("theta must be between 0 and pi.")

        phi = phi % (2 * np.pi)
        return (theta, phi)

    def sanity_check_traj(self, a: float, p:np.ndarray, e:np.ndarray, xI:np.ndarray):
        """Sanity check on parameters output from the trajectory module.

        Make sure parameters are within allowable ranges.

        args:
            a: Dimensionless spin of massive black hole.
            p: Array of semi-latus rectum values produced by
                the trajectory module.
            e: Array of eccentricity values produced by
                the trajectory module.
            xI: Array of cosine(inclination) values produced by the trajectory module.

        Raises:
            ValueError: If any of the trajectory points are not allowed.
            warn: If any points in the trajectory are allowable,
                but outside calibration region.

        """

        if np.any(e < 0.0):
            raise ValueError("Members of e array are less than zero.")

        if np.any(p < 0.0):
            raise ValueError("Members of p array are less than zero.")

        if np.any((a < 0.0) | (a > 1.0)):
            raise ValueError("Members of a array are not within the range [0, 1].")

        if np.any(abs(xI) > 1.0):
            raise ValueError("Members of xI array have a magnitude greater than one.")

class SchwarzschildEccentric(SphericalHarmonic):
    """
    Schwarzschild eccentric base class.

    Args:
        use_gpu: If True, will allocate arrays on the GPU. Default is False.
        lmax: Maximum l value for the model. Default is 10.
        nmax: Maximum n value for the model. Default is 30.
        ndim: Number of phases in the model. Default is 2.
    """
    def __init__(
            self,
            *args:Optional[list],
            use_gpu: bool=False,
            lmax:int = 10,
            nmax:int = 30,
            ndim:int = 2,
            **kwargs: Optional[dict]
        ):
        # some descriptive information
        self.background = "Schwarzschild"
        """str: The spacetime background for this model. Is Schwarzschild."""
        self.descriptor = "eccentric"
        """str: Description of the inspiral trajectory properties for this model. Is eccentric."""
        self.frame = "source"
        """str: Frame in which source is generated. Is source frame."""
        self.needs_Y = False
        """bool: If True, model expects inclination parameter Y (rather than xI). Is False."""

        # set mode index settings
        self.lmax = lmax
        self.nmax = nmax

        self.ndim = ndim

        SphericalHarmonic.__init__(self, *args, use_gpu=use_gpu, **kwargs)

    @property
    def gpu_capability(self):
        """Confirms GPU capability"""
        return True

    def sanity_check_init(self, M: float, mu: float, a: float, p0: float, e0: float, xI: float) -> tuple[float, float]:
        r"""Sanity check initial parameters.

        Make sure parameters are within allowable ranges.

        args:
            M: Massive black hole mass in solar masses.
            mu: compact object mass in solar masses.
            a: Dimensionless spin of massive black hole :math:`(a = 0)`.
            p0: Initial semilatus rectum (dimensionless)
                :math:`(10\leq p_0\leq 16 + 2e_0)`. See the documentation for
                more information on :math:`p_0 \leq 10.0`.
            e0: Initial eccentricity :math:`(0\leq e_0\leq0.7)`.
            xI: Initial cosine(inclination) :math:`(x_I = 1)`.

        Returns:
            (a_fix, xI_fix): a and xI in the correct convention (a >= 0).
        
        Raises:
            ValueError: If any of the parameters are not allowed.

        """
        for val, key in [[M, "M"], [p0, "p0"], [e0, "e0"], [mu, "mu"]]:
            test = val < 0.0
            if test:
                raise ValueError("{} is negative. It must be positive.".format(key))

        if e0 > 0.75:
            raise ValueError(
                "Initial eccentricity above 0.75 not allowed. (e0={})".format(e0)
            )

        if e0 < 0.0:
            raise ValueError(
                "Initial eccentricity below 0.0 not physical. (e0={})".format(e0)
            )

        if p0 < 10.0:
            if p0 < 7 * (6.0 + 2 * e0) - 41.9:
                raise ValueError(
                    "This p0 ({}) and e0 ({}) combination is outside of our domain of validity.".format(
                        p0, e0
                    )
                )

        if p0 > 16.0 + 2 * e0:
            raise ValueError(
                "Initial p0 is too large (p0={}). Must be 10 <= p0 <= 16 + 2 * e.".format(
                    p0
                )
            )

        if a != 0.:
            raise ValueError(
                "Spin must be zero for Schwarzschild inspirals."
            )

        if abs(xI) != 1.:
            raise ValueError(
                "For equatorial orbits, xI must be either 1 or -1."
            )
        
        return a, xI


class KerrEccentricEquatorial(SphericalHarmonic):
    """
    Kerr eccentric equatorial base class.

    Args:
        use_gpu: If True, will allocate arrays on the GPU. Default is False.
        lmax: Maximum l value for the model. Default is 10.
        nmax: Maximum n value for the model. Default is 55.
        ndim: Number of phases in the model. Default is 2.
    """

    def __init__(
            self,
            *args: Optional[list],
            use_gpu:bool=False,
            lmax:int= 10,
            nmax:int = 55,
            ndim:int = 2,
            **kwargs:Optional[dict]
        ):
        # some descriptive information
        self.background = "Kerr"
        """str: The spacetime background for this model. Is Kerr."""
        self.descriptor = "eccentric equatorial"
        """str: Description of the inspiral trajectory properties for this model. Is eccentric equatorial."""
        self.frame = "source"
        """str: Frame in which source is generated. Is source frame."""
        self.needs_Y = False
        """bool: If True, model expects inclination parameter Y (rather than xI). Is False."""

        # set mode index settings
        self.lmax = lmax
        self.nmax = nmax

        self.ndim = ndim

        SphericalHarmonic.__init__(self, *args, use_gpu=use_gpu, **kwargs)

    @property
    def gpu_capability(self):
        """Confirms GPU capability"""
        return True

    def sanity_check_init(self, M: float, mu:float, a:float, p0:float, e0:float, xI:float) -> tuple[float, float]:
        r"""Sanity check initial parameters.

        Make sure parameters are within allowable ranges.

        args:
            M: Massive black hole mass in solar masses.
            mu: compact object mass in solar masses.
            a: Dimensionless spin of massive black hole.
            p0: Initial semilatus rectum (dimensionless)
                :math:`(10\leq p_0\leq 16 + 2e_0)`. See the documentation for
                more information on :math:`p_0 \leq 10.0`.
            e0: Initial eccentricity :math:`(0\leq e_0\leq0.7)`.
            xI: Initial cosine(inclination) :math:`(|x_I| = 1)`.

        Returns:
            (a_fix, xI_fix): a and xI in the correct convention (a >= 0).
        
        Raises:
            ValueError: If any of the parameters are not allowed.

        """
        # TODO: update function when grids replaced

        for val, key in [[M, "M"], [p0, "p0"], [e0, "e0"], [mu, "mu"]]:
            test = val < 0.0
            if test:
                raise ValueError("{} is negative. It must be positive.".format(key))
        
        if a < 0:
            # flip convention
            few_logger.warning(
                "Negative spin magnitude detected. Flipping sign of a and xI to match convention."
            )
            a = -a
            xI = -xI

        if a > 0.999:
            raise ValueError(
                "Larger black hole spin magnitude above 0.999 is outside of our domain of validity."
            )

        # transform parameters and check they are within bounds
        a_sign = a * xI
        xI_in = abs(xI)
        grid_coords = kerrecceq_forward_map(a_sign, p0, e0, xI_in)

        if np.isnan(grid_coords[0]):
            raise ValueError(
                f"This value of p0 ({p0}) is too close to the separatrix for our model."
            )
        elif grid_coords[0] > 1.000001:
            raise ValueError(
                f"This value of p0 ({p0}) is outside of our domain of validity."
            )
        if grid_coords[1] < -1e-6 or grid_coords[1] > 1.000001:
            raise ValueError(
                f"This a ({a}), p0 ({p0}) and e0 ({e0}) combination is outside of our domain of validity."
                )

        if abs(xI) != 1.:
            raise ValueError(
                "For equatorial orbits, xI must be either 1 or -1."
            )
        
        return a, xI


class Pn5AAK(ParallelModuleBase):
    """Base class for Pn5AAK waveforms.

    This class contains some basic checks and information for AAK waveforms
    with a 5PN trajectory model. Please see :class:`few.waveform.Pn5AAKWaveform`
    for more details.

    args:
        use_gpu: If True, will allocate arrays on the GPU.
            Default is False.

    """

    def __init__(self, *args:Optional[list], use_gpu: bool=False, **kwargs:Optional[dict]):
        ParallelModuleBase.__init__(self, *args, use_gpu=use_gpu, **kwargs)

        # some descriptive information
        self.background = "Kerr"
        """str: The spacetime background for this model. Is Kerr."""
        self.descriptor = "eccentric inclined"
        """str: Description of the inspiral trajectory properties for this model. Is eccentric inclined."""
        self.frame = "detector"
        """str: Frame in which source is generated. Is detector frame."""
        self.needs_Y = True
        """bool: If True, model expects inclination parameter Y (rather than xI). Is True."""

    @classmethod
    def module_references(cls) -> list[REFERENCE]:
        """Return citations related to this module"""
        return [REFERENCE.PN5] + super(Pn5AAK, cls).module_references()

    def sanity_check_angles(self, qS: float, phiS: float, qK: float, phiK: float):
        """Sanity check on viewing angles.

        Make sure parameters are within allowable ranges.

        args:
            qS (double): Sky location polar angle in ecliptic
                coordinates.
            phiS (double): Sky location azimuthal angle in
                ecliptic coordinates.
            qK (double): Initial BH spin polar angle in ecliptic
                coordinates.
            phiK (double): Initial BH spin azimuthal angle in
                ecliptic coordinates.

        Returns:
            tuple: (qS, phiS, qK, phiK). phiS and phiK are wrapped.

        Raises:
            ValueError: If any of the angular values are not allowed.

        """
        if qS < 0.0 or qS > np.pi:
            raise ValueError("qS must be between 0 and pi.")

        if qK < 0.0 or qK > np.pi:
            raise ValueError("qK must be between 0 and pi.")

        phiS = phiS % (2 * np.pi)
        phiK = phiK % (2 * np.pi)
        return (qS, phiS, qK, phiK)

    def sanity_check_traj(self, p: np.ndarray, e: np.ndarray, Y: np.ndarray):
        r"""Sanity check on parameters output from thte trajectory module.

        Make sure parameters are within allowable ranges.

        args:
            p (1D np.ndarray): Array of semi-latus rectum values produced by
                the trajectory module.
            e (1D np.ndarray): Array of eccentricity values produced by
                the trajectory module.
            Y (1D np.ndarray): Array of cos:math:`\iota` values produced by
                the trajectory module.

        Raises:
            ValueError: If any of the trajectory points are not allowed.
            warn: If any points in the trajectory are allowable,
                but outside calibration region.

        """

        if np.any(e < 0.0):
            raise ValueError("Members of e array are less than zero.")

        if np.any(p < 0.0):
            raise ValueError("Members of p array are less than zero.")

        if np.any(Y < -1.0) or np.any(Y > 1.0):
            raise ValueError(
                "Members of Y array are greater than 1.0 or less than -1.0."
            )

    def sanity_check_init(self, M: float, mu: float, a: float, p0: float, e0: float, Y0: float):
        r"""Sanity check initial parameters.

        Make sure parameters are within allowable ranges.

        args:
            M: Massive black hole mass in solar masses.
            m: compact object mass in solar masses.
            a: Dimensionless spin of massive black hole :math:`(0 \leq a \leq 1)`.
            p0: Initial semilatus rectum (dimensionless)
                :math:`(10\leq p_0\leq 16 + 2e_0)`. See the documentation for
                more information on :math:`p_0 \leq 10.0`.
            e0: Initial eccentricity :math:`(0\leq e_0\leq0.7)`.
            Y0: Initial cos:math:`\iota` :math:`(-1.0\leq Y_0\leq1.0)`.

        Raises:
            ValueError: If any of the parameters are not allowed.

        """

        for val, key in [[M, "M"], [p0, "p0"], [e0, "e0"], [mu, "mu"]]:
            test = val < 0.0
            if test:
                raise ValueError("{} is negative. It must be positive.".format(key))

        if a < 0:
            # flip convention
            few_logger.warning(
                "Negative spin magnitude detected. Flipping sign of a and Y0 to match convention."
            )
            a = -a
            Y0 = -Y0

        if mu / M > 1e-4:
            few_logger.warning(
                "Mass ratio is outside of generally accepted range for an extreme mass ratio (1e-4). (q={})".format(
                    mu / M
                )
            )

        if Y0 > 1.0 or Y0 < -1.0:
            raise ValueError(
                "Y0 is greater than 1 or less than -1. Must be between -1 and 1."
            )

        return a, Y0