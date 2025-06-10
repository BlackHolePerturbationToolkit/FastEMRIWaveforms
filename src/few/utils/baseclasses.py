# Collection of base classes for FastEMRIWaveforms Packages

"""
The :code:`few.utils.baseclasses` module contains abstract base classes for the
various modules. When creating new modules, these classes should be used to maintain
a common interface and pass information related to each model.
"""

from __future__ import annotations

import types
from typing import Optional, Sequence, TypeVar, Union

import numpy as np

# Python imports
from ..cutils import Backend
from ..utils.citations import REFERENCE, Citable
from ..utils.globals import get_backend, get_first_backend, get_logger
from ..utils.mappings.kerrecceq import kerrecceq_forward_map

xp_ndarray = TypeVar("xp_ndarray")
"""Generic alias for backend ndarray"""

BackendLike = Union[str, Backend, None]
"""Type hint to declare a backend in constructor."""


class ParallelModuleBase(Citable):
    """
    Base class for modules that can use a GPU (or revert back to CPU).

    This class mainly handles backend selection. Each backend offers accelerated
    computations on a specific device (cpu, CUDA 11.x enabled GPU, CUDA 12.x enabled GPU).

    args:
        force_backend (str, optional): Name of the backend to use
    """

    _backend_name: str

    def __init__(self, /, force_backend: BackendLike = None):
        if force_backend is not None:
            if isinstance(force_backend, Backend):
                force_backend = force_backend.name
            self._backend_name = get_backend(force_backend).name
        else:
            self._backend_name = get_first_backend(self.supported_backends()).name

    @property
    def backend(self) -> Backend:
        """Access the underlying backend."""
        return get_backend(self._backend_name)

    @classmethod
    def supported_backends(cls) -> Sequence[str]:
        """List of backends supported by a parallel module by order of preference."""
        raise NotImplementedError(
            "Class {} does not implement the supported_backends method.".format(cls)
        )

    @staticmethod
    def CPU_ONLY() -> list[str]:
        """List of supported backend for CPU only class"""
        return ["cpu"]

    @staticmethod
    def GPU_RECOMMENDED() -> list[str]:
        """List of supported backends for GPU-recommended class with CPU support"""
        return ["cuda12x", "cuda11x", "cpu"]

    @staticmethod
    def CPU_RECOMMENDED_WITH_GPU_SUPPORT() -> list[str]:
        """List of supported backends for CPU-recommended class with GPU support"""
        return ["cpu", "cuda12x", "cuda11x"]

    @staticmethod
    def GPU_ONLY() -> list[str]:
        """List of supported backends for GPU-only class"""
        return ["cuda12x", "cuda11x"]

    @property
    def xp(self) -> types.ModuleType:
        """Return the module providing ndarray capabilities"""
        return self.backend.xp

    @property
    def backend_name(self) -> str:
        """Return the name of current backend"""
        return self.backend.name

    ParallelModuleDerivate = TypeVar(
        "ParallelModuleDerivate", bound="ParallelModuleBase"
    )

    def build_with_same_backend(
        self,
        module_class: type[ParallelModuleDerivate],
        args: Optional[list] = None,
        kwargs: Optional[dict] = None,
    ) -> ParallelModuleDerivate:
        """
        Build an instance of `module_class` with same backend as current object.

        args:
          module_class: class of the object to be built, must derive from ParallelModuleBase
          args (list, optional): positional arguments for module_class constructor
          kwargs (dict, optional): keyword arguments for module_class constructor
                                   (the 'force_backend' argument will be ignored and replaced)
        """
        args = [] if args is None else args
        return module_class(*args, **self.adapt_backend_kwargs(kwargs=kwargs))

    def adapt_backend_kwargs(self, kwargs: Optional[dict] = None) -> dict:
        """Adapt a set of keyword arguments to add/set 'force_backend' to current backend"""
        if kwargs is None:
            kwargs = {}
        kwargs["force_backend"] = self.backend_name
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
    """

    lmax: int
    """Maximum l value for the model."""

    nmax: int
    """Maximum n value for the model."""

    num_modes: int
    """Total number of modes in the model"""

    num_teuk_modes: int
    """Number of Teukolsky modes in the model."""

    m0sort: xp_ndarray
    """1D Array of sorted mode indices with m=0 first, then m<0, then m>0."""

    l_arr_no_mask: xp_ndarray
    """1D Array of l values for each mode before masking."""
    m_arr_no_mask: xp_ndarray
    """1D Array of m values for each mode before masking."""
    n_arr_no_mask: xp_ndarray
    """1D Array of n values for each mode before masking."""

    lmn_indices: dict[int, int]
    """Dictionary of mode indices to mode number."""

    m0mask: xp_ndarray
    """1D Array Mask for m != 0."""

    num_m_zero_up: int
    """Number of modes with m >= 0."""
    num_m0: int
    """Number of modes with m == 0."""
    num_m_1_up: int
    """Number of modes with m > 0."""

    l_arr: xp_ndarray
    """1D Array of l values for each mode."""
    m_arr: xp_ndarray
    """1D Array of m values for each mode."""
    n_arr: xp_ndarray
    """1D Array of n values for each mode."""

    m_zero_up_mask: xp_ndarray
    """1D Mask for m >= 0."""

    unique_l: xp_ndarray
    """1D Array of unique l values."""
    unique_m: xp_ndarray
    """1D Array of unique m values."""
    num_unique_lm: int
    """Number of unique (l,m) values."""

    index_map: dict[tuple[int, int, int], int]
    """Maps mode index to mode tuple."""
    special_index_map: dict[tuple[int, int, int], int]
    """Maps mode index to mode tuple with m > 0."""
    index_map_arr: xp_ndarray
    """Array mapping mode tuple to mode index - used for fast indexing. Returns -1 if mode does not exist."""
    special_index_map_arr: xp_ndarray
    """Array mapping mode tuple to mode index with m > 0 - used for fast indexing. Returns -1 if mode does not exist."""

    def __init__(
        self, lmax: int = 10, nmax: int = 30, force_backend: BackendLike = None
    ):
        ParallelModuleBase.__init__(self, force_backend=force_backend)

        self.lmax = lmax
        self.nmax = nmax

        # fill all lmn mode values
        md = []
        for l in range(2, self.lmax + 1):
            for m in range(0, l + 1):
                for n in range(-self.nmax, self.nmax + 1):
                    md.append([l, m, n])

        # total number of modes in the model
        self.num_modes = len(md)
        self.num_teuk_modes = self.num_modes

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

        # sorts the mode indexes
        md = self.xp.asarray(md).T[:, m0sort].astype(self.xp.int32)

        # store l m and n values
        self.l_arr_no_mask = md[0]
        self.m_arr_no_mask = md[1]
        self.n_arr_no_mask = md[2]

        # adjust with .get method for cupy
        if self.backend.uses_cupy:
            lmn_indices = {tuple(md_i): i for i, md_i in enumerate(md.T.get())}
        else:
            lmn_indices = {tuple(md_i): i for i, md_i in enumerate(md.T)}

        self.lmn_indices = lmn_indices

        # store the mask as m != 0 is True
        self.m0mask = self.m_arr_no_mask != 0

        # number of m >= 0
        self.num_m_zero_up = len(self.m_arr_no_mask)

        # number of m == 0
        self.num_m0 = len(self.xp.arange(self.num_teuk_modes)[m0mask])

        # number of m > 0
        self.num_m_1_up = self.num_m_zero_up - self.num_m0

        # create final arrays to include -m modes
        self.l_arr = self.xp.concatenate(
            [self.l_arr_no_mask, self.l_arr_no_mask[self.m0mask]]
        )
        self.m_arr = self.xp.concatenate(
            [self.m_arr_no_mask, -self.m_arr_no_mask[self.m0mask]]
        )
        self.n_arr = self.xp.concatenate(
            [self.n_arr_no_mask, self.n_arr_no_mask[self.m0mask]]
        )

        # mask for m >= 0
        self.m_zero_up_mask = self.m_arr >= 0

        # find unique sets of (l,m)
        # create inverse array to build full (l,m,n) from unique l and m
        # also adjust for cupy
        if self.backend.uses_cupy:
            temp, self.inverse_lm = np.unique(
                np.asarray([self.l_arr.get(), self.m_arr.get()]).T,
                axis=0,
                return_inverse=True,
            )
        else:
            temp, self.inverse_lm = np.unique(
                np.asarray([self.l_arr, self.m_arr]).T, axis=0, return_inverse=True
            )

        # unique values of l and m
        unique_l, unique_m = self.xp.asarray(temp).T
        self.unique_l = unique_l
        self.unique_m = unique_m

        # number of unique values
        self.num_unique_lm = len(self.unique_l)

        # creates special maps to the modes
        self.index_map = {}
        self.special_index_map = {}  # maps the minus m values to positive m
        self.index_map_arr = (
            self.xp.zeros(
                (self.lmax + 1, self.lmax * 2 + 1, self.nmax * 2 + 1),
                dtype=self.xp.int32,
            )
            - 1
        )
        self.special_index_map_arr = (
            self.xp.zeros(
                (self.lmax + 1, self.lmax * 2 + 1, self.nmax * 2 + 1),
                dtype=self.xp.int32,
            )
            - 1
        )
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
            sp_i = i if i < self.num_modes else i - self.num_m_1_up

            if m >= 0:
                self.special_index_map[(l, m, n)] = sp_i
                self.special_index_map_arr[l, m, n] = sp_i
            else:
                self.special_index_map[(l, m, -n)] = sp_i
                self.special_index_map_arr[l, m, -n] = sp_i

        # TODO make this more efficient
        # mode indices for all positive m-modes
        self.mode_indexes = self.xp.linspace(
            0, self.num_teuk_modes - 1, self.num_teuk_modes, dtype=int
        )
        # mode indices for all negative m-modes
        self.negative_mode_indexes = self.xp.linspace(
            0, self.num_teuk_modes - 1, self.num_teuk_modes, dtype=int
        )
        for i, (l, m, n) in enumerate(
            zip(self.l_arr_no_mask, self.m_arr_no_mask, self.n_arr_no_mask)
        ):
            self.negative_mode_indexes[i] = self.special_index_map[
                (l.item(), -m.item(), n.item())
            ]

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

    def sanity_check_traj(self, a: float, p: np.ndarray, e: np.ndarray, xI: np.ndarray):
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

        if np.any(abs(a) > 1.0):
            raise ValueError("Members of a array have a magnitude greater than one.")

        if np.any(abs(xI) > 1.0):
            raise ValueError("Members of xI array have a magnitude greater than one.")


class SchwarzschildEccentric(SphericalHarmonic):
    """
    Schwarzschild eccentric base class.

    Args:
        lmax: Maximum l value for the model. Default is 10.
        nmax: Maximum n value for the model. Default is 30.
        ndim: Number of phases in the model. Default is 2.
    """

    background: str = "Schwarzschild"
    """The spacetime background for this model."""

    descriptor: str = "eccentric"
    """Description of the inspiral trajectory properties for this model."""

    frame: str = "source"
    """Frame in which source is generated. Is source frame."""

    needs_Y: bool = False
    """If True, model expects inclination parameter Y (rather than xI)."""

    ndim: int
    """Number of phases in the model."""

    def __init__(
        self,
        /,
        lmax: int = 10,
        nmax: int = 30,
        ndim: int = 2,
        force_backend: BackendLike = None,
    ):
        SphericalHarmonic.__init__(
            self, lmax=lmax, nmax=nmax, force_backend=force_backend
        )

        self.ndim = ndim

    @classmethod
    def supported_backends(cls):
        return cls.GPU_RECOMMENDED()

    def sanity_check_init(
        self, m1: float, m2: float, a: float, p0: float, e0: float, xI: float
    ) -> tuple[float, float]:
        r"""Sanity check initial parameters.

        Make sure parameters are within allowable ranges.

        args:
            m1: Massive black hole mass in solar masses.
            m2: compact object mass in solar masses.
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
        for val, key in [[m1, "m1"], [p0, "p0"], [e0, "e0"], [m2, "m2"]]:
            test = val < 0.0
            if test:
                raise ValueError("{} is negative. It must be positive.".format(key))

        if m1 < m2:
            raise ValueError(
                "Massive black hole mass must be larger than the compact object mass. (m1={}, m2={})".format(
                    m1, m2
                )
            )

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

        if a != 0.0:
            raise ValueError("Spin must be zero for Schwarzschild inspirals.")

        if abs(xI) != 1.0:
            raise ValueError("For equatorial orbits, xI must be either 1 or -1.")

        return a, xI


class KerrEccentricEquatorial(SphericalHarmonic):
    """
    Kerr eccentric equatorial base class.

    Args:
        lmax: Maximum l value for the model. Default is 10.
        nmax: Maximum n value for the model. Default is 55.
        ndim: Number of phases in the model. Default is 2.
    """

    background: str = "Kerr"
    """The spacetime background for this model."""

    descriptor: str = "eccentric equatorial"
    """Description of the inspiral trajectory properties for this model."""

    frame: str = "source"
    """Frame in which source is generated. Is source frame."""

    needs_Y: bool = False
    """If True, model expects inclination parameter Y (rather than xI)."""

    ndim: int
    """Number of phases in the model."""

    def __init__(
        self,
        lmax: int = 10,
        nmax: int = 55,
        ndim: int = 2,
        force_backend: BackendLike = None,
    ):
        SphericalHarmonic.__init__(
            self, lmax=lmax, nmax=nmax, force_backend=force_backend
        )

        self.ndim = ndim

    @classmethod
    def supported_backends(cls):
        return cls.GPU_RECOMMENDED()

    def sanity_check_init(
        self, m1: float, m2: float, a: float, p0: float, e0: float, xI: float
    ) -> tuple[float, float]:
        r"""Sanity check initial parameters.

        Make sure parameters are within allowable ranges.

        args:
            m1: Massive black hole mass in solar masses.
            m2: compact object mass in solar masses.
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

        for val, key in [[m1, "m1"], [p0, "p0"], [e0, "e0"], [m2, "m2"]]:
            test = val < 0.0
            if test:
                raise ValueError("{} is negative. It must be positive.".format(key))

        if m1 < m2:
            raise ValueError(
                "Massive black hole mass must be larger than the compact object mass. (m1={}, m2={})".format(
                    m1, m2
                )
            )

        if xI < 0:
            # flip convention
            get_logger().warning(
                "Negative inclination detected. Flipping sign of a and xI to match convention."
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

        if abs(xI) != 1.0:
            raise ValueError("For equatorial orbits, xI must be either 1 or -1.")

        return a, xI


class Pn5AAK(Citable):
    """Base class for Pn5AAK waveforms.

    This class contains some basic checks and information for AAK waveforms
    with a 5PN trajectory model. Please see :class:`few.waveform.Pn5AAKWaveform`
    for more details.
    """

    background: str = "Kerr"
    """The spacetime background for this model."""

    descriptor: str = "eccentric inclined"
    """Description of the inspiral trajectory properties for this model."""

    frame: str = "detector"
    """Frame in which source is generated. Is detector frame."""

    needs_Y: bool = True
    """If True, model expects inclination parameter Y (rather than xI)."""

    @classmethod
    def module_references(cls) -> list[REFERENCE]:
        """Return citations related to this module"""
        return [REFERENCE.PN5] + super().module_references()

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

    def sanity_check_init(
        self, m1: float, m2: float, a: float, p0: float, e0: float, Y0: float
    ):
        r"""Sanity check initial parameters.

        Make sure parameters are within allowable ranges.

        args:
            m1: Massive black hole mass in solar masses.
            m2: compact object mass in solar masses.
            a: Dimensionless spin of massive black hole :math:`(0 \leq a \leq 1)`.
            p0: Initial semilatus rectum (dimensionless)
                :math:`(10\leq p_0\leq 16 + 2e_0)`. See the documentation for
                more information on :math:`p_0 \leq 10.0`.
            e0: Initial eccentricity :math:`(0\leq e_0\leq0.7)`.
            Y0: Initial cos:math:`\iota` :math:`(-1.0\leq Y_0\leq1.0)`.

        Raises:
            ValueError: If any of the parameters are not allowed.

        """

        for val, key in [[m1, "m1"], [p0, "p0"], [e0, "e0"], [m2, "m2"]]:
            test = val < 0.0
            if test:
                raise ValueError("{} is negative. It must be positive.".format(key))

        if m1 < m2:
            raise ValueError(
                "Massive black hole mass must be larger than the compact object mass. (m1={}, m2={})".format(
                    m1, m2
                )
            )

        if a < 0:
            # flip convention
            get_logger().warning(
                "Negative spin magnitude detected. Flipping sign of a and Y0 to match convention."
            )
            a = -a
            Y0 = -Y0

        if m2 / m1 > 1e-4:
            get_logger().warning(
                "Mass ratio is outside of generally accepted range for an extreme mass ratio (1e-4). (q={})".format(
                    m2 / m1
                )
            )

        if Y0 > 1.0 or Y0 < -1.0:
            raise ValueError(
                "Y0 is greater than 1 or less than -1. Must be between -1 and 1."
            )

        return a, Y0
