# Collection of base classes for FastEMRIWaveforms Packages

"""Base classes shared by the FEW modules: backend handling and the waveform-model traits.

- :class:`FewBackendConsumer`: backend selection of every backend-aware FEW module (maps
  ``force_backend`` onto FEW's registered backends).
- :class:`SphericalHarmonic`: the ``(l, m, k, n)`` mode bookkeeping of spherical-harmonic
  models (mode arrays, index maps, the ``+m`` / ``-m`` symmetry) and the generic trajectory /
  viewing-angle checks.
- Model classes with their descriptive traits and initial-parameter checks:
  :class:`SchwarzschildEccentric`, :class:`KerrEccentricEquatorial` (the production
  ``FastKerrEccentricEquatorialFlux`` model and its amplitude module),
  :class:`KerrGeneric` (work in progress) and :class:`Pn5AAK` (AAK waveforms).
- Type helpers ``xp_ndarray`` and ``BackendLike``.

When creating new modules, these classes should be used to keep a common interface and to
carry the information of each model.
"""

from __future__ import annotations

import types
from typing import Optional, Sequence, TypeVar, Union

import numpy as np

# Python imports
from ..cutils import FEWBackend
from ..utils.citations import REFERENCE, Citable
from ..utils.mappings.kerrecceq import kerrecceq_forward_map

xp_ndarray = TypeVar("xp_ndarray")
"""Generic alias for backend ndarray"""

BackendLike = Union[str, FEWBackend, None]
"""Type hint to declare a backend in constructor."""

# TODO: will change this to BackendConsumer in GBT 
from gpubackendtools.parallelbase import ParallelModuleBase


class FewBackendConsumer(ParallelModuleBase):
    """Base class of the FEW modules that run on a compute backend (CPU or CUDA).

    Resolves ``force_backend`` to one of FEW's registered backends (``few_cpu``,
    ``few_cuda11x``, ``few_cuda12x``, ``few_cuda13x``; registered by ``import few``) and
    passes it to :class:`gpubackendtools.parallelbase.ParallelModuleBase`, which provides
    :attr:`backend`, :attr:`xp` (numpy or cupy) and ``build_with_same_backend``. Subclasses
    implement the classmethod ``supported_backends`` (bare flavour names in order of
    preference, e.g. ``GPU_RECOMMENDED()``).

    Args:
        force_backend: Backend to use:

            - ``None`` (default): the first available FEW backend among
              ``supported_backends()`` (each name prefixed with ``few_``);
            - a ``"few_<flavour>"`` string, e.g. ``"few_cuda13x"``;
            - a bare flavour, e.g. ``"cuda13x"`` or ``"cpu"``: mapped to ``few_<flavour>``;
            - another package's backend name ``"<pkg>_<flavour>"`` (e.g. ``"gbt_cpu"``, as
              passed down by a parent object): the flavour is kept and FEW's backend of that
              flavour is used;
            - a backend instance or a ``(package, flavour)`` tuple: passed to the base class
              unchanged.

    Raises:
        ValueError: From the base class, if the resolved name is not a registered backend.
        gpubackendtools.gpubackendtools.BackendAccessException: From the backend registry,
            if the requested backend (or every supported one) is unavailable.

    Example:
        .. code-block:: python

            from few.utils.ylm import GetYlms

            ylm = GetYlms(force_backend="cpu")  # resolved to the few_cpu backend
            ylm.backend.name
    """
    def __init__(self, force_backend: Optional[str | FEWBackend] =  None):
        if force_backend is None:
            # the base would take the first backend of ANY package matching FEW's bare
            # names (e.g. gbt_cpu); pick the first available FEW one instead
            from gpubackendtools.globals import get_first_backend

            names = [b if b.startswith("few_") else "few_" + b for b in self.supported_backends()]
            force_backend_in = ("few", get_first_backend(names).name.split("few_", 1)[1])
        elif isinstance(force_backend, str):
            if force_backend.startswith("few_"):
                force_backend_in = ("few", force_backend[len("few_"):])
            elif "_" in force_backend:
                # another package's resolved name (e.g. a parent built on gbt_cpu):
                # keep the device part, use FEW's backend for it
                force_backend_in = ("few", force_backend.split("_", 1)[1])
            else:
                force_backend_in = ("few", force_backend)
        else:
            force_backend_in = force_backend
        super().__init__(force_backend=force_backend_in)


class SphericalHarmonic(FewBackendConsumer):
    r"""Base class for waveforms constructed in a spherical harmonic basis.

    This class creates shared traits between different implementations of the
    same model. Particularly, this class includes descriptive traits as well as
    the sanity check class method that should be used in all implementations of
    this model. This method can be overwritten if necessary.

    As we assume the amplitudes have been remapped to a spherical harmonic basis,
    we use -2 spin-weighted spherical harmonics (:math:`(s=-2)Y_{l,m}`) in place
    of the more generic angular function from Eq. :eq:`emri_wave_eq`.

    The constructor enumerates the modes :math:`2 \le l \le` ``lmax``,
    :math:`0 \le m \le l`, :math:`|k| \le` ``kmax``, :math:`|n| \le` ``nmax`` (the
    :math:`m \ge 0` "Teukolsky" modes), sorts them with :math:`m = 0` first, appends a
    :math:`-m` copy of every :math:`m > 0` mode for the full arrays (``l_arr``, ...), and
    builds the index maps used by the amplitude modules and the mode selector. The arrays
    live on the backend (numpy or cupy).

    Args:
        lmax: Maximum ``l``. Default ``10``.
        kmax: Maximum ``|k|``. Default ``0`` (equatorial).
        nmax: Maximum ``|n|``. Default ``0``.
        force_backend: See :class:`FewBackendConsumer`. Default ``None``.

    Attributes:
        inverse_lm (np.ndarray): For each entry of the full arrays, the index of its
            ``(l, m)`` in ``unique_l`` / ``unique_m`` (host array, from ``np.unique``).
        mode_indexes (xp.ndarray): ``arange(num_teuk_modes)``, the indices of the
            :math:`m \ge 0` modes.
        negative_mode_indexes (xp.ndarray): For each :math:`m > 0` mode
            :math:`(l, m, k, n)`, the index of the stored mode :math:`(l, m, -k, -n)`, whose
            :math:`(-m, -k, -n)` partner is :math:`(l, -m, k, n)`; for :math:`m = 0` the
            mode itself. Used by the amplitude modules for :math:`m < 0` requests and
            negative :math:`x_I`.
    """

    lmax: int
    """Maximum l value for the model."""

    kmax: int
    """Maximum k value for the model. Currently always 0 for equatorial orbits."""

    nmax: int
    """Maximum n value for the model."""

    num_modes: int
    """Number of (l, m, k, n) modes with m >= 0 in the model."""

    num_teuk_modes: int
    """Number of Teukolsky (m >= 0) modes in the model; equal to ``num_modes``."""

    m0sort: xp_ndarray
    """1D array of the enumeration indices of the m >= 0 modes, sorted m = 0 first, then m > 0."""

    l_arr_no_mask: xp_ndarray
    """1D Array of l values of the m >= 0 modes (``m0sort`` order), before the -m copies."""
    m_arr_no_mask: xp_ndarray
    """1D Array of m values of the m >= 0 modes (``m0sort`` order), before the -m copies."""
    k_arr_no_mask: xp_ndarray
    """1D Array of k values of the m >= 0 modes (``m0sort`` order), before the -m copies."""
    n_arr_no_mask: xp_ndarray
    """1D Array of n values of the m >= 0 modes (``m0sort`` order), before the -m copies."""

    m0mask: xp_ndarray
    """1D Array Mask for m != 0."""

    num_m_zero_up: int
    """Number of modes with m >= 0."""
    num_m0: int
    """Number of modes with m == 0."""
    num_m_1_up: int
    """Number of modes with m > 0."""

    l_arr: xp_ndarray
    """1D Array of l values for each mode: the m >= 0 modes, then a -m copy of each m > 0 mode."""
    m_arr: xp_ndarray
    """1D Array of m values for each mode (same layout as ``l_arr``)."""
    k_arr: xp_ndarray
    """1D Array of k values for each mode. Currently always 0 for equatorial orbits."""
    n_arr: xp_ndarray
    """1D Array of n values for each mode (same layout as ``l_arr``)."""

    m_zero_up_mask: xp_ndarray
    """1D Mask for m >= 0."""

    unique_l: xp_ndarray
    """1D Array of unique l values."""
    unique_m: xp_ndarray
    """1D Array of unique m values."""
    num_unique_lm: int
    """Number of unique (l,m) values."""

    index_map: dict[tuple[int, int, int, int], int]
    """Maps each mode tuple (l, m, k, n), m of either sign, to its index in the full arrays."""
    special_index_map: dict[tuple[int, int, int, int], int]
    """Maps a mode tuple to the index of its stored m >= 0 mode: (l, m, k, n) itself for m >= 0,
    and (l, -m, -k, -n) -> index of (l, m, k, n) for the m < 0 partners."""
    index_map_arr: xp_ndarray
    """Array form of ``index_map``, indexed ``[l, m, k, n]`` (negative m, k, n wrap around);
    -1 where the mode does not exist. Shape ``(lmax+1, 2 lmax+1, 2 kmax+1, 2 nmax+1)``."""
    special_index_map_arr: xp_ndarray
    """Array form of ``special_index_map`` (same indexing and shape); -1 where the mode does
    not exist."""

    def __init__(
        self, lmax: int = 10, kmax: int = 0, nmax: int = 0, force_backend: BackendLike = None
    ):
        FewBackendConsumer.__init__(self, force_backend=force_backend)

        self.lmax = lmax
        self.kmax = kmax
        self.nmax = nmax

        # fill all lmn mode values
        md = []
        for l in range(2, self.lmax + 1):
            for m in range(0, l + 1):
                for k in range(-self.kmax, self.kmax + 1):
                    for n in range(-self.nmax, self.nmax + 1):
                        md.append([l, m, k, n])

        # total number of modes in the model
        self.num_modes = len(md)
        self.num_teuk_modes = self.num_modes

        # mask for m == 0
        m0mask = self.xp.array(
            [
                m == 0
                for l in range(2, self.lmax + 1)
                for m in range(0, l + 1)
                for k in range(-self.kmax, self.kmax + 1)
                for n in range(-self.nmax, self.nmax + 1)
            ]
        )

        # sorts so that order is m=0, then m>0
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
        self.k_arr_no_mask = md[2]  # k is always 0 for equatorial orbits
        self.n_arr_no_mask = md[3]

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
        self.k_arr = self.xp.concatenate(
            [self.k_arr_no_mask, self.k_arr_no_mask[self.m0mask]]
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
                (self.lmax + 1, self.lmax * 2 + 1, self.kmax * 2 + 1, self.nmax * 2 + 1),
                dtype=self.xp.int32,
            )
            - 1
        )
        self.special_index_map_arr = (
            self.xp.zeros(
                (self.lmax + 1, self.lmax * 2 + 1, self.kmax * 2 + 1, self.nmax * 2 + 1),
                dtype=self.xp.int32,
            )
            - 1
        )
        for i, (l, m, k, n) in enumerate(zip(self.l_arr, self.m_arr, self.k_arr, self.n_arr)):
            try:
                l = l.item()
                m = m.item()
                k = k.item()
                n = n.item()

            except AttributeError:
                pass

            # regular map: mode tuple to index
            self.index_map[(l, m, k, n)] = i
            self.index_map_arr[l, m, k, n] = i
            # special map that gives m < 0 indices as m > 0 indices
            sp_i = i if i < self.num_modes else i - self.num_m_1_up

            if m >= 0:
                self.special_index_map[(l, m, k, n)] = sp_i
                self.special_index_map_arr[l, m, k, n] = sp_i
            else:
                self.special_index_map[(l, m, -k, -n)] = sp_i
                self.special_index_map_arr[l, m, -k, -n] = sp_i

        # TODO make this more efficient
        # mode indices for all positive m-modes
        self.mode_indexes = self.xp.linspace(
            0, self.num_teuk_modes - 1, self.num_teuk_modes, dtype=int
        )
        # mode indices for all negative m-modes
        self.negative_mode_indexes = self.xp.linspace(
            0, self.num_teuk_modes - 1, self.num_teuk_modes, dtype=int
        )
        for i, (l, m, k, n) in enumerate(
            zip(self.l_arr_no_mask, self.m_arr_no_mask, self.k_arr_no_mask, self.n_arr_no_mask)
        ):
            self.negative_mode_indexes[i] = self.special_index_map[
                (l.item(), -m.item(), k.item(), n.item())
            ]

    def sanity_check_viewing_angles(self, theta: float, phi: float):
        """Wrap the azimuthal viewing angle; ``theta`` is not checked.

        Args:
            theta (float): Source-frame polar viewing angle in rad (returned unchanged; the
                range check is commented out).
            phi (float): Source-frame azimuthal viewing angle in rad.

        Returns:
            tuple: ``(theta, phi % (2 pi))``.
        """
        # if theta < 0.0 or theta > np.pi:
        #    raise ValueError("theta must be between 0 and pi.")

        phi = phi % (2 * np.pi)
        return (theta, phi)

    def sanity_check_traj(self, a: float, p: np.ndarray, e: np.ndarray, xI: np.ndarray):
        r"""Check the trajectory output of a spherical-harmonic waveform.

        Called by :meth:`~few.waveform.base.SphericalHarmonicWaveformBase._generate_waveform`
        after the inspiral.

        Args:
            a: Dimensionless spin of the primary.
            p: Semi-latus rectum along the trajectory, in units of :math:`M`.
            e: Eccentricity along the trajectory.
            xI: :math:`x_I = \cos I` along the trajectory.

        Raises:
            ValueError: If any ``e < 0``, any ``p < 0``, ``|a| > 1`` or any ``|xI| > 1``.
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
    """Schwarzschild eccentric model traits (``a = 0``, equatorial, eccentric).

    Base of the Schwarzschild amplitude modules and waveforms
    (:class:`~few.waveform.waveform.FastSchwarzschildEccentricFlux` and related).

    Args:
        lmax: Maximum ``l``. Default ``10``.
        nmax: Maximum ``|n|``. Default ``30``.
        force_backend: See :class:`FewBackendConsumer`. Default ``None``.
    """

    background: str = "Schwarzschild"
    """The spacetime background for this model."""

    descriptor: str = "eccentric"
    """Description of the inspiral trajectory properties for this model."""

    frame: str = "source"
    """Frame in which source is generated. Is source frame."""

    needs_Y: bool = False
    """If True, model expects inclination parameter Y (rather than xI)."""

    def __init__(
        self,
        /,
        lmax: int = 10,
        nmax: int = 30,
        force_backend: BackendLike = None,
    ):
        SphericalHarmonic.__init__(
            self, lmax=lmax, nmax=nmax, force_backend=force_backend
        )

    @classmethod
    def supported_backends(cls):
        """Backend preference: ``GPU_RECOMMENDED()`` (CUDA 13/12/11, then CPU)."""
        return cls.GPU_RECOMMENDED()

    def sanity_check_init(
        self, m1: float, m2: float, a: float, p0: float, e0: float, xI: float
    ) -> tuple[float, float]:
        r"""Check the initial parameters against the Schwarzschild eccentric model's domain.

        Args:
            m1: Primary (massive black hole) mass in solar masses.
            m2: Secondary (compact object) mass in solar masses, ``m2 <= m1``.
            a: Dimensionless spin; must be ``0``.
            p0: Initial semi-latus rectum in units of :math:`M`:
                :math:`p_0 \le 16 + 2e_0`, and for :math:`p_0 < 10` also
                :math:`p_0 \ge 7(6 + 2e_0) - 41.9`.
            e0: Initial eccentricity, :math:`0 \le e_0 \le 0.75`.
            xI: Initial :math:`x_I`; ``+1`` or ``-1``.

        Returns:
            tuple[float, float]: ``(a, xI)`` unchanged.

        Raises:
            ValueError: If a mass, ``p0`` or ``e0`` is negative, ``m1 < m2``, ``e0`` or
                ``p0`` is out of range, ``a != 0`` or ``|xI| != 1``.
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
    """Kerr eccentric equatorial model traits (the ``FastKerrEccentricEquatorialFlux`` model).

    Base of :class:`~few.amplitude.ampinterp2d.AmpInterpKerrEccEq` and of
    :class:`~few.waveform.waveform.FastKerrEccentricEquatorialFlux`; the defaults give
    63 ``(l, m >= 0)`` pairs x 111 values of ``n`` = 6993 Teukolsky modes. Internal
    convention: equatorial orbits with ``xI = +1`` and a signed spin ``a`` in
    ``[-0.999, 0.999]``, negative for retrograde orbits (the waveforms map ``xI0 = -1`` to
    ``(-a, +1)`` before :meth:`sanity_check_init`).

    Args:
        lmax: Maximum ``l``. Default ``10``.
        nmax: Maximum ``|n|``. Default ``55``.
        force_backend: See :class:`FewBackendConsumer`. Default ``None``.
    """

    background: str = "Kerr"
    """The spacetime background for this model."""

    descriptor: str = "eccentric equatorial"
    """Description of the inspiral trajectory properties for this model."""

    frame: str = "source"
    """Frame in which source is generated. Is source frame."""

    needs_Y: bool = False
    """If True, model expects inclination parameter Y (rather than xI)."""

    def __init__(
        self,
        lmax: int = 10,
        nmax: int = 55,
        force_backend: BackendLike = None,
    ):
        SphericalHarmonic.__init__(
            self, lmax=lmax, nmax=nmax, force_backend=force_backend
        )

    @classmethod
    def supported_backends(cls):
        """Backend preference: ``GPU_RECOMMENDED()`` (CUDA 13/12/11, then CPU)."""
        return cls.GPU_RECOMMENDED()

    def sanity_check_init(
        self, m1: float, m2: float, a: float, p0: float, e0: float, xI: float
    ) -> tuple[float, float]:
        r"""Check the initial parameters against the KerrEccEq flux grid.

        Called by :meth:`~few.waveform.base.SphericalHarmonicWaveformBase._generate_waveform`
        (after it has mapped ``xI0 < 0`` to ``(-a, +1)``). ``xI < 0`` is flipped here too, to
        ``(-a, -xI)``. The point is then mapped with
        :func:`~few.utils.mappings.kerrecceq.kerrecceq_forward_map` (flux grid) and must
        satisfy: ``u`` not NaN (else "too close to the separatrix") and ``u <= 1.000001``
        (else "outside of our domain"), and ``-1e-6 <= w <= 1.000001``. Points inside the
        grid's inner edge (``u < 0``) are not rejected here; the trajectory's start check
        (:meth:`~few.trajectory.ode.flux.KerrEccEqFlux.isvalid_pex`) rejects them.

        Args:
            m1: Primary (massive black hole) mass in solar masses.
            m2: Secondary (compact object) mass in solar masses, ``m2 <= m1``.
            a: Dimensionless spin, signed (negative for retrograde orbits with ``xI = 1``);
                at most ``0.999`` after the flip.
            p0: Initial semi-latus rectum in units of :math:`M`.
            e0: Initial eccentricity.
            xI: Initial :math:`x_I`, ``+1`` or ``-1``.

        Returns:
            tuple[float, float]: ``(a, xI)`` in the internal convention: ``xI = +1`` and a
            signed ``a``.

        Raises:
            ValueError: If a mass, ``p0`` or ``e0`` is negative, ``m1 < m2``, ``a > 0.999``
                after the flip, the grid checks above fail, or ``|xI| != 1``.

        Warning:
            The ``xI < 0`` branch logs through ``get_logger``, which this module does not
            import (``NameError``); the waveforms never reach it because they flip
            ``xI0 < 0`` first.
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

# TODO check: keep/remove/document -- only used by the WIP AmpInterpKerrGeneric; only a = 0.7
class KerrGeneric(SphericalHarmonic):
    """
    Kerr generic base class example.

    Args:
        lmax: Maximum l value for the model. Default is 6.
        kmax: Maximum k value for the model. Default is 6.
        nmax: Maximum n value for the model. Default is 30.
    """

    background: str = "Kerr"
    """The spacetime background for this model."""

    descriptor: str = "generic"
    """Description of the inspiral trajectory properties for this model."""

    frame: str = "source"
    """Frame in which source is generated. Is source frame."""

    needs_Y: bool = False
    """If True, model expects inclination parameter Y (rather than xI)."""

    def __init__(
        self,
        lmax: int = 6,
        kmax: int = 6,
        nmax: int = 30,
        force_backend: BackendLike = None,
    ):
        SphericalHarmonic.__init__(
            self, lmax=lmax, kmax=kmax, nmax=nmax, force_backend=force_backend
        )

    @classmethod
    def supported_backends(cls):
        return cls.GPU_RECOMMENDED()

    def sanity_check_init(
        self, m1: float, m2: float, a: float, p0: float, e0: float, xI: float
    ) -> tuple[float, float]:
        r"""Sanity check initial parameters.

        Make sure parameters are within allowable ranges.

        Args:
            m1: Massive black hole mass in solar masses.
            m2: compact object mass in solar masses.
            a: Dimensionless spin of massive black hole; must be exactly ``0.7`` (after the
                flip of a negative ``xI``).
            p0: Initial semilatus rectum (dimensionless); only checked for sign.
            e0: Initial eccentricity; only checked for sign.
            xI: Initial cosine(inclination) :math:`(|x_I| \le 1)`.

        Returns:
            (a_fix, xI_fix): a and xI in the correct convention (xI >= 0).

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

        if a != 0.7:
            raise ValueError(
                "Model currently only supports spin 0.7"
            )

        if abs(xI) > 1.0:
            raise ValueError("xI must be between 1 or -1.")

        return a, xI


class Pn5AAK(Citable):
    r"""Base class for Pn5AAK waveforms.

    This class contains some basic checks and information for AAK waveforms
    with a 5PN trajectory model (detector-frame output, inclination parameter
    :math:`Y = \cos\iota`). Please see :class:`few.waveform.waveform.Pn5AAKWaveform`
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
        r"""Check the sky and spin angles of an AAK waveform; wrap the azimuths.

        Args:
            qS (float): Sky location polar angle in rad (ecliptic coordinates).
            phiS (float): Sky location azimuthal angle in rad.
            qK (float): Initial spin polar angle in rad.
            phiK (float): Initial spin azimuthal angle in rad.

        Returns:
            tuple: ``(qS, phiS % (2 pi), qK, phiK % (2 pi))``.

        Raises:
            ValueError: If ``qS`` or ``qK`` is outside :math:`[0, \pi]`.
        """
        if qS < 0.0 or qS > np.pi:
            raise ValueError("qS must be between 0 and pi.")

        if qK < 0.0 or qK > np.pi:
            raise ValueError("qK must be between 0 and pi.")

        phiS = phiS % (2 * np.pi)
        phiK = phiK % (2 * np.pi)
        return (qS, phiS, qK, phiK)

    def sanity_check_traj(self, p: np.ndarray, e: np.ndarray, Y: np.ndarray):
        r"""Check the trajectory output of an AAK waveform.

        Args:
            p: Semi-latus rectum along the trajectory, in units of :math:`M`.
            e: Eccentricity along the trajectory.
            Y: :math:`Y = \cos\iota` along the trajectory.

        Raises:
            ValueError: If any ``e < 0``, any ``p < 0``, or any ``Y`` outside ``[-1, 1]``.
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
        r"""Check the initial parameters of an AAK waveform.

        A negative spin is flipped to ``(-a, -Y0)`` (convention ``a >= 0``). ``p0`` and
        ``e0`` are only checked for sign.

        Args:
            m1: Primary (massive black hole) mass in solar masses.
            m2: Secondary (compact object) mass in solar masses, ``m2 <= m1``.
            a: Dimensionless spin of the primary.
            p0: Initial semi-latus rectum in units of :math:`M`.
            e0: Initial eccentricity.
            Y0: Initial :math:`Y_0 = \cos\iota`, in ``[-1, 1]``.

        Returns:
            tuple[float, float]: ``(a, Y0)`` with ``a >= 0``.

        Raises:
            ValueError: If a mass, ``p0`` or ``e0`` is negative, ``m1 < m2``, or
                ``|Y0| > 1``.

        Warning:
            The two warnings (negative ``a``; mass ratio ``m2 / m1 > 1e-4``) log through
            ``get_logger``, which this module does not import, so those cases raise
            ``NameError``.
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
