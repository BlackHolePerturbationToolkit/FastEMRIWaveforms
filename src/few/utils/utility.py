# Utilities to aid in FastEMRIWaveforms Packages

"""Analysis and helper utilities of FEW.

- Waveform comparison: :func:`get_overlap` and :func:`get_mismatch` (flat-noise inner
  product).
- Frame helpers of :class:`~few.waveform.waveform.GenerateEMRIWaveform` (also called by the
  LISA Analysis Tools EMRI response): :func:`get_viewing_angles` (source-frame viewing
  angles) and :func:`get_polarization_angle`.
- Inverse problems on a trajectory module, solved by Brent's method: :func:`get_p_at_t`
  (initial ``p0`` giving a requested duration), :func:`get_m2_at_t` (secondary mass) and
  the generic :func:`get_at_t`.
- Root finders: the numba scalar :func:`_brentq_jit` and cubic solver :func:`_solveCubic`
  (used by :mod:`few.utils.geodesic` and the flux ODEs), and the vectorised
  :func:`brentq_array` (batched trajectories, experimental).
- Legacy pointer helpers :func:`wrapper` / :func:`pointer_adjust`, duplicates of
  :mod:`few.cutils.wrappers`.
"""

from math import acos, cos, sqrt
from typing import Optional, Tuple

import numpy as np
from numba import njit
from scipy.optimize import brentq

from .exceptions import TrajectoryOffGridException
from .globals import get_logger

from .constants import PI, YRSID_SI

# TODO: need to make sure this works
from gpubackendtools import get_first_backend


def get_overlap(
    time_series_1: np.ndarray, time_series_2: np.ndarray, use_gpu: bool = False
) -> float:
    r"""Return the normalised overlap of two time series (flat-noise inner product).

    If the lengths differ, both series are truncated to the shorter length (from index 0)
    and a warning is logged. With :math:`\tilde a, \tilde b` the FFTs of the two series,

    .. math::

        \gamma_{a,b} = \mathrm{Re}\,\frac{\sum_k \tilde a_k^* \tilde b_k}
            {\sqrt{\sum_k |\tilde a_k|^2 \sum_k |\tilde b_k|^2}},

    i.e. the zero-lag normalised correlation, without noise weighting or time / phase
    maximisation.

    Args:
        time_series_1: First time series (real or complex, e.g. :math:`h_+ - i h_\times`);
            converted with ``xp.asarray`` if needed.
        time_series_2: Second time series.
        use_gpu: Use cupy instead of numpy. Default ``False``.

    Returns:
        float: The overlap, in :math:`[-1, 1]`.

    Example:
        >>> import numpy as np
        >>> from few.utils.utility import get_overlap
        >>> h = np.exp(2j * np.pi * 1e-3 * 10.0 * np.arange(1000))
        >>> round(float(get_overlap(h, h)), 12)
        1.0
    """

    # adjust arrays based on GPU usage
    if use_gpu:
        import cupy as xp
    else:
        xp = np

    if not isinstance(time_series_1, xp.ndarray):
        time_series_1 = xp.asarray(time_series_1)
    if not isinstance(time_series_2, xp.ndarray):
        time_series_2 = xp.asarray(time_series_2)

    # get the lesser of the two lengths
    min_len = int(np.min([len(time_series_1), len(time_series_2)]))

    if len(time_series_1) != len(time_series_2):
        import logging

        get_logger().warning(
            "The two time series are not the same length ({} vs {}). The calculation will run with length {} starting at index 0 for both arrays.".format(
                len(time_series_1), len(time_series_2), min_len
            ),
            stack_info=get_logger().getEffectiveLevel() <= logging.DEBUG,
        )

    # chop off excess length on a longer array
    # take fft
    time_series_1_fft = xp.fft.fft(time_series_1[:min_len])
    time_series_2_fft = xp.fft.fft(time_series_2[:min_len])

    # autocorrelation
    ac = xp.dot(time_series_1_fft.conj(), time_series_2_fft) / xp.sqrt(
        xp.dot(time_series_1_fft.conj(), time_series_1_fft)
        * xp.dot(time_series_2_fft.conj(), time_series_2_fft)
    )

    # if using cupy, it will return a dimensionless array
    return ac.item().real if use_gpu else ac.real


def get_viewing_angles(qS: float, phiS: float, qK: float, phiK: float) -> Tuple[float, float]:
        r"""Return the source-frame viewing angles of an observer from the sky and spin angles.

        With :math:`\hat R` the unit vector of the sky position and :math:`\hat S` that of the
        primary's spin, the polar viewing angle is :math:`\theta = \arccos(-\hat R\cdot\hat S)`
        (the angle between the spin and the direction to the observer); the azimuthal angle
        is fixed to :math:`\phi = -\pi/2` by the definition of the source frame. Used by
        :meth:`~few.waveform.waveform.GenerateEMRIWaveform.__call__` (detector-frame
        waveforms) and by the LISA Analysis Tools EMRI response.

        Args:
            qS: Polar angle of the sky location in rad (FEW documents ecliptic coordinates;
                only the relative orientation of sky and spin matters).
            phiS: Azimuthal angle of the sky location in rad.
            qK: Polar angle of the primary's spin in rad, in the same frame as ``qS``.
            phiK: Azimuthal angle of the primary's spin in rad.

        Returns:
            tuple[float, float]: ``(theta, phi)`` in rad, ``theta`` in :math:`[0, \pi]` and
            ``phi = -pi/2``.

        Example:
            >>> import numpy as np
            >>> from few.utils.utility import get_viewing_angles
            >>> theta, phi = get_viewing_angles(np.pi / 2, 0.0, 0.0, 0.0)
            >>> round(float(theta), 12), round(float(phi), 12)
            (1.570796326795, -1.570796326795)
        """

        cqS = np.cos(qS)
        sqS = np.sin(qS)

        cphiS = np.cos(phiS)
        sphiS = np.sin(phiS)

        cqK = np.cos(qK)
        sqK = np.sin(qK)

        cphiK = np.cos(phiK)
        sphiK = np.sin(phiK)

        # sky location vector
        R = np.array([sqS * cphiS, sqS * sphiS, cqS])

        # spin vector
        S = np.array([sqK * cphiK, sqK * sphiK, cqK])

        # get viewing angles
        phi = -np.pi / 2.0  # by definition of source frame

        theta = np.arccos(-np.dot(R, S))  # normalized vector

        return (theta, phi)


def get_polarization_angle(qS: float, phiS: float, qK: float, phiK: float) -> float:
        r"""Return the polarization angle that rotates source-frame polarizations to the SSB basis.

        :math:`\psi = -\mathrm{atan2}(u, d)` with
        :math:`u = \cos q_S \sin q_K \cos(\phi_S - \phi_K) - \cos q_K \sin q_S` and
        :math:`d = \sin q_K \sin(\phi_S - \phi_K)`; :math:`\psi = \pi/2` when :math:`d = 0`
        (the ``psi_ldc`` convention of the LISA Data Challenges). Used by
        :meth:`~few.waveform.waveform.GenerateEMRIWaveform._to_SSB_frame` and by the LISA
        Analysis Tools EMRI response.

        Args:
            qS: Polar angle of the sky location in rad.
            phiS: Azimuthal angle of the sky location in rad.
            qK: Polar angle of the primary's spin in rad, in the same frame as ``qS``.
            phiK: Azimuthal angle of the primary's spin in rad.

        Returns:
            float: :math:`\psi` in rad, in :math:`[-\pi, \pi]`.
        """

        cqS = np.cos(qS)
        sqS = np.sin(qS)

        # cphiS = np.cos(phiS)
        # sphiS = np.sin(phiS)

        cqK = np.cos(qK)
        sqK = np.sin(qK)

        # cphiK = np.cos(phiK)
        # sphiK = np.sin(phiK)

        # get polarization angle

        up_ldc = cqS * sqK * np.cos(phiS - phiK) - cqK * sqS
        dw_ldc = sqK * np.sin(phiS - phiK)

        if dw_ldc != 0.0:
            psi_ldc = -np.arctan2(up_ldc, dw_ldc)

        else:
            psi_ldc = 0.5 * np.pi

        return psi_ldc


def get_mismatch(
    time_series_1: np.ndarray, time_series_2: np.ndarray, use_gpu: bool = False
) -> float:
    """Return the mismatch ``1 - overlap`` of two time series.

    See :func:`get_overlap` for the (flat-noise, zero-lag) overlap and the truncation of
    series of different lengths.

    Args:
        time_series_1: First time series.
        time_series_2: Second time series.
        use_gpu: Use cupy instead of numpy. Default ``False``.

    Returns:
        float: The mismatch, in :math:`[0, 2]`.

    Example:
        >>> import numpy as np
        >>> from few.utils.utility import get_mismatch
        >>> h = np.exp(2j * np.pi * 1e-3 * 10.0 * np.arange(1000))
        >>> abs(float(get_mismatch(h, h))) < 1e-12
        True
    """
    overlap = get_overlap(time_series_1, time_series_2, use_gpu=use_gpu)
    return 1.0 - overlap


@njit(fastmath=False)
def _solveCubic(A2, A1, A0):
    """Roots of :math:`r^3 + A_2 r^2 + A_1 r + A_0 = 0` (Cardano / trigonometric form).

    Used by the equatorial branch of :func:`~few.utils.geodesic.ELQ_to_pex`. With three real
    roots it returns ``(rp, ra, r3)`` = (middle, largest, smallest root), i.e. periapsis,
    apoapsis and the third root of the radial potential. With one real root (non-negative
    discriminant) ``ra = rp`` is the real part of the complex pair and ``r3`` the real root,
    which is not meaningful for bound orbits.
    """
    # Coefficients
    a = 1.0  # coefficient of r^3
    b = A2  # coefficient of r^2
    c = A1  # coefficient of r^1
    d = A0  # coefficient of r^0

    # Calculate p and q
    p = (3.0 * a * c - b * b) / (3.0 * a * a)
    q = (2.0 * b * b * b - 9.0 * a * b * c + 27.0 * a * a * d) / (27.0 * a * a * a)

    # Calculate discriminant
    discriminant = q * q / 4.0 + p * p * p / 27.0

    if discriminant >= 0:
        # One real root and two complex conjugate roots
        u = (-q / 2.0 + sqrt(discriminant)) ** (1 / 3)
        v = (-q / 2.0 - sqrt(discriminant)) ** (1 / 3)
        root = u + v - b / (3.0 * a)
        # cout << "Real Root: " << root << endl

        # imaginaryPart(-sqrt(3.0) / 2.0 * (u - v), 0.5 * (u + v))
        imaginaryPart = 0.5 * (u + v)
        root2 = -0.5 * (u + v) - b / (3.0 * a) + imaginaryPart
        root3 = -0.5 * (u + v) - b / (3.0 * a) - imaginaryPart
        # cout << "Complex Root 1: " << root2 << endl
        # cout << "Complex Root 2: " << root3 << endl
        ra = -0.5 * (u + v) - b / (3.0 * a)
        rp = -0.5 * (u + v) - b / (3.0 * a)
        r3 = root
    # } else if (discriminant == 0) {
    #     # All roots are real and at least two are equal
    #     u = cbrt(-q/2.)
    #     v = cbrt(-q/2.)
    #     root = u + v - b/(3.*a)
    #     # cout << "Real Root: " << root << endl
    #     # cout << "Real Root (equal to above): " << root << endl
    #     # complex<double> root2 = -0.5 * (u + v) - b / (3 * a)
    #     # cout << "Complex Root: " << root2 << endl
    #     *ra = -0.5 * (u + v) - b / (3. * a)
    #     *rp = -0.5 * (u + v) - b / (3. * a)
    #     *r3 = root
    else:
        # All three roots are real and different
        r = sqrt(-p / 3.0)
        theta = acos(-q / (2.0 * r * r * r))
        root1 = 2.0 * r * cos(theta / 3.0) - b / (3.0 * a)
        root2 = 2.0 * r * cos((theta + 2.0 * PI) / 3.0) - b / (3.0 * a)
        root3 = 2.0 * r * cos((theta - 2.0 * PI) / 3.0) - b / (3.0 * a)

        ra = root1
        rp = root3
        r3 = root2

    return rp, ra, r3
    # cout << "ra: " << *ra << endl
    # cout << "rp: " << *rp << endl
    # cout << "r3: " << *r3 << endl


@njit(fastmath=False)
def _brentq_jit(f, a, b, args, tol):
    """Brent's bracketed root finder for numba code (scalar).

    Args:
        f: numba-compiled function called as ``f(x, args)``.
        a (float): One end of the bracket.
        b (float): Other end; ``f(a)`` and ``f(b)`` must have opposite signs.
        args (tuple): Extra parameters passed to ``f``.
        tol (float): Absolute tolerance; iteration stops when half the bracket is below
            ``2 eps |b| + tol / 2`` (``eps`` the double-precision machine epsilon).

    Returns:
        float: The root (``a`` or ``b`` directly if ``f`` vanishes there). There is no
        iteration limit.

    Raises:
        ValueError: If ``f(a)`` and ``f(b)`` have the same sign.
    """
    # Machine epsilon for double precision
    eps = 2.220446049250313e-16

    fa = f(a, args)
    fb = f(b, args)

    # Check that f(a) and f(b) have different signs
    if fa == 0.0 or fb == 0.0:
        return a if fa == 0.0 else b

    if fa * fb > 0.0:
        raise ValueError("f(a) and f(b) must have different signs.")

    c = a
    fc = fa
    d = b - a
    e = d

    while True:
        if abs(fc) < abs(fb):
            a = b
            b = c
            c = a
            fa = fb
            fb = fc
            fc = fa

        tol1 = 2.0 * eps * abs(b) + 0.5 * tol
        xm = 0.5 * (c - b)

        if abs(xm) <= tol1 or fb == 0.0:
            # within tolerance -> return root
            return b

        # Check if bisection is forced
        if abs(e) >= tol1 and abs(fa) > abs(fb):
            s = fb / fa
            if a == c:
                # Linear interpolation
                p = 2.0 * xm * s
                q = 1.0 - s
            else:
                # Inverse quadratic interpolation
                q = fa / fc
                r = fb / fc
                p = s * (2.0 * xm * q * (q - r) - (b - a) * (r - 1.0))
                q = (q - 1.0) * (r - 1.0) * (s - 1.0)

            if p > 0.0:
                q = -q
            else:
                p = -p

            if (2.0 * p < 3.0 * xm * q - abs(tol1 * q)) and (p < abs(0.5 * e * q)):
                # Accept interpolation
                d = p / q
            else:
                # Bisection step
                d = xm
                e = d
        else:
            # Bisection step
            d = xm
            e = d

        a = b
        fa = fb
        if abs(d) > tol1:
            b += d
        else:
            b += tol1 if xm > 0 else -tol1

        fb = f(b, args)
        if fb * fc > 0.0:
            c = a
            fc = fa
            d = b - a
            e = d


def get_at_t(
    traj_module: object,
    traj_args: list[float],
    bounds: list[float],
    t_out: float,
    index_of_interest: int,
    traj_kwargs: Optional[dict] = None,
    xtol: float = 2e-12,
    rtol: float = 8.881784197001252e-16,
) -> float:
    """Find the value of one trajectory argument that gives a trajectory of duration ``t_out``.

    Solves ``t_final(value) = t_out`` with :func:`scipy.optimize.brentq`: for each trial
    value the argument list is rebuilt (``value`` inserted at ``index_of_interest``) and the
    trajectory is run with ``T = 2 t_out``; its last time minus ``t_out`` (in seconds) is
    the root function, ``-t_out`` if the trajectory is empty (it started at the
    separatrix). Used by :func:`get_p_at_t` and :func:`get_m2_at_t`.

    Args:
        traj_module: Trajectory module, e.g. :class:`~few.trajectory.inspiral.EMRIInspiral`,
            whose first output is the time array in seconds.
        traj_args: Trajectory arguments without the unknown; must be a ``list`` (the trial
            value is inserted into a copy).
        bounds: ``[low, high]`` bracket of the unknown; the root function must change sign
            over it.
        t_out: Requested duration in years.
        index_of_interest: Position of the unknown in the full argument list.
        traj_kwargs: Keyword arguments of ``traj_module``. Default ``None`` (empty). The
            dict is modified in place: its ``"T"`` is set to ``2 t_out``.
        xtol: Absolute tolerance of :func:`scipy.optimize.brentq`. Default ``2e-12``
            (the scipy default).
        rtol: Relative tolerance of :func:`scipy.optimize.brentq`. Default
            ``8.881784197001252e-16`` (four machine epsilons, the scipy default).

    Returns:
        float: The value of the unknown.

    Raises:
        ValueError: From :func:`scipy.optimize.brentq` if the root function has the same sign
            at both bounds.
    """
    if traj_kwargs is None:
        traj_kwargs = {}

    def get_time_root(val, traj, inj_args, traj_kwargs, t_out, ind_interest):
        """Root function of :func:`get_at_t`: final trajectory time minus ``t_out`` (in s)."""
        inputs = inj_args.copy()
        inputs.insert(ind_interest, val)
        traj_kwargs["T"] = t_out * 2.0
        out = traj(*inputs, **traj_kwargs)
        try:
            return out[0][-1] - t_out * YRSID_SI
        except IndexError:  # trajectory must have started at p_sep
            return -t_out * YRSID_SI

    root = brentq(
        get_time_root,
        bounds[0],
        bounds[1],
        xtol=xtol,
        rtol=rtol,
        args=(traj_module, traj_args, traj_kwargs, t_out, index_of_interest),
    )
    return root


def get_p_at_t(
    traj_module: object,
    t_out: float,
    traj_args: list[float],
    index_of_p: int = 3,
    index_of_a: int = 2,
    index_of_e: int = 4,
    index_of_x: int = 5,
    bounds: list[Optional[float]] = None,
    lower_bound_buffer: float = 1e-6,
    upper_bound_maximum: float = 1e6,
    **kwargs,
) -> float:
    """Find the initial semi-latus rectum ``p0`` that gives an inspiral of duration ``t_out``.

    Uses Brent's method (:func:`get_at_t`) on the (assumed unique) ``p0`` whose trajectory
    reaches the plunge after exactly ``t_out`` years. Default bounds come from the ODE's
    ``func.min_p`` / ``func.max_p`` at ``(e0, x0, a)`` (with ``enforce_schwarz_sep`` in
    ``traj_kwargs``, the lower bound is at most ``6 + 2 e0``); ``lower_bound_buffer`` is then
    added to the lower bound and an infinite upper bound is replaced by
    ``upper_bound_maximum``. Before the root search the lower bound is probed: a trajectory
    that leaves the flux grid (:class:`~few.utils.exceptions.TrajectoryOffGridException`)
    raises it by ``0.01`` and retries, until it reaches the upper bound.

    Args:
        traj_module: Trajectory module with a ``func`` attribute exposing ``min_p`` /
            ``max_p`` (e.g. :class:`~few.trajectory.inspiral.EMRIInspiral`); its first output
            is the time in seconds.
        t_out: Requested duration in years.
        traj_args: Trajectory arguments without ``p0``, as a ``list``; for the stock models
            ``[m1, m2, a, e0, x0]``.
        index_of_p: Position of ``p0`` in the full argument list ``(m1, m2, a, p0, e0, x0)``.
            Default ``3``.
        index_of_a: Position of ``a`` in the full argument list. Default ``2``.
        index_of_e: Position of ``e0`` in the full argument list. Default ``4``.
        index_of_x: Position of ``x0`` in the full argument list. Default ``5``.
        bounds: ``[low, high]`` search range of ``p0``; either entry ``None`` (or ``bounds``
            ``None``, the default) uses the ODE limits above. A given list is modified in
            place.
        lower_bound_buffer: Added to the lower bound so that the search does not start on
            the separatrix. Default ``1e-6``.
        upper_bound_maximum: Replaces an infinite upper bound. Default ``1e6``.
        **kwargs: Forwarded to :func:`get_at_t` (``traj_kwargs``, ``xtol``, ``rtol``).

    Returns:
        float: ``p0`` in units of :math:`M`.

    Raises:
        ValueError: If the trajectory from the lower bound already lasts at least
            ``t_out`` ("No solution found within the bounds of the interpolants."), or from
            :func:`scipy.optimize.brentq` if the bracket has no sign change.

    Note:
        The probing trajectory is run as ``traj_module(m1, m2, a, p, e0, x0, T=...)`` built
        from ``traj_args[0]``, ``traj_args[1]`` and the three indices, without
        ``traj_kwargs``.

    Example:
        .. code-block:: python

            from few.trajectory.inspiral import EMRIInspiral
            from few.trajectory.ode import KerrEccEqFlux
            from few.utils.utility import get_p_at_t

            traj = EMRIInspiral(func=KerrEccEqFlux)
            # [m1, m2, a, e0, x0]; p0 is solved for (index 3 of the full list)
            p0 = get_p_at_t(traj, 1.0, [1e6, 10.0, 0.9, 0.3, 1.0], index_of_p=3)
    """

    # fix indexes for p
    if index_of_a > index_of_p:
        index_of_a -= 1
    if index_of_e > index_of_p:
        index_of_e -= 1
    if index_of_x > index_of_p:
        index_of_x -= 1

    if "traj_kwargs" in kwargs and "enforce_schwarz_sep" in kwargs["traj_kwargs"]:
        enforce_schwarz_sep = kwargs["traj_kwargs"]["enforce_schwarz_sep"]

    else:
        enforce_schwarz_sep = False

    # fix bounds
    if bounds is None:
        bounds = [None, None]

    if bounds[0] is None:
        if not enforce_schwarz_sep:
            bounds[0] = traj_module.func.min_p(
                traj_args[index_of_e], x=traj_args[index_of_x], a=traj_args[index_of_a]
            )
        else:
            bounds[0] = min(
                traj_module.func.min_p(
                    traj_args[index_of_e],
                    x=traj_args[index_of_x],
                    a=traj_args[index_of_a],
                ),
                6 + 2 * traj_args[index_of_e],
            )

    if bounds[1] is None:
        bounds[1] = traj_module.func.max_p(
            traj_args[index_of_e], x=traj_args[index_of_x], a=traj_args[index_of_a]
        )
    # Adding a buffer to prevent starting trajectories on the separatrix
    bounds[0] += lower_bound_buffer

    # Prevent the rootfinder from starting at infinity
    if bounds[1] == float("inf"):
        bounds[1] = upper_bound_maximum

    # With the varying bounds of eccentricity used for KerrEccEqFlux,
    # it is possible to have no solution within the bounds of the interpolants
    # It might also be possible for the trajectory to evolve off of the grid
    while bounds[0] < bounds[1]:
        try:
            traj_pars = [
                traj_args[0],
                traj_args[1],
                traj_args[index_of_a],
                bounds[0],
                traj_args[index_of_e],
                traj_args[index_of_x],
            ]
            t, p, e, xI, Phi_phi, Phi_theta, Phi_r = traj_module(
                *traj_pars, T=t_out * 1.001
            )
            if t[-1] >= t_out * YRSID_SI:
                raise ValueError(
                    "No solution found within the bounds of the interpolants."
                )
            break
        except TrajectoryOffGridException:
            # Trajectory is off the grid
            # Increase lower bound and try again
            bounds[0] += 1e-2

    root = get_at_t(traj_module, traj_args, bounds, t_out, index_of_p, **kwargs)
    return root


def get_m2_at_t(
    traj_module: object,
    t_out: float,
    traj_args: list[float],
    index_of_m2: int = 1,
    bounds: list[Optional[float]] = None,
    **kwargs,
) -> float:
    """Find the secondary mass ``m2`` that gives an inspiral of duration ``t_out``.

    Uses Brent's method (:func:`get_at_t`) on the (assumed unique) ``m2`` whose trajectory
    plunges after exactly ``t_out`` years. Both bounds are checked first: the inspiral at the
    lower bound must last longer than ``t_out``, the one at the upper bound shorter.

    Args:
        traj_module: Trajectory module (e.g. :class:`~few.trajectory.inspiral.EMRIInspiral`)
            whose first output is the time in seconds.
        t_out: Requested duration in years.
        traj_args: Trajectory arguments without ``m2``, as a ``list``:
            ``[m1, a, p0, e0, x0]``.
        index_of_m2: Position of ``m2`` in the full argument list. Default ``1``; the bound
            checks always build ``(m1, m2, a, p0, e0, x0)``, so other values are not
            supported consistently.
        bounds: ``[low, high]`` search range of ``m2`` in solar masses. Default ``None``
            means ``[0.1, m1]``; a single ``None`` entry is replaced by its default.
        **kwargs: Forwarded to :func:`get_at_t` (``traj_kwargs``, ``xtol``, ``rtol``).

    Returns:
        float: ``m2`` in solar masses.

    Raises:
        ValueError: If either bound check fails ("No solution found within the bounds for
            secondary mass."), or from :func:`scipy.optimize.brentq`.

    Example:
        .. code-block:: python

            from few.trajectory.inspiral import EMRIInspiral
            from few.trajectory.ode import KerrEccEqFlux
            from few.utils.utility import get_m2_at_t

            traj = EMRIInspiral(func=KerrEccEqFlux)
            # [m1, a, p0, e0, x0]; m2 is solved for (index 1 of the full list)
            m2 = get_m2_at_t(traj, 1.0, [1e6, 0.9, 10.0, 0.3, 1.0], index_of_m2=1)
    """

    # fix bounds
    if bounds is None:
        bounds = [1e-1, traj_args[0]]

    elif bounds[0] is None:
        bounds[0] = 1e-1

    elif bounds[1] is None:
        bounds[1] = traj_args[0]

    # Check lower bound
    traj_pars = [
        traj_args[0],
        bounds[0],
        traj_args[1],
        traj_args[2],
        traj_args[3],
        traj_args[4],
    ]
    t, p, e, xI, Phi_phi, Phi_theta, Phi_r = traj_module(*traj_pars, T=t_out * 1.001)
    if t[-1] <= t_out * YRSID_SI:
        raise ValueError("No solution found within the bounds for secondary mass.")

    # Check lower bound
    traj_pars = [
        traj_args[0],
        bounds[1],
        traj_args[1],
        traj_args[2],
        traj_args[3],
        traj_args[4],
    ]
    t, p, e, xI, Phi_phi, Phi_theta, Phi_r = traj_module(*traj_pars, T=t_out * 1.001)
    if t[-1] >= t_out * YRSID_SI:
        raise ValueError("No solution found within the bounds for secondary mass.")

    root = get_at_t(traj_module, traj_args, bounds, t_out, index_of_m2, **kwargs)
    return root


# data history is saved here nased on version nunber
# record_by_version = {
#     "1.0.0": 3981654,
#     "1.1.0": 3981654,
#     "1.1.1": 3981654,
#     "1.1.2": 3981654,
#     "1.1.3": 3981654,
#     "1.1.4": 3981654,
#     "1.1.5": 3981654,
#     "1.2.0": 3981654,
#     "1.2.1": 3981654,
#     "1.2.2": 3981654,
#     "1.3.0": 3981654,
#     "1.3.1": 3981654,
#     "1.3.2": 3981654,
#     "1.3.3": 3981654,
#     "1.3.4": 3981654,
#     "1.3.5": 3981654,
#     "1.3.6": 3981654,
#     "1.3.7": 3981654,
#     "1.4.0": 3981654,
#     "1.4.1": 3981654,
#     "1.4.2": 3981654,
#     "1.4.3": 3981654,
#     "1.4.4": 3981654,
#     "1.4.5": 3981654,
#     "1.4.6": 3981654,
#     "1.4.7": 3981654,
#     "1.4.8": 3981654,
#     "1.4.9": 3981654,
#     "1.4.10": 3981654,
#     "1.4.11": 3981654,
#     "1.5.0": 3981654,
#     "1.5.1": 3981654,
# }


# TODO: deal with this and make sure it is in gpubackendtools and import what is needed. 

# TODO check: keep/remove/document -- duplicate of few.cutils.wrappers.wrapper; no callers
def wrapper(*args, **kwargs):
    """Function to convert array and C/C++ class arguments to ptrs

    This function checks the object type. If it is a cupy or numpy array,
    it will determine its pointer by calling the proper attributes. If you design
    a Cython class to be passed through python, it must have a :code:`ptr`
    attribute.

    If you use this function, you must convert input arrays to size_t data type in Cython and
    then properly cast the pointer as it enters the c++ function. See the
    Cython codes
    `here <https://github.com/BlackHolePerturbationToolkit/FastEMRIWaveforms/tree/master/src/few/cutils>`_
    for examples.

    args:
        *args (list): list of the arguments for a function.
        **kwargs (dict): dictionary of keyword arguments to be converted.

    returns:
        Tuple: (targs, tkwargs) where t indicates target (with pointer values
            rather than python objects).

    """
    # declare target containers
    targs = []
    tkwargs = {}

    best_backend = get_first_backend(["cuda13x", "cuda12x", "cuda11x", "cpu"])

    # args first
    for arg in args:
        if best_backend.uses_cupy:
            # cupy arrays
            if isinstance(arg, best_backend.xp.ndarray):
                targs.append(arg.data.mem.ptr)
                continue

        # numpy arrays
        if isinstance(arg, np.ndarray):
            targs.append(arg.__array_interface__["data"][0])
            continue

        try:
            # cython classes
            targs.append(arg.ptr)
            continue
        except AttributeError:
            # regular argument
            targs.append(arg)

    # kwargs next
    for key, arg in kwargs.items():
        if best_backend.uses_cupy:
            # cupy arrays
            if isinstance(arg, best_backend.xp.ndarray):
                tkwargs[key] = arg.data.mem.ptr
                continue

        if isinstance(arg, np.ndarray):
            # numpy arrays
            tkwargs[key] = arg.__array_interface__["data"][0]
            continue

        try:
            # cython classes
            tkwargs[key] = arg.ptr
            continue
        except AttributeError:
            # other arguments
            tkwargs[key] = arg

    return (targs, tkwargs)


# TODO check: keep/remove/document -- duplicate of few.cutils.wrappers.pointer_adjust; no callers
def pointer_adjust(func):
    """Decorator function for cupy/numpy agnostic cython

    This decorator applies :func:`few.utils.utility.wrapper` to functions
    via the decorator construction.

    If you use this decorator, you must convert input arrays to size_t data type in Cython and
    then properly cast the pointer as it enters the c++ function. See the
    Cython codes
    `here <https://github.com/BlackHolePerturbationToolkit/FastEMRIWaveforms/tree/master/src/few/cutils>`_
    for examples.

    """

    def func_wrapper(*args, **kwargs):
        # get pointers
        targs, tkwargs = wrapper(*args, **kwargs)
        return func(*targs, **tkwargs)

    return func_wrapper


def brentq_array(f, a, b, args, tol, maxiter=200):
    """Vectorised bracketed root finder: one root per element of ``a`` / ``b``.

    Anderson-Bjorck regula falsi with a bisection safeguard, run on the whole batch with
    per-element convergence masks (the batched counterpart of the scalar
    :func:`_brentq_jit` / ``scipy.optimize.brentq`` calls of the trajectory; part of the
    experimental trajectory batching of ``docs/traj-batching.md``, not called by FEW
    itself). An element converges when its bracket width is at most ``tol + 4 eps |c|``
    (``c`` the latest iterate).

    Args:
        f: Callable ``f(x, args)`` returning an array of the shape of ``x``, evaluated
            elementwise on numpy arrays.
        a (np.ndarray): Lower ends of the brackets (copied, as float).
        b (np.ndarray): Upper ends, same shape; ``f(a) * f(b) <= 0`` elementwise.
        args: Passed through to ``f`` unchanged (e.g. per-element parameters).
        tol (float): Absolute ``x`` tolerance.
        maxiter (int, optional): Maximum number of iterations. Default ``200``.

    Returns:
        np.ndarray: The roots, float, shape of ``a``; an end where ``f`` vanishes is returned
        directly.

    Raises:
        ValueError: If an element has no sign change, or if not every element converged
            within ``maxiter`` iterations.
    """
    eps = 2.220446049250313e-16
    a = np.array(a, dtype=float, copy=True)
    b = np.array(b, dtype=float, copy=True)
    fa = np.asarray(f(a, args), dtype=float).copy()
    fb = np.asarray(f(b, args), dtype=float).copy()
    if np.any(fa * fb > 0.0):
        raise ValueError("brentq_array: f(a) and f(b) must have different signs for every element.")
    out = np.where(fa == 0.0, a, np.where(fb == 0.0, b, np.nan))
    done = (fa == 0.0) | (fb == 0.0)
    side = np.zeros(a.shape, dtype=int)            # which end was kept last (-1 a, +1 b)
    for _ in range(maxiter):
        if np.all(done):
            return out
        live = ~done
        # regula falsi point, bisection where it would be degenerate or outside
        with np.errstate(divide="ignore", invalid="ignore"):
            c = (a * fb - b * fa) / (fb - fa)
        bad = ~np.isfinite(c) | (c <= np.minimum(a, b)) | (c >= np.maximum(a, b))
        c = np.where(bad, 0.5 * (a + b), c)
        fc = np.asarray(f(c, args), dtype=float)
        root = live & (fc == 0.0)
        out[root] = c[root]
        done |= root
        live = ~done
        left = live & (fa * fc < 0.0)              # root in [a, c]: move b
        right = live & ~left                       # root in [c, b]: move a
        # Anderson-Bjorck: scale the retained end's value when the same end is kept twice
        with np.errstate(divide="ignore", invalid="ignore"):
            m_ab = 1.0 - fc / fb
            m_aa = 1.0 - fc / fa
        m_ab = np.where(m_ab > 0.0, m_ab, 0.5)
        m_aa = np.where(m_aa > 0.0, m_aa, 0.5)
        fa = np.where(left & (side == -1), fa * m_ab, fa)
        fb = np.where(right & (side == 1), fb * m_aa, fb)
        b = np.where(left, c, b)
        fb = np.where(left, fc, fb)
        a = np.where(right, c, a)
        fa = np.where(right, fc, fa)
        side = np.where(left, -1, np.where(right, 1, side))
        conv = live & (np.abs(b - a) <= tol + 4 * eps * np.abs(c))
        out[conv] = c[conv]
        done |= conv
    if not np.all(done):
        raise ValueError("brentq_array: maximum iterations exceeded.")
    return out
