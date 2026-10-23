r"""The generic EMRI trajectory module :class:`EMRIInspiral` and a 0PA frequency helper.

:class:`EMRIInspiral` integrates any :class:`~few.trajectory.ode.base.ODEBase` model (stock:
``KerrEccEqFlux``, ``SchwarzEccFlux``, ``PN5``) with the DOP853 integrator of
:mod:`few.trajectory.integrate` and exposes the integrator's dense output for the waveform
generators. In the FEW waveform models it is the ``inspiral_generator`` of
:class:`~few.waveform.base.SphericalHarmonicWaveformBase`; on the LISA global-fit EMRI path
(``FastKerrEccentricEquatorialFlux`` with ``KerrEccEqFlux``) it runs on the CPU in the main
process and in the :mod:`few.trajectory.pool` workers.

Conventions: masses ``m1`` (primary) and ``m2`` (secondary) in solar masses, spin ``a``
dimensionless (range set by the model, :math:`|a| \le 0.999` for ``KerrEccEqFlux``),
semi-latus rectum ``p`` in units of :math:`M = m_1 + m_2`, eccentricity ``e``, inclination
coordinate ``x`` (:math:`x_I = \cos I`, ``+1`` prograde and ``-1`` retrograde for equatorial
models; :math:`Y = \cos\iota` for models with ``convert_Y``, e.g. ``PN5``), phases in
radians, times in seconds, ``T`` in years.
"""

# Generic trajectory module for Fast EMRI Waveforms (any ODEBase model)

from typing import Type, Union

import numpy as np

from ..utils.citations import REFERENCE
from ..utils.constants import MTSUN_SI, PI
from ..utils.geodesic import (
    ELQ_to_pex,
    get_fundamental_frequencies,
    get_kerr_geo_constants_of_motion,
)
from ..utils.globals import get_logger
from ..utils.mappings.pn import Y_to_xI

# Python imports
from .base import TrajectoryBase
from .integrate import get_integrator
from .ode.base import ODEBase


class EMRIInspiral(TrajectoryBase):
    r"""EMRI trajectory module: integrate an ODE model from initial orbital parameters.

    Integrates a generic ODE model (:mod:`few.trajectory.ode`) with the DOP853 Runge-Kutta
    integrator of :mod:`few.trajectory.integrate`, adaptively (default) or with fixed steps.
    An adaptive integration also builds a dense-output (continuous) solution that can be
    evaluated at any time, through ``inspiral_generator.eval_integrator_spline`` and
    ``inspiral_generator.eval_integrator_derivative_spline``. Calling the module
    (:meth:`~few.trajectory.base.TrajectoryBase.__call__`) runs :meth:`get_inspiral`.

    The integration runs on the radiation-reaction time :math:`\tilde t = \nu\, t / M`
    with :math:`\nu = \mu / M = m_1 m_2 / (m_1 + m_2)^2`, and integrates the phases as
    :math:`\nu \Phi`; the outputs are converted back to seconds and radians. It stops at
    ``T`` or when ``p`` reaches the model's separatrix buffer
    (``p_sep + func.separatrix_buffer_dist``); the final point is then placed exactly at
    ``T``, or on the buffer by root-finding on the dense output.

    The integrator built here is ``inspiral_generator`` (an
    :class:`~few.trajectory.integrate.APEXIntegrate`, or an
    :class:`~few.trajectory.integrate.AELQIntegrate` with
    ``integrate_constants_of_motion=True``); it keeps the state of the last integration
    (``trajectory``, ``integrator_t_cache``, ``integrator_spline_coeff``), which the
    waveform generators read after each call. ``specific_kwarg_keys`` lists the call keys
    forwarded from :meth:`__call__` to the integrator.

    Args:
        func: The ODE model: an :class:`~few.trajectory.ode.base.ODEBase` subclass (not an
            instance) or the name of a stock model (``"KerrEccEqFlux"``,
            ``"SchwarzEccFlux"``, ``"PN5"``; see ``few.trajectory.ode._STOCK_TRAJECTORY_OPTIONS``).
            Keyword-only and required.
        integrate_constants_of_motion: If ``True``, integrate the constants of motion
            ``(E, L, Q)`` instead of ``(p, e, x)``; the model must support it
            (``supports_ELQ``). Default ``False``.
        enforce_schwarz_sep: If ``True``, stop at the Schwarzschild separatrix
            :math:`p_s = 6 + 2e` (plus the buffer) instead of the Kerr separatrix. This can
            help at high spin or eccentricity where, e.g., PN fluxes become inaccurate.
            Default ``False``.
        convert_to_pex: With ``integrate_constants_of_motion``, convert the output from
            ``(E, L, Q)`` back to ``(p, e, x)``. Default ``True``.
        rootfind_separatrix: Accepted for compatibility; has no effect (it is forwarded to
            the ODE constructor, which ignores it). The final point is always placed by
            root-finding. Default ``True``.
        *args: Forwarded to :class:`~few.trajectory.base.TrajectoryBase` (ignored).
        **kwargs: Forwarded to :class:`~few.trajectory.integrate.Integrate`
            (``buffer_length``, default 10000; ``max_iter``, default ``buffer_length``;
            ``downsample``) and from there to the ODE constructor (e.g. ``downsample`` and
            ``flux_output_convention`` of :class:`~few.trajectory.ode.flux.KerrEccEqFlux`).
            Keys nobody uses are swallowed by :class:`~few.trajectory.ode.base.ODEBase`
            (e.g. ``force_backend``, ``max_init_len``, ``DENSE_STEPPING`` from a waveform's
            ``inspiral_kwargs``).

    Raises:
        ValueError: If ``func`` is a string that is not a stock model.
        AssertionError: If ``integrate_constants_of_motion`` is set and the model does not
            support ELQ integration.
        few.utils.exceptions.FileManagerException: If the model's data file cannot be
            found or fetched.

    Attributes:
        func (ODEBase): The ODE model instance (``inspiral_generator.func``).
        integrate_constants_of_motion (bool): Whether ``(E, L, Q)`` are integrated.
        convert_to_pex (bool): Whether ELQ output is converted back to ``(p, e, x)``.
        enforce_schwarz_sep (bool): Whether the Schwarzschild separatrix is used.

    Example:
        .. code-block:: python

            from few.trajectory.inspiral import EMRIInspiral
            from few.trajectory.ode import KerrEccEqFlux

            traj = EMRIInspiral(func=KerrEccEqFlux)
            # m1, m2 [Msun], a, p0 [M], e0, xI0; T in years
            t, p, e, xI, Phi_phi, Phi_theta, Phi_r = traj(1e6, 1e2, 0.9, 10.0, 0.8, 1.0, T=4.0)

            # dense output at new times [s]: columns p, e, x, Phi_phi, Phi_theta, Phi_r
            import numpy as np
            t_new = np.linspace(0.0, t[-1], 1000)
            y_new = traj.inspiral_generator.eval_integrator_spline(t_new)
    """

    def __init__(
        self,
        *args,
        func: Union[str, Type[ODEBase]],
        integrate_constants_of_motion: bool = False,
        enforce_schwarz_sep: bool = False,
        convert_to_pex: bool = True,
        rootfind_separatrix: bool = True,
        **kwargs,
    ):
        TrajectoryBase.__init__(self, *args, **kwargs)

        self.enforce_schwarz_sep = enforce_schwarz_sep
        self.inspiral_generator = get_integrator(
            func,
            integrate_constants_of_motion=integrate_constants_of_motion,
            enforce_schwarz_sep=enforce_schwarz_sep,
            rootfind_separatrix=rootfind_separatrix,
            **kwargs,
        )
        """Integrate: Integrator instance of the trajectory, from
        :func:`~few.trajectory.integrate.get_integrator`: an
        :class:`~few.trajectory.integrate.APEXIntegrate`, or an
        :class:`~few.trajectory.integrate.AELQIntegrate` with
        ``integrate_constants_of_motion``."""

        self.func = self.inspiral_generator.func

        self.specific_kwarg_keys = [
            "T",
            "dt",
            "err",
            "DENSE_STEPPING",
            "buffer_length",
            "integrate_backwards",
            "max_step_size",
        ]
        """list[str]: Call keywords forwarded to the integrator on every call (they may
        change from call to call)."""

        self.integrate_constants_of_motion = (
            self.inspiral_generator.integrate_constants_of_motion
        )

        self.convert_to_pex = convert_to_pex

    @classmethod
    def module_references(cls) -> list[REFERENCE]:
        """Return the citations of this module (the Kerr separatrix plus the base list).

        Returns:
            list[REFERENCE]: ``[REFERENCE.KERR_SEPARATRIX]`` followed by the parent's.
        """
        return [REFERENCE.KERR_SEPARATRIX] + super().module_references()

    @property
    def npoints(self):
        """int: Number of points of the last trajectory (``inspiral_generator.npoints``)."""
        return self.inspiral_generator.npoints

    @property
    def tolerance(self) -> float:
        """Intended to be the absolute tolerance of the integrator.

        It currently returns ``inspiral_generator.npoints`` (the number of trajectory
        points), not the tolerance; use ``inspiral_generator.tolerance`` for the
        tolerance (``err`` of the last call).
        """
        return self.inspiral_generator.npoints

    @property
    def dense_stepping(self) -> bool:
        """If ``True``, the last trajectory used fixed steps (``DENSE_STEPPING``)."""
        return self.inspiral_generator.dense_stepping

    @property
    def integrate_backwards(self) -> bool:
        """If ``True``, the last trajectory was integrated backwards in time."""
        return self.inspiral_generator.integrate_backwards

    @property
    def trajectory(self):
        """np.ndarray: The last trajectory, shape ``(npoints, nparams + 1)``.

        Column 0 is the time in seconds, columns 1-3 the orbital state (``p, e, x`` for a
        pex integration), columns 4-6 the phases ``Phi_phi, Phi_theta, Phi_r`` in rad.
        """
        return self.inspiral_generator.trajectory

    @property
    def integrator_spline_t(self):
        """np.ndarray: Knot times of the dense output in seconds, shape ``(npoints,)``."""
        return self.inspiral_generator.integrator_t_cache

    @property
    def integrator_spline_coeff(self):
        r"""np.ndarray: DOP853 dense-output coefficients, shape ``(npoints - 1, 6, 8)``.

        One set of 8 coefficients per segment and per integrated quantity
        ``(p, e, x, nu * Phi_phi, nu * Phi_theta, nu * Phi_r)``: the phase rows carry the
        integration scaling by :math:`\nu = m_1 m_2 / (m_1 + m_2)^2` (see
        :attr:`integrator_spline_phase_coeff`). Evaluate with
        :meth:`~few.trajectory.dopr853.DOPR853.eval` at times in seconds against
        :attr:`integrator_spline_t`.
        """
        return self.inspiral_generator.integrator_spline_coeff

    @property
    def integrator_spline_phase_coeff(self):
        """np.ndarray: Phase rows of the dense output in radians, shape ``(npoints - 1, 3, 8)``.

        :attr:`integrator_spline_coeff` ``[:, 3:6]`` divided by the mass ratio. The offset
        that ``eval_integrator_spline`` adds after a backwards integration is not included.
        Read by :class:`~few.waveform.base.SphericalHarmonicWaveformBase` as the phase
        input of the mode summation (with :attr:`integrator_spline_t` as the knots).
        """
        return (
            self.inspiral_generator.integrator_spline_coeff[:, 3:6]
            / self.inspiral_generator.massratio
        )

    def get_inspiral(
        self,
        m1: float,
        m2: float,
        a: float,
        y1: float,
        y2: float,
        y3: float,
        *args,
        Phi_phi0: float = 0.0,
        Phi_theta0: float = 0.0,
        Phi_r0: float = 0.0,
        **kwargs,
    ) -> tuple[np.ndarray]:
        r"""Integrate one inspiral from its initial (or, backwards, final) parameters.

        Called by :meth:`~few.trajectory.base.TrajectoryBase.__call__`, which supplies the
        call keywords. Steps: check the start point with ``func.isvalid_pex``; set
        ``a = 0`` for a Schwarzschild model (and ``a = 1e-6``, with a warning, for a
        non-equatorial Kerr model with ``a < 1e-6``, negative spins included); require
        ``|x0| = 1`` for an
        equatorial model; flip the sign of ``Phi_phi0`` for ``x0 = -1``; set ``e0 = 0``
        for a circular model; convert to ``(E, L, Q)`` when integrating constants of
        motion; negate the initial phases for a backwards integration; then run
        :meth:`~few.trajectory.integrate.Integrate.run_inspiral` on the state
        ``(y1, y2, y3, nu Phi_phi0, nu Phi_theta0, nu Phi_r0)`` with
        :math:`\nu = m_1 m_2 / (m_1 + m_2)^2`.

        Args:
            m1: Mass of the massive black hole in solar masses.
            m2: Mass of the compact object in solar masses.
            a: Dimensionless spin of the massive black hole.
            y1: Initial semi-latus rectum ``p0`` in units of :math:`M = m_1 + m_2`.
            y2: Initial eccentricity ``e0``.
            y3: Initial inclination coordinate ``x0``: :math:`x_I = \cos I` (``+1`` or
                ``-1`` for an equatorial model such as ``KerrEccEqFlux``), or
                :math:`Y = \cos\iota` for a model with ``convert_Y`` (``PN5``).
            *args: Extra ODE arguments, passed to the model as ``additional_args`` of
                :meth:`~few.trajectory.ode.base.ODEBase.add_fixed_parameters`
                (``[0.0]`` when none are given).
            Phi_phi0: Initial azimuthal phase :math:`\Phi_\phi` in rad. Default ``0.0``.
            Phi_theta0: Initial polar phase :math:`\Phi_\theta` in rad. Default ``0.0``.
            Phi_r0: Initial radial phase :math:`\Phi_r` in rad. Default ``0.0``.
            **kwargs: Must contain the call keys ``T`` (years), ``dt`` (s), ``err``,
                ``DENSE_STEPPING``, ``buffer_length``, ``integrate_backwards`` and
                ``max_step_size`` (as added by :meth:`__call__`); other keys are ignored.

        Returns:
            tuple[np.ndarray, ...]: ``(t, p, e, x, Phi_phi, Phi_theta, Phi_r)``, 1-D arrays
            of length :attr:`npoints`: ``t`` in seconds (from 0), ``p`` in units of
            :math:`M`, the phases in rad. With ``integrate_constants_of_motion`` and
            ``convert_to_pex=False`` the columns 1-3 are ``(E, L, Q)``.

        Raises:
            ValueError: If the start point is outside the model's ``x``, ``e`` or ``a``
                bounds (``isvalid_pex``), if the integration exceeds ``max_iter`` steps or
                fails 100 consecutive steps, or (backwards) if it starts too close to the
                separatrix.
            AssertionError: If ``p0`` is outside the model's ``p`` bounds for this
                ``(e0, x0, a)`` (``isvalid_pex`` uses ``assert``).
            RuntimeError: If ``|x0| != 1`` for an equatorial model whose ``isvalid_x``
                accepts it, or if the root-finding of the final point does not converge.
            few.utils.exceptions.TrajectoryOffGridException: If a ``KerrEccEqFlux``
                trajectory leaves the flux grid in ``e`` or ``a`` (not caught by the
                integrator's retry loop, so it reaches the caller).
        """

        # transfer kwargs from parent class
        temp_kwargs = {key: kwargs[key] for key in self.specific_kwarg_keys}
        args_in = np.asarray(args)

        # correct for issue in Cython pass
        if len(args_in) == 0:
            args_in = np.array([0.0])

        fill_value = 1e-6

        # fix for specific requirements of different odes
        background = self.inspiral_generator.background
        equatorial = self.inspiral_generator.equatorial
        circular = self.inspiral_generator.circular

        p0 = y1
        e0 = y2
        x0 = y3

        if temp_kwargs["integrate_backwards"]:
            self.func.isvalid_pex(p=p0, e=e0, x=x0, a=a, p_buffer=[-1e-6, 0])
        else:
            self.func.isvalid_pex(p=p0, e=e0, x=x0, a=a)

        if background == "Schwarzschild":
            a = 0.0
        elif a < fill_value:
            if background == "Kerr" and not equatorial:
                get_logger().warning(
                    "Our model with spin breaks near a = 0. Adjusting to a = 1e-6."
                )
                a = fill_value

        if equatorial:
            if abs(x0) != 1:
                raise RuntimeError(
                    "Magnitude of orbital inclination cosine x0 needs to be one for equatorial inspiral."
                )

        if x0 == -1:
            Phi_phi0 = -1 * Phi_phi0  # flip initial azimuthal phase for retrograde

        if circular:
            e0 = 0.0

        # if integrating constants of motion, convert from pex to ELQ now
        if self.integrate_constants_of_motion:
            if self.inspiral_generator.convert_Y:
                x0 = Y_to_xI(a, p0, e0, x0)
            y1, y2, y3 = get_kerr_geo_constants_of_motion(a, p0, e0, x0)

        # flip initial phases if integrating backwards
        if temp_kwargs["integrate_backwards"]:
            Phi_phi0 = -1 * Phi_phi0
            Phi_theta0 = -1 * Phi_theta0
            Phi_r0 = -1 * Phi_r0

        mu = m1 * m2 / (m1 + m2)
        M = m1 + m2

        y0 = np.array(
            [y1, y2, y3, Phi_phi0 * (mu / M), Phi_theta0 * (mu / M), Phi_r0 * (mu / M)]
        )

        # this will return in coordinate time
        out = self.inspiral_generator.run_inspiral(
            m1, m2, a, y0, args_in, **temp_kwargs
        )
        if self.integrate_constants_of_motion and self.convert_to_pex:
            out_ELQ = out.copy()
            pex = ELQ_to_pex(a, out[:, 1].copy(), out[:, 2].copy(), out[:, 3].copy())
            out[:, 1] = pex[0]
            out[:, 2] = pex[1]
            if self.inspiral_generator.convert_Y:
                out[:, 3] = out_ELQ[:, 2] / np.sqrt(out_ELQ[:, 2] ** 2 + out_ELQ[:, 3])
            else:
                out[:, 3] = pex[2]

        t, p, e, x, Phi_phi, Phi_theta, Phi_r = out.T.copy()
        return t, p, e, x, Phi_phi, Phi_theta, Phi_r

    # TODO check: keep/remove/document -- no callers; add_fixed_parameters(..., False) -> TypeError
    def get_rhs_ode(
        self,
        m1: float,
        m2: float,
        a: float,
        y1: float,
        y2: float,
        y3: float,
        *args,
        Phi_phi0: float = 0.0,
        Phi_theta0: float = 0.0,
        Phi_r0: float = 0.0,
        **kwargs,
    ) -> tuple[np.ndarray]:
        r"""Compute the right hand side of the ordinary differential equation.

        This is a convenience function for interfacing with the call method of the ODE class.

        Args:
            m1: Mass of massive black hole in solar masses.
            m2: Mass of compact object in solar masses.
            a: Dimensionless spin of massive black hole.
            y1: Semi-latus rectum ``p`` in units of M.
            y2: Eccentricity ``e``.
            y3: Inclination coordinate ``x`` (as for :meth:`get_inspiral`).
            *args: Added for flexibility.
            Phi_phi0: Initial phase for :math:`\Phi_\phi`.
                Default is 0.0.
            Phi_theta0: Initial phase for :math:`\Phi_\Theta`.
                Default is 0.0.
            Phi_r0: Initial phase for :math:`\Phi_r`.
                Default is 0.0.
            **kwargs: kwargs passed from parent.

        Returns:
            np.ndarray: The ODE output for the state ``(y1, y2, y3, Phi_phi0, Phi_theta0,
            Phi_r0)`` (the model's ``__call__``), not a trajectory.
        """

        fill_value = 1e-6

        # fix for specific requirements of different odes
        background = self.inspiral_generator.background
        equatorial = self.inspiral_generator.equatorial
        circular = self.inspiral_generator.circular

        self.inspiral_generator.func.add_fixed_parameters(m1, m2, a, False)

        p0 = y1
        e0 = y2
        x0 = y3

        if background == "Schwarzschild":
            a = 0.0
        elif a < fill_value:
            if background == "Kerr" and not equatorial:
                get_logger().warning(
                    "Our model with spin breaks near a = 0. Adjusting to a = 1e-6."
                )
                a = fill_value

        if equatorial:
            if abs(x0) != 1:
                raise RuntimeError(
                    "Magnitude of orbital inclination cosine x0 needs to be one for equatorial inspiral."
                )

        if x0 == -1:
            Phi_phi0 = -1 * Phi_phi0  # flip initial azimuthal phase for retrograde

        if circular:
            e0 = 0.0

        # if integrating constants of motion, convert from pex to ELQ now
        if self.integrate_constants_of_motion:
            if self.inspiral_generator.convert_Y:
                x0 = Y_to_xI(a, p0, e0, x0)
            y1, y2, y3 = get_kerr_geo_constants_of_motion(a, p0, e0, x0)

        y0 = np.array([y1, y2, y3, Phi_phi0, Phi_theta0, Phi_r0])

        y0_and_args = np.concatenate(([y0], args))
        out = self.inspiral_generator.func(y0_and_args)
        # out = self.inspiral_generator.func(np.r_[y0, *args])

        return out


def get_0PA_frequencies(
    m1: float,
    m2: float,
    a: Union[float, np.ndarray],
    p: Union[float, np.ndarray],
    e: Union[float, np.ndarray],
    x: Union[float, np.ndarray],
    use_gpu: bool = False,
) -> tuple[Union[float, np.ndarray]]:
    r"""Get the orbital frequencies of the 0PA (adiabatic) phase evolution in Hz.

    Evaluates the dimensionless Kerr geodesic frequencies with
    :func:`~few.utils.geodesic.get_fundamental_frequencies` and converts them to Hz,
    :math:`f = \hat\Omega / (2 \pi M T_\odot)` with :math:`M = m_1 + m_2` (the 0PA
    convention :math:`m_1 \Omega_{geo} = M \Omega_{0PA}`) and :math:`T_\odot` =
    ``MTSUN_SI``.

    Args:
        m1: Mass of the massive black hole in solar masses.
        m2: Mass of the secondary body in solar masses.
        a: Dimensionless spin of the massive black hole. If the other parameters are
            arrays and the spin is scalar, it is cast to a 1-D array.
        p: Semi-latus rectum :math:`p` in units of :math:`M`.
        e: Eccentricity :math:`e`.
        x: Cosine of the inclination, :math:`x = \cos{I}` (not :math:`Y = \cos{\iota}`).
        use_gpu: Forwarded to :func:`~few.utils.geodesic.get_fundamental_frequencies`
            (``True`` evaluates with cupy). Default ``False``.

    Returns:
        tuple: ``(f_phi, f_theta, f_r)`` in Hz (the code names them ``OmegaPhi``,
        ``OmegaTheta``, ``OmegaR``), scalars or 1-D arrays following the inputs.

    Example:
        .. code-block:: python

            from few.trajectory.inspiral import get_0PA_frequencies

            f_phi, f_theta, f_r = get_0PA_frequencies(1e6, 1e1, 0.9, 10.0, 0.3, 1.0)
    """
    # we have m1 \Omega_geo = M \Omega_0PA
    OmegaPhi, OmegaTheta, OmegaR = get_fundamental_frequencies(
        a, p, e, x, use_gpu=use_gpu
    )
    M = m1 + m2
    OmegaPhi = OmegaPhi / (M * MTSUN_SI) / (2 * PI)
    OmegaTheta = OmegaTheta / (M * MTSUN_SI) / (2 * PI)
    OmegaR = OmegaR / (M * MTSUN_SI) / (2 * PI)

    return (OmegaPhi, OmegaTheta, OmegaR)
