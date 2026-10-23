"""The :class:`ODEBase` base class of the trajectory right-hand sides, and its property reader.

An ODE model maps the integrated state ``y = (p, e, x, nu Phi_phi, nu Phi_theta,
nu Phi_r)`` (or ``(E, L, Q, ...)``) to its derivative with respect to the
radiation-reaction time (see :mod:`few.trajectory.integrate`). :meth:`ODEBase.__call__`
wraps the model's ``evaluate_rhs`` with the bound checks, the optional ELQ-to-pex Jacobian,
the user hooks ``modify_rhs_before_Jacobian`` / ``modify_rhs`` and the sign flip of a
backwards integration. :func:`get_ode_properties` reads a model's class-level properties.
"""

from typing import Optional, Type, Union

import numpy as np

from ...utils.geodesic import ELQ_to_pex, get_separatrix
from ...utils.mappings.jacobian import ELdot_to_PEdot_Jacobian
from ...utils.mappings.pn import Y_to_xI


class ODEBase:
    r"""Base class of the trajectory ODE models (the right-hand side and its domain).

    To define a new trajectory model, subclass it and implement :meth:`evaluate_rhs`;
    override the properties (:attr:`equatorial`, :attr:`background`,
    :attr:`separatrix_buffer_dist`, ...) and the bound methods (``min_p``, ``isvalid_*``,
    :meth:`distance_to_outer_boundary`) that differ from the defaults. Pass the class as
    ``func`` to :class:`~few.trajectory.inspiral.EMRIInspiral`, which instantiates it.

    The right-hand side is with respect to the radiation-reaction time
    :math:`\tilde t = \nu t / M`, :math:`\nu = m_1 m_2 / (m_1 + m_2)^2`: the first three
    entries are :math:`d(p, e, x)/d\tilde t` (or of :math:`(E, L, Q)`), the last three the
    dimensionless orbital frequencies (the derivative of :math:`\nu \Phi`), as returned by
    :func:`~few.utils.geodesic.get_fundamental_frequencies`.

    Args:
        *args: Accepted and ignored.
        use_ELQ (bool, optional): If ``True``, the model takes and returns the constants of
            motion ``(E, L, Q)``; requires :attr:`supports_ELQ`. Default ``False``.
        integrate_backwards (bool, optional): If ``True``, :meth:`__call__` negates the
            right-hand side. Reset by the integrator on every trajectory. Default ``False``.
        **kwargs: Accepted and ignored (this is where unused trajectory keywords such as
            ``rootfind_separatrix``, ``force_backend`` or ``max_init_len`` end up).

    Raises:
        AssertionError: If ``use_ELQ`` is set and the model does not support ELQ.

    Example:
        .. code-block:: python

            from few.trajectory.ode import KerrEccEqFlux
            from few.trajectory.inspiral import EMRIInspiral

            class ModifiedKerrEccEqFlux(KerrEccEqFlux):
                def modify_rhs(self, ydot, y):
                    ydot[0] *= 1 + self.additional_args[0]   # modify in place

            traj = EMRIInspiral(func=ModifiedKerrEccEqFlux)
            out = traj(1e6, 1e1, 0.9, 10.0, 0.3, 1.0, 1e-2, T=1.0)  # extra ODE argument 1e-2
    """

    _flux_output_convention = "pex"

    def __init__(self, *args, use_ELQ=False, integrate_backwards=False, **kwargs):
        self.flux_output_convention = "pex"
        if use_ELQ:
            assert self.supports_ELQ, "This ODE does not support ELQ evaluation."
        self.use_ELQ = use_ELQ
        """
        bool: If True, the ODE will take as input (and output derivatives of) the integrals of motion (E, L, Q). Defaults to False.
        """
        self.num_add_args = 0
        """int: Number of additional arguments being passed to the ODE function."""

        self.integrate_backwards = integrate_backwards
        """bool: If True, the ODE corresponds to integrating backwards in time. Defaults to False."""

    @property
    def convert_Y(self):
        """bool: If ``True``, the inclination coordinate is ``Y = cos(iota)`` and is converted
        to ``x_I`` where needed. Default ``False``.
        """
        return False

    @property
    def equatorial(self):
        """bool: If ``True``, the inclination coordinate is ``+1`` or ``-1``. Default ``False``."""
        return False

    @property
    def circular(self):
        """bool: If ``True``, the eccentricity is ``0``. Default ``False``."""
        return False

    @property
    def supports_ELQ(self):
        """bool: If ``True``, the ODE can take (and return derivatives of) the integrals of
        motion ``(E, L, Q)`` when built with ``use_ELQ=True``. Default ``False``.
        """
        return False

    @property
    def background(self):
        """str: The background spacetime, ``"Kerr"`` or ``"Schwarzschild"``. Default ``"Kerr"``."""
        return "Kerr"

    @property
    def separatrix_buffer_dist(self):
        """float: Distance in ``p`` above the separatrix at which the trajectory stops.

        A value of 0 would stop the trajectory on the separatrix. Default ``0.05``.
        """
        return 0.05

    @property
    def nparams(self):
        """int: Number of integrated quantities. Default ``6`` (three orbital elements, three
        phases).
        """
        return 6

    @property
    def flux_output_convention(self):
        """str: Coordinate convention of the fluxes returned by :meth:`evaluate_rhs`.

        ``"pex"`` (default) or ``"ELQ"``; with ``"ELQ"`` and a pex integration,
        :meth:`__call__` applies the Jacobian to ``(pdot, edot)``. Setting it validates the
        value (``ValueError`` otherwise) and updates ``apply_Jacobian_bool``. Models whose
        fluxes are not interpolated (e.g. PN5) leave it at ``"pex"``.
        """
        return self._flux_output_convention

    @flux_output_convention.setter
    def flux_output_convention(self, value):
        if value not in ["pex", "ELQ"]:
            raise ValueError(
                f"Invalid flux output convention: {value}. Must be 'pex' or 'ELQ'."
            )
        self._flux_output_convention = value
        self.apply_Jacobian_bool = (
            self.flux_output_convention == "ELQ" and not self.use_ELQ
        )

    def add_fixed_parameters(
        self, m1: float, m2: float, a: float, additional_args=None
    ):
        """Set the per-trajectory constants of the model (called by ``run_inspiral``).

        Sets ``self.massratio`` to :math:`m_1 m_2 / (m_1 + m_2)^2`, ``self.a``,
        ``self.additional_args`` and ``self.num_add_args``.

        Args:
            m1: Mass of the massive black hole in solar masses.
            m2: Mass of the compact object in solar masses.
            a: Dimensionless spin.
            additional_args (array-like, optional): Extra model arguments
                (``num_add_args = len(additional_args)``). Default ``None`` (0 arguments).
        """
        self.massratio = m1 * m2 / (m1 + m2) ** 2
        self.a = a
        self.additional_args = additional_args

        if additional_args is None:
            self.num_add_args = 0
        else:
            self.num_add_args = len(additional_args)

    def evaluate_rhs(self, y: np.ndarray, **kwargs) -> NotImplementedError:
        """Evaluate the model's right-hand side at ``y`` (implemented by each model).

        Called by :meth:`__call__` only for in-bounds states, after
        :meth:`cache_values_and_check_bounds` has set ``p_sep_cache`` (and ``xI_cache``).
        The fixed parameters are on the instance (``self.a``, ``self.massratio``,
        ``self.additional_args``).

        Args:
            y: State, ``(p, e, x)`` or ``(E, L, Q)`` first, then the phases.
            **kwargs: Forwarded from :meth:`__call__`.

        Returns:
            list or np.ndarray: Six values: the derivatives of the three orbital
            quantities (in the model's :attr:`flux_output_convention`) and the three
            dimensionless frequencies ``(Omega_phi, Omega_theta, Omega_r)``.

        Raises:
            NotImplementedError: In the base class.
        """
        raise NotImplementedError

    def modify_rhs_before_Jacobian(
        self, ydot: np.ndarray, y: np.ndarray, **kwargs
    ) -> np.ndarray:
        """Hook to modify the right-hand side before the ELQ-to-pex Jacobian is applied.

        Useful to modify fluxes of the integrals of motion ("ELQ") while integrating a
        trajectory in "pex". :meth:`__call__` ignores the return value: modify ``ydot`` in
        place. By default it does nothing (returns ``ydot``).

        Args:
            ydot: The right-hand side, shape ``(6,)``, modified in place.
            y: The state.
            **kwargs: Forwarded from :meth:`__call__`.

        Returns:
            np.ndarray: ``ydot``.
        """
        return ydot

    def min_p(
        self,
        e: Union[float, np.ndarray],
        x: Union[float, np.ndarray] = 1,
        a: Optional[Union[float, np.ndarray]] = 0,
    ) -> Union[float, np.ndarray]:
        """Return the minimum semi-latus rectum ``p`` of the model at ``(e, x, a)``.

        Models with their own interpolants override it to return the grid boundary. By
        default the domain is assumed rectilinear: ``p_sep + separatrix_buffer_dist``.

        Args:
            e: Eccentricity.
            x: Inclination coordinate. Default ``1``.
            a: Dimensionless spin. Default ``0``.

        Returns:
            float or np.ndarray: Minimum ``p`` in units of :math:`M`.
        """
        return get_separatrix(a, e, x) + self.separatrix_buffer_dist

    def max_p(
        self,
        e: Union[float, np.ndarray],
        x: Union[float, np.ndarray] = 1.0,
        a: Optional[Union[float, np.ndarray]] = 0.0,
    ) -> Union[float, np.ndarray]:
        """Return the maximum semi-latus rectum ``p`` of the model at ``(e, x, a)``.

        Models with their own interpolants override it to return the grid boundary. By
        default there is no bound: ``np.inf`` for a Python ``float`` ``e``, else
        ``np.full_like(e, np.inf)`` (an array of ``inf`` for a float array; pass floats,
        since an integer ``e`` gives an integer result that cannot hold ``inf``).

        Args:
            e: Eccentricity.
            x: Inclination coordinate. Default ``1.0``.
            a: Dimensionless spin. Default ``0.0``.

        Returns:
            float or np.ndarray: Maximum ``p`` in units of :math:`M`.
        """
        if isinstance(e, float):
            return np.inf
        else:
            return np.full_like(e, np.inf)

    def min_e(
        self,
        p: Union[float, np.ndarray],
        x: Union[float, np.ndarray] = 1.0,
        a: Optional[Union[float, np.ndarray]] = 0.0,
    ) -> Union[float, np.ndarray]:
        """Return the minimum eccentricity of the model at ``(p, x, a)``.

        Models with their own interpolants override it to return the grid boundary. By
        default circular orbits are allowed: ``0`` (an array of zeros for a non-float ``p``).

        Args:
            p: Semi-latus rectum in units of :math:`M`.
            x: Inclination coordinate. Default ``1.0``.
            a: Dimensionless spin. Default ``0.0``.

        Returns:
            float or np.ndarray: Minimum eccentricity.
        """
        if isinstance(p, float):
            return 0
        else:
            return np.zeros_like(p)

    def max_e(
        self,
        p: Union[float, np.ndarray],
        x: Union[float, np.ndarray] = 1,
        a: Optional[Union[float, np.ndarray]] = 0,
    ) -> Union[float, np.ndarray]:
        """Return the maximum eccentricity of the model at ``(p, x, a)``.

        Models with their own interpolants override it to return the grid boundary. By
        default there is no bound: ``np.inf`` for a Python ``float`` ``p``, else
        ``np.full_like(p, np.inf)`` (an array of ``inf`` for a float array; pass floats,
        since an integer ``p`` gives an integer result that cannot hold ``inf``).

        Args:
            p: Semi-latus rectum in units of :math:`M`.
            x: Inclination coordinate. Default ``1``.
            a: Dimensionless spin. Default ``0``.

        Returns:
            float or np.ndarray: Maximum eccentricity.
        """
        if isinstance(p, float):
            return np.inf
        else:
            return np.full_like(p, np.inf)

    def isvalid_x(self, x: float):
        """Check the inclination coordinate against the model's domain (no check here).

        Models raise ``ValueError`` for an invalid ``x``.
        """
        pass

    def isvalid_e(self, e: float, e_buffer=[0, 0]):
        """Check the eccentricity against the model's domain (no check here).

        Models raise ``ValueError`` for an ``e`` outside their range shrunk by ``e_buffer``.
        """
        pass

    def isvalid_a(self, a: float, a_buffer=[0, 0]):
        """Check the spin against the model's domain (no check here).

        Models raise ``ValueError`` for an ``a`` outside their range shrunk by ``a_buffer``.
        """
        pass

    def bounds_p(self, e=0, x=1, a=0, p_buffer=[0, 0]):
        """Return ``[min_p + p_buffer[0], max_p - p_buffer[1]]`` after checking ``x, e, a``.

        Args:
            e (float, optional): Eccentricity. Default ``0``.
            x (float, optional): Inclination coordinate. Default ``1``.
            a (float, optional): Dimensionless spin. Default ``0``.
            p_buffer (list[float], optional): Margins at the lower and upper bound.
                Default ``[0, 0]``.

        Returns:
            list[float]: The allowed ``p`` range in units of :math:`M`.
        """
        self.isvalid_x(x)
        self.isvalid_e(e)
        self.isvalid_a(a)
        return [self.min_p(e, x, a) + p_buffer[0], self.max_p(e, x, a) - p_buffer[1]]

    def isvalid_pex(
        self, p=20, e=0, x=1, a=0, p_buffer=[0, 0], e_buffer=[0, 0], a_buffer=[0, 0]
    ):
        """Check that ``(p, e, x, a)`` lies in the model's domain (start-point check).

        Called by :meth:`~few.trajectory.inspiral.EMRIInspiral.get_inspiral` before every
        integration.

        Args:
            p (float, optional): Semi-latus rectum in units of :math:`M`. Default ``20``.
            e (float, optional): Eccentricity. Default ``0``.
            x (float, optional): Inclination coordinate. Default ``1``.
            a (float, optional): Dimensionless spin. Default ``0``.
            p_buffer (list[float], optional): Margins on the ``p`` range. Default
                ``[0, 0]``.
            e_buffer (list[float], optional): Margins on the ``e`` range. Default
                ``[0, 0]``.
            a_buffer (list[float], optional): Margins on the ``a`` range. Default
                ``[0, 0]``.

        Raises:
            ValueError: From the model's ``isvalid_x`` / ``isvalid_e`` / ``isvalid_a``.
            AssertionError: If ``p`` is outside :meth:`bounds_p` (an ``assert``, so the
                check disappears under ``python -O``).
        """
        self.isvalid_x(x)
        self.isvalid_e(e, e_buffer=e_buffer)
        self.isvalid_a(a, a_buffer=a_buffer)
        pmin, pmax = self.bounds_p(e, x, a, p_buffer=p_buffer)
        assert p >= pmin and p <= pmax, (
            f"Interpolation: p {p} out of bounds. Must be between {pmin} and {pmax}."
        )

    def modify_rhs(self, ydot: np.ndarray, y: np.ndarray, **kwargs) -> np.ndarray:
        """Hook to modify the right-hand side after any Jacobian transform.

        :meth:`__call__` ignores the return value: modify ``ydot`` in place (e.g.
        ``ydot[0] *= 1 + self.additional_args[0]``). By default it does nothing (returns
        ``ydot``).

        Args:
            ydot: The right-hand side, shape ``(6,)``, modified in place.
            y: The state.
            **kwargs: Forwarded from :meth:`__call__`.

        Returns:
            np.ndarray: ``ydot``.
        """
        return ydot

    def interpolate_flux_grids(self, *args) -> NotImplementedError:
        """Interpolate the fluxes from the model's precomputed grids (implemented by each model).

        Handles the parameter transformations to the grid coordinates. To reuse a stock
        model's interpolated fluxes in a custom model, subclass the stock model and call
        this method from :meth:`evaluate_rhs`.

        Implementations should also check that the inputs lie within the grids; otherwise
        the fluxes are silently extrapolated during the trajectory evaluation.

        Raises:
            NotImplementedError: In the base class.
        """
        raise NotImplementedError

    def distance_to_outer_boundary(self, y: np.ndarray) -> float:
        """Return the distance of ``y`` to the outer boundary of the model's grid.

        Needed for backwards integration, which stops (and root-finds the final point) once
        this becomes negative. Stock models with interpolated fluxes override it; the
        default returns the positive constant ``1e10``, so a backwards integration then only
        stops at ``T``.

        Args:
            y: State.

        Returns:
            float: Positive inside the grid, negative outside.
        """
        return 1e10

    def get_pex(self, y: np.ndarray) -> tuple[float]:
        """Return ``(p, e, x)`` of a state, converting from ``(E, L, Q)`` when ``use_ELQ``.

        Args:
            y: State.

        Returns:
            tuple[float]: ``(p, e, x)``.
        """
        if self.use_ELQ:
            E, L, Q = y[:3]
            p, e, x = ELQ_to_pex(self.a, E, L, Q)
        else:
            p, e, x = y[:3]
        return p, e, x

    def cache_values_and_check_bounds(self, y: np.ndarray) -> bool:
        """Check that ``y`` is physical (``e >= 0`` and ``p`` above the separatrix).

        Caches the separatrix in ``self.p_sep_cache`` (used by the stock ``evaluate_rhs``)
        and, for models with ``convert_Y``, the converted ``x_I`` in ``self.xI_cache``.

        Args:
            y: State.

        Returns:
            bool: ``True`` if in bounds.
        """

        p, e, x_or_Y = self.get_pex(y)

        # first: check the eccentricity
        in_bounds = e >= 0
        if in_bounds:
            # second: check the separatrix
            if self.convert_Y:
                x = Y_to_xI(self.a, p, e, x_or_Y)
                self.xI_cache = x
            else:
                x = x_or_Y

            self.p_sep_cache = get_separatrix(self.a, e, x)
            in_bounds = p > self.p_sep_cache

        return in_bounds

    def __call__(
        self,
        y: Union[list, np.ndarray],
        out: Optional[np.ndarray] = None,
        **kwargs: Optional[dict],
    ) -> np.ndarray:
        """Evaluate the full right-hand side at ``y`` (the integrator's per-stage call).

        Order: :meth:`cache_values_and_check_bounds`; :meth:`evaluate_rhs` if in bounds,
        else NaN everywhere (the integrator then retries with a smaller step);
        :meth:`modify_rhs_before_Jacobian`; the ELQ-to-pex Jacobian
        (:func:`~few.utils.mappings.jacobian.ELdot_to_PEdot_Jacobian`) on the first two
        entries when the model's fluxes are "ELQ" and the integration is in "pex";
        :meth:`modify_rhs`; and a sign flip when ``integrate_backwards``.

        Args:
            y: State (``p, e, x`` or ``E, L, Q`` first; only the first three entries are
                read by the stock models).
            out: Output array of 6 values, written in place. Default ``None`` allocates
                ``np.zeros(6)``.
            **kwargs: Forwarded to :meth:`evaluate_rhs` and the two hooks.

        Returns:
            np.ndarray: ``out``: the derivatives of the three orbital quantities and the
            three dimensionless frequencies, with respect to the radiation-reaction time.

        Raises:
            ValueError: From the model (e.g. ``KerrEccEqFlux`` out of the grid in ``p``).
            few.utils.exceptions.TrajectoryOffGridException: From ``KerrEccEqFlux`` out of
                the grid in ``e`` or ``a``.

        Example:
            .. code-block:: python

                from few.trajectory.ode import KerrEccEqFlux

                rhs = KerrEccEqFlux()
                rhs.add_fixed_parameters(1e6, 1e1, 0.9)
                pdot, edot, xdot, Om_phi, Om_theta, Om_r = rhs([10.0, 0.3, 1.0])
        """
        in_bounds = self.cache_values_and_check_bounds(y)

        if out is None:
            out = np.zeros(6)

        if in_bounds:
            out[:] = self.evaluate_rhs(y, **kwargs)
        else:
            out *= np.nan

        self.modify_rhs_before_Jacobian(out, y, **kwargs)

        if self.apply_Jacobian_bool:  # implicitly this means that y contains (p, e, x)
            out[:2] = ELdot_to_PEdot_Jacobian(self.a, *y[:3], *out[:2])

        self.modify_rhs(out, y, **kwargs)

        if self.integrate_backwards:
            out *= -1.0

        return out

    def __reduce__(self):
        #  to ensure pickleability of the trajectory & waveform modules
        #  TODO: re-examine this in future, this is a band-aid fix that breaks
        #  if the user adds their own args/kwargs to their class
        #  Or optionally, we can ask the user to define this as well (not ideal)
        return self.__class__, (self.use_ELQ,)


def _properties(cls: type) -> list[str]:
    """Return the names of the properties defined directly on ``cls`` (not inherited)."""
    return [key for key, value in cls.__dict__.items() if isinstance(value, property)]


def get_ode_properties(inst_cls: Type[ODEBase]):
    """Return the property values of an ODE model as a dict.

    Reads the properties defined on the model's first direct base class, updated with those
    defined on the model's own class (properties from further up the hierarchy are not
    read). Each getter is evaluated on the class, not the instance, so only constant
    properties give meaningful values. Called by
    :class:`~few.trajectory.integrate.Integrate` (``ode_info``) and used in the trajectory
    tutorial.

    Args:
        inst_cls: Despite the annotation, an ODE model INSTANCE (its ``__class__`` is
            read).

    Returns:
        dict: ``{property name: value}``, e.g. ``equatorial``, ``background``,
        ``separatrix_buffer_dist``, ``nparams``.

    Example:
        .. code-block:: python

            from few.trajectory.ode import KerrEccEqFlux
            from few.trajectory.ode.base import get_ode_properties

            get_ode_properties(KerrEccEqFlux())
    """
    cls = inst_cls.__class__

    # first get all the properties of ODEBase
    parent = cls.__bases__[0]
    parentprops = _properties(parent)
    props = {pkey: getattr(parent, pkey).fget(parent) for pkey in parentprops}

    # now update with what is changed by this subclass
    childprops = _properties(cls)
    props.update({ckey: getattr(cls, ckey).fget(cls) for ckey in childprops})
    return props
