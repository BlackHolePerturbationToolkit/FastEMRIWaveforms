r"""DOP853 integration of the trajectory ODE: stepping, stopping, dense output.

:class:`Integrate` drives a :class:`~few.trajectory.dopr853.DOPR853` stepper over an
:class:`~few.trajectory.ode.base.ODEBase` model. One call of :meth:`Integrate.run_inspiral`
integrates one trajectory: adaptive steps with absolute error control (or fixed steps),
retry-and-halve on failed steps, a stop at ``T`` or at the separatrix buffer, and a final
point placed on the stopping boundary by root-finding on the dense output. The results stay
on the instance (:attr:`Integrate.trajectory`, :attr:`Integrate.integrator_t_cache`,
:attr:`Integrate.integrator_spline_coeff`) and are evaluated with
:meth:`Integrate.eval_integrator_spline` / :meth:`Integrate.eval_integrator_derivative_spline`.

Units: the integration variable is the radiation-reaction time
:math:`\tilde t = \nu\, t / (M T_\odot)` with :math:`\nu = m_1 m_2 / (m_1 + m_2)^2`
(``massratio``), :math:`M = m_1 + m_2` in solar masses and :math:`T_\odot` = ``MTSUN_SI``;
the ODE state is ``(p, e, x, nu * Phi_phi, nu * Phi_theta, nu * Phi_r)`` (or
``(E, L, Q, ...)`` for :class:`AELQIntegrate`). Stored times are in seconds
(``Msec = MTSUN_SI * M / nu`` seconds per unit of :math:`\tilde t`) and stored phases are
divided by :math:`\nu` (radians) after the integration.

Entry points: :func:`get_integrator` (used by :class:`~few.trajectory.inspiral.EMRIInspiral`)
returns an :class:`APEXIntegrate` (pex variables, the production path) or an
:class:`AELQIntegrate`. :class:`BatchedIntegrate` is an experimental CPU reference that
steps many trajectories at once (see ``docs/traj-batching.md``); it is not used by the
waveform generators.
"""

# DOP853 integrator classes of the FEW trajectory module

from __future__ import annotations

from typing import Optional, Tuple, Type

import numpy as np
from scipy.optimize import brentq

from ..utils.constants import MTSUN_SI, YRSID_SI

# Python imports
from ..utils.geodesic import (
    ELQ_to_pex,
    get_separatrix,
)
from ..utils.globals import get_logger
from .dopr853 import DOPR853
from .ode import _STOCK_TRAJECTORY_OPTIONS
from .ode.base import ODEBase, get_ode_properties

# xtol [s] of the brentq root-find that places the final point on the separatrix buffer
INNER_THRESHOLD = 1e-10
# unused
PERCENT_STEP = 0.25
# maxiter of that root-find
MAX_ITER = 1000


def get_integrator(func, integrate_constants_of_motion=False, **kwargs):
    """Build the integrator for an ODE model: pex or ELQ variables.

    Called by :class:`~few.trajectory.inspiral.EMRIInspiral` at construction.

    Args:
        func (str or type): The ODE model class (an
            :class:`~few.trajectory.ode.base.ODEBase` subclass) or a stock model name.
        integrate_constants_of_motion (bool, optional): If ``True``, integrate
            ``(E, L, Q)``. Default ``False``.
        **kwargs: Forwarded to the :class:`Integrate` constructor.

    Returns:
        Integrate: An :class:`AELQIntegrate` when ``integrate_constants_of_motion`` is set,
        else an :class:`APEXIntegrate`.
    """
    if integrate_constants_of_motion:
        return AELQIntegrate(
            func=func,
            integrate_constants_of_motion=integrate_constants_of_motion,
            **kwargs,
        )
    else:
        return APEXIntegrate(
            func=func,
            integrate_constants_of_motion=integrate_constants_of_motion,
            **kwargs,
        )


def digest_func(func):
    """Resolve a stock model name to its ODE class; return a class unchanged.

    Args:
        func (str or type): A name in ``few.trajectory.ode._STOCK_TRAJECTORY_OPTIONS``
            (``"KerrEccEqFlux"``, ``"SchwarzEccFlux"``, ``"PN5"``) or an ODE class.

    Returns:
        type: The ODE class.

    Raises:
        ValueError: If ``func`` is a string that is not a stock model name.
    """
    if isinstance(func, str):
        try:
            func = _STOCK_TRAJECTORY_OPTIONS[func]
        except KeyError:
            raise ValueError(f"The trajectory function {func} could not be found.")
    return func


class Integrate:
    r"""DOP853 trajectory integrator over an ODE model (base of the pex / ELQ variants).

    Builds the ODE model instance ``func(use_ELQ=integrate_constants_of_motion,
    downsample=downsample, **kwargs)`` and a :class:`~few.trajectory.dopr853.DOPR853`
    stepper on :meth:`_dopr_ode_wrap`. Each :meth:`run_inspiral` call re-initialises the
    buffers and integrates one trajectory in the radiation-reaction time (see the module
    docstring for the units). Subclasses define :meth:`get_pex`, the map from the
    integrated state to ``(p, e, x)`` used by the stopping checks.

    Args:
        func: The ODE model class, or a stock model name (resolved by
            :func:`digest_func`).
        integrate_constants_of_motion: If ``True``, the model integrates ``(E, L, Q)``
            instead of ``(p, e, x)`` (``use_ELQ``). Default ``False``.
        downsample (optional): Forwarded to the model constructor (flux-grid downsampling
            of :class:`~few.trajectory.ode.flux.KerrEccEqFlux`). Default ``None``.
        buffer_length: Rows preallocated for the trajectory and dense-output arrays;
            they grow by 100 rows whenever they fill. Default ``10000``.
        enforce_schwarz_sep: If ``True``, stop at the Schwarzschild separatrix
            :math:`p_s = 6 + 2e` instead of the Kerr one. Default ``False``.
        max_iter: Maximum number of accepted adaptive steps per trajectory. Default
            ``None`` means ``buffer_length``.
        **kwargs: Forwarded to the model constructor.

    Raises:
        ValueError: If ``func`` is an unknown stock model name.

    Attributes:
        func (ODEBase): The ODE model instance.
        base_func (type): The ODE model class.
        ode_info (dict): The model's property values
            (:func:`~few.trajectory.ode.base.get_ode_properties`).
        dopr (DOPR853): The stepper (``abstol`` and ``fix_step`` are reset by every
            :meth:`initialize_integrator`).
        buffer_length (int): Current buffer length (grows during an integration).
        max_iter (int): Step limit of an adaptive integration.
        enforce_schwarz_sep (bool): Whether the Schwarzschild separatrix is used.

    Note:
        ``dopr`` keeps its step-size controller memory (``errOldTemp``,
        ``previousRejectTemp``) across trajectories, so consecutive trajectories from one
        instance differ at the integrator-tolerance level from those of a fresh instance;
        :func:`few.trajectory.pool.reset_stepper` removes it.
    """

    def __init__(
        self,
        func: Type[ODEBase],
        integrate_constants_of_motion: bool = False,
        downsample=None,
        buffer_length: int = 10000,
        enforce_schwarz_sep: bool = False,
        max_iter: Optional[int] = None,
        **kwargs,
    ):
        get_logger().debug(
            "Initializing integrator with func=%s, downsample=%s, buffer_length=%s",
            func,
            downsample,
            buffer_length,
        )
        self.buffer_length = buffer_length

        func = digest_func(func)

        self.base_func = func

        # load func
        self.func = func(
            use_ELQ=integrate_constants_of_motion, downsample=downsample, **kwargs
        )

        self.ode_info = get_ode_properties(self.func)

        self.enforce_schwarz_sep = enforce_schwarz_sep

        self.dopr = DOPR853(
            self._dopr_ode_wrap,
            stopping_criterion=None,  # stopping_criterion,
            tmax=1e9,
            max_step=1e6,
        )

        if max_iter is None:
            self.max_iter = buffer_length
        else:
            self.max_iter = max_iter

        # assert np.all(self.backgrounds == self.backgrounds[0])
        # assert np.all(self.equatorial == self.equatorial[0])
        # assert np.all(self.circular == self.circular[0])

    @property
    def nparams(self) -> int:
        """Dimensionality of the ODE state (``func.nparams``; 6 for the stock models)."""
        return self.func.nparams

    @property
    def num_add_args(self) -> int:
        """Number of additional arguments of the ODE (``func.num_add_args``)."""
        return self.func.num_add_args

    @property
    def convert_Y(self):
        """bool: Whether the model's inclination coordinate is ``Y`` (``func.convert_Y``)."""
        return self.func.convert_Y

    @property
    def background(self):
        """str: Spacetime background, ``"Kerr"`` or ``"Schwarzschild"`` (``func.background``)."""
        return self.func.background

    @property
    def equatorial(self):
        """bool: Whether the model is restricted to equatorial orbits (``func.equatorial``)."""
        return self.func.equatorial

    @property
    def circular(self):
        """bool: Whether the model is restricted to circular orbits (``func.circular``)."""
        return self.func.circular

    @property
    def integrate_constants_of_motion(self):
        """bool: Whether ``(E, L, Q)`` are integrated (``func.use_ELQ``)."""
        return self.func.use_ELQ

    @property
    def separatrix_buffer_dist(self):
        """float: Stop distance in ``p`` above the separatrix (``func.separatrix_buffer_dist``)."""
        return self.func.separatrix_buffer_dist

    def distance_to_outer_boundary(self, y):
        """Return the model's distance to the outer edge of its grid (backwards stop check).

        Args:
            y (np.ndarray): Integrated state.

        Returns:
            float: ``func.distance_to_outer_boundary(y)``; negative once outside.
        """
        return self.func.distance_to_outer_boundary(y)

    def _dopr_ode_wrap(self, t, y, ydot, additionalArgs):
        """Adapt the single-system model call to the stepper's ``ode(x, y, ydot, args)``.

        Evaluates ``func`` on column 0 of ``y`` (shape ``(nparams, 1)``) and writes into
        column 0 of ``ydot`` in place; ``t`` and ``additionalArgs`` are unused (the model
        is autonomous and holds its own fixed parameters).

        Returns:
            np.ndarray: ``ydot``.
        """
        self.func(y[:, 0], out=ydot[:, 0])
        return ydot

    def take_step(
        self, t: float, h: float, y: np.ndarray
    ) -> Tuple[float, float, np.ndarray]:
        """Attempt one DOP853 step (thin wrapper of :meth:`DOPR853.take_step_single`).

        Not used by :meth:`integrate`, which calls the stepper directly. Needs
        ``tmax_dimensionless`` from a previous :meth:`run_inspiral`.

        Args:
            t: Current time (radiation-reaction units).
            h: Proposed step size.
            y: Current state, updated in place when the step is accepted.

        Returns:
            tuple[bool, float, float]: ``(accepted, t_new, h_new)`` (the return annotation,
            ``Tuple[float, float, np.ndarray]``, is wrong): whether the step was accepted,
            the new time (``t + h`` if accepted, else ``t``) and the next proposed step size.
        """
        return self.dopr.take_step_single(t, h, y, self.tmax_dimensionless, None)

    def log_failed_step(self, lastError: Exception):
        """Count a failed step and give up after 100 consecutive failures.

        Increments ``bad_num`` (reset to 0 by every accepted step) and ``bad_num_total``
        and stores ``last_error``.

        Args:
            lastError: The exception (instance or class) of the failed attempt.

        Raises:
            Exception: ``lastError`` itself, once ``bad_num`` reaches 100 (usually a
                ``ValueError``).
        """
        # check for number of tries to fix this
        self.bad_num += 1
        self.bad_num_total += 1
        self.last_error = lastError
        if self.bad_num >= 100:
            raise self.last_error

    def tune_initial_step_size(self, t0: float, y0: np.ndarray, h_max: float) -> float:
        """Estimate the first adaptive step size (adapted from scipy's ``select_initial_step``).

        Uses the absolute tolerance ``dopr.abstol`` as the scale and one explicit Euler
        probe ``y0 + h0 * f(y0)``; if the probe is out of the model's bounds (the model
        raises ``ValueError``) the first estimate ``h0`` is returned.

        Args:
            t0: Initial time (radiation-reaction units).
            y0: Initial state, shape ``(nparams,)``.
            h_max: Largest step permitted.

        Returns:
            float: ``min(100 * h0, h1, tmax_dimensionless - t0, h_max)``, or ``h0`` (see
            above).
        """

        # Initialize step size based on the problem's scale
        scale = self.dopr.abstol

        f0 = self.func(y0)
        interval_length = self.tmax_dimensionless - t0

        d0 = np.linalg.norm(y0 / scale) / self.nparams**0.5
        d1 = np.linalg.norm(f0 / scale) / self.nparams**0.5

        if d0 < 1e-5 or d1 < 1e-5:
            h0 = 1e-6
        else:
            h0 = 0.01 * d0 / d1

        h0 = min(h0, interval_length)

        y1 = y0 + h0 * f0

        # It is possible for the trajectory to have already stepped past any boundaries, leading to error
        # In this case it is hard to determine a good step size, so we just return the first estimate for h0
        try:
            f1 = self.func(y1)
        except ValueError:
            return h0

        d2 = np.linalg.norm((f1 - f0) / scale) / h0 / self.nparams**0.5

        if d1 <= 1e-15 and d2 <= 1e-15:
            h1 = max(1e-6, h0 * 1e-3)
        else:
            h1 = (0.01 / max(d1, d2)) ** (1 / 8)

        return min(100 * h0, h1, interval_length, h_max)

    def integrate(self, t0: float, y0: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Run the stepping loop from ``(t0, y0)`` and fill the trajectory buffers.

        Adaptive mode: the first step comes from :meth:`tune_initial_step_size` (capped at
        a tenth of the interval and at ``max_step_size``); a step that raises
        ``ValueError``, returns a NaN step size or state, or exceeds ``max_step_size`` is
        retried from the last accepted point with half the step (counted by
        :meth:`log_failed_step`); a step rejected by the error control is retried with the
        controller's smaller step (and also counted by :meth:`log_failed_step`, so 100
        consecutive rejections abort). Each accepted step stores its dense-output coefficients
        (:meth:`DOPR853.prep_evaluate_single`) and the point (time in seconds). The loop ends
        when :meth:`stop_integrate_check` fires or the time passes ``tmax_dimensionless``;
        the coefficients of that last step are kept, the point is not, and
        :meth:`finishing_function` places the final point. Fixed-step mode
        (``DENSE_STEPPING``) takes steps of ``dt_dimensionless`` without dense output.

        Args:
            t0: Initial time (radiation-reaction units; 0 from :meth:`run_inspiral`).
            y0: Initial state, shape ``(nparams,)`` (phases scaled by the mass ratio).

        Returns:
            None: The return annotation (``Tuple[np.ndarray, np.ndarray]``) is wrong; the
            results are in :attr:`trajectory`, :attr:`integrator_t_cache` and
            :attr:`integrator_spline_coeff`.

        Raises:
            ValueError: If an adaptive integration exceeds ``max_iter`` accepted steps, or
                (re-raised by :meth:`log_failed_step`) after 100 consecutive failures.
                Exceptions other than ``ValueError`` raised inside a step (e.g.
                :class:`~few.utils.exceptions.TrajectoryOffGridException`) are not retried
                and propagate immediately.
        """

        # store initial values
        t = t0
        h = self.dt_dimensionless

        if not self.dopr.fix_step:
            # we do not allow an initial step larger than ~10% of the interval to ensure there are a few steps taken
            # this is necessary for later in waveform generation (computing cubic splines).
            h = self.tune_initial_step_size(
                t0, y0, min(self.max_step_size, self.tmax_dimensionless / 10)
            )

        t_prev = t0
        y_prev = y0.copy()
        y = y0.copy()

        # setup integrator
        self.bad_num = 0
        self.bad_num_total = 0

        # add the first point
        self.save_point(t0, y)
        self._integrator_t_cache[0] = t0

        # run
        niter = 0
        while t < self.tmax_dimensionless:
            if not self.dopr.fix_step:
                if niter >= self.max_iter:
                    raise ValueError("Integration did not converge within max_iter.")

            try:
                # take a step
                t_old = t
                h_old = h
                y_old = y.copy()
                status, t, h = self.dopr.take_step_single(
                    t_old, h_old, y, self.tmax_dimensionless, None
                )

            except ValueError as e:
                t = t_prev
                y[:] = y_prev[:]
                h = h_old / 2
                self.log_failed_step(e)
                continue

            if np.isnan(h):
                t = t_prev
                y[:] = y_prev[:]
                h = h_old / 2

                self.log_failed_step(ValueError)
                continue

            if h > self.max_step_size:
                # if the step size is larger than the maximum, reduce the initial guess
                t = t_prev
                y[:] = y_prev[:]
                h = h_old / 2
                self.log_failed_step(ValueError)
                continue

            # or if any quantity is nan, step back and take a smaller step.
            if np.any(
                np.isnan(y)
            ):  # an Elliptic function returned a nan and it raised an exception
                # self.integrator.reset_solver()
                t = t_prev
                y[:] = y_prev[:]
                h /= 2

                self.log_failed_step(ValueError)
                continue

            if not status:  # the step failed, but we retain the step size as this is the estimate for the next attempt
                self.log_failed_step(ValueError)
                continue

            if not self.dopr.fix_step:
                # prepare output spline information
                spline_info = self.dopr.prep_evaluate_single(
                    t_old, y_old, h_old, t, y, None
                )
            else:
                spline_info = None

            # or if any quantity is nan, step back and take a smaller step.
            if np.any(
                np.isnan(y)
            ):  # an Elliptic function returned a nan and it raised an exception
                # self.integrator.reset_solver()
                t = t_prev
                y[:] = y_prev[:]
                h /= 2

                self.log_failed_step(ValueError)
                continue

            # if it made it here, reset bad num
            self.bad_num = 0

            # action check allows user to add adjustments easily
            self.action_function(t, y)

            # stop check determines when to stop the integration
            stop_check = self.stop_integrate_check(t, y)

            if stop_check:
                if not self.dopr.fix_step:
                    self.dopr_spline_output[self.traj_step - 1, :] = spline_info
                    self._integrator_t_cache[self.traj_step] = t * self.Msec

                # go back to last values
                y[:] = y_prev[:]
                t = t_prev
                break

            # for adaptive stepping, the last point in the trajectory will be at t = tmax
            if t > self.tmax_dimensionless:
                if not self.dopr.fix_step:
                    self.dopr_spline_output[self.traj_step - 1, :] = spline_info
                    self._integrator_t_cache[self.traj_step] = t * self.Msec
                break

            # We did not stop or cross tmax, so save the point
            self.save_point(
                t * self.Msec, y, spline_output=spline_info
            )  # adds time in seconds

            t_prev = t
            y_prev[:] = y[:]

            niter += 1

        if hasattr(self, "finishing_function"):
            self.finishing_function(t, y)

    def action_function(self, t: float, y: np.ndarray) -> None:
        """Hook called after every accepted step (a no-op here; override to act on it).

        Args:
            t: Time of the accepted step (radiation-reaction units).
            y: State after the step; may be modified in place.
        """
        return None

    def initialize_integrator(
        self,
        err: float = 1e-11,
        DENSE_STEPPING: bool = False,
        integrate_backwards: bool = False,
        max_step_size: Optional[float] = None,
        **kwargs,
    ):
        """Reset the stepper settings and the output buffers for a new trajectory.

        Sets ``dopr.abstol`` and ``dopr.fix_step``, the backwards flag (also on the model,
        which then negates its right-hand side), ``max_step_size``, and allocates
        ``trajectory_arr`` ``(buffer_length, nparams + 1)``, ``_integrator_t_cache``
        ``(buffer_length,)`` and ``dopr_spline_output`` ``(buffer_length, 6, 8)``.

        Args:
            err: Absolute tolerance of the DOP853 error control. Default ``1e-11``.
            DENSE_STEPPING: If ``True``, use fixed steps. Default ``False``.
            integrate_backwards: If ``True``, integrate backwards. Default ``False``.
            max_step_size: Largest step in radiation-reaction units. Default ``None``
                means no limit.
            **kwargs: Ignored (e.g. the call's ``buffer_length``: the constructor's value
                is used).
        """
        self.dopr.abstol = err
        self.dopr.fix_step = DENSE_STEPPING
        self.integrate_backwards = integrate_backwards
        self.func.integrate_backwards = integrate_backwards

        if max_step_size is None:
            self.max_step_size = np.inf
        else:
            self.max_step_size = max_step_size

        self.trajectory_arr = np.zeros((self.buffer_length, self.nparams + 1))
        self._integrator_t_cache = np.zeros((self.buffer_length,))
        self.dopr_spline_output = np.zeros(
            (self.buffer_length, 6, 8)
        )  # 3 parameters + 3 phases, 8 coefficients
        self.traj_step = 0

    @property
    def tolerance(self) -> float:
        """Absolute tolerance of the integrator (``dopr.abstol``, the call's ``err``)."""
        return self.dopr.abstol

    @property
    def dense_stepping(self) -> bool:
        """If ``True``, the last trajectory used fixed steps (``dopr.fix_step``)."""
        return self.dopr.fix_step

    @property
    def npoints(self) -> int:
        """Number of points of the last trajectory."""
        return self.traj_step

    @property
    def trajectory(self) -> np.ndarray:
        """The last trajectory, shape ``(npoints, nparams + 1)`` (a view of the buffer).

        Column 0 is the time in seconds, then the integrated state; after
        :meth:`run_inspiral` the phase columns 4-6 are in radians.
        """
        return self.trajectory_arr[: self.traj_step]

    @property
    def integrator_t_cache(self) -> np.ndarray:
        """Knot times of the dense output in seconds, shape ``(npoints,)``."""
        return self._integrator_t_cache[: self.traj_step]

    @property
    def integrator_spline_coeff(self) -> np.ndarray:
        """DOP853 dense-output coefficients, shape ``(npoints - 1, nparams, 8)``.

        Segment ``i`` spans ``integrator_t_cache[i:i + 2]``; the phase rows (3-5) carry the
        mass-ratio scaling of the integration (divide by ``massratio`` for radians).
        """
        return self.dopr_spline_output[: self.traj_step - 1]

    def save_point(
        self, t: float, y: np.ndarray, spline_output: Optional[np.ndarray] = None
    ) -> None:
        """Append a point (and the coefficients of the segment ending there) to the buffers.

        Grows the buffers by 100 rows when they are full.

        Args:
            t: Time in seconds (0 for the first point).
            y: State at ``t``, shape ``(nparams,)``.
            spline_output: Dense-output coefficients of the step that ended at ``t``, shape
                ``(nparams, 8)``, stored as segment ``traj_step - 1`` (with ``t`` as its
                end knot). Default ``None`` stores no coefficients.
        """
        self.trajectory_arr[self.traj_step, 0] = t
        self.trajectory_arr[self.traj_step, 1:] = y

        if spline_output is not None:
            assert self.traj_step >= 0
            self._integrator_t_cache[self.traj_step] = t
            self.dopr_spline_output[self.traj_step - 1, :] = spline_output

        self.traj_step += 1
        if self.traj_step >= self.buffer_length:
            # increase by 100
            buffer_increment = 100
            get_logger().debug(
                "trajectory buffer full [size: %d], increasing by %d",
                self.buffer_length,
                buffer_increment,
            )

            self.trajectory_arr = np.concatenate(
                [self.trajectory_arr, np.zeros((buffer_increment, self.nparams + 1))],
                axis=0,
            )
            if not self.dopr.fix_step:
                self.dopr_spline_output = np.concatenate(
                    [
                        self.dopr_spline_output,
                        np.zeros((buffer_increment, self.nparams, 8)),
                    ],
                    axis=0,
                )
                self._integrator_t_cache = np.concatenate(
                    [
                        self._integrator_t_cache,
                        np.zeros(
                            buffer_increment,
                        ),
                    ]
                )
            self.buffer_length += buffer_increment

    def eval_integrator_spline(self, t_new: np.ndarray) -> np.ndarray:
        """Evaluate the dense-output trajectory at new times.

        Called by the waveform generators (upsampling, LAT's direct EMRI path) and by the
        final-point root-finding. Outside the integration a phase column is divided by the
        mass ratio (radians) and, after a backwards integration, shifted by
        ``trajectory[0, 4:7] + trajectory[-1, 4:7]`` to match the forward phase convention.

        Args:
            t_new: Times in seconds, shape ``(N,)``.

        Returns:
            np.ndarray: Shape ``(N, 6)``, columns ``p, e, x, Phi_phi, Phi_theta, Phi_r``
            (phases in rad; in mass-ratio-scaled units while ``generating_trajectory``).
            Rows with ``t_new < 0`` or ``t_new > integrator_t_cache.max()`` are zero
            (before the phase shift).
        """
        t_old = self.integrator_t_cache

        result = np.zeros((t_new.size, 6))
        t_in_mask = (t_new >= 0.0) & (t_new <= t_old.max())

        result[t_in_mask, :] = self.dopr.eval(
            t_new[t_in_mask], t_old, self.integrator_spline_coeff
        )

        if not self.generating_trajectory:
            result[:, 3:6] /= self.massratio

        # backwards integration requires an additional adjustment to match forwards phase conventions
        if self.integrate_backwards and not self.generating_trajectory:
            result[:, 3:6] += self.trajectory[0, 4:7] + self.trajectory[-1, 4:7]
        return result

    def eval_integrator_derivative_spline(self, t_new: np.ndarray, order: int = 1):
        r"""Evaluate time derivatives of the dense-output trajectory at new times.

        The phase columns are divided by the mass ratio outside the integration, so
        ``order=1`` gives the angular frequencies in rad/s (the waveform generators divide
        ``[:, 3:6]`` by :math:`2\pi` for the orbital frequencies in Hz). No backwards-phase
        shift is applied (it is a constant, so it only matters for ``order=0``).

        Args:
            t_new: Times in seconds, shape ``(N,)``, within ``integrator_t_cache``.
            order: Derivative order, 0 to 3. Default ``1``.

        Returns:
            np.ndarray: Shape ``(N, 6)``: the ``order``-th derivatives with respect to time
            in seconds of ``p, e, x, Phi_phi, Phi_theta, Phi_r``.

        Raises:
            ValueError: If any ``t_new`` is outside the knot range.
        """
        t_old = self.integrator_t_cache
        result = self.dopr.eval_derivative(
            t_new, t_old, self.integrator_spline_coeff, order=order
        )

        if not self.generating_trajectory:
            result[:, 3:6] /= self.massratio

        return result

    def run_inspiral(
        self,
        m1: float,
        m2: float,
        a: float,
        y0: np.ndarray,
        additional_args: list | np.ndarray,
        T: float = 1.0,
        dt: float = 10.0,
        **kwargs,
    ) -> np.ndarray:
        r"""Integrate one trajectory (called by :class:`~few.trajectory.inspiral.EMRIInspiral`).

        Sets the unit conversions (``massratio`` :math:`\nu = m_1 m_2 / (m_1 + m_2)^2`,
        ``tmax_dimensionless``, ``dt_dimensionless``, ``Msec``), resets the buffers
        (:meth:`initialize_integrator`), loads ``(m1, m2, a, additional_args)`` into the
        model, runs :meth:`integrate` from ``t = 0``, then divides the stored phases by
        :math:`\nu` (and, backwards, shifts them to the forward phase convention).

        Args:
            m1: Mass of the massive black hole in solar masses.
            m2: Mass of the compact object in solar masses.
            a: Dimensionless spin of the massive black hole.
            y0: Initial state, shape ``(nparams,)``: ``(p0, e0, x0)`` (or ``(E, L, Q)``)
                followed by the initial phases multiplied by :math:`\nu`.
            additional_args: Extra model arguments, passed to
                :meth:`~few.trajectory.ode.base.ODEBase.add_fixed_parameters`.
            T: Maximum duration in years. Default ``1.0``.
            dt: Fixed step in seconds (used with ``DENSE_STEPPING``). Default ``10.0``.
            **kwargs: Forwarded to :meth:`initialize_integrator` (``err``,
                ``DENSE_STEPPING``, ``integrate_backwards``, ``max_step_size``; others are
                ignored).

        Returns:
            np.ndarray: :attr:`trajectory`, shape ``(npoints, nparams + 1)``: time in
            seconds, the state, and the phases in rad (a view of the internal buffer, which
            the next call overwrites).

        Raises:
            ValueError: From :meth:`integrate`, or for a backwards integration whose start
                is closer to the separatrix than ``separatrix_buffer_dist``.
            AssertionError: If ``len(y0) != nparams``.
        """
        # Indicate we are generating a trajectory (alters behaviour of integrator spline)
        self.generating_trajectory = True

        self.moves_check = 0
        self.initialize_integrator(**kwargs)
        mu = m1 * m2 / (m1 + m2)
        M = m1 + m2
        self.massratio = mu / M
        # Compute the adimensionalized time steps and max time
        self.tmax_dimensionless = T * YRSID_SI / (M * MTSUN_SI) * self.massratio
        self.dt_dimensionless = dt / (M * MTSUN_SI) * self.massratio
        self.Msec = MTSUN_SI * M / self.massratio
        self.a = a
        assert self.nparams == len(y0)

        self.func.add_fixed_parameters(m1, m2, a, additional_args=additional_args)

        self.integrate(0.0, y0)

        orb_params = [y0[0], y0[1], y0[2]]
        if self.integrate_constants_of_motion:
            orb_params = ELQ_to_pex(self.a, y0[0], y0[1], y0[2])

        # Error if we start too close to separatrix.
        if self.integrate_backwards:
            if not self.enforce_schwarz_sep:
                p_sep = get_separatrix(self.a, orb_params[1], orb_params[2])
            else:
                p_sep = 6 + 2 * orb_params[1]
            if (orb_params[0] - p_sep) < self.separatrix_buffer_dist - 1e-6:
                # Raise a warning
                raise ValueError(
                    f"p_f is too close to separatrix. It must start above p_sep + {self.separatrix_buffer_dist}. Started at {orb_params[0]}, separatrix {p_sep}, starting p {orb_params[0]} should be larger than {p_sep + self.separatrix_buffer_dist - INNER_THRESHOLD}."
                )

        # scale phases here by the mass ratio so the cache is accurate
        self.trajectory_arr[:, 4:7] /= self.massratio

        # backwards integration requires an additional manipulation to match forwards phase convention
        if self.integrate_backwards:
            self.trajectory_arr[:, 4:7] -= (
                self.trajectory_arr[0, 4:7]
                + self.trajectory_arr[self.traj_step - 1, 4:7]
            )

        # Restore normal spline behaviour
        self.generating_trajectory = False
        return self.trajectory

    def stop_integrate_check(self, t: float, y: np.ndarray) -> bool:
        """Check the stopping condition after an accepted step.

        Forwards: stop once ``p - p_sep < separatrix_buffer_dist`` (Kerr separatrix
        :func:`~few.utils.geodesic.get_separatrix`, or :math:`6 + 2e` with
        ``enforce_schwarz_sep``). Backwards: stop once
        :meth:`distance_to_outer_boundary` is negative.

        Args:
            t: Time (unused).
            y: State after the step.

        Returns:
            bool or None: ``True`` to stop; ``None`` (falsy) otherwise.
        """

        if self.integrate_backwards:
            # this function handles the pex/ELQ conversion internally, in case of ELQ-specific outer boundaries
            distance_to_grid_boundary = self.distance_to_outer_boundary(y)
            if distance_to_grid_boundary < 0:
                return True
        else:
            p, e, x = self.get_pex(y)
            if not self.enforce_schwarz_sep:
                p_sep = get_separatrix(self.a, e, x)
            else:
                p_sep = 6 + 2 * e

            if p - p_sep < self.separatrix_buffer_dist:
                return True

    def inner_func_forward(self, t_step):
        """Return the distance of the dense output at ``t_step`` from the stop boundary in ``p``.

        The root-finding target of a forward stop: ``p - (p_sep + separatrix_buffer_dist)``.
        Caches the evaluated state in ``_y_inner_cache``.

        Args:
            t_step (float): Time in seconds.

        Returns:
            float: Positive above the boundary, zero on it.
        """
        self._y_inner_cache = self.eval_integrator_spline(
            np.array(
                [
                    t_step,
                ]
            )
        )[0]

        p, e, x = self.get_pex(self._y_inner_cache)
        # get the separatrix value at this new step
        if not self.enforce_schwarz_sep:
            p_sep = get_separatrix(self.a, e, x)
        else:
            p_sep = 6 + 2 * e

        return p - (p_sep + self.separatrix_buffer_dist)  # we want this to go to zero

    def inner_func_backward(self, t_step):
        """Return the distance of the dense output at ``t_step`` from the grid's outer edge.

        The root-finding target of a backwards stop (:meth:`distance_to_outer_boundary`).
        Caches the evaluated state in ``_y_inner_cache``.

        Args:
            t_step (float): Time in seconds.

        Returns:
            float: Positive inside the grid, negative outside.
        """

        self._y_inner_cache = self.eval_integrator_spline(
            np.array(
                [
                    t_step,
                ]
            )
        )[0]

        return self.distance_to_outer_boundary(self._y_inner_cache)

    def finishing_function(self, t: float, y: np.ndarray):
        """Place the final point after the stepping loop ends (adaptive mode only).

        If the loop stopped before ``tmax`` (a boundary was crossed), calls
        :meth:`_finishing_function_stop`; otherwise :meth:`_finishing_function_at_tmax`.
        Does nothing with fixed steps.

        Args:
            t: Time of the last accepted point (radiation-reaction units).
            y: State at that point.
        """
        if self.dopr.fix_step:
            return

        # advance the step counter by one temporarily to read the last spline value
        self.traj_step += 1

        if (
            t < self.tmax_dimensionless
        ):  # don't step to the separatrix if we already hit the time window
            self._finishing_function_stop(t)
        else:
            self._finishing_function_at_tmax()

    def _finishing_function_stop(self, t: float):
        """Place the final point on the stop boundary by root-finding on the last segment.

        If the last segment also passes ``tmax`` and the boundary is not yet crossed at
        ``tmax``, the final point goes at ``tmax`` instead. Otherwise ``brentq``
        (``xtol=INNER_THRESHOLD``, ``rtol=1e-13``, ``maxiter=MAX_ITER``) finds the crossing
        time between the last point and the segment's end knot.

        Args:
            t: Time of the last accepted point (radiation-reaction units).

        Raises:
            RuntimeError: If the root-finding does not converge.
        """
        if self.integrate_backwards:
            distance_func = self.inner_func_backward
        else:
            distance_func = self.inner_func_forward

        # if tmax occurs before the last point, we need to determine if the crossing is before t=tmax
        # essentially, both stopping conditions may have occured in the same step, and we need to check
        # which of them happened first.
        if self.integrator_t_cache[-1] > self.tmax_dimensionless * self.Msec:
            # first, check if the crossing has happened by t=tmax (this is unlikely but can happen)
            distance_at_tmax = distance_func(self.tmax_dimensionless * self.Msec)
            if distance_at_tmax > 0:
                # the trajectory didnt cross the boundary before t=tmax, so stop at t=tmax.
                # just place the point at y_at_tmax above.
                # we do not pass any spline information here as it has already been computed in the main integrator loop
                self.traj_step -= 1  # revert the step counter to place the last (t, y) in the right place (spline info not overwritten)
                self.save_point(
                    self.tmax_dimensionless * self.Msec,
                    self._y_inner_cache,
                    spline_output=None,
                )
                # exit the finishing function
                return

        # the trajectory crosses the boundary before t=tmax. Root-find to get the crossing time.
        result = brentq(
            distance_func,
            t * self.Msec,  # lower bound: the current point
            self.integrator_t_cache[
                -1
            ],  # upper bound: the knot that passed the boundary
            maxiter=MAX_ITER,
            xtol=INNER_THRESHOLD,
            rtol=1e-13,
            full_output=True,
        )

        if result[1].converged:
            t_out = result[0]
            y_out = self.eval_integrator_spline(
                np.array(
                    [
                        t_out,
                    ]
                )
            )[0]

            self.traj_step -= 1  # revert the step counter to place the last (t, y) in the right place (spline info not overwritten)
            self.save_point(t_out, y_out, spline_output=None)
        else:
            raise RuntimeError(
                "Separatrix root-finding operation did not converge within MAX_ITER."
            )

    def _finishing_function_at_tmax(self):
        """Place the final point at ``tmax`` on the dense output of the last segment."""
        y_finish = self.eval_integrator_spline(
            np.array(
                [
                    self.tmax_dimensionless * self.Msec,
                ]
            )
        )[0]

        self.traj_step -= 1  # revert the step counter to place the last (t, y) in the right place (spline info not overwritten)
        # we do not pass any spline information here as it has already been computed in the main integrator loop
        self.save_point(
            self.tmax_dimensionless * self.Msec, y_finish, spline_output=None
        )  # adds time in seconds

    @classmethod
    def get_pex(self, y: np.ndarray) -> np.ndarray:
        """Map an integrated state to ``(p, e, x)`` (implemented by the subclasses).

        Args:
            y: Integrated state.

        Returns:
            np.ndarray: ``(p, e, x)``.

        Raises:
            NotImplementedError: Always, in the base class.
        """
        raise NotImplementedError


class APEXIntegrate(Integrate):
    """:class:`Integrate` in the orbital elements ``(p, e, x)`` (the default variant).

    Built by :func:`get_integrator` for ``integrate_constants_of_motion=False``; this is
    the integrator of every FEW waveform model and of the LISA global-fit EMRI path. Takes
    the :class:`Integrate` arguments.
    """

    def get_pex(self, y):
        """Return the first three state entries ``(p, e, x)`` (a view of ``y``)."""
        return y[:3]


class AELQIntegrate(Integrate):
    """:class:`Integrate` in the constants of motion ``(E, L, Q)``.

    Built by :func:`get_integrator` for ``integrate_constants_of_motion=True``; the model
    must support ELQ (``supports_ELQ``). Takes the :class:`Integrate` arguments.
    """

    def get_pex(self, y):
        """Convert ``(E, L, Q)`` to ``(p, e, x)`` with :func:`~few.utils.geodesic.ELQ_to_pex`.

        Args:
            y (np.ndarray): State whose first three entries are ``(E, L, Q)``.

        Returns:
            tuple: ``(p, e, x)``.
        """
        p, e, x = ELQ_to_pex(self.a, y[0], y[1], y[2])
        return p, e, x


class BatchedIntegrate:
    """Integrate many EMRI trajectories at once (experimental CPU reference).

    Each DOPR853 step advances every active system, with the right-hand side from
    ``func.evaluate_rhs_batch`` (no Python loop over systems inside a step). It is the CPU
    reference of a planned device-resident integrator (``docs/traj-batching.md``) and is not
    wired into :class:`~few.trajectory.inspiral.EMRIInspiral` or the waveform generators.

    Per system it follows :class:`Integrate` exactly: the same initial step tuning,
    controller, retry-and-halve rules, separatrix-buffer stop and final-point placement.
    The final point is placed by a scalar :class:`Integrate` loaded with that system's state
    (once per system), so the result is bit-comparable with the scalar path.

    Scope: forward integration, pex variables (``use_ELQ=False``), adaptive stepping.
    A fresh DOPR853 is used per :meth:`run` (the scalar DOPR853 keeps controller state
    across trajectories; compare against a fresh scalar integrator).

    Args:
        func (str or type): ODE model (class or stock name) with an
            ``evaluate_rhs_batch`` method, in practice
            :class:`~few.trajectory.ode.flux.KerrEccEqFlux`. Default ``"KerrEccEqFlux"``.
        err: Absolute tolerance of the error control. Default ``1e-11``.
        buffer_length: Initial rows per system of the output buffers (grown by 100).
            Default ``1000``.
        max_step_size: Largest step in radiation-reaction units. Default ``None`` means no
            limit.
        max_iter: Accepted-step limit per system. Default ``None`` means ``buffer_length``.
        **kwargs: Forwarded to :func:`get_integrator` for the scalar helper integrator
            (``force_backend`` defaults to ``"cpu"``; the ODE ignores it).

    Raises:
        NotImplementedError: If the model has no ``evaluate_rhs_batch``, or its class
            overrides ``evaluate_rhs``, ``modify_rhs`` or ``modify_rhs_before_Jacobian``
            (the batched RHS only reproduces the stock ``KerrEccEqFlux``).

    Example:
        .. code-block:: python

            import numpy as np
            from few.trajectory.integrate import BatchedIntegrate

            bi = BatchedIntegrate("KerrEccEqFlux", err=1e-11)
            y0 = np.array([[10.0, 12.0], [0.3, 0.2], [1.0, 1.0],
                           [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])   # (6, S): pex + phases
            out = bi.run(1e6, 1e1, 0.9, y0, T=1.0)
            out.trajectories[0]   # (n_0, 7): t [s], p, e, x, Phi_phi, Phi_theta, Phi_r
    """

    def __init__(self, func="KerrEccEqFlux", err: float = 1e-11, buffer_length: int = 1000,
                 max_step_size: Optional[float] = None, max_iter: Optional[int] = None, **kwargs):
        kwargs.setdefault("force_backend", "cpu")
        self._scalar = get_integrator(func, integrate_constants_of_motion=False,
                                      buffer_length=buffer_length, **kwargs)
        self.func = self._scalar.func
        if not hasattr(self.func, "evaluate_rhs_batch"):
            raise NotImplementedError(f"{type(self.func).__name__} has no evaluate_rhs_batch")
        # evaluate_rhs_batch reproduces ODEBase.__call__ for the STOCK physics only: refuse
        # subclasses that override the right-hand side or its documented hooks
        from .ode.base import ODEBase
        from .ode.flux import KerrEccEqFlux

        cls = type(self.func)
        if (cls.evaluate_rhs is not KerrEccEqFlux.evaluate_rhs
                or cls.modify_rhs is not ODEBase.modify_rhs
                or cls.modify_rhs_before_Jacobian is not ODEBase.modify_rhs_before_Jacobian):
            raise NotImplementedError(
                f"{cls.__name__} overrides evaluate_rhs / modify_rhs hooks; BatchedIntegrate "
                "only reproduces the stock KerrEccEqFlux right-hand side")
        self.err = err
        self.buffer_length = int(buffer_length)
        self.max_step_size = np.inf if max_step_size is None else float(max_step_size)
        self.max_iter = self.buffer_length if max_iter is None else int(max_iter)

    # --- RHS through the batched ODE ----------------------------------------------------------
    def _ode(self, t, y, ydot, a_sub):
        """Stepper callback: write ``evaluate_rhs_batch(y, a_sub)`` into ``ydot`` ``(6, S)``."""
        out, _status = self.func.evaluate_rhs_batch(y, a_sub)
        ydot[:] = out
        return ydot

    def _rhs(self, y, a):
        """Return ``(ydot (6, S), status (S,))`` from ``func.evaluate_rhs_batch``."""
        return self.func.evaluate_rhs_batch(y, a)

    def _tune_initial_step(self, y0, a, tmax, h_max):
        """Vectorised copy of :meth:`Integrate.tune_initial_step_size` (t0 = 0)."""
        n = y0.shape[0]
        scale = self.err
        f0, _ = self._rhs(y0, a)
        d0 = np.linalg.norm(y0 / scale, axis=0) / n ** 0.5
        d1 = np.linalg.norm(f0 / scale, axis=0) / n ** 0.5
        with np.errstate(divide="ignore", invalid="ignore"):
            h0 = np.where((d0 < 1e-5) | (d1 < 1e-5), 1e-6, 0.01 * d0 / d1)
        h0 = np.minimum(h0, tmax)
        f1, st = self._rhs(y0 + h0 * f0, a)
        with np.errstate(divide="ignore", invalid="ignore"):
            d2 = np.linalg.norm((f1 - f0) / scale, axis=0) / h0 / n ** 0.5
            h1 = np.where((d1 <= 1e-15) & (d2 <= 1e-15), np.maximum(1e-6, h0 * 1e-3),
                          (0.01 / np.maximum(d1, d2)) ** (1 / 8))
        h = np.minimum.reduce([100 * h0, h1, tmax, h_max])
        return np.where(st != 0, h0, h)      # scalar: an off-bounds f1 returns h0

    def run(self, m1, m2, a, y0, T: float = 1.0, dt: float = 10.0):
        """Integrate ``S`` systems forward from ``y0``.

        This is the :meth:`Integrate.run_inspiral`-level contract: ``y0`` is used as given
        (no ``isvalid_pex`` check, no retrograde ``Phi_phi0`` flip as in
        :meth:`~few.trajectory.inspiral.EMRIInspiral.get_inspiral`).

        Args:
            m1 (float or np.ndarray): Massive black hole mass in solar masses, scalar or
                ``(S,)``.
            m2 (float or np.ndarray): Compact object mass in solar masses, scalar or
                ``(S,)``.
            a (float or np.ndarray): Dimensionless spin, scalar or ``(S,)``.
            y0 (np.ndarray): Initial states, shape ``(6, S)``: ``p, e, x`` and the physical
                initial phases in rad (scaled by the mass ratio internally).
            T: Maximum duration in years. Default ``1.0``.
            dt: Accepted for signature parity; inert for adaptive stepping (as in the scalar
                path). Default ``10.0``.

        Returns:
            types.SimpleNamespace: With fields

            - ``trajectories``: list of ``S`` arrays ``(n_j, 7)``: t [s], p, e, x, Phi_phi,
              Phi_theta, Phi_r (rad), as :attr:`Integrate.trajectory` after
              :meth:`Integrate.run_inspiral`;
            - ``status``: ``(S,)`` int, -1 separatrix buffer, -2 tmax, -3 step-size collapse
              (100 failed tries), -4 ``max_iter`` reached;
            - ``nsteps``: ``(S,)`` int, the ``n_j``;
            - ``spline_t``: list of knot-time arrays [s];
            - ``spline_coeff``: list of ``(n_j - 1, nparams, 8)`` dense-output coefficients,
              as :attr:`Integrate.integrator_spline_coeff` (phase rows mass-ratio scaled).
        """
        from types import SimpleNamespace

        y0 = np.array(y0, dtype=float, copy=True)
        nparams, S = y0.shape
        m1, m2, a = (np.broadcast_to(np.asarray(v, dtype=float), (S,)).copy() for v in (m1, m2, a))
        M = m1 + m2
        massratio = m1 * m2 / M ** 2
        tmax = T * YRSID_SI / (M * MTSUN_SI) * massratio
        Msec = MTSUN_SI * M / massratio
        sep_buffer = self.func.separatrix_buffer_dist
        y = y0.copy()
        y[3:6] *= massratio                            # integrated phases carry the mass ratio

        dopr = DOPR853(self._ode, stopping_criterion=None, tmax=1e9, max_step=1e6)
        dopr.abstol = self.err
        dopr.fix_step = False

        buf = self.buffer_length
        traj = np.zeros((S, buf, nparams + 1))
        tcache = np.zeros((S, buf))
        spl = np.zeros((S, buf, nparams, 8))
        step = np.zeros(S, dtype=int)
        status = np.zeros(S, dtype=int)
        bad = np.zeros(S, dtype=int)
        niter = np.zeros(S, dtype=int)

        def save(j, tj, yj, spline_j=None):
            nonlocal traj, tcache, spl, buf
            if step[j] >= buf - 1:
                grow = 100
                traj = np.concatenate([traj, np.zeros((S, grow, nparams + 1))], axis=1)
                tcache = np.concatenate([tcache, np.zeros((S, grow))], axis=1)
                spl = np.concatenate([spl, np.zeros((S, grow, nparams, 8))], axis=1)
                buf += grow
            traj[j, step[j], 0] = tj
            traj[j, step[j], 1:] = yj
            if spline_j is not None:
                tcache[j, step[j]] = tj
                spl[j, step[j] - 1] = spline_j
            step[j] += 1

        t = np.zeros(S)
        h = self._tune_initial_step(y, a, tmax, np.minimum(self.max_step_size, tmax / 10))
        t_prev, y_prev = t.copy(), y.copy()
        for j in range(S):
            save(j, 0.0, y[:, j])
            tcache[j, 0] = 0.0

        active = np.ones(S, dtype=bool)
        while np.any(active):
            idx = np.flatnonzero(active)
            over = idx[niter[idx] >= self.max_iter]     # per system: flag, never abort the batch
            if over.size:
                status[over] = -4
                active[over] = False
                idx = np.flatnonzero(active)
                if idx.size == 0:
                    break
            t_old, h_old, y_old = t[idx].copy(), h[idx].copy(), y[:, idx].copy()
            xT, hT, yT = t_old.copy(), h_old.copy(), y_old.copy()
            flag = dopr.take_step(xT, hT, yT, tmax[idx], a[idx], inds=idx)
            yT = np.asarray(yT).reshape(nparams, idx.size)

            # scalar order of checks (Integrate.integrate)
            nan_h = np.isnan(hT)
            big_h = ~nan_h & (hT > self.max_step_size)
            nan_y = ~nan_h & ~big_h & np.any(np.isnan(yT), axis=0)
            retry = nan_h | big_h | nan_y
            hT = np.where(nan_h | big_h, h_old / 2, np.where(nan_y, hT / 2, hT))
            xT = np.where(retry, t_prev[idx], xT)
            yT[:, retry] = y_prev[:, idx][:, retry]
            bad[idx[retry]] += 1
            failed = ~retry & ~flag
            bad[idx[failed]] += 1
            if np.any(bad[idx] >= 100):
                status[idx[bad[idx] >= 100]] = -3
                active[idx[bad[idx] >= 100]] = False

            ok = ~retry & flag
            t[idx], h[idx], y[:, idx] = xT, hT, yT
            if not np.any(ok):
                continue
            oi = idx[ok]
            # prep_evaluate reuses the stage coefficients of EVERY column of this take_step:
            # evaluate on all of them, keep the successful ones
            with np.errstate(invalid="ignore", over="ignore"):
                spline_all = dopr.prep_evaluate(t_old, y_old, h_old, t[idx], y[:, idx].copy(), a[idx])
            spline_info = np.asarray(spline_all)[ok]
            bad[oi] = 0
            p, e, x = y[0, oi], y[1, oi], y[2, oi]
            p_sep = np.array([get_separatrix(float(aj), float(ej), float(xj)) for aj, ej, xj in zip(a[oi], e, x)])
            stop = (p - p_sep) < sep_buffer
            for k, j in enumerate(oi):
                if stop[k]:
                    spl[j, step[j] - 1] = spline_info[k]
                    tcache[j, step[j]] = t[j] * Msec[j]
                    y[:, j] = y_prev[:, j]
                    t[j] = t_prev[j]
                    status[j] = -1
                    active[j] = False
                elif t[j] > tmax[j]:
                    spl[j, step[j] - 1] = spline_info[k]
                    tcache[j, step[j]] = t[j] * Msec[j]
                    status[j] = -2
                    active[j] = False
                else:
                    save(j, t[j] * Msec[j], y[:, j], spline_info[k])
                    t_prev[j] = t[j]
                    y_prev[:, j] = y[:, j]
                    niter[j] += 1

        # final point: the scalar finishing functions on each system's own state
        sc = self._scalar
        trajectories, spline_t, spline_coeff = [], [], []
        for j in range(S):
            if status[j] in (-3, -4):
                out = traj[j, :step[j]].copy()
                out[:, 4:7] /= massratio[j]              # physical phases, as for every system
                trajectories.append(out)
                spline_t.append(tcache[j, :step[j]].copy())
                spline_coeff.append(spl[j, :max(step[j] - 1, 0)].copy())
                continue
            sc.trajectory_arr = traj[j].copy()
            sc._integrator_t_cache = tcache[j].copy()
            sc.dopr_spline_output = spl[j].copy()
            sc.traj_step = int(step[j])
            sc.buffer_length = traj.shape[1]
            sc.tmax_dimensionless, sc.Msec, sc.massratio, sc.a = tmax[j], Msec[j], massratio[j], a[j]
            sc.integrate_backwards = False
            sc.generating_trajectory = True
            sc.dopr.fix_step = False
            sc.func.add_fixed_parameters(m1[j], m2[j], a[j])
            sc.finishing_function(t[j], y[:, j].copy())
            out = sc.trajectory_arr[: sc.traj_step].copy()
            out[:, 4:7] /= massratio[j]
            trajectories.append(out)
            spline_t.append(sc._integrator_t_cache[: sc.traj_step].copy())
            spline_coeff.append(sc.dopr_spline_output[: sc.traj_step - 1].copy())
        return SimpleNamespace(trajectories=trajectories, status=status,
                               nsteps=np.array([tr.shape[0] for tr in trajectories]),
                               spline_t=spline_t, spline_coeff=spline_coeff)
