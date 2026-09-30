# Pn5-based Generic Kerr trajectory module for Fast EMRI Waveforms

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

INNER_THRESHOLD = 1e-10
PERCENT_STEP = 0.25
MAX_ITER = 1000


def get_integrator(func, integrate_constants_of_motion=False, **kwargs):
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
    if isinstance(func, str):
        try:
            func = _STOCK_TRAJECTORY_OPTIONS[func]
        except KeyError:
            raise ValueError(f"The trajectory function {func} could not be found.")
    return func


class Integrate:
    """Custom integrator class.

    Flexible integration options. # TODO: discuss options

    Args:
        func: ODE base function to integrate.
        integrate_constants_of_motion: If ``True``, integrate in ELQ rather than orbital parameters.
        buffer_length: Initial buffer length for trajectory. Will adjust itself if needed.
        enforce_schwarz_sep: If ``True``, enforce the Schwarzschild separatrix as the separatrix.
            :math:`p_s = 6 + 2e`.
        max_iter: Maximum number of iterations for the integrator.
        **kwargs: Keyword arguments for the ODE function, ``func``.

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
        """Dimensionality of ODE"""
        return self.func.nparams

    @property
    def num_add_args(self) -> int:
        """Number of additional arguments to the ODE"""
        return self.func.num_add_args

    @property
    def convert_Y(self):
        """Convert to Y if needed."""
        return self.func.convert_Y

    @property
    def background(self):
        """Spacetime background (spin or no spin)"""
        return self.func.background

    @property
    def equatorial(self):
        """Orbit limited to equatorial plane."""
        return self.func.equatorial

    @property
    def circular(self):
        """Circular orbit."""
        return self.func.circular

    @property
    def integrate_constants_of_motion(self):
        """Integrating in ELQ."""
        return self.func.use_ELQ

    @property
    def separatrix_buffer_dist(self):
        """Buffer distance from separatrix."""
        return self.func.separatrix_buffer_dist

    def distance_to_outer_boundary(self, y):
        """Distance to outer boundary of interpolation grid."""
        return self.func.distance_to_outer_boundary(y)

    def _dopr_ode_wrap(self, t, y, ydot, additionalArgs):
        self.func(y[:, 0], out=ydot[:, 0])
        return ydot

    def take_step(
        self, t: float, h: float, y: np.ndarray
    ) -> Tuple[float, float, np.ndarray]:
        """Take a step of the integrator.

        Args:
            t: Time.
            h: Step size.
            y: Current position of integrator.

        Returns:
            (new time, new step size, new position)

        """
        return self.dopr.take_step_single(t, h, y, self.tmax_dimensionless, None)

    def log_failed_step(self, lastError: Exception):
        """Add to and check failed step count.

        Raises:
            ValueError: Too many failed tries.

        """
        # check for number of tries to fix this
        self.bad_num += 1
        self.bad_num_total += 1
        self.last_error = lastError
        if self.bad_num >= 100:
            raise self.last_error

    def tune_initial_step_size(self, t0: float, y0: np.ndarray, h_max: float) -> float:
        """Tune the initial step size of the integrator if using adaptive stepping.

        Adapted from scipy.integrate._ivp.common.select_initial_step

        Args:
            t0: Initial time.
            y0: Initial state.
            h_max: Largest step size permitted.

        Returns:
            Initial step size.

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
        """Integrate from (t0, y0).

        Args:
            t0: Initial time.
            y0: Initial position.

        Returns:
            (Time array, Position array).


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
        """Act on the integrator.

        Args:
            t: Time.
            y: Current position of integrator.

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
        """Setup the integrator.

        Args:
            err: Absolute tolerance of the integrator.
            DENSE_STEPPING: If ``True``, use fixed stepping.
            integrate_backwards: If ``True``, integrate backwards.
            **kwargs: For future interoperability.

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
        """Absolute tolerance of the integrator."""
        return self.dopr.abstol

    @property
    def dense_stepping(self) -> bool:
        """If ``True``, trajectory is using fixed stepping."""
        return self.dopr.fix_step

    @property
    def npoints(self) -> int:
        """Number of points in the trajectory."""
        return self.traj_step

    @property
    def trajectory(self) -> np.ndarray:
        """Trajectory array."""
        return self.trajectory_arr[: self.traj_step]

    @property
    def integrator_t_cache(self) -> np.ndarray:
        """Spline coefficient time array."""
        return self._integrator_t_cache[: self.traj_step]

    @property
    def integrator_spline_coeff(self) -> np.ndarray:
        """Spline coefficients from dopr backend."""
        return self.dopr_spline_output[: self.traj_step - 1]

    def save_point(
        self, t: float, y: np.ndarray, spline_output: Optional[np.ndarray] = None
    ) -> None:
        """Save point in trajectory array.

        Args:
            t: Time.
            y: Current position of integrator.
            spline_output: Spline coefficients from the backend.

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
        """Evaluate integration at new time array.

        Args:
            t_new: New time array.

        Returns:
            New trajectory.

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
        """Evaluate integration derivatives at new time array.

        Args:
            t_new: New time array.
            order: Order of derivative to evaluate. Defaults to 1.

        Returns:
            New trajectory derivatives.

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
        """Run inspiral integration.

        Args:
            m1: Larger mass in solar masses.
            m2: Small mass in solar masses.
            a: Dimensionless spin of central black hole.
            y0: Initial set of coordinates for integration.

        Returns:
            Trajectory array for integrated coefficients.


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
        """Stop the inspiral when close to the separatrix (forwards integration)
            or when close to the outer grid boundary (backwards integration).

        Args:
            t: Time.
            y: Current position of integrator.

        Returns:
            ``True`` if integration should be stopped. ``False`` otherwise.

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
        """
        Evaluates the distance from the inner boundary at t=t_step.
        Also caches the state of the system as self._y_inner_cache.
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
        """
        Evaluates the distance from the outer boundary at t=t_step.
        Also caches the state of the system as self._y_inner_cache.
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
        """
        This function is called when the integrator has finished due to reaching a stopping condition. The
        function identifies the stopping condition and places the final point at the stopping boundary accordingly.
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
        """
        If the integrator stops due to the separatrix stopping condition, place a point at the inner
        boundary and finish integration.
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
        """
        If the integrator walked past tmax during the main loop, place a point at tmax and finish integration.
        """
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
        """Handles integrator-specific conversion from y to pex and returns the result."""
        raise NotImplementedError


class APEXIntegrate(Integrate):
    """
    An subclass of Integrate for integrating in pex coordinates.
    """

    def get_pex(self, y):
        return y[:3]


class AELQIntegrate(Integrate):
    """
    An subclass of Integrate for integrating in ELQ coordinates.
    """

    def get_pex(self, y):
        p, e, x = ELQ_to_pex(self.a, y[0], y[1], y[2])
        return p, e, x


class BatchedIntegrate:
    """Integrate many EMRI trajectories at once: each DOPR853 step advances every active
    system, with the right-hand side from ``func.evaluate_rhs_batch`` (no Python loop over
    systems inside a step). The CPU reference of the device-resident integrator.

    Per system it follows :class:`Integrate` exactly: the same initial step tuning,
    controller, retry-and-halve rules, separatrix-buffer stop and final-point placement.
    The final point is placed by a scalar :class:`Integrate` loaded with that system's state
    (once per system), so the result is bit-comparable with the scalar path.

    Scope: forward integration, pex variables (``use_ELQ=False``), adaptive stepping.
    A fresh DOPR853 is used per :meth:`run` (the scalar DOPR853 keeps controller state
    across trajectories; compare against a fresh scalar integrator).
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
        out, _status = self.func.evaluate_rhs_batch(y, a_sub)
        ydot[:] = out
        return ydot

    def _rhs(self, y, a):
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
        """Integrate ``S`` systems. ``y0`` is ``(6, S)`` pex + phases (physical phases).

        This is the :meth:`Integrate.run_inspiral`-level contract: ``y0`` is used as given
        (no ``isvalid_pex`` check, no retrograde ``Phi_phi0`` flip as in
        ``EMRIInspiral.get_inspiral``); ``m1, m2, a`` broadcast to ``S``; ``dt`` is accepted
        for signature parity and is inert for adaptive stepping (as in the scalar path).

        Returns a namespace with ``trajectories`` (list of ``(n_j, 7)``: t [s], p, e, x,
        Phi_phi, Phi_theta, Phi_r, as :attr:`Integrate.trajectory` after
        :meth:`Integrate.run_inspiral`), ``status`` (-1 separatrix buffer, -2 tmax,
        -3 step-size collapse (100 failed tries), -4 max_iter reached), ``nsteps``, and the
        dense output per system: ``spline_t`` (knot times [s]) and ``spline_coeff``
        (``(n_j - 1, nparams, 8)``, as :attr:`Integrate.integrator_spline_coeff`).
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
