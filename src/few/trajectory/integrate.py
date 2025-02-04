# Pn5-based Generic Kerr trajectory module for Fast EMRI Waveforms

# Copyright (C) 2020 Michael L. Katz, Alvin J.K. Chua, Niels Warburton, Scott A. Hughes
# Based on implementation from Fujita & Shibata 2020
# See specific code documentation for proper citation.

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


import os
from typing import Tuple, List, Union, Type, Optional

import numpy as np
from scipy.optimize import brentq

# Python imports
from ..utils.utility import (
    check_for_file_download,
    ELQ_to_pex,
    get_kerr_geo_constants_of_motion,
    get_separatrix_interpolant,
)
from ..utils.constants import *
from ..utils.citations import *

from .ode.base import ODEBase, get_ode_properties
from .ode import _STOCK_TRAJECTORY_OPTIONS

# get path to this file
dir_path = os.path.dirname(os.path.realpath(__file__))

from ..utils.utility import get_separatrix

INNER_THRESHOLD = 1e-8
PERCENT_STEP = 0.25
MAX_ITER = 1000

from .dopr853 import DOPR853

def get_integrator(func, file_directory=None, integrate_constants_of_motion=False, **kwargs):
    if file_directory is None:
        file_directory = dir_path + "/../../few/files/"

    if integrate_constants_of_motion:
        return AELQIntegrate(
            func=func,
            file_directory=file_directory,
            integrate_constants_of_motion=integrate_constants_of_motion,
            **kwargs,
        )
    else:
        return APEXIntegrate(
            func=func,
            file_directory=file_directory,
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
    def __init__(
        self,
        func: Type[ODEBase],
        integrate_constants_of_motion: bool=False,
        buffer_length: int = 1000,
        rootfind_separatrix: bool=True,
        enforce_schwarz_sep: bool=False,
        file_directory: Optional[str]=None,
        **kwargs
    ):
        self.buffer_length = buffer_length
        self.file_directory = file_directory

        func = digest_func(func)
        
        self.base_func = func
        self.func = func(file_directory=file_directory, use_ELQ=integrate_constants_of_motion, **kwargs)

        self.ode_info = get_ode_properties(self.func)

        self.rootfind_separatrix = rootfind_separatrix
        self.enforce_schwarz_sep = enforce_schwarz_sep

        self.dopr = DOPR853(
            self._dopr_ode_wrap,
            stopping_criterion=None,  # stopping_criterion,
            tmax=1e9,
            max_step=1e6,
        )

        # assert np.all(self.backgrounds == self.backgrounds[0])
        # assert np.all(self.equatorial == self.equatorial[0])
        # assert np.all(self.circular == self.circular[0])

    def __reduce__(self):
        return (self.__class__, (self.base_func, self.integrate_constants_of_motion, self.buffer_length, self.rootfind_separatrix, self.enforce_schwarz_sep, self.file_directory))

    @property
    def nparams(self) -> int:
        return self.func.nparams

    @property
    def num_add_args(self) -> int:
        return self.func.num_add_args

    @property
    def file_dir(self) -> str:
        return str(self.func.file_dir)

    @property
    def convert_Y(self):
        return self.func.convert_Y

    @property
    def background(self):
        return self.func.background

    @property
    def equatorial(self):
        return self.func.equatorial

    @property
    def circular(self):
        return self.func.circular

    @property
    def integrate_constants_of_motion(self):
        return self.func.use_ELQ

    @property
    def separatrix_buffer_dist(self):
        return self.func.separatrix_buffer_dist

    def _dopr_ode_wrap(self, t, y, ydot, additionalArgs):
        self.func(y[:,0], out=ydot[:,0])
        if self.integrate_backwards:
            ydot *= -1
        return ydot

    def take_step(
        self, t: float, h: float, y: np.ndarray
    ) -> Tuple[float, float, np.ndarray]:
        return self.dopr.take_step_single(t, h, y, self.tmax_dimensionless, None)

    def log_failed_step(self):
        # check for number of tries to fix this
        self.bad_num += 1
        if self.bad_num >= 100:
            raise ValueError("error, reached bad limit.\n")

    def integrate(self, t0: float, y0: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        t = t0
        h = self.dt_dimensionless

        t_prev = t0
        y_prev = y0.copy()
        y = y0.copy()

        self.bad_num = 0
        # add the first point
        self.save_point(0.0, y)
        self._integrator_t_cache[0] = 0.0
        # breakpoint()
        while t < self.tmax_dimensionless:

            try:
                t_old = t
                h_old = h
                y_old = y.copy()
                status, t, h = self.dopr.take_step_single(
                    t_old, h_old, y, self.tmax_dimensionless, None
                )

            except (
                ValueError
            ):  # an Elliptic function returned a nan and it raised an exception

                # self.integrator.reset_solver()
                t = t_prev
                y[:] = y_prev[:]
                h = h_old / 2
                self.log_failed_step()
                continue

            if np.isnan(h):
                t = t_prev
                y[:] = y_prev[:]
                h = h_old / 2
                self.log_failed_step()
                continue

            if not status:  # what does this do?
                self.bad_num += 1
                self.log_failed_step()
                continue

            if not self.dopr.fix_step:
                spline_info = self.dopr.prep_evaluate_single(
                    t_old, y_old, h_old, t, y, None
                )
            else:
                spline_info = None

            # or if any quantity is nan, step back and take a smaller step.
            if np.any(np.isnan(y)):  # an Elliptic function returned a nan and it raised an exception

                # self.integrator.reset_solver()
                t = t_prev
                y[:] = y_prev[:]
                h /= 2

                self.log_failed_step()
                continue

            # if it made it here, reset bad num
            self.bad_num = 0

            # should not be needed but is safeguard against stepping past maximum allowable time
            # the last point in the trajectory will be at t = tmax
            if t > self.tmax_dimensionless:
                if not self.dopr.fix_step:
                    self.dopr_spline_output[self.traj_step - 1, :] = spline_info
                    self._integrator_t_cache[self.traj_step] = t * self.Msec
                break

            # if status is 9 meaning inside the separatrix
            action = self.action_function(t, y)

            if action == "stop":
                if not self.dopr.fix_step:
                    self.dopr_spline_output[self.traj_step - 1, :] = spline_info
                    self._integrator_t_cache[self.traj_step] = t * self.Msec

                # go back to last values
                y[:] = y_prev[:]
                t = t_prev
                break

            self.save_point(
                t * self.Msec, y, spline_output=spline_info
            )  # adds time in seconds

            t_prev = t
            y_prev[:] = y[:]

        if hasattr(self, "finishing_function"):
            self.finishing_function(t, y)

    def initialize_integrator(
        self, err=1e-11, DENSE_STEPPING=False, integrate_backwards=False, **kwargs
    ):
        self.dopr.abstol = err
        self.dopr.fix_step = DENSE_STEPPING
        self.integrate_backwards = integrate_backwards
        try:
            self.trajectory_arr = np.zeros((self.buffer_length, self.nparams + 1))
        except TypeError:
            breakpoint()
        self._integrator_t_cache = np.zeros((self.buffer_length, ))
        self.dopr_spline_output = np.zeros(
            (self.buffer_length, 6, 8)
        )  # 3 parameters + 3 phases, 8 coefficients
        self.traj_step = 0

    @property
    def trajectory(self):
        return self.trajectory_arr[: self.traj_step]

    @property
    def integrator_t_cache(self):
        return self._integrator_t_cache[: self.traj_step]

    @property
    def integrator_spline_coeff(self):
        return self.dopr_spline_output[: self.traj_step - 1]

    def save_point(
        self, t: float, y: np.ndarray, spline_output: List[np.ndarray] = None
    ):

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

            self.trajectory_arr = np.concatenate(
                [self.trajectory_arr, np.zeros((buffer_increment, self.nparams + 1))], axis=0
            )
            if not self.dopr.fix_step:
                self.dopr_spline_output = np.concatenate(
                    [self.dopr_spline_output, np.zeros((buffer_increment, self.nparams, 8))], axis=0
                )
                self._integrator_t_cache = np.concatenate([self._integrator_t_cache, np.zeros(buffer_increment,)])
            self.buffer_length += buffer_increment

    def eval_integrator_spline(self, t_new: np.ndarray):
        t_old = self.integrator_t_cache

        result = np.zeros((t_new.size, 6))
        t_in_mask = (t_new >= 0.) & (t_new <= t_old.max())

        result[t_in_mask,:] = self.dopr.eval(t_new[t_in_mask], t_old, self.integrator_spline_coeff)

        if not self.generating_trajectory:
            result[:,3:6] /= self.epsilon

        # backwards integration requires an additional adjustment to match forwards phase conventions
        if self.integrate_backwards and not self.generating_trajectory:
            result[:,3:6] += (self.trajectory[0,4:7] + self.trajectory[-1,4:7])
        return result

    def eval_integrator_derivative_spline(self, t_new: np.ndarray, order: int=1):
        t_old = self.integrator_t_cache
        result = self.dopr.eval_derivative(t_new, t_old, self.integrator_spline_coeff, order=order)

        if not self.generating_trajectory:
            result[:,3:6] /= self.epsilon

        return result

    def run_inspiral(self, M, mu, a, y0, additional_args, T=1.0, dt=10.0, **kwargs):
        # Indicate we are generating a trajectory (alters behaviour of integrator spline)
        self.generating_trajectory = True

        self.moves_check = 0
        self.initialize_integrator(**kwargs)
        self.epsilon = mu / M
        # Compute the adimensionalized time steps and max time
        self.tmax_dimensionless = T * YRSID_SI / (M * MTSUN_SI) * self.epsilon
        self.dt_dimensionless = dt / (M * MTSUN_SI) * self.epsilon
        self.Msec = MTSUN_SI * M / self.epsilon
        self.a = a
        assert self.nparams == len(y0)

        self.func.add_fixed_parameters(
            M, mu, a, additional_args=additional_args
        )

        self.integrate(0.0, y0)

        orb_params = [y0[0], y0[1], y0[2]]
        if self.integrate_constants_of_motion:
            orb_params = ELQ_to_pex(self.a, y0[0], y0[1], y0[2])

        # Create a warning in case we start too close to separatrix.
        if self.integrate_backwards:
            if not self.enforce_schwarz_sep:
                p_sep = get_separatrix_interpolant(self.a, orb_params[1], orb_params[2])
            else:
                p_sep = 6 + 2 * orb_params[1]
            if (
                orb_params[0] - p_sep
            ) < self.separatrix_buffer_dist - INNER_THRESHOLD:
                # Raise a warning
                raise ValueError(
                    f"Warning: p_f is too close to separatrix. It must start above p_sep + {self.separatrix_buffer_dist}."
                )

        # scale phases here by the mass ratio so the cache is accurate
        self.trajectory_arr[:, 4:7] /= self.epsilon

        # backwards integration requires an additional manipulation to match forwards phase convention
        if self.integrate_backwards:
            self.trajectory_arr[:, 4:7] -= (self.trajectory_arr[0,4:7] + self.trajectory_arr[self.traj_step - 1, 4:7])

        # Restore normal spline behaviour
        self.generating_trajectory = False
        return self.trajectory


class APEXIntegrate(Integrate):
    def get_p_sep(self, y: np.ndarray) -> float:
        p = y[0]
        e = y[1]
        x = y[2]

        if self.a == 0.0:
            p_sep = 6.0 + 2.0 * e

        else:
            p_sep = get_separatrix_interpolant(self.a, e, x)
        return p_sep

    def action_function(
        self, t: float, y: np.ndarray
    ) -> str:  # Stop the inspiral when close to the separatrix
        # TODO: implement a function for checking the outer grid bounds for backwards integration.
        p = y[0]

        if self.integrate_backwards:
            pass
        else:
            if not self.enforce_schwarz_sep:
                p_sep = self.get_p_sep(y)
            else:
                p_sep = 6 + 2*y[1]

            if p - p_sep < self.separatrix_buffer_dist + INNER_THRESHOLD:
                return "stop"

            # if p < 10.0 and self.moves_check < 1:
            #     self.integrator.currently_running_ode_index = 1
            #     print(f"Switched to index: {self.integrator.currently_running_ode_index}")
            #     self.moves_check += 1
            # elif p < 8.0 and self.moves_check < 2:
            #     self.integrator.currently_running_ode_index = 0
            #     print(f"Switched to index: {self.integrator.currently_running_ode_index}")
            #     self.moves_check += 1

            return None

    def end_stepper(self, t: float, y: np.ndarray, ydot: np.ndarray, factor: float):
        # estimate the step to the breaking point and multiply by PERCENT_STEP
        if not self.enforce_schwarz_sep:
            p_sep = self.get_p_sep(y)
        else:
            p_sep = 6 + 2*y[1]
        p = y[0]
        pdot = ydot[0]
        step_size = PERCENT_STEP / factor * ((p_sep + self.separatrix_buffer_dist - p) / pdot)

        # copy current values
        temp_y = y + ydot * step_size

        temp_t = t + step_size
        temp_p = temp_y[0]
        temp_stop = temp_p - p_sep

        return (temp_t, temp_y, temp_stop)

    def finishing_function_euler_step(self, t: float, y: np.ndarray):
        """
        We use this stepper for DENSE_STEPPING=1 because no trajectory spline is available.
        """
        # Issue with likelihood computation if this step ends at an arbitrary value inside separatrix + DIST_TO_SEPARATRIX.
        #     // To correct for this we self-integrate from the second-to-last point in the integation to
        #     // within the INNER_THRESHOLD with respect to separatrix +  DIST_TO_SEPARATRIX

        if (
            t < self.tmax_dimensionless
        ):  # don't step to the separatrix if we hit the time window
            # update p_sep (fixes part of issue #17)
            if not self.enforce_schwarz_sep:
                p_sep = self.get_p_sep(y)
            else:
                p_sep = 6 + 2*y[1]

            p = y[0]
            ydot = np.zeros(self.nparams)
            y_temp = np.zeros(self.nparams)

            # set initial values
            factor = 1.0
            iteration = 0
            while p - p_sep > self.separatrix_buffer_dist + INNER_THRESHOLD:
                # Same function in the integrator
                ydot = self.integrator.get_derivatives(y)
                t_temp, y_temp, temp_stop = self.end_stepper(t, y, ydot, factor)
                if (
                    temp_stop > self.separatrix_buffer_dist
                    or self.integrate_backwards == True
                ):
                    # update points
                    t = t_temp
                    y[:] = y_temp[:]
                    if not self.enforce_schwarz_sep:
                        p_sep = self.get_p_sep(y)
                    else:
                        p_sep = 6 + 2*y[1]
                    p = y[0]
                else:
                    # all variables stay the same

                    # decrease step
                    factor *= 2.0

                iteration += 1

                if iteration > MAX_ITER:
                    raise ValueError(
                        "Could not find workable step size in finishing function."
                    )

            self.save_point(t * self.Msec, y)

    def inner_func(self, t_step):
        # evaluate the dense output at y_step
        y_step = self.eval_integrator_spline(np.array([t_step,]))[0]

        # get the separatrix value at this new step
        if not self.enforce_schwarz_sep:
            p_sep = self.get_p_sep(y_step)
        else:
            p_sep = 6 + 2 * y_step[1]

        return y_step[0] - (p_sep + self.separatrix_buffer_dist)  # we want this to go to zero

    def finishing_function(self, t: float, y: np.ndarray):
        # Tune the step-size of a dense integration step such that the trajectory terminates within INNER_THRESHOLD of p_sep + DIST_TO_SEPARATRIX

        # advance the step counter by one temporarily to read the last spline value
        if not self.dopr.fix_step:
            self.traj_step += 1
        if (
            t < self.tmax_dimensionless
        ):  # don't step to the separatrix if we already hit the time window
            if (
                self.rootfind_separatrix and not self.dopr.fix_step
            ):  # use a root-finder and the full integration routine to tune the finish

                # if tmax occurs before the last point, we need to determine if the crossing is before t=tmax
                if self.integrator_t_cache[-1] > self.tmax_dimensionless * self.Msec:

                    # first, check if the crossing has happened by t=tmax (this is unlikely but can happen)
                    y_at_tmax = self.eval_integrator_spline(np.array([self.tmax_dimensionless * self.Msec]))[0]

                    if not self.enforce_schwarz_sep:
                        p_sep_at_tmax = self.get_p_sep(y_at_tmax)
                    else:
                        p_sep_at_tmax = 6 + 2*y_at_tmax[1]

                    if (y_at_tmax[0] - (p_sep_at_tmax + self.separatrix_buffer_dist)) > 0:
                        # the trajectory didnt cross the boundary before t=tmax, so stop at t=tmax.
                        # just place the point at y_at_tmax above.
                        # we do not pass any spline information here as it has already been computed in the main integrator loop
                        self.traj_step -= 1  # revert the step counter to place the last (t, y) in the right place (spline info not overwritten)
                        self.save_point(
                            self.tmax_dimensionless * self.Msec, y_at_tmax, spline_output=None
                        )
                else:
                    # the trajectory crosses the boundary before t=tmax. Root-find to get the crossing time.
                    result = brentq(
                        self.inner_func,
                        t * self.Msec,  # lower bound: the current point
                        self.integrator_t_cache[-1],  # upper bound: the knot that passed the boundary
                        # args=(y_out,),
                        maxiter=MAX_ITER,
                        xtol=INNER_THRESHOLD,
                        rtol=1e-10,
                        full_output=True,
                    )

                    if result[1].converged:
                        t_out = result[0]
                        y_out = self.eval_integrator_spline(np.array([t_out,]))[0]

                        self.traj_step -= 1  # revert the step counter to place the last (t, y) in the right place (spline info not overwritten)
                        self.save_point(
                            t_out, y_out, spline_output=None
                        )
                    else:
                        raise RuntimeError(
                            "Separatrix root-finding operation did not converge within MAX_ITER."
                        )

            # else:
            #     self.finishing_function_euler_step(
            #         t, y
            #     )  # weird step-halving Euler thing

        else:  # If integrator walked past tmax during main loop, place a point at tmax and finish integration
            if not self.dopr.fix_step:
                y_finish = self.eval_integrator_spline(np.array([self.tmax_dimensionless * self.Msec,]))[0]

                self.traj_step -= 1  # revert the step counter to place the last (t, y) in the right place (spline info not overwritten)
                # we do not pass any spline information here as it has already been computed in the main integrator loop
                self.save_point(
                    self.tmax_dimensionless * self.Msec, y_finish, spline_output=None
                )  # adds time in seconds
            else:
                # if another fixed step does not fit in the time window, just finish integration
                pass

class AELQIntegrate(Integrate):
    def get_p_sep(self, y: np.ndarray) -> float:
        e = y[0]
        x = y[1]

        if self.a == 0.0:
            p_sep = 6.0 + 2.0 * e

        else:
            p_sep = get_separatrix_interpolant(self.a, e, x)
        return p_sep

    def action_function(
        self, t: float, y: np.ndarray
    ) -> str:  # Stop the inspiral when close to the separatrix
        # TODO: implement a function for checking the outer grid bounds for backwards integration.
        E = y[0]
        L = y[1]
        Q = y[2]

        p, e, x = ELQ_to_pex(self.a, E, L, Q)

        if self.integrate_backwards:
            pass
        else:
            if not self.enforce_schwarz_sep:
                p_sep = self.get_p_sep([e, x])
            else:
                p_sep = 6 + 2*e

            if p - p_sep < self.separatrix_buffer_dist + INNER_THRESHOLD:
                return "stop"

            return None

    def end_stepper(self, t: float, y: np.ndarray, ydot: np.ndarray, factor: float):
        # estimate the step to the breaking point and multiply by PERCENT_STEP
        p, e, x = ELQ_to_pex(self.a, y[0], y[1], y[2])
        if not self.enforce_schwarz_sep:
            p_sep = self.get_p_sep([e, x])
        else:
            p_sep = 6 + 2*e

        Edot, Ldot = ydot[0], ydot[1]

        E_sep, L_sep, _ = get_kerr_geo_constants_of_motion(
            self.a, p_sep + self.separatrix_buffer_dist, e, x
        )
        step_size_E = PERCENT_STEP / factor * ((E_sep - y[0]) / Edot)
        step_size_L = PERCENT_STEP / factor * ((L_sep - y[1]) / Ldot)

        step_size = (step_size_E**2 + step_size_L**2) ** 0.5

        # copy current values
        temp_y = y + ydot * step_size

        temp_t = t + step_size
        temp_E, temp_L = temp_y[0], temp_y[1]
        temp_stop = ((temp_E - E_sep) ** 2 + (temp_L - L_sep) ** 2) ** 0.5

        return (temp_t, temp_y, temp_stop)

    def finishing_function_euler_step(self, t: float, y: np.ndarray):
        """
        We use this stepper for DENSE_STEPPING=1 because no trajectory spline is available.
        """

        if (
            t < self.tmax_dimensionless
        ):  # don't step to the separatrix if we hit the time window
            # update p_sep (fixes part of issue #17)
            p, e, x = ELQ_to_pex(self.a, y[0], y[1], y[2])

            if not self.enforce_schwarz_sep:
                p_sep = self.get_p_sep([e, x])
            else:
                p_sep = 6 + 2*e

            ydot = np.zeros(self.nparams)
            y_temp = np.zeros(self.nparams)

            # set initial values
            factor = 1.0
            iteration = 0
            while p - p_sep > self.separatrix_buffer_dist + INNER_THRESHOLD:
                # Same function in the integrator
                ydot = self.integrator.get_derivatives(y)
                t_temp, y_temp, temp_stop = self.end_stepper(t, y, ydot, factor)
                if (
                    temp_stop > self.separatrix_buffer_dist
                    or self.integrate_backwards == True
                ):
                    # update points
                    t = t_temp
                    y[:] = y_temp[:]
                    p, e, x = ELQ_to_pex(self.a, y[0], y[1], y[2])

                    if not self.enforce_schwarz_sep:
                        p_sep = self.get_p_sep([e, x])
                    else:
                        p_sep = 6 + 2*e

                else:
                    # all variables stay the same

                    # decrease step
                    factor *= 2.0

                iteration += 1

                if iteration > MAX_ITER:
                    raise ValueError(
                        "Could not find workable step size in finishing function."
                    )

            self.save_point(t * self.Msec, y)

    def inner_func(self, t_step):
        # evaluate the dense output at y_step
        y_step = self.eval_integrator_spline(np.array([t_step,]))[0]
        p, e, x = ELQ_to_pex(self.a, y_step[0], y_step[1], y_step[2])

        # get the separatrix value at this new step
        if not self.enforce_schwarz_sep:
            p_sep = self.get_p_sep([e, x])
        else:
            p_sep = 6 + 2 * e

        return p - (p_sep + self.separatrix_buffer_dist)  # we want this to go to zero

    def finishing_function(self, t: float, y: np.ndarray):
        # Tune the step-size of a dense integration step such that the trajectory terminates within INNER_THRESHOLD of p_sep + DIST_TO_SEPARATRIX

        # advance the step counter by one temporarily to read the last spline value
        if not self.dopr.fix_step:
            self.traj_step += 1
        if (
            t < self.tmax_dimensionless
        ):  # don't step to the separatrix if we already hit the time window
            if (
                self.rootfind_separatrix and not self.dopr.fix_step
            ):  # use a root-finder and the full integration routine to tune the finish

                # if tmax occurs before the last point, we need to determine if the crossing is before t=tmax
                if self.integrator_t_cache[-1] > self.tmax_dimensionless * self.Msec:

                    # first, check if the crossing has happened by t=tmax (this is unlikely but can happen)
                    y_at_tmax = self.eval_integrator_spline(np.array([self.tmax_dimensionless * self.Msec]))[0]
                    p, e, x = ELQ_to_pex(self.a, y_at_tmax[0], y_at_tmax[1], y_at_tmax[2])
                    if not self.enforce_schwarz_sep:
                        p_sep_at_tmax = self.get_p_sep([e, x])
                    else:
                        p_sep_at_tmax = 6 + 2*e

                    if (p - (p_sep_at_tmax + self.separatrix_buffer_dist)) > 0:
                        # the trajectory didnt cross the boundary before t=tmax, so stop at t=tmax.
                        # just place the point at y_at_tmax above.
                        # we do not pass any spline information here as it has already been computed in the main integrator loop
                        self.traj_step -= 1  # revert the step counter to place the last (t, y) in the right place (spline info not overwritten)
                        self.save_point(
                            self.tmax_dimensionless * self.Msec, y_at_tmax, spline_output=None
                        )
                else:
                    # the trajectory crosses the boundary before t=tmax. Root-find to get the crossing time.
                    result = brentq(
                        self.inner_func,
                        t * self.Msec,  # lower bound: the current point
                        self.integrator_t_cache[-1],  # upper bound: the knot that passed the boundary
                        # args=(y_out,),
                        maxiter=MAX_ITER,
                        xtol=INNER_THRESHOLD,
                        rtol=1e-10,
                        full_output=True,
                    )

                    if result[1].converged:
                        t_out = result[0]
                        y_out = self.eval_integrator_spline(np.array([t_out,]))[0]

                        self.traj_step -= 1  # revert the step counter to place the last (t, y) in the right place (spline info not overwritten)
                        self.save_point(
                            t_out, y_out, spline_output=None
                        )
                    else:
                        raise RuntimeError(
                            "Separatrix root-finding operation did not converge within MAX_ITER."
                        )

            else:
                self.finishing_function_euler_step(
                    t, y
                )  # weird step-halving Euler thing

        else:  # If integrator walked past tmax during main loop, place a point at tmax and finish integration
            if not self.dopr.fix_step:
                y_finish = self.eval_integrator_spline(np.array([self.tmax_dimensionless * self.Msec,]))[0]

                self.traj_step -= 1  # revert the step counter to place the last (t, y) in the right place (spline info not overwritten)
                # we do not pass any spline information here as it has already been computed in the main integrator loop
                self.save_point(
                    self.tmax_dimensionless * self.Msec, y_finish, spline_output=None
                )  # adds time in seconds
            else:
                # if another fixed step does not fit in the time window, just finish integration
                pass
