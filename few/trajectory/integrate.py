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
import warnings
from typing import Tuple, List, Union
import time

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

# Cython/C++ imports
from pyInspiral import pyInspiralGenerator, pyDerivative

# Python imports
from ..utils.baseclasses import TrajectoryBase
from ..utils.utility import check_for_file_download, get_ode_function_options, ELQ_to_pex, get_kerr_geo_constants_of_motion, get_separatrix_interpolant
from ..utils.constants import *
from ..utils.citations import *


# get path to this file
dir_path = os.path.dirname(os.path.realpath(__file__))

from ..utils.utility import get_separatrix
# from ..utils.utility import get_separatrix as get_sep
# from few.utils.spline import TricubicSpline
# # create a TricubicSpline to interpolate get_separatrix
# a = np.linspace(0.0, 0.998, 100)
# e = np.linspace(0.0, 0.99, 100)
# x = np.linspace(-1.0, 1.0, 100)
# XYZ = np.meshgrid(a, e, x, indexing='ij')
# sep = ( get_sep(XYZ[0].flatten(), XYZ[1].flatten(), XYZ[2].flatten())).reshape((100,100,100))
# get_separatrix = TricubicSpline(a, e, x, sep)

DIST_TO_SEPARATRIX = 0.05
INNER_THRESHOLD = 1e-8
PERCENT_STEP = 0.25
MAX_ITER = 1000

KERR = 1
SCHWARZSCHILD = 2

def get_integrator(func, nparams, few_dir, **kwargs):

    ode_info = get_ode_function_options()
    
    if isinstance(func, str):
        num_add_args = ode_info[func]['num_add_args']
        func = [func]

    integrator = pyInspiralGenerator(
        nparams,
        num_add_args,
    )

    for func_i in func:
        assert isinstance(func_i, str)
        if func_i not in ode_info:
            raise ValueError(
                f"func not available. Options are {list(ode_info.keys())}."
            )
        integrator.add_ode(func_i.encode(), few_dir.encode())

    
    if not integrator.integrate_phases[0]:
        nparams -= 3
    if integrator.integrate_constants_of_motion[0]:
        return AELQIntegrate(func, nparams, few_dir, num_add_args=num_add_args,**kwargs)
    else:
        return APEXIntegrate(func, nparams, few_dir, num_add_args=num_add_args,**kwargs)

def fun_wrap(t, y, integrator):
    # print("YEYA", t, y)
    p = y[0]
    p_sep = integrator.get_p_sep(y)

    # print("UGH", p - p_sep - DIST_TO_SEPARATRIX)
    if y[1] < 0.0 or p - p_sep - DIST_TO_SEPARATRIX < 0:
        ydot = np.zeros_like(y)
        return ydot

    ydot = integrator.integrator.get_derivatives(y)
    return ydot


class Stopper:
    def __init__(self, integrator):
        self.integrator = integrator
        self.terminal = True

    def __call__(self, t, y, *args):
        p_sep = self.integrator.get_p_sep(y)
        p = y[0]
        # print("SEP", p - p_sep - DIST_TO_SEPARATRIX)
        return p - p_sep - DIST_TO_SEPARATRIX


class Integrate:
    def __init__(
        self,
        func: Union[str, List],
        nparams: int,
        few_dir: str,
        buffer_length: int = 1000,
        num_add_args: int = 0,
        rootfind_separatrix = True,
    ):
        self.buffer_length = buffer_length
        self.integrator = pyInspiralGenerator(
            nparams,
            num_add_args,
        )
        self.dense_integrator = pyInspiralGenerator(
            nparams,
            num_add_args,
        )
        self.ode_info = ode_info = get_ode_function_options()
        self.few_dir = few_dir

        self.rootfind_separatrix = rootfind_separatrix

        if isinstance(func, str):
            func = [func]
        self.func = func
        for func_i in self.func:
            assert isinstance(func_i, str)
            if func_i not in ode_info:
                raise ValueError(
                    f"func not available. Options are {list(ode_info.keys())}."
                )
            self.integrator.add_ode(func_i.encode(), few_dir.encode())
            self.dense_integrator.add_ode(func_i.encode(), few_dir.encode())

            # make sure all files needed for the ode specifically are downloaded
            for fp in ode_info[func_i]["files"]:
                try:
                    check_for_file_download(fp, few_dir)
                except FileNotFoundError:
                    raise ValueError(
                        f"File required for this ODE ({fp}) was not found in the proper folder ({few_dir + 'few/files/'}) or on zenodo."
                    )

        assert np.all(self.backgrounds == self.backgrounds[0])
        assert np.all(self.equatorial == self.equatorial[0])
        assert np.all(self.circular == self.circular[0])

    @property
    def nparams(self) -> int:
        return self.integrator.nparams

    @property
    def num_add_args(self) -> int:
        return self.integrator.num_add_args

    # @property
    # def few_dir(self) -> str:
    #     return str(self.integrator.few_dir)

    @property
    def convert_Y(self):
        return self.integrator.convert_Y

    @property
    def backgrounds(self):
        return self.integrator.backgrounds

    @property
    def equatorial(self):
        return self.integrator.equatorial

    @property
    def circular(self):
        return self.integrator.circular

    @property
    def integrate_constants_of_motion(self):
        return self.integrator.integrate_constants_of_motion

    @property
    def integrate_ODE_phases(self):
        return self.integrator.integrate_phases

    # @property
    # def func(self) -> str:
    #     return str(self.integrator.func_name)

    def __setstate__(self, d):
        self.__dict__ = d

        ode_info = get_ode_function_options()

        for func_i in self.func:
            assert isinstance(func_i, str)
            if func_i not in ode_info:
                raise ValueError(
                    f"func not available. Options are {list(ode_info.keys())}."
                )
            self.integrator.add_ode(func_i.encode(), self.few_dir.encode())
            self.dense_integrator.add_ode(func_i.encode(), self.few_dir.encode())

            # make sure all files needed for the ode specifically are downloaded
            for fp in ode_info[func_i]["files"]:
                try:
                    check_for_file_download(fp, self.few_dir)
                except FileNotFoundError:
                    raise ValueError(
                        f"File required for this ODE ({fp}) was not found in the proper folder ({self.few_dir + 'few/files/'}) or on zenodo."
                    )


    def take_step(
        self, t: float, h: float, y: np.ndarray
    ) -> Tuple[float, float, np.ndarray]:
        return self.integrator.take_step(t, h, y)

    def reset_solver(self):
        self.integrator.reset_solver()

        # check for number of tries to fix this
        self.bad_num += 1
        if self.bad_num >= self.bad_limit:
            raise ValueError("error, reached bad limit.\n")

    def integrate_backwards(self, t0: float, y0: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        t = t0
        h = self.dt_dimensionless

        t_prev = t0
        y_prev = y0.copy()
        y = y0.copy()

        # add the first point
        self.save_point(0., y)
        
        while t < self.tmax_dimensionless:
            status, t, h = self.integrator.take_step(t, h, y, self.tmax_dimensionless)
                               
            # should not be needed but is safeguard against stepping past maximum allowable time
            # the last point in the trajectory will be at t = tmax
            if t > self.tmax_dimensionless:
                break

            self.save_point(t * self.Msec, y)  # adds time in seconds

            t_prev = t
            y_prev[:] = y[:]

    def integrate(self, t0: float, y0: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        t = t0
        h = self.dt_dimensionless

        # for _ in range(100):
        #     st = time.perf_counter()
        #     out = solve_ivp(
        #         fun_wrap,
        #         (0.0, self.tmax_dimensionless),
        #         y0,
        #         method="DOP853",
        #         events=Stopper(self),
        #         args=(self,),
        #         first_step=self.dt_dimensionless,
        #         max_step=self.Msec * 1e5,
        #     )
        #     et = time.perf_counter()
        #     print("Init", (et - st))
        # breakpoint()

        t_prev = t0
        y_prev = y0.copy()
        y = y0.copy()

        # control it if it keeps returning nans and what not
        bad_num = 0

        total = 0.0

        # add the first point
        self.save_point(0., y)
        # breakpoint()
        while t < self.tmax_dimensionless:
            try:
                status, t, h = self.integrator.take_step(t, h, y, self.tmax_dimensionless)
            except ValueError:  # an Elliptic function returned a nan and it raised an exception
                # print("We got a NAN!")
                self.integrator.reset_solver()
                t = t_prev
                y[:] = y_prev[:]
                h /= 2
                continue
                               
            # or if any quantity is nan, step back and take a smaller step.
            if np.any(np.isnan(y)):
                self.integrator.reset_solver()
                t = t_prev
                y[:] = y_prev[:]
                h /= 2
                continue

            # if it made it here, reset bad num
            self.bad_num = 0

            # should not be needed but is safeguard against stepping past maximum allowable time
            # the last point in the trajectory will be at t = tmax
            if t > self.tmax_dimensionless:
                break

            # if status is 9 meaning inside the separatrix
            if status == 9:
                stop = True
            else:
                action = self.action_function(t, y)

            if action == "stop":
                # go back to last values
                y[:] = y_prev[:]
                t = t_prev
                break

            self.save_point(t * self.Msec, y)  # adds time in seconds

            t_prev = t
            y_prev[:] = y[:]

        if hasattr(self, "finishing_function"):
            self.finishing_function(t, y)

    def initialize_integrator(self, err=1e-10, DENSE_STEPPING=False, use_rk4=False, **kwargs):
        self.integrator.set_integrator_kwargs(err, DENSE_STEPPING, not use_rk4)
        self.dense_integrator.set_integrator_kwargs(err, True, not use_rk4)  # always dense step for final stage

        self.integrator.initialize_integrator()
        self.dense_integrator.initialize_integrator()
        self.trajectory_arr = np.zeros((self.buffer_length, self.nparams + 1))
        self.traj_step = 0
    
    @property
    def trajectory(self):
        return self.trajectory_arr[: self.traj_step]

    def save_point(self, t: float, y: np.ndarray):
        
        self.trajectory_arr[self.traj_step, 0] = t
        self.trajectory_arr[self.traj_step, 1:] = y

        self.traj_step += 1
        if self.traj_step >= self.buffer_length:
            # increase by 100
            self.trajectory_arr = np.concatenate(
                [self.trajectory_arr, np.zeros((100, self.nparams + 1))], axis=0
            )
            self.buffer_length = self.trajectory_arr.shape[0]

    def run_inspiral(self, M, mu, a, y0, additional_args, T=1., dt=10., **kwargs):
        self.moves_check = 0
        self.initialize_integrator(**kwargs)
        self.epsilon = mu/M
        # Compute the adimensionalized time steps and max time
        self.tmax_dimensionless = T*YRSID_SI / (M * MTSUN_SI) *self.epsilon
        self.dt_dimensionless = dt / (M * MTSUN_SI) * self.epsilon
        self.Msec = MTSUN_SI * M / self.epsilon
        self.a = a
        assert self.nparams == len(y0)
        assert (self.num_add_args == len(additional_args)) or (
            self.num_add_args == 0 and len(additional_args) == 1
        )

        self.bool_integrate_backwards = kwargs['integrate_backwards'] 
        self.integrator.add_parameters_to_holder(M, mu, a, self.bool_integrate_backwards, additional_args)
        self.dense_integrator.add_parameters_to_holder(M, mu, a, self.bool_integrate_backwards, additional_args)

        t0 = 0.0
        if self.bool_integrate_backwards:
            self.integrate_backwards(t0, y0)
        else:
            self.integrate(t0,y0)
        self.integrator.destroy_integrator_information()
        self.dense_integrator.destroy_integrator_information()

        # Create a warning in case we start to close to separatrix. 
        orb_params = [y0[0],y0[1],y0[2]]
        if self.integrate_constants_of_motion:
            orb_params = ELQ_to_pex(self.a, y0[0], y0[1], y0[2])
        
        p_sep = get_separatrix_interpolant(self.a, orb_params[1], orb_params[2])
        if (orb_params[0] - p_sep) < DIST_TO_SEPARATRIX + INNER_THRESHOLD and self.bool_integrate_backwards:
            # Raise a warning
            warnings.warn("Warning: initial p_f is too close to separatrix. May not be compatible with forwards integration.")

        return self.trajectory


class APEXIntegrate(Integrate):
    def get_p_sep(self, y: np.ndarray) -> float:
        p = y[0]
        e = y[1]
        x = y[2]

        if self.a == 0.0:
            p_sep = 6.0 + 2.0 * e

        else:
            # p_sep = get_separatrix(self.a, e, x)
            p_sep = get_separatrix_interpolant(self.a, e, x)
        return p_sep

    def action_function(
        self, t: float, y: np.ndarray
    ) -> str:  # Stop the inspiral when close to the separatrix
        p_sep = self.get_p_sep(y)
        p = y[0]
        if p - p_sep < DIST_TO_SEPARATRIX + INNER_THRESHOLD and self.bool_integrate_backwards == False:
            return "stop"

        # if p < 10.0 and self.moves_check < 1:
        #     self.integrator.currently_running_ode_index = 1
        #     print(f"Switched to index: {self.integrator.currently_running_ode_index}")
        #     self.moves_check += 1
        # elif p < 8.0 and self.moves_check < 2:
        #     self.integrator.currently_running_ode_index = 0
        #     print(f"Switched to index: {self.integrator.currently_running_ode_index}")
        #     self.moves_check += 1

    def end_stepper(self, t: float, y: np.ndarray, ydot: np.ndarray, factor: float):
        # estimate the step to the breaking point and multiply by PERCENT_STEP
        p_sep = self.get_p_sep(y)
        p = y[0]
        pdot = ydot[0]
        step_size = PERCENT_STEP / factor * ((p_sep + DIST_TO_SEPARATRIX - p) / pdot)

        # copy current values
        temp_y = y + ydot * step_size

        temp_t = t + step_size
        temp_p = temp_y[0]
        temp_stop = temp_p - p_sep

        return (temp_t, temp_y, temp_stop)

    def finishing_function_euler_step(self, t: float, y: np.ndarray):
        # Issue with likelihood computation if this step ends at an arbitrary value inside separatrix + DIST_TO_SEPARATRIX.
        #     // To correct for this we self-integrate from the second-to-last point in the integation to
        #     // within the INNER_THRESHOLD with respect to separatrix +  DIST_TO_SEPARATRIX

        if t < self.tmax_dimensionless:  # don't step to the separatrix if we hit the time window
            # update p_sep (fixes part of issue #17)
            p_sep = self.get_p_sep(y)
            p = y[0]
            ydot = np.zeros(self.nparams)
            y_temp = np.zeros(self.nparams)

            # set initial values
            factor = 1.0
            iteration = 0
            while p - p_sep > DIST_TO_SEPARATRIX + INNER_THRESHOLD:
                # Same function in the integrator
                ydot = self.integrator.get_derivatives(y)
                t_temp, y_temp, temp_stop = self.end_stepper(t, y, ydot, factor)
                if temp_stop > DIST_TO_SEPARATRIX or self.bool_integrate_backwards == True:
                    # update points
                    t = t_temp
                    y[:] = y_temp[:]
                    p_sep = self.get_p_sep(y)
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

    def inner_func(self, h, t, y, y_out):
        y_out[:] = y[:].copy()

        # careful here, if the step is too big it will return a ValueError. One to watch out for in future
        try:
            status, self.t_final_temp, self.h_final_temp = self.dense_integrator.take_step(t, h, y_out, self.tmax_dimensionless)
        except ValueError:
            return -np.inf  # for the root finder

        p_sep = self.get_p_sep(y_out)

        # edge effect check: stops the loop if we cross the maximum time during this tuning process
        # this may be disabled for performance? probably negligible cost to trajectory as only if statements
        if self.t_final_temp > self.tmax_dimensionless:
            if y_out[0] > p_sep + DIST_TO_SEPARATRIX:
                final_h = self.h_final_temp - (self.t_final_temp - self.tmax_dimensionless)
                y_out[:] = y[:].copy()

                status, self.t_final_temp, self.h_final_temp = self.dense_integrator.take_step(t, final_h, y_out, self.tmax_dimensionless)

                raise RuntimeError("We need to stop - hit the maximum time before the separatrix.")

        return y_out[0] - (p_sep + DIST_TO_SEPARATRIX)  # we want this to go to zero

    def finishing_function(self, t: float, y: np.ndarray):
        # Tune the step-size of a dense integration step such that the trajectory terminates within INNER_THRESHOLD of p_sep + DIST_TO_SEPARATRIX 

        if t < self.tmax_dimensionless:  # don't step to the separatrix if we already hit the time window
            if self.rootfind_separatrix:  # use a root-finder and the full integration routine to tune the finish
                p_sep = self.get_p_sep(y)
                p = y[0]
                ydot = self.dense_integrator.get_derivatives(y)
                pdot = ydot[0]
                y_temp = y.copy()
                y_out = y_temp.copy()

                step_size = (p_sep + DIST_TO_SEPARATRIX - p) / pdot  # roughly work out the scale of the step

                try:
                    result = brentq(self.inner_func, step_size / 2, step_size * 2, args=(t, y_temp, y_out),maxiter=MAX_ITER, xtol=INNER_THRESHOLD, rtol=1e-8, full_output=True)

                    if result[1].converged:
                        self.save_point(self.t_final_temp * self.Msec, y_out)
                    else:
                        raise RuntimeError("Separatrix root-finding operation did not converge within MAX_ITER.")

                except RuntimeError as e:  # for the edge case where t -> tmax while we are tuning this
                    self.save_point(self.t_final_temp * self.Msec, y_out)

            else:
                self.finishing_function_euler_step(t, y)  # weird step-halving Euler thing

class AELQIntegrate(Integrate):
    def get_p_sep(self, y: np.ndarray) -> float:
        p, e, x = ELQ_to_pex(self.a, y[0], y[1], y[2])
        
        if self.a == 0.0:
            p_sep = 6.0 + 2.0 * e

        else:
            p_sep = get_separatrix(self.a, e, x)
        return p_sep, (p, e, x)  # return pex for later on

    def action_function(
        self, t: float, y: np.ndarray
    ) -> str:  # Stop the inspiral when close to the separatrix
        p_sep, (p,e,x) = self.get_p_sep(y)

        if p - p_sep < DIST_TO_SEPARATRIX + INNER_THRESHOLD:
            return "stop"

    def end_stepper(self, t: float, y: np.ndarray, ydot: np.ndarray, factor: float):
        # estimate the step to the breaking point and multiply by PERCENT_STEP
        p_sep, (p, e, x) = self.get_p_sep(y)
        Edot, Ldot = ydot[0], ydot[1]

        E_sep, L_sep, _ = get_kerr_geo_constants_of_motion(self.a, p_sep + DIST_TO_SEPARATRIX, e, x)
        step_size_E = PERCENT_STEP / factor * ((E_sep - y[0]) / Edot)
        step_size_L = PERCENT_STEP / factor * ((L_sep - y[1]) / Ldot)

        step_size = (step_size_E**2 + step_size_L**2)**0.5

        # copy current values
        temp_y = y + ydot * step_size

        temp_t = t + step_size
        temp_E, temp_L = temp_y[0], temp_y[1]
        temp_stop = ((temp_E - E_sep)**2 + (temp_L - L_sep)**2)**0.5

        return (temp_t, temp_y, temp_stop)

    def finishing_function(self, t: float, y: np.ndarray):
        # Issue with likelihood computation if this step ends at an arbitrary value inside separatrix + DIST_TO_SEPARATRIX.
        #     // To correct for this we self-integrate from the second-to-last point in the integation to
        #     // within the INNER_THRESHOLD with respect to separatrix +  DIST_TO_SEPARATRIX

        if t < self.tmax_dimensionless:  # don't step to the separatrix if we hit the time window
            # update p_sep (fixes part of issue #17)
            p_sep, (p,e,x) = self.get_p_sep(y)

            ydot = np.zeros(self.nparams)
            y_temp = np.zeros(self.nparams)

            # set initial values
            factor = 1.0
            iteration = 0

            while (p - p_sep > DIST_TO_SEPARATRIX + INNER_THRESHOLD):
                # Same function in the integrator
                ydot = self.integrator.get_derivatives(y)
                t_temp, y_temp, temp_stop = self.end_stepper(t, y, ydot, factor)

                if temp_stop > 0:
                    # update points
                    p_sep_temp, (p_temp,etemp,xtemp) = self.get_p_sep(y_temp)
                    if (p_temp - p_sep_temp < DIST_TO_SEPARATRIX):
                        factor *=2
                    else:
                        t = t_temp
                        y[:] = y_temp[:]
                        p_sep, (p,e,x) = p_sep_temp, (p_temp,etemp,xtemp)
                else:
                    # all variables stay the same
                    # decrease step
                    factor *= 2.0

                iteration += 1

                if iteration > MAX_ITER:
                    raise ValueError(
                        "Could not find workable step size in finishing function."
                    )
            # print(p - p_sep , DIST_TO_SEPARATRIX + INNER_THRESHOLD, temp_stop)
            self.save_point(t * self.Msec, y)