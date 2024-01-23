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

# Cython/C++ imports
from pyInspiral import pyInspiralGenerator, pyDerivative

# Python imports
from ..utils.baseclasses import TrajectoryBase
from ..utils.utility import check_for_file_download, get_ode_function_options
from ..utils.constants import *
from ..utils.citations import *


# get path to this file
dir_path = os.path.dirname(os.path.realpath(__file__))

from ..utils.utility import get_separatrix


DIST_TO_SEPARATRIX = 0.1
INNER_THRESHOLD = 1e-8
PERCENT_STEP = 0.25
MAX_ITER = 1000

KERR = 1
SCHWARZSCHILD = 2


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
        dt: float = 10.0,
        T: float = 2.0,
        buffer_length: int = 1000,
        num_add_args: int = 0,
    ):
        self.tmax_seconds = T * YRSID_SI
        self.dt_seconds = dt
        self.buffer_length = buffer_length
        self.integrator = pyInspiralGenerator(
            nparams,
            num_add_args,
        )
        self.ode_info = ode_info = get_ode_function_options()

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

    @property
    def few_dir(self) -> str:
        return str(self.integrator.few_dir)

    @property
    def backgrounds(self):
        return self.integrator.backgrounds

    @property
    def equatorial(self):
        return self.integrator.equatorial

    @property
    def circular(self):
        return self.integrator.circular

    # @property
    # def func(self) -> str:
    #     return str(self.integrator.func_name)

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
        while t < self.tmax_dimensionless:
            status, t, h = self.integrator.take_step(t, h, y, self.tmax_dimensionless)

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

    def initialize_integrator(self):
        self.integrator.initialize_integrator()
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

    def run_inspiral(self, M, mu, a, y0, additional_args, **kwargs):
        self.moves_check = 0
        self.initialize_integrator()

        # Compute the adimensionalized time steps and max time
        self.tmax_dimensionless = self.tmax_seconds / (M * MTSUN_SI)
        self.dt_dimensionless = self.dt_seconds / (M * MTSUN_SI)
        self.Msec = MTSUN_SI * M
        self.a = a
        assert self.nparams == len(y0)
        assert (self.num_add_args == len(additional_args)) or (
            self.num_add_args == 0 and len(additional_args) == 1
        )

        self.integrator.add_parameters_to_holder(M, mu, a, additional_args)
        t0 = 0.0
        self.integrate(t0, y0)

        self.integrator.destroy_integrator_information()

        return self.trajectory


class APEXIntegrate(Integrate):
    def get_p_sep(self, y: np.ndarray) -> float:
        p = y[0]
        e = y[1]
        x = y[2]

        if self.a == 0.0:
            p_sep = 6.0 + 2.0 * e

        else:
            p_sep = get_separatrix(a, e, x)

        return p_sep

    def action_function(
        self, t: float, y: np.ndarray
    ) -> str:  # Stop the inspiral when close to the separatrix
        p = y[0]

        if p < 10.0 and self.moves_check < 1:
            self.integrator.currently_running_ode_index = 1
            print(f"Switched to index: {self.integrator.currently_running_ode_index}")
            self.moves_check += 1
        elif p < 8.0 and self.moves_check < 2:
            self.integrator.currently_running_ode_index = 0
            print(f"Switched to index: {self.integrator.currently_running_ode_index}")
            self.moves_check += 1

        if p > 7.5:
            return None

        p_sep = self.get_p_sep(y)
        if p - p_sep < DIST_TO_SEPARATRIX + INNER_THRESHOLD:
            return "stop"

        return None

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

    def finishing_function(self, t: float, y: np.ndarray):
        # Issue with likelihood computation if this step ends at an arbitrary value inside separatrix + DIST_TO_SEPARATRIX.
        #     // To correct for this we self-integrate from the second-to-last point in the integation to
        #     // within the INNER_THRESHOLD with respect to separatrix +  DIST_TO_SEPARATRIX

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
            if temp_stop > DIST_TO_SEPARATRIX:
                # update points
                t = t_temp
                y[:] = y_temp[:]
                p_sep = self.get_p_sep(y)
                p = y[0]
            else:
                # all variables stay the same

                # decrease step
                factor *= 0.5

            iteration += 1

            if iteration > MAX_ITER:
                raise ValueError(
                    "Could not find workable step size in finishing function."
                )
