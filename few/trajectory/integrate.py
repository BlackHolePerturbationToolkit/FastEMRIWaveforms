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

import numpy as np
from scipy.interpolate import CubicSpline

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

class Integrate:
    def __init__(self, nparams: int, dt: float=10.0, T: float=1.0, buffer_length: int=1000):
        self.nparams = nparams
        self.integrator = c_integration_class
        self.tmax_seconds = T * YRSID_SI
        self.dt_seconds = dt
        self.buffer_length = buffer_length
    
    def take_step(self, t: float, h: float, y: np.ndarray) -> Tuple(float, float, np.ndarray):
        return self.integrator.take_step(t, h, y)

    def reset_solver(self):
        self.integrator.reset_solver()

        # check for number of tries to fix this
        self.bad_num += 1
        if (self.bad_num >= self.bad_limit):
            raise ValueError("error, reached bad limit.\n")

    def integrate(self, t0: float, y0: np.ndarray) -> Tuple(np.ndarray, np.ndarray):

        t = t0
        h = self.dt_dimensionless

        t_prev = t0
        y_prev = y0.copy()
        y = y0.copy()

        # control it if it keeps returning nans and what not
        bad_num = 0

        while (t < tmax):
            status, t, h = self.take_step(t, h, y)

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
            if (t > tmax)
                break

            # if status is 9 meaning inside the separatrix
            if status == 9:
                stop = True
            else:
                stop = self.stopping_function(y)

            if stop:
                # go back to last values
                y[:] = y_prev[:]
                t = t_prev
                break

            self.save_point(t * Msec, y)  # adds time in seconds

            t_prev = t
            y_prev[:] = y[:] 

        if self.hasattr("finishing_function"):
            self.finishing_function(t, y)

    def initial_integrator(self):
        self.integrator.initialize_integrator()
        self.trajectory = np.zeros((self.buffer_length, self.nparams + 1))
        self.traj_step = 0

    def save_point(self, t: float, y: np.ndarray):
        self.trajectory[self.traj_step, 0] = t
        self.trajectory[self.traj_step, 1:] = y

        self.traj_step += 1
        if self.traj_step >= self.buffer_length:
            # increase by 100
            self.trajectory = np.concatenate([self.trajectory, np.zeros((100, self.nparams + 1))], axis=0)
            self.buffer_length = self.trajectory.shape[0]

    def run_inspiral(self, M, mu, a, y0):
        self.initialize_integrator()

        # Compute the adimensionalized time steps and max time
        self.tmax_dimensionless = self.tmax_seconds / (M * MTSUN_SI)
        self.dt_dimensionless = self.dt_seconds / (M * MTSUN_SI)
        self.Msec = MTSUN_SI * M
        self.a = a

        self.integrator.add_parameters_to_holder(M, mu, a)
        self.integrate()
        
        self.integrator.destroy_integrator_information()
        assert len(y0) == nparams

class APEXIntegrate(Integrate):
    def get_p_sep(self, y: np.ndarray) -> float:
        p = y[0]
        e = y[1]
        x = y[2]
        
        if self.a == 0.0:
            p_sep = 6.0 + 2. * e
        
        else:
            p_sep = get_separatrix(a, e, x)

        return p_sep

    def stopping_function(self, t: float, y: np.ndarray) -> bool:    # Stop the inspiral when close to the separatrix
        p_sep = self.get_p_sep(y)
        p = y[0]
        if (p - p_sep < DIST_TO_SEPARATRIX)
            return true
        else
            return false

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

    def finishing_function(self, t: float, y: np.ndarray)
        # Issue with likelihood computation if this step ends at an arbitrary value inside separatrix + DIST_TO_SEPARATRIX.
        #     // To correct for this we self-integrate from the second-to-last point in the integation to
        #     // within the INNER_THRESHOLD with respect to separatrix +  DIST_TO_SEPARATRIX

        # update p_sep (fixes part of issue #17)
        p_sep = self.get_p_sep(y)
        p = y[0]
        ydot = np.zeros(nparams)
        y_temp = np.zeros(nparams)

        # set initial values
        factor = 1.0
        iteration = 0

        while ((p - p_sep > DIST_TO_SEPARATRIX + INNER_THRESHOLD))
            # Same function in the integrator
            ydot = self.integrator.get_derivatives(y)
            t_temp, y_temp, temp_stop = self.end_stepper(t, y, ydot, factor)
            if (temp_stop > DIST_TO_SEPARATRIX):
                # update points
                t = t_temp
                y[:] = y_temp[:]
            else:
                # all variables stay the same

                # decrease step
                factor *= 0.5

            iteration += 1

            if iteration > MAX_ITER:
                raise ValueError("Could not find workable step size in finishing function.")
