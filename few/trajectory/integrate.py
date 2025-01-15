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

# Python imports
from ..utils.utility import (
    check_for_file_download,
    ELQ_to_pex,
    get_kerr_geo_constants_of_motion,
    get_separatrix_interpolant,
)
from ..utils.constants import *
from ..utils.citations import *

from ..utils.baseclasses import ParallelModuleBase

from .ode.base import ODEBase, get_ode_properties
from .ode import _STOCK_TRAJECTORY_OPTIONS

# get path to this file
dir_path = os.path.dirname(os.path.realpath(__file__))

from ..utils.utility import get_separatrix

INNER_THRESHOLD = 1e-8
PERCENT_STEP = 0.25
MAX_ITER = 1000

import time

from .dopr853 import DOPR853

def brentq_array(f, a, b, args, tol):
    a = a.copy()
    b = b.copy()

    # Machine epsilon for double precision
    eps = 2.220446049250313e-16

    converged = np.zeros(len(a), dtype=bool)
    outputs = np.zeros(len(a))

    fa = f(a, args)
    fb = f(b, args)
    
    # Check that f(a) and f(b) have different signs

    # we write the above as a vectorised operation to find indices that have converged
    fa_mask = fa == 0.0
    outputs[fa_mask] = a[fa_mask]
    fb_mask = fb == 0.0
    outputs[fb_mask] = b[fb_mask]
    converged[fa_mask | fb_mask] = 1

    if np.any((fa * fb > 0.0)[~converged]):
        raise ValueError("f(a) and f(b) must have different signs.")

    c = a.copy()
    fc = fa.copy()
    d = b - a
    e = d.copy()

    p = np.empty_like(a)
    q = np.empty_like(a)
    
    max_iter = 100
    iter = 0
    while np.any(~converged):
        if iter > max_iter:
            raise ValueError("Maximum number of iterations exceeded.")
        
        afc_gt_afb_mask = np.abs(fc) < np.abs(fb)

        a[afc_gt_afb_mask] = b[afc_gt_afb_mask]
        b[afc_gt_afb_mask] = c[afc_gt_afb_mask]
        c[afc_gt_afb_mask] = a[afc_gt_afb_mask]
        fa[afc_gt_afb_mask] = fb[afc_gt_afb_mask]
        fb[afc_gt_afb_mask] = fc[afc_gt_afb_mask]
        fc[afc_gt_afb_mask] = fa[afc_gt_afb_mask]

        tol1 = 2.0 * eps * np.abs(b) + 0.5 * tol
        xm = 0.5 * (c - b)
        check_convergence_mask = ((np.abs(xm) <= tol1) | (fb == 0.0)) & ~converged
        outputs[check_convergence_mask] = b[check_convergence_mask]
        converged[check_convergence_mask] = 1

        # Check if bisection is forced
        no_bisect_mask = ((np.abs(e) >= tol1) & (abs(fa) > abs(fb)))

        s = fb / fa

        # linear_interpolation
        a_eq_c_mask = a == c
        linear_mask = no_bisect_mask & a_eq_c_mask
        p[linear_mask] = (2.0 * xm * s)[linear_mask]
        q[linear_mask] = 1.0 - s[linear_mask]

        # Inverse quadratic interpolation
        inverse_quad_mask = no_bisect_mask & ~a_eq_c_mask
        q[inverse_quad_mask] = (fa / fc)[inverse_quad_mask]
        r = fb / fc
        p[inverse_quad_mask] = (s * (2.0 * xm * q * (q - r) - (b - a) * (r - 1.0)))[inverse_quad_mask]
        q[inverse_quad_mask] = ((q - 1.0) * (r - 1.0) * (s - 1.0))[inverse_quad_mask]

        p_pos_mask = (p > 0.0)
        q[p_pos_mask] *= -1
        p[~p_pos_mask] *= -1

        accept_interp_mask = (2.0 * p < 3.0 * xm * q - np.abs(tol1 * q)) & (p < np.abs(0.5 * e * q))
        d[accept_interp_mask] = (p / q)[accept_interp_mask]
        d[~accept_interp_mask] = xm[~accept_interp_mask]
        e[~accept_interp_mask] = d[~accept_interp_mask]

        d[~no_bisect_mask] = xm[~no_bisect_mask]
        e[~no_bisect_mask] = d[~no_bisect_mask]

        a = b.copy()
        fa = fb.copy()
        tolmask = np.abs(d) > tol1
        b[tolmask] += d[tolmask]
        add = tol1 * (2 * (xm > 0) - 1)
        b[~tolmask] += add[~tolmask]

        fb = f(b, args)
        final_upd_mask = fb * fc > 0.0
        c[final_upd_mask] = a[final_upd_mask]
        fc[final_upd_mask] = fa[final_upd_mask]
        d[final_upd_mask] = (b - a)[final_upd_mask]
        e[final_upd_mask] = d[final_upd_mask]

        iter += 1

    return outputs


def get_integrator(func, file_directory=None, integrate_constants_of_motion=False, **kwargs):
    if file_directory is None:
        file_directory = dir_path + "/../../few/files/"
    # ode_info = get_ode_function_options()

    # if isinstance(func, str):
    #     num_add_args = ode_info[func]["num_add_args"]
    #     func = [func]

    # integrator = pyInspiralGenerator(
    #     nparams,
    #     num_add_args,
    # )

    # for func_i in func:
    #     assert isinstance(func_i, str)
    #     if func_i not in ode_info:
    #         raise ValueError(
    #             f"func not available. Options are {list(ode_info.keys())}."
    #         )
    #         # make sure all files needed for the ode specifically are downloaded
    #     for fp in ode_info[func_i]["files"]:
    #         try:
    #             check_for_file_download(fp, file_directory)
    #         except FileNotFoundError:
    #             raise ValueError(
    #                 f"File required for the ODE ({fp}) was not found in the proper folder ({file_directory}) and could not be downloaded."
    #             )

    #     integrator.add_ode(func_i.encode(), file_directory.encode())

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

class Integrate(ParallelModuleBase):
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
        ParallelModuleBase.__init__(self, **kwargs)

        self.buffer_length = buffer_length
        self.file_directory = file_directory

        if isinstance(func, str):
            try:
                func = _STOCK_TRAJECTORY_OPTIONS[func]
            except KeyError:
                raise ValueError(f"The trajectory function {func} could not be found.")

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
        return (self.__class__, (self.base_func, self.integrate_constants_of_motion, self.buffer_length, self.rootfind_separatrix, self.enforce_schwarz_sep, self.file_directory, self.use_gpu))

    @property
    def gpu_capability(self):
        """Confirms GPU capability"""
        return True

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
        self.func(y, out=ydot)
        if self.integrate_backwards:
            ydot *= -1
        return ydot

    # def take_step(
    #     self, t: float, h: float, y: np.ndarray
    # ) -> Tuple[float, float, np.ndarray]:
    #     return self.dopr.take_step_single(t, h, y, self.tmax_dimensionless, None)

    def take_step(
        self, t: np.ndarray, h: np.ndarray, y: np.ndarray
    ) -> np.ndarray:
        return self.dopr.take_step(t, h, y, self.tmax_dimensionless, None)

    def integrate(self, t0: float, y0: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        
        t = t0.copy()
        h = self.dt_dimensionless * self.xp.ones((self.numSys,))
        
        t_prev = t0.copy()
        y_prev = y0.copy()
        h_prev = h.copy()
        y = y0.copy()

        t_finish = t.copy()
        y_finish = y.copy()

        # add the first point
        self.save_point(self.xp.zeros(self.numSys), y)
        self.integrator_t_cache[0] = 0.0

        # explanation of the status codes
        # 0: continue integration
        # -1: reached stopping criterion
        # -2: reached maximum time
        # -3: failed integration
        # >= 1: event boundary reached (not yet implemented)

        # the integrator will advance all trajectories, then check events and the stopping criterion
        # if events are reached by trajectories, the integrator will evaluate them all together
        # integration will continue until all trajectories reach either the stopping criterion or the max time
        # those at the stopping criterion are then evaluated together

        final_time = 0.0
        step_time = 0.0
        prep_time = 0.0
        tmax_mask_time = 0.0
        action_time = 0.0
        sep_mask_time = 0.0
        save_time = 0.0

        while self.xp.any(self.integrate_mask):
            # print(self.traj_step.max())
            st=time.perf_counter()
            try:
                status = self.take_step(
                    t, h, y
                )
            except (
                ValueError
            ) as e:  # an Elliptic function returned a nan and it raised an exception
                # TODO: improve handling of these errors (optional nan handling)
                raise e

            step_time += time.perf_counter() - st
                # t = t_prev
                # y[:] = y_prev[:]
                # h = h_old / 2
                # self.failed_steps[self.integrate_mask] += 1
                # continue
            
            #if any quantity is nan, step those trajectories back and take a smaller step.
            fail_mask = self.xp.any((self.xp.isnan(h)) | (self.xp.isnan(y)), axis=0) | ~status
            
            # move failed points back to their previous values
            t[fail_mask] = t_prev[fail_mask]
            y[:, fail_mask] = y_prev[:, fail_mask]

            # if h returned as nan, we halve the previous step size
            h_nan_mask = self.xp.isnan(h)
            h[h_nan_mask] = h_prev[h_nan_mask] / 2
            # h was adjusted to a new step size, so we update h_prev to match it for the next iteration            
            h_prev[fail_mask] = h[fail_mask]
            self.failed_steps[fail_mask] += 1

            # if all trajectories failed just skip to the start of the next iteration
            # if self.xp.all(fail_mask == self.integrate_mask):
            #     continue
            
            st = time.perf_counter()

            if not self.dopr.fix_step:
                spline_info = self.dopr.prep_evaluate(
                    t_prev, y_prev, h_prev, t, y, None
                )
            else:
                spline_info = None

            prep_time += time.perf_counter() - st

            # if it made it here, reset bad num
            self.failed_steps[~fail_mask] = 0

            st = time.perf_counter()
            # the last point in the trajectory will be at t = tmax
            past_tmax_mask = (t > self.tmax_dimensionless) & (~fail_mask) & self.integrate_mask
            self.status_arr[past_tmax_mask] = -2
            if self.xp.any(past_tmax_mask) and not self.dopr.fix_step:
                step_here = self.traj_step[past_tmax_mask]
                # print(upd_inds)
                # self.xp.put_along_axis(self._integrator_t_cache, upd_inds, (t * self.Msec)[past_tmax_mask], axis=0)
                self.dopr_spline_output[step_here - 1, past_tmax_mask] = spline_info[past_tmax_mask]

                # self.xp.put_along_axis(self.dopr_spline_output, upd_inds[:,:,None,None], spline_info[past_tmax_mask], axis=0)
                self._integrator_t_cache[step_here, past_tmax_mask] = (t * self.Msec)[past_tmax_mask]

                t_finish[past_tmax_mask] = t[past_tmax_mask]
                y_finish[:, past_tmax_mask] = y[:, past_tmax_mask]
            
            tmax_mask_time += time.perf_counter() - st
            # if status is 1 then trajectory went inside the separatrix

            st = time.perf_counter()
            actions = self.action_function(t, y)
            action_time += time.perf_counter() - st
            
            st = time.perf_counter()
            separatrix_mask = ((actions == 1) & ~past_tmax_mask) & (~fail_mask) & self.integrate_mask
            self.status_arr[separatrix_mask] = -1
            if self.xp.any(separatrix_mask) and not self.dopr.fix_step:
                step_here = self.traj_step[separatrix_mask]
                self.dopr_spline_output[step_here - 1, separatrix_mask] = spline_info[separatrix_mask]
                self._integrator_t_cache[step_here, separatrix_mask] = (t * self.Msec)[separatrix_mask]

                # go back to last values
                y_finish[:, separatrix_mask] = y_prev[:, separatrix_mask]
                t_finish[separatrix_mask] = t_prev[separatrix_mask]
            sep_mask_time += time.perf_counter() - st
            save_mask = (~past_tmax_mask & ~separatrix_mask) & ~fail_mask  & self.integrate_mask
            # breakpoint()
            # print('------')
            # print("NEW POINT:", self.traj_step)
            # print("STATUS:", status)
            # print("PAST MAX T:", past_tmax_mask)
            # print("SAVING POINTS:", save_mask)
            # print(t_prev, t, t_finish)
            # print(t, y_prev[0], y[0], fail_mask, status, save_mask)
            st = time.perf_counter()
            if self.xp.any(save_mask):
                self.save_point(
                    t * self.Msec, y, spline_output=spline_info, locs=save_mask,
                )  # adds time in seconds

                # update previous values to current values
                t_prev[save_mask] = t[save_mask]
                y_prev[:,save_mask] = y[:,save_mask]
                h_prev[save_mask] = h[save_mask]
            save_time += time.perf_counter() - st

            # check fails here and stop trajectories that have failed too many times
            self.status_arr[self.failed_steps >= 50] = -3

            # re-check which trajectories are still integrating
            self.integrate_mask = self.status_arr >= 0
            # print("CONTINUE NEXT:", self.integrate_mask, "STATUS:", self.status_arr)
            
        if hasattr(self, "finishing_function"):
            # print("DIST:", y_finish[0] - (self.get_p_sep(y_finish) + self.func.separatrix_buffer_dist))
            # breakpoint()
            st = time.perf_counter()
            self.finishing_function(t_finish, y_finish)
            final_time += time.perf_counter() - st

        # print("STEP TIME:", step_time)
        # print("PREP TIME:", prep_time)
        # print("TMAX MASK TIME:", tmax_mask_time)
        # print("ACTION TIME:", action_time)
        # print("SEP MASK TIME:", sep_mask_time)
        # print("SAVE TIME:", save_time)
        # print("FINAL TIME:", final_time)


    def initialize_integrator(
        self, err=1e-11, DENSE_STEPPING=False, integrate_backwards=False, **kwargs
    ):  
        self.dopr.reset(self.numSys, self.nparams)

        self.dopr.abstol = err
        self.dopr.fix_step = DENSE_STEPPING
        self.integrate_backwards = integrate_backwards

        self.trajectory_arr = self.xp.zeros((self.buffer_length, self.nparams + 1, self.numSys))
        self._integrator_t_cache = self.xp.zeros((self.buffer_length, self.numSys))
        self.dopr_spline_output = self.xp.zeros(
            (self.buffer_length, self.numSys, 6, 8)
        )  # 3 parameters + 3 phases, 8 coefficients
        self.traj_step = self.xp.zeros((self.numSys,), dtype=np.int32)

        # failed steps
        self.failed_steps = self.xp.zeros((self.numSys,), dtype=np.int32)

        # store the status of each ODE here
        self.status_arr = self.xp.zeros((self.numSys,), dtype=np.int32)
        self.integrate_mask = self.xp.ones((self.numSys,), dtype=np.bool)

    @property
    def trajectory(self):
        return self.trajectory_arr[: self.traj_step.max()]
    
    @property
    def integrator_t_cache(self):
        return self._integrator_t_cache[: self.traj_step.max()]

    @property
    def integrator_spline_coeff(self):
        return self.dopr_spline_output[: self.traj_step.max() - 1]

    def save_point(
        self, t: float, y: np.ndarray, spline_output: np.ndarray = None, locs: Optional[np.ndarray[bool]]=None, inds: Optional[np.ndarray[int]]=None 
    ):

        if locs is not None:
            t = t[locs]
            y = y[:, locs]
            if spline_output is not None:
                spline_output = spline_output[locs]

            inds = self.xp.where(locs)[0]
        elif locs is None and inds is None:
            inds = self.xp.arange(self.numSys)

        step_here = self.traj_step[inds]

        self.trajectory_arr[step_here, 0, inds] = t
        self.trajectory_arr[step_here, 1:, inds] = y.T
        # self.xp.put_along_axis(self.trajectory_arr, self.traj_step[inds][:,None,None], self.xp.vstack((t[inds], y[:,inds])), axis=0)

        if spline_output is not None:
            assert self.xp.all(self.traj_step >= 0)
            self._integrator_t_cache[step_here, inds] = t
            self.dopr_spline_output[step_here - 1, inds] = spline_output
            # self.xp.put_along_axis(self._integrator_t_cache, self.traj_step[inds][:,None], t[inds], axis=0)
            # self.xp.put_along_axis(self.dopr_spline_output, self.traj_step[inds][:,None,None,None] - 1, spline_output[inds], axis=0)

        self.traj_step[inds] += 1
        if self.xp.any(self.traj_step >= self.buffer_length):
            # increase by 100
            buffer_increment = 100

            self.trajectory_arr = self.xp.concatenate(
                [self.trajectory_arr, self.xp.zeros((buffer_increment, self.numSys, self.nparams + 1))], axis=0
            )
            if not self.dopr.fix_step:
                self.dopr_spline_output = self.xp.concatenate(
                    [self.dopr_spline_output, self.xp.zeros((buffer_increment, self.numSys, self.nparams, 8))], axis=0
                )
                self._integrator_t_cache = self.xp.concatenate([self._integrator_t_cache, self.xp.zeros(buffer_increment, self.numSys)])
            self.buffer_length += buffer_increment

    def eval_integrator_spline(self, t_new: np.ndarray, inds=None):
        if inds is None:
            inds = self.xp.arange(self.numSys)

        t_old = self.integrator_t_cache[:,inds]

        result = self.xp.zeros((*t_new.shape, self.nparams))
        # t_in_mask = (t_new >= 0.) & (t_new <= t_old.max(0))
        # print(t_in_mask)
        # print(t_in_mask.shape)
        # print(t_new, t_old.max())
        result[:,:] = self.dopr.eval(self.xp.atleast_2d(t_new), t_old, self.integrator_spline_coeff[:,inds], self.traj_step[inds] - 1)

        if not self.generating_trajectory:
            result[:,:,3:6] /= self.epsilon

        # backwards integration requires an additional adjustment to match forwards phase conventions
        if self.integrate_backwards and not self.generating_trajectory:
            result[:,:,3:6] += (self.trajectory[0,4:7] + self.trajectory[self.traj_step,4:7])
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
        
        self.numSys = y0.shape[1]

        self.moves_check = 0
        self.initialize_integrator(**kwargs)
        self.epsilon = mu / M
        # Compute the adimensionalized time steps and max time
        self.tmax_dimensionless = T * YRSID_SI / (M * MTSUN_SI) * self.epsilon
        self.dt_dimensionless = dt / (M * MTSUN_SI) * self.epsilon
        self.Msec = MTSUN_SI * M / self.epsilon
        self.a = a

        assert self.nparams == y0.shape[0]
        
        self.func.add_fixed_parameters(
            M, mu, a, additional_args=additional_args
        )

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
                    f"p_f is too close to separatrix. It must start above p_sep + {self.separatrix_buffer_dist}."
                )

        self.integrate(self.xp.zeros(self.numSys), y0)

        # scale phases here by the mass ratio so the cache is accurate
        self.trajectory_arr[:, 4:7] /= self.epsilon

        # backwards integration requires an additional manipulation to match forwards phase convention
        if self.integrate_backwards:
            self.trajectory_arr[:, 4:7] -= (self.trajectory_arr[0,4:7] + self.trajectory_arr[self.traj_step - 1, 4:7])

        # Restore normal spline behaviour
        self.generating_trajectory = False
        return self.trajectory


class APEXIntegrate(Integrate):
    def get_p_sep(self, y: np.ndarray, inds: Optional[np.ndarray]=None) -> float:
        if inds is None:
            inds = self.xp.arange(self.numSys)

        p = y[0]
        e = y[1]
        x = y[2]

        p_sep = get_separatrix(self.a[inds], e, x)
        return p_sep

    def action_function(
        self, t: float, y: np.ndarray
    ) -> str:  # Stop the inspiral when close to the separatrix
        # TODO: implement a function for checking the outer grid bounds for backwards integration.
        actions_out = self.xp.zeros(self.numSys)

        p = y[0]

        if self.integrate_backwards:
            pass
        else:
            if not self.enforce_schwarz_sep:
                p_sep = self.get_p_sep(y)
            else:
                p_sep = 6 + 2*y[1]
    
            past_sep_mask = p - p_sep < self.separatrix_buffer_dist
            actions_out[past_sep_mask] = 1

            # if p < 10.0 and self.moves_check < 1:
            #     self.integrator.currently_running_ode_index = 1
            #     print(f"Switched to index: {self.integrator.currently_running_ode_index}")
            #     self.moves_check += 1
            # elif p < 8.0 and self.moves_check < 2:
            #     self.integrator.currently_running_ode_index = 0
            #     print(f"Switched to index: {self.integrator.currently_running_ode_index}")
            #     self.moves_check += 1

        return actions_out

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

    def inner_func(self, t_step, args):
        inds = args[0]
        # evaluate the dense output at y_step
        y_step = self.eval_integrator_spline(self.xp.array([t_step,]), inds=inds)[0].T
        # get the separatrix value at this new step
        if not self.enforce_schwarz_sep:
            p_sep = self.get_p_sep(y_step, inds=inds)
        else:
            p_sep = 6 + 2 * y_step[1]

        return y_step[0] - (p_sep + self.separatrix_buffer_dist)  # we want this to go to zero

    def finishing_function(self, t: float, y: np.ndarray):
        # Tune the step-size of a dense integration step such that the trajectory terminates within INNER_THRESHOLD of p_sep + DIST_TO_SEPARATRIX
        if self.dopr.fix_step:
            # no rootfinding performed for the fixed stepper
            return
        
        # advance the step counter by one temporarily to read the last spline value
        self.traj_step += 1

        # mask based on finishing condition
        sep_mask = self.status_arr == -1
        tmax_mask = self.status_arr == -2

        if (
            self.rootfind_separatrix
        ):  # use a root-finder and the full integration routine to tune the finish
            
            # if tmax occurs before the last point, we need to determine if the crossing is before t=tmax
            last_point_above_mask = (self._integrator_t_cache[self.traj_step] > self.tmax_dimensionless * self.Msec) & sep_mask
            if self.xp.any(last_point_above_mask):

                # first, check if the crossing has happened by t=tmax (this is unlikely but can happen)
                y_at_tmax = self.eval_integrator_spline(self.xp.array([self.tmax_dimensionless * self.Msec]))[0, :, last_point_above_mask]

                if not self.enforce_schwarz_sep:
                    p_sep_at_tmax = self.get_p_sep(y_at_tmax)
                else:
                    p_sep_at_tmax = 6 + 2*y_at_tmax[1]

                save_here_inds = last_point_above_mask & (y_at_tmax[0] - (p_sep_at_tmax + self.separatrix_buffer_dist) > 0)

                # the trajectory didnt cross the boundary before t=tmax, so stop at t=tmax.
                # just place the point at y_at_tmax above.
                # we do not pass any spline information here as it has already been computed in the main integrator loop
                self.traj_step[save_here_inds] -= 1  # revert the step counter to place the last (t, y) in the right place (spline info not overwritten)
                self.save_point(
                    self.tmax_dimensionless * self.Msec, y_at_tmax, spline_output=None, inds=save_here_inds
                )
            
                sep_mask[save_here_inds] = False  # remove these points from the separatrix mask as they have now been processed

            # now we perform root-finding for those points that crossed the boundary before t=tmax
            # the trajectory crosses the boundary before t=tmax. Root-find to get the crossing time.
            if self.xp.any(sep_mask):
                # t_out = brentq(
                #     self.inner_func,
                #     (t * self.Msec)[sep_mask],  # lower bound: the current point
                #     self._integrator_t_cache.max(axis=0)[sep_mask],  # upper bound: the knot that passed the boundary
                #     # args=(y_out,),
                #     maxiter=MAX_ITER,
                #     xtol=INNER_THRESHOLD,
                # )
                t_out = brentq_array(
                    self.inner_func,
                    (t * self.Msec)[sep_mask],  # lower bound: the current point
                    self._integrator_t_cache.max(axis=0)[sep_mask],  # upper bound: the knot that passed the boundary,
                    args=(sep_mask,),
                    tol=INNER_THRESHOLD,
                )
                y_out = self.eval_integrator_spline(t_out[None, :], inds=sep_mask)[0]

                self.traj_step[sep_mask] -= 1  # revert the step counter to place the last (t, y) in the right place (spline info not overwritten)
                self.save_point(
                    t_out, y_out.T, spline_output=None, inds=sep_mask
                )

        elif not self.rootfind_separatrix:
            raise NotImplementedError
            self.finishing_function_euler_step(
                t, y
            )  # weird step-halving Euler thing (not yet updated)

        # else:  # If integrator walked past tmax during main loop, place a point at tmax and finish integration
        if self.xp.any(tmax_mask):
            t_out = (self.tmax_dimensionless * self.Msec)[tmax_mask]

            y_finish = self.eval_integrator_spline(t_out[None, :], inds=tmax_mask)[0]

            self.traj_step[tmax_mask] -= 1  # revert the step counter to place the last (t, y) in the right place (spline info not overwritten)
            # we do not pass any spline information here as it has already been computed in the main integrator loop
            self.save_point(
                t_out, y_finish.T, spline_output=None, inds=tmax_mask
            )  # adds time in seconds

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
