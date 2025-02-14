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


import numpy as np

# Python imports
from .base import TrajectoryBase
from ..utils.utility import (
    ELQ_to_pex,
    get_kerr_geo_constants_of_motion,
)
from ..utils.pn_map import Y_to_xI
from ..utils.citations import REFERENCE

from .integrate import get_integrator

from typing import Type, Union
from .ode.base import ODEBase

from few.utils.globals import get_logger


class EMRIInspiral(TrajectoryBase):
    """EMRI trajectory module.

    This module implements generic trajectories by integrating with a
    DOP853 Runge-Kutta integrator (see :mod:`few.trajectory.integrate`).
    Both adaptive (default) and fixed timesteps are supported. For an adaptive
    integration, a continuous solution is generated that can be evaluated
    at any point in time.

    The trajectory operates on generic ODE functions that are defined in
    :mod:`few.trajectory.ode`. Flux-based trajectories (which interpolate data grids)
    can be found in :mod:`few.trajectory.ode.flux`. A generic Post-Newtonian trajectory
    is also provided in :mod:`few.trajectory.ode.pn5`. Users can subclass
    :class:`few.trajectory.ode.base.ODEBase` to define their own ODE functions.
    See the documentation for examples on how to do this.

    args:
        func: Function name for the ode to use in the integration.
            This must be given as a keyword argument, even though it is required. To get inbuilt
            stock options for this argument, check :const:`few.trajectory.ode._STOCK_TRAJECTORY_OPTIONS`.
        integrate_constants_of_motion: If True, the trajectory will integrate the constants of motion (E, L, Q).
            Default is False.
        enforce_schwarz_sep: Enforce the separatrix of Schwarzschild
            spacetime. This can mitigate issues at higher spin and/or higher
            eccentricity where (e.g.) PN approximations become increasingly inaccurate.
            Default is False.
        convert_to_pex: Convert the output from ELQ to pex coordinates (only used if integrating constants of motion).
            Default is True.
        rootfind_separatrix: Finish trajectory by performing a numerical root-finding operation to place the final point
            of the trajectory. If False, performs Euler integration to the final point. Default is True.
        *args: Any arguments for parent
            class :class:`few.trajectory.base.TrajectoryBase`
        **kwargs: Any keyword arguments for the integrator and ODE classes
            :class:`few.trajectory.integrate.Integrate` and
            :class:`few.trajectory.ode.ODEBase`.

    Raises:
        ValueError: :code:`func` kwarg not given or not available.
        ValueError: File necessary for ODE not found.
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
        """class: Integrator class for the trajectory."""

        self.func = self.inspiral_generator.func

        self.specific_kwarg_keys = [
            "T",
            "dt",
            "err",
            "DENSE_STEPPING",
            "buffer_length",
            "integrate_backwards",
        ]
        """dict: Specific keywords that need to transferred to the inspiral function that can be adjusted with each call."""

        self.integrate_constants_of_motion = (
            self.inspiral_generator.integrate_constants_of_motion
        )

        self.convert_to_pex = convert_to_pex

    @classmethod
    def module_references(cls) -> list[REFERENCE]:
        """Return citations related to this module"""
        return [REFERENCE.KERR_SEPARATRIX] + super(
            EMRIInspiral, cls
        ).module_references()

    @property
    def trajectory(self):
        return self.inspiral_generator.trajectory

    @property
    def integrator_spline_t(self):
        return self.inspiral_generator.integrator_t_cache

    @property
    def integrator_spline_coeff(self):
        return self.inspiral_generator.integrator_spline_coeff

    @property
    def integrator_spline_phase_coeff(self):
        return (
            self.inspiral_generator.integrator_spline_coeff[:, 3:6]
            / self.inspiral_generator.epsilon
        )

    def get_inspiral(
        self,
        M: float,
        mu: float,
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
        r"""Generate the inspiral.

        This is the function for calling the creation of the trajectory.
        Inputs define the output time spacing.

        args:
            M: Mass of massive black hole in solar masses.
            mu: Mass of compact object in solar masses.
            a: Dimensionless spin of massive black hole.
            p0: Initial semi-latus rectum in terms units of M (p/M).
            e0: Initial eccentricity (dimensionless).
            x0: Initial :math:`\cos{\iota}`. **Note**: This value is different from :math:`x_I`
            used in the relativistic waveforms.
            *args: Added for flexibility.
            Phi_phi0: Initial phase for :math:`\Phi_\phi`.
                Default is 0.0.
            Phi_theta0: Initial phase for :math:`\Phi_\Theta`.
                Default is 0.0.
            Phi_r0: Initial phase for :math:`\Phi_r`.
                Default is 0.0.
            **kwargs: kwargs passed from parent.

        Returns:
            Tuple of (t, p, e, x, Phi_phi, Phi_theta, Phi_r).

        """

        fill_value = 1e-6

        # fix for specific requirements of different odes
        background = self.inspiral_generator.background
        equatorial = self.inspiral_generator.equatorial
        circular = self.inspiral_generator.circular

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

        # transfer kwargs from parent class
        temp_kwargs = {key: kwargs[key] for key in self.specific_kwarg_keys}
        args_in = np.asarray(args)

        # correct for issue in Cython pass
        if len(args_in) == 0:
            args_in = np.array([0.0])

        # flip initial phases if integrating backwards
        if temp_kwargs["integrate_backwards"]:
            Phi_phi0 = -1 * Phi_phi0
            Phi_theta0 = -1 * Phi_theta0
            Phi_r0 = -1 * Phi_r0

        y0 = np.array(
            [y1, y2, y3, Phi_phi0 * (mu / M), Phi_theta0 * (mu / M), Phi_r0 * (mu / M)]
        )

        # this will return in coordinate time
        out = self.inspiral_generator.run_inspiral(M, mu, a, y0, args_in, **temp_kwargs)
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

    def get_rhs_ode(
        self,
        M: float,
        mu: float,
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

        args:
            M: Mass of massive black hole in solar masses.
            mu: Mass of compact object in solar masses.
            a: Dimensionless spin of massive black hole.
            p0: Initial semi-latus rectum in terms units of M (p/M).
            e0: Initial eccentricity (dimensionless).
            x0: Initial :math:`\cos{\iota}`. **Note**: This value is different from :math:`x_I`
            used in the relativistic waveforms.
            *args: Added for flexibility.
            Phi_phi0: Initial phase for :math:`\Phi_\phi`.
                Default is 0.0.
            Phi_theta0: Initial phase for :math:`\Phi_\Theta`.
                Default is 0.0.
            Phi_r0: Initial phase for :math:`\Phi_r`.
                Default is 0.0.
            **kwargs: kwargs passed from parent.

        Returns:
            Tuple of (t, p, e, x, Phi_phi, Phi_theta, Phi_r).
        """

        fill_value = 1e-6

        # fix for specific requirements of different odes
        background = self.inspiral_generator.background
        equatorial = self.inspiral_generator.equatorial
        circular = self.inspiral_generator.circular

        self.inspiral_generator.func.add_fixed_parameters(M, mu, a, False)

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
