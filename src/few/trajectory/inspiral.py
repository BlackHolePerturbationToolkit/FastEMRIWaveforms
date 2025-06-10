# Pn5-based Generic Kerr trajectory module for Fast EMRI Waveforms

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
            "max_step_size",
        ]
        """dict: Specific keywords that need to transferred to the inspiral function that can be adjusted with each call."""

        self.integrate_constants_of_motion = (
            self.inspiral_generator.integrate_constants_of_motion
        )

        self.convert_to_pex = convert_to_pex

    @classmethod
    def module_references(cls) -> list[REFERENCE]:
        """Return citations related to this module"""
        return [REFERENCE.KERR_SEPARATRIX] + super().module_references()

    @property
    def npoints(self):
        """Number of points in the trajectory."""
        return self.inspiral_generator.npoints

    @property
    def tolerance(self) -> float:
        """Absolute tolerance of the integrator."""
        return self.inspiral_generator.npoints

    @property
    def dense_stepping(self) -> bool:
        """If ``True``, trajectory is using fixed stepping."""
        return self.inspiral_generator.dense_stepping

    @property
    def integrate_backwards(self) -> bool:
        """If ``True``, integrate backwards."""
        return self.inspiral_generator.integrate_backwards

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
        r"""Generate the inspiral.

        This is the function for calling the creation of the trajectory.
        Inputs define the output time spacing.

        args:
            m1: Mass of massive black hole in solar masses.
            m2: Mass of compact object in solar masses.
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

        args:
            m1: Mass of massive black hole in solar masses.
            m2: Mass of compact object in solar masses.
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
    r"""Get frequencies for 0PA phase evolution in Hertz

    arguments:
        m1: Mass of the massive black hole in solar masses.
        m2: Mass of the secondary body in solar masses.
        a: Dimensionless spin of massive
            black hole. If other parameters are arrays and the spin is scalar,
            it will be cast to a 1D array.
        p: Values of separation,
            :math:`p`.
        e: Values of eccentricity,
            :math:`e`.
        x: Values of cosine of the
            inclination, :math:`x=\cos{I}`. Please note this is different from
            :math:`Y=\cos{\iota}`.

    returns:
        Tuple of (OmegaPhi, OmegaTheta, OmegaR). These are 1D arrays or scalar values depending on inputs.

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
