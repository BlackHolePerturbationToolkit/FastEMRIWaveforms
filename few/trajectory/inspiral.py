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
from scipy.interpolate import CubicSpline, make_interp_spline
from scipy.integrate import cumulative_simpson

# Cython/C++ imports
from pyInspiral import pyInspiralGenerator, pyDerivative

# Python imports
from few.utils.baseclasses import TrajectoryBase
from few.utils.utility import check_for_file_download, get_ode_function_options, ELQ_to_pex, get_kerr_geo_constants_of_motion, get_fundamental_frequencies, Y_to_xI
from few.utils.constants import *
from few.utils.citations import *

from .integrate import APEXIntegrate, AELQIntegrate, get_integrator


# get path to this file
dir_path = os.path.dirname(os.path.realpath(__file__))


class EMRIInspiral(TrajectoryBase):
    """EMRI trajectory module.

    This module implements generic trajectories by integrating with an
    RK8 integrator. It can also adjust the output time-spacing using
    cubic spline interpolation. Additionally, it gives the option for
    a dense trajectory, which integrates the trajectory at the user-defined
    timestep.

    The trajectory operates on generic ODE functions that are defined in
    :code:`src/ode_base.cc` and :code:`include/ode_base.hh`. The ODE can be either
    a function or a C++ class that has constructor function, destructor function,
    and function for get the derivatives called :code:`deriv_func`. Implemented examples
    for the Schwarzchild eccentric flux-based trajectory (ODE class) and
    the 5PN trajecotry (ODE function) can be found in the :code:`ode_base` files.
    The ODE as a function or the :code:`deriv_func` method must take an exact
    set of arguments::

        __deriv__
        func(double* pdot, double* edot, double* xdot,
             double* Omega_phi, double* Omega_theta, double* Omega_r,
             double epsilon, double a, double p, double e, double Y, double* additional_args)

    :code:`pdot, edot, xdot, Omega_phi, Omega_theta`
    The :code:`__deriv__` decorator let's the installer know which functions are actually the
    derivative functions, in case there are other auxillary functions in the file.
    The user then uses :code:`#define` lines to indicate to the code limitations
    or settings specific to that ODE. See the tutorial documentation for more detail.


    args:
        func (str): Function name for the ode to use in the integration.
            This must be given as a keyword argument, even though it is required. To get
            the options for this argument, use :func:`few.utils.utility.get_ode_function_options`.
            Stock options include :code:`"SchwarzEccFlux"` and :code:`"pn5"`.
        enforce_schwarz_sep (bool, optional): Enforce the separatrix of Schwarzschild
            spacetime. This helps to midigate issues at higher spin and/or higher
            eccentricity where the PN approximations are more likely to fail.
            Default is ``False``.
        *args (list): Any arguments for parent
            class :class:`few.utils.baseclasses.TrajectoryBase` or
            class :class:`few.utils.baseclasses.Pn5AAK`.
        **kwargs (dict): Any keyword arguments for parent
            class :class:`few.utils.baseclasses.TrajectoryBase` or
            class :class:`few.utils.baseclasses.Pn5AAK`.

    attributes:
        num_add_args (int): Number of additional arguments for the ODE function.
        background (str): "Either Schwarzschild" or "Kerr".
        equatorial (bool): True if equatorial orbit.
        circular (bool): True if circular orbit.
        convert_Y (bool): If the ODE is integrated in :math:`Y` rather than
            :math:`x_I`.
        files (list): List of files necessary for this ODE.
        citations (list): list of additional citations for this ODE.
        enforce_schwarz_sep (bool): Enforce the separatrix of Schwarzschild
            spacetime.
        inspiral_generator (func): Inspiral C/C++ wrapped function.
        func (str): ODE function name.
        specific_kwarg_keys (dict): Specific keywords that need to transferred
            to the inspiral function that can be adjusted with each call.

    Raises:
        ValueError: :code:`func` kwarg not given or not available.
        ValueError: File necessary for ODE not found.
    """

    def __init__(
        self,
        *args,
        func=None,
        enforce_schwarz_sep=False,
        test_new_version=True,
        convert_to_pex=True,
        numerically_integrate_phases=True,
        rootfind_separatrix=True,
        **kwargs,
    ):
        few_dir = dir_path + "/../../"

        if func is None:
            raise ValueError("Must provide func kwarg.")

        TrajectoryBase.__init__(self, *args, **kwargs)

        self.enforce_schwarz_sep = enforce_schwarz_sep
        
        nparams = 6
        self.inspiral_generator = get_integrator(func, nparams, few_dir, rootfind_separatrix=rootfind_separatrix)

        self.func = self.inspiral_generator.func

        self.specific_kwarg_keys = [
            "T",
            "dt",
            "err",
            "DENSE_STEPPING",
            "max_init_len",
            "use_rk4",
            "integrate_backwards"
        ]

        self.get_derivative = pyDerivative(self.func[0], few_dir.encode())

        self.integrate_constants_of_motion = self.inspiral_generator.integrate_constants_of_motion
        self.integrate_ODE_phases = self.inspiral_generator.integrate_ODE_phases
        self.convert_to_pex = convert_to_pex
        self.numerically_integrate_phases = numerically_integrate_phases

    def attributes_EMRIInspiral(self):
        """
        attributes:
            inspiral_generator (obj): C++ class for inspiral trajectory generation.
            specific_kwarg_keys (list): specific kwargs from
                :class:`few.utils.baseclasses.TrajectoryBase` that apply to this
                inspiral generator.
        """

    @property
    def citation(self):
        """Return citation for this class"""
        citations_out = (
            larger_few_citation
            + few_citation
            + few_software_citation
            + kerr_separatrix_citation
        )

        for citation in self.citations:
            citations_out += globals()[citation]
        return citations_out

    def get_inspiral(
        self,
        M,
        mu,
        a,
        y1,
        y2,
        y3,
        *args,
        Phi_phi0=0.0,
        Phi_theta0=0.0,
        Phi_r0=0.0,
        **kwargs,
    ):
        """Generate the inspiral.

        This is the function for calling the creation of the trajectory.
        Inputs define the output time spacing.

        This class can be used on its own. However, it is generally accessed
        through the __call__ method associated with its base class:
        (:class:`few.utils.baseclasses.TrajectoryBase`).

        args:
            M (double): Mass of massive black hole in solar masses.
            mu (double): Mass of compact object in solar masses.
            a (double): Dimensionless spin of massive black hole.
            p0 (double): Initial semi-latus rectum in terms units of M (p/M).
            e0 (double): Initial eccentricity (dimensionless).
            x0 (double): Initial :math:`\cos{\iota}`. **Note**: This value is different from :math:`x_I`
            used in the relativistic waveforms.
            *args (list, placeholder): Added for flexibility.
            Phi_phi0 (double, optional): Initial phase for :math:`\Phi_\phi`.
                Default is 0.0.
            Phi_theta0 (double, optional): Initial phase for :math:`\Phi_\Theta`.
                Default is 0.0.
            Phi_r0 (double, optional): Initial phase for :math:`\Phi_r`.
                Default is 0.0.
            **kwargs (dict, optional): kwargs passed from parent.

        Returns:
            tuple: Tuple of (t, p, e, x, Phi_phi, Phi_theta, Phi_r).

        """

        fill_value = 1e-6

        # fix for specific requirements of different odes
        background = self.inspiral_generator.backgrounds[0]
        equatorial = self.inspiral_generator.equatorial[0]
        circular = self.inspiral_generator.circular[0]

        p0 = y1
        e0 = y2
        x0 = y3

        if background == "Schwarzschild":
            a = 0.0
        elif a < fill_value:
            if background == "Kerr" and not equatorial:
                warnings.warn(
                    "Our model with spin breaks near a = 0. Adjusting to a = 1e-6.".format(
                        fill_value
                    )
                )
                a = fill_value

        if equatorial:
            if abs(x0) != 1:
                raise RuntimeError("Magnitude of orbital inclination cosine x0 needs to be one for equatorial inspiral.")

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

        if self.integrate_ODE_phases:
            y0 = np.array([y1, y2, y3, Phi_phi0*(mu/M), Phi_theta0*(mu/M), Phi_r0*(mu/M)])
        else:
            y0 = np.array([y1, y2, y3])

        # this will return in coordinate time
        out = self.inspiral_generator.run_inspiral(M, mu, a, y0, args_in, **temp_kwargs)
        if self.integrate_constants_of_motion and self.convert_to_pex:
            out_ELQ = out.copy()
            pex = ELQ_to_pex(a, out[:,1].copy(), out[:,2].copy(), out[:,3].copy())
            out[:,1] = pex[0]
            out[:,2] = pex[1]
            if self.inspiral_generator.convert_Y:
                out[:,3] = out_ELQ[:,2] / np.sqrt(out_ELQ[:,2]**2 + out_ELQ[:,3])
            else:
                out[:,3] = pex[2]

        # handle no phase integration in ODE   # TODO move to the integrate class? This is an integration method...
        if not self.integrate_ODE_phases:
            # if performing phase integration numerically, perform it here
            if self.numerically_integrate_phases:
                t = out[:,0]
                ups_p, ups_e, ups_x = out[:,1:4].T
                t_spline = t.copy()

                # either use cached x or map Y to x if required
                if self.inspiral_generator.integrate_constants_of_motion:
                    ups_x_freq = pex[2].copy()
                else:
                    if self.inspiral_generator.convert_Y:
                        ups_x_freq = Y_to_xI(a, ups_p.copy(), ups_e.copy(), ups_x.copy())
                    else:
                        ups_x_freq = ups_x.copy()

                frequencies = np.array(get_fundamental_frequencies(a, ups_p.copy(), ups_e.copy(), ups_x_freq))/(M*MTSUN_SI)
                if temp_kwargs["use_rk4"]:
                    # breakpoint()
                    cs = CubicSpline(t_spline, frequencies.T)
                    phase_array = cs.antiderivative()(t).T
                else:
                    # get derivatives  TODO ELQ
                    pdot = np.asarray([self.inspiral_generator.integrator.get_derivatives(tr_h.copy()) for tr_h in out[:,1:4]])[:,0] / (M*MTSUN_SI)

                    if kwargs['integrate_backwards'] == True: 
                        # TODO: FIX THIS. DOESN'T WORK FOR BACKWARDS INTEGRATION. 
                        cs = CubicSpline(ups_p, (frequencies/pdot/(mu/M)), axis=1)
                        phase_array = cs.antiderivative()(ups_p) # Integrate directly like a savage. Nice magic Christian.    
 
                        # phase_array = np.array([phase_array[i][-1] - phase_array[i] for i in range(len(phase_array))]) # Reshift frequencies
                    else:
                        cs = CubicSpline(-1*ups_p, (frequencies/pdot/(mu/M)), axis=1)
                        phase_array = -1*cs.antiderivative()(-1*ups_p)

                # to check whether we are outside the interpolation range
                # if self.integrate_constants_of_motion:
                    # print("ELQ deriv check")
                    # ELQ_dot  = np.asarray([self.inspiral_generator.integrator.get_derivatives(el.copy()) for el in out_ELQ[:,1:]])
                # to_spline = frequencies /ELQ_dot[:,0] * MTSUN_SI
                # cs_ELQ = CubicSpline(out_ELQ[::-1,1], -to_spline[:,::-1].T)
                # cs_E = CubicSpline(t_spline, out_ELQ[::-1,1])
                # phase_array = cs_ELQ.antiderivative()(cs_E(t)).T * M
                
                # CUMULATIVE SIMPS
                # phase_array = cumulative_simpson(frequencies, x=t_spline, axis=-1, initial=0)  # initial = 0 sets the zero phase
                # phase_array_ELQ = cumulative_simpson(to_spline[:,::-1], x=out_ELQ[::-1,1], axis=-1, initial=0) # initial = 0 sets the zero phase                
                
                # PLOTS
                # import matplotlib.pyplot as plt
                # plt.figure(); plt.plot(out[:,1],out[:,2],'.'); plt.axvline(self.inspiral_generator.get_p_sep(out[-1])[0]+0.1); plt.show()
                # plt.figure(); plt.plot(out[:,1],out[:,2],'.'); plt.plot(ups_p, ups_e); plt.show()
                
                # plt.figure(); plt.semilogy(t,np.abs(1-phase_array[0] / phase_array_ELQ[0]),'.'); plt.show()
                
                # then we add on the initial phases to all phase values
                phase_array += np.array([Phi_phi0, Phi_theta0, Phi_r0])[:,None]

                out = np.vstack((t_spline, ups_p, ups_e, ups_x, phase_array)).T
            # otherwise just return empty arrays for the phases
            else:
                phase_array = np.zeros((out.shape[0],3))
                out = np.hstack((out, phase_array))

            return tuple(out.T.copy())
        else:
            t, p, e, x, Phi_phi, Phi_theta, Phi_r = out.T.copy()
            return t, p, e, x, Phi_phi/(mu/M), Phi_theta/(mu/M), Phi_r/(mu/M)


    def get_rhs_ode(
        self,
        M,
        mu,
        a,
        y1,
        y2,
        y3,
        *args,
        Phi_phi0=0.0,
        Phi_theta0=0.0,
        Phi_r0=0.0,
        **kwargs,
    ):
        """Generate the right hand side of the ordinary differential equation.

        This is the function for calling the creation of the trajectory.
        Inputs define the output time spacing.

        This class can be used on its own. However, it is generally accessed
        through the __call__ method associated with its base class:
        (:class:`few.utils.baseclasses.TrajectoryBase`).

        args:
            M (double): Mass of massive black hole in solar masses.
            mu (double): Mass of compact object in solar masses.
            a (double): Dimensionless spin of massive black hole.
            p0 (double): Initial semi-latus rectum in terms units of M (p/M).
            e0 (double): Initial eccentricity (dimensionless).
            x0 (double): Initial :math:`\cos{\iota}`. **Note**: This value is different from :math:`x_I`
            used in the relativistic waveforms.
            *args (list, placeholder): Added for flexibility.
            Phi_phi0 (double, optional): Initial phase for :math:`\Phi_\phi`.
                Default is 0.0.
            Phi_theta0 (double, optional): Initial phase for :math:`\Phi_\Theta`.
                Default is 0.0.
            Phi_r0 (double, optional): Initial phase for :math:`\Phi_r`.
                Default is 0.0.
            **kwargs (dict, optional): kwargs passed from parent.

        Returns:
            tuple: Tuple of (t, p, e, x, Phi_phi, Phi_theta, Phi_r).

        """

        fill_value = 1e-6

        # fix for specific requirements of different odes
        background = self.inspiral_generator.backgrounds[0]
        equatorial = self.inspiral_generator.equatorial[0]
        circular = self.inspiral_generator.circular[0]

        p0 = y1
        e0 = y2
        x0 = y3

        if background == "Schwarzschild":
            a = 0.0
        elif a < fill_value:
            if background == "Kerr" and not equatorial:
                warnings.warn(
                    "Our model with spin breaks near a = 0. Adjusting to a = 1e-6.".format(
                        fill_value
                    )
                )
                a = fill_value

        if equatorial:
            if abs(x0) != 1:
                raise RuntimeError("Magnitude of orbital inclination cosine x0 needs to be one for equatorial inspiral.")

        if x0 == -1:
            Phi_phi0 = -1 * Phi_phi0  # flip initial azimuthal phase for retrograde

        if circular:
            e0 = 0.0

        # if integrating constants of motion, convert from pex to ELQ now
        if self.integrate_constants_of_motion:
            if self.inspiral_generator.convert_Y:
                x0 = Y_to_xI(a, p0, e0, x0)
            y1, y2, y3 = get_kerr_geo_constants_of_motion(a, p0, e0, x0)

        args_in = np.asarray(args)

        # correct for issue in Cython pass
        if len(args_in) == 0:
            args_in = np.array([0.0])

        if self.integrate_ODE_phases:
            y0 = np.array([y1, y2, y3, Phi_phi0, Phi_theta0, Phi_r0])
        else:
            y0 = np.array([y1, y2, y3])
   
        # this will return the derivatives in coordinate time
        out  = self.get_derivative(mu/M, a, y0, args_in)
        
        return out
