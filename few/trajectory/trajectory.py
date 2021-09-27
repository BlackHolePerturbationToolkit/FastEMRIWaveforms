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
from pyInspiral import pyInspiralGenerator

# Python imports
from few.utils.baseclasses import TrajectoryBase
from few.utils.utility import check_for_file_download, get_ode_function_options
from few.utils.constants import *
from few.utils.citations import *


# get path to this file
dir_path = os.path.dirname(os.path.realpath(__file__))


class EMRIInspiral(TrajectoryBase):
    """Pn5-based trajectory module.

    This module implements a Pn5-based trajectory by integrating with an
    RK8 integrator. It can also adjust the output time-spacing using
    cubic spline interpolation. Additionally, it gives the option for
    a dense trajectory, which integrates the trajectory at the user-defined
    timestep.

    # TODO: transfer sanity checks
    # TODO: update

    The trajectory calculations are based on
    `Fujita & Shibata 2020<https://arxiv.org/abs/2008.13554>`_.

    args:
        func (str): Function name for the ode to use in the integration. To get
            the options for this argument, use :func:`few.utils.utility.get_ode_function_options`.
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

    """

    def __init__(self, func, *args, enforce_schwarz_sep=False, **kwargs):

        TrajectoryBase.__init__(self, *args, **kwargs)

        ode_info = get_ode_function_options()

        if func not in ode_info:
            raise ValueError(f"func not available. Options are {list(ode_info.keys())}.")

        self.enforce_schwarz_sep = enforce_schwarz_sep

        for key, item in ode_info[func].items():
            setattr(self, key, item)

        self.inspiral_generator = pyInspiralGenerator(func, enforce_schwarz_sep, self.num_add_args, self.convert_Y)

        self.func = func

        self.specific_kwarg_keys = [
            "T",
            "dt",
            "err",
            "DENSE_STEPPING",
            "max_init_len",
            "use_rk4",
        ]

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
        # TODO: adjust this properly
        return (
            larger_few_citation
            + few_citation
            + few_software_citation
            + Pn5_citation
            + kerr_separatrix_citation
        )

    def get_inspiral(
        self,
        M,
        mu,
        a,
        p0,
        e0,
        x0,
        *args,
        Phi_phi0=0.0,
        Phi_theta0=0.0,
        Phi_r0=0.0,
        **kwargs
    ):
        """Generate the inspiral.

        # TODO: fix
        This is the function for calling the creation of the Pn5-based
        trajectory. Inputs define the output time spacing. This class can be
        used on its own. However, it is generally accessed through the __call__
        method associated with its base class:
        (:class:`few.utils.baseclasses.TrajectoryBase`). See its documentation
        for information on a more flexible interface to the trajectory modules.

        **Please note:** the 5PN trajectory and AAK waveform take the parameter
        :math:`Y\equiv\cos{\iota}=L/\sqrt{L^2 + Q}` rather than :math:`x_I` as is accepted
        for relativistic waveforms and in the generic waveform interface discussed above.
        The generic waveform interface directly converts :math:`x_I` to :math:`Y`.

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
            tuple: Tuple of (t, p, e, x, Phi_phi, Phi_r, Phi_theta).

        """

        fill_value = 1e-6

        if self.background == "Schwarzschild":
            a = 0.0
        elif a < fill_value:
            warnings.warn(
                "Our model with spin breaks near a = 0. Adjusting to a = 1e-6.".format(
                    fill_value
                )
            )
            a = fill_value

        if self.equatorial:
            x0 = 1.0

        if self.circular:
            e0 = 0.0

        # transfer kwargs from parent class
        temp_kwargs = {key: kwargs[key] for key in self.specific_kwarg_keys}

        args_in = np.asarray(args)

        # correct for issue in Cython pass
        if len(args_in) == 0:
            args_in = np.array([0.0])

        # this will return in coordinate time
        t, p, e, x, Phi_phi, Phi_theta, Phi_r = self.inspiral_generator(
            M, mu, a, p0, e0, x0, Phi_phi0, Phi_theta0, Phi_r0, args_in, **temp_kwargs
        )
        return (t, p, e, x, Phi_phi, Phi_theta, Phi_r)
