# Flux-based Schwarzschild Eccentric trajectory module for Fast EMRI Waveforms

# Copyright (C) 2020 Michael L. Katz, Alvin J.K. Chua, Niels Warburton, Scott A. Hughes
#
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

import numpy as np
from scipy.interpolate import CubicSpline

# Cython/C++ imports
from pyFLUX import pyFluxGenerator

# Python imports
from few.utils.baseclasses import TrajectoryBase, SchwarzschildEccentric
from few.utils.utility import check_for_file_download
from few.utils.constants import *
from few.utils.citations import *


# get path to this file
dir_path = os.path.dirname(os.path.realpath(__file__))


class RunSchwarzEccFluxInspiral(TrajectoryBase, SchwarzschildEccentric):
    """Flux-based trajectory module.

    This module implements a flux-based trajectory by integrating with an
    RK8 integrator. It can also adjust the output time-spacing using
    cubic spline interpolation. Additionally, it gives the option for
    a dense trajectory, which integrates the trajectory at the user-defined
    timestep.

    args:
        *args (list): Any arguments for parent
            class :class:`few.utils.baseclasses.TrajectoryBase` or
            class :class:`few.utils.baseclasses.SchwarzschildEccentric`.
        **kwargs (dict): Any keyword arguments for parent
            class :class:`few.utils.baseclasses.TrajectoryBase` or
            class :class:`few.utils.baseclasses.SchwarzschildEccentric`.

    """

    def __init__(self, *args, **kwargs):

        TrajectoryBase.__init__(self, *args, **kwargs)
        SchwarzschildEccentric.__init__(self, *args, **kwargs)
        few_dir = dir_path + "/../../"

        fp = "AmplitudeVectorNorm.dat"
        check_for_file_download(fp, few_dir)
        fp = "FluxNewMinusPNScaled_fixed_y_order.dat"
        check_for_file_download(fp, few_dir)

        self.flux_generator = pyFluxGenerator(few_dir)

        self.specific_kwarg_keys = [
            "T",
            "dt",
            "err",
            "DENSE_STEPPING",
            "max_init_len",
            "use_rk4",
        ]

    def attributes_RunSchwarzEccFluxInspiral(self):
        """
        attributes:
            flux_generator (obj): C++ class for flux trajectory generation.
            specific_kwarg_keys (list): specific kwargs from
                :class:`few.utils.baseclasses.TrajectoryBase` that apply to this
                inspiral generator.
        """

    @property
    def citation(self):
        """Return citation for this class"""
        return few_citation

    def get_inspiral(self, M, mu, p0, e0, *args, **kwargs):
        """Generate the inspiral.

        This is the function for calling the creation of the flux-based
        trajectory. Inputs define the output time spacing. This class can be
        used on its own. However, it is generally accessed through the __call__
        method associated with its base class:
        (:class:`few.utils.baseclasses.TrajectoryBase`). See its documentation
        for information on a more flexible interface to the trajectory modules.

        args:
            M (double): Mass of massive black hole in solar masses.
            mu (double): Mass of compact object in solar masses.
            p0 (double): Initial semi-latus rectum in terms units of M (p/M).
                This model can handle (p0 <= 18.0).
            e0 (double): Initial eccentricity (dimensionless).
                This model can handle (e0 <= 0.7).

            err (double, optional): Tolerance for integrator. Default is 1e-10.
                Decreasing this parameter will give more steps over the
                trajectory, but if it is too small, memory issues will occur as
                the trajectory length will blow up. We recommend not adjusting
                this parameter.
            *args (list, placeholder): Added for flexibility.
            **kwargs (dict, optional): kwargs passed from parent.
        Returns:
            tuple: Tuple of (t, p, e, Phi_phi, Phi_r, flux_norm).

        """

        # transfer kwargs from parent class
        temp_kwargs = {key: kwargs[key] for key in self.specific_kwarg_keys}

        # this will return in coordinate time
        # must include flux normalization in case normalization is desired
        t, p, e, Phi_phi, Phi_r, amp_norm = self.flux_generator(
            M, mu, p0, e0, **temp_kwargs
        )
        return (t, p, e, Phi_phi, Phi_r, amp_norm)
