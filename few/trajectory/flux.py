"""
This module contains flux-based or adiabatic trajectory modules.
TODO: more info (?), Specific papers to cite
"""

import numpy as np
from scipy.interpolate import CubicSpline

from pyFLUX import flux_inspiral, pyFluxCarrier

from few.utils.baseclasses import TrajectoryBase, SchwarzschildEccentric

MTSUN_SI = 4.925491025543575903411922162094833998e-6


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

    attributes:
        flux carrier (obj): Unaccessible from python. It carries c++ classes
            for the integration.
        specific_kwarg_keys (list): specific kwargs from
            :class:`few.utils.baseclasses.TrajectoryBase` that apply this
            inspiral generator.

    """

    def __init__(self, *args, **kwargs):
        TrajectoryBase.__init__(self, *args, **kwargs)
        SchwarzschildEccentric.__init__(self, *args, **kwargs)
        self.flux_carrier = pyFluxCarrier()

        self.specific_kwarg_keys = [
            "tmax",
            "dt",
            "err",
            "DENSE_STEPPING",
            "max_init_len",
        ]

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
        self.sanity_check_init(p0, e0)

        temp_kwargs = {key: kwargs[key] for key in self.specific_kwarg_keys}

        # this will return in coordinate time
        t, p, e, Phi_phi, Phi_r, amp_norm = flux_inspiral(
            M, mu, p0, e0, self.flux_carrier, **temp_kwargs
        )
        return (t, p, e, Phi_phi, Phi_r, amp_norm)


if __name__ == "__main__":
    flux = RunFluxInspiral()

    M = 1e6
    mu = 1e1
    p0 = 8.0
    e0 = 0.6
    DENSE_STEPPING = 1
    max_init_len = int(1e7)
    step_eps = 1e-11

    check = flux(
        M,
        mu,
        p0,
        e0,
        DENSE_STEPPING=DENSE_STEPPING,
        max_init_len=max_init_len,
        step_eps=step_eps,
    )
    import pdb

    pdb.set_trace()
