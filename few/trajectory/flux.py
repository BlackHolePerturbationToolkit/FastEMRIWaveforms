"""
This module contains flux-based or adiabatic trajectory modules.
TODO: more info (?), Specific papers to cite
"""

import numpy as np
from scipy.interpolate import CubicSpline

from pyFLUX import flux_inspiral, pyFluxCarrier

from few.utils.baseclasses import TrajectoryBase, SchwarzschildEccentric

MTSUN_SI = 4.925491025543575903411922162094833998e-6

import os

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
        few_dir = dir_path + "/../../"

        # check if necessary files are in the few_dir
        file_list = os.listdir(few_dir + "few/files/")

        if "AmplitudeVectorNorm.dat" not in file_list:
            raise FileNotFoundError(
                "The file AmplitudeVectorNorm.dat did not open sucessfully. Make sure it is located in the proper directory (Path/to/Installation/few/files/)."
            )

        if "FluxNewMinusPNScaled_fixed_y_order.dat" not in file_list:
            raise FileNotFoundError(
                "The file FluxNewMinusPNScaled_fixed_y_order.dat did not open sucessfully. Make sure it is located in the proper directory (Path/to/Installation/few/files/)."
            )

        self.flux_carrier = pyFluxCarrier(few_dir)

        self.specific_kwarg_keys = ["T", "dt", "err", "DENSE_STEPPING", "max_init_len"]

    @property
    def citation(self):
        return """
                @article{Hughes:2005qb,
                    author = "Hughes, Scott A. and Drasco, Steve and Flanagan, Eanna E. and Franklin, Joel",
                    title = "{Gravitational radiation reaction and inspiral waveforms in the adiabatic limit}",
                    eprint = "gr-qc/0504015",
                    archivePrefix = "arXiv",
                    doi = "10.1103/PhysRevLett.94.221101",
                    journal = "Phys. Rev. Lett.",
                    volume = "94",
                    pages = "221101",
                    year = "2005"
                }
            """

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
