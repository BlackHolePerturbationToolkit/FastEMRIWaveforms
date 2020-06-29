import numpy as np
from scipy.interpolate import CubicSpline

from pyFLUX import flux_inspiral, pyFluxCarrier

MTSUN_SI = 4.925491025543575903411922162094833998e-6


class RunFluxInspiral:
    def __init__(self):
        """Flux-based trajectory module.

        This module implements a flux-based trajectory by integrating with an
        RK8 integrator. It can also adjust the output time-spacing using
        cubic spline interpolation. Additionally, it gives the option for
        a dense trajectory, which integrates the trajectory at the user-defined
        timestep.

        """

        self.flux_carrier = pyFluxCarrier()

    def __call__(
        self,
        M,
        mu,
        p0,
        e0,
        err=1e-10,
        in_coordinate_time=True,
        dt=-1,
        T=1.0,
        new_t=None,
        spline_kwargs={},
        DENSE_STEPPING=0,
        max_init_len=1000,
        upsample=False,
        step_eps=1e-11,
    ):
        """Call function for this class.

        This is the function for calling the creation of the flux-based
        trajectory. Inputs define the output time spacing.

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
            in_coordinate_time (bool, optional): If True, the trajectory will be
                outputted in coordinate time. If False, the trajectory will be
                outputted in units of M. Default is True.
            dt (double, optional): Time step for output waveform in seconds. Also sets
                initial step for integrator. Default is 10.0.
            T (double, optional): Total observation time in years. Sets the maximum time
                for the integrator to run. Default is 1.0.
            new_t (1D np.ndarray, optional): If given, this represents the final
                time array at which the trajectory is analyzed. This is
                performed by using a cubic spline on the integrator output.
                Default is None.
            spline_kwargs (dict, optional): If using upsampling, spline_kwargs
                provides the kwargs desired for scipy.interpolate.CubicSpline.
                Default is {}.
            DENSE_STEPPING (int, optional): If 1, the trajectory used in the
                integrator will be densely stepped at steps of :obj:`dt`. If 0,
                the integrator will determine its stepping. Default is 0.
            max_init_len (int, optional): Sets the allocation of memory for
                trajectory parameters. This should be the maximum length
                expected for a trajectory. Trajectories with default settings
                will be ~100 points. Default is 1000.
            upsample (bool, optional): If True, upsample, with a cubic spline,
                the trajectories from 0 to T in steps of dt. Default is False.

        Returns:
            tuple: Tuple of (t, p, e, Phi_phi, Phi_r, flux_norm).

        Raises:
            ValueError: If input parameters are not allowed in this model.

        """

        if e0 > 0.7:
            raise ValueError("e0 = {} not allowed. e0 must be <= 0.7.".format(e0))

        if p0 > 18.0:
            raise ValueError("p0 = {} not allowed. p0 must be <= 18.0.".format(p0))

        # this will return in coordinate time
        t, p, e, Phi_phi, Phi_r, amp_norm = flux_inspiral(
            M,
            mu,
            p0,
            e0,
            self.flux_carrier,
            tmax=T,
            dt=dt,
            err=err,
            DENSE_STEPPING=DENSE_STEPPING,
            max_init_len=max_init_len,
            step_eps=step_eps,
        )

        if in_coordinate_time is False:
            Msec = M * MTSUN_SI
            t = t / Msec

        if not upsample:
            return (t, p, e, Phi_phi, Phi_r, amp_norm)

        spline_p = CubicSpline(t, p, **spline_kwargs)
        spline_e = CubicSpline(t, e, **spline_kwargs)
        spline_Phi_phi = CubicSpline(t, Phi_phi, **spline_kwargs)
        spline_Phi_r = CubicSpline(t, Phi_r, **spline_kwargs)
        spline_amp_norm = CubicSpline(t, amp_norm, **spline_kwargs)

        if new_t is not None:
            if isinstance(new_t, np.ndarray) is False:
                raise ValueError("new_t parameter, if provided, must be numpy array.")

        elif dt != -1:
            new_t = np.arange(0.0, T + dt, dt)

        else:
            raise ValueError(
                "If upsampling trajectory, must provide dt or new_t array."
            )

        if new_t[-1] > t[-1]:
            print("Warning: new_t array goes beyond generated t array.")
        return (
            new_t,
            spline_p(new_t),
            spline_e(new_t),
            spline_Phi_phi(new_t),
            spline_Phi_r(new_t),
            spline_amp_norm(new_t),
        )


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
