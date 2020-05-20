import numpy as np
from scipy.interpolate import CubicSpline

from pyFLUX import flux_inspiral, pyFluxCarrier

MTSUN_SI = 4.925491025543575903411922162094833998e-6


class RunFluxInspiral:
    def __init__(self):

        self.flux_carrier = pyFluxCarrier()

    def __call__(
        self,
        M,
        mu,
        p0,
        e0,
        err=1e-10,
        in_coordinate_time=True,
        dt=None,
        T=None,
        new_t=None,
        spline_kwargs={},
    ):
        # this will return in coordinate time
        t, p, e, Phi_phi, Phi_r = flux_inspiral(
            M, mu, p0, e0, self.flux_carrier, err=err
        )

        if in_coordinate_time is False:
            Msec = M * MTSUN_SI
            t = t / Msec

        if dt is None and T is None and new_t is None:
            return (t, p, e, Phi_phi, Phi_r)

        if dt is not None or T is not None or new_t is not None:
            spline_p = CubicSpline(t, p, **spline_kwargs)
            spline_e = CubicSpline(t, e, **spline_kwargs)
            spline_Phi_phi = CubicSpline(t, Phi_phi, **spline_kwargs)
            spline_Phi_r = CubicSpline(t, Phi_r, **spline_kwargs)

        if new_t is not None:
            if isinstance(new_t, np.ndarray) is False:
                raise ValueError("new_t parameter, if provided, must be numpy array.")

            if new_t[-1] > t[-1]:
                print("Warning: new_t array goes beyond generated t array.")

            return (
                new_t,
                spline_p(new_t),
                spline_e(new_t),
                spline_Phi_phi(new_t),
                spline_Phi_r(new_t),
            )

        elif dt is not None or T is not None:
            if dt is None or T is None:
                raise ValueError("If providing dt or T, need to provide both.")

            new_t = np.arange(0.0, T + dt, dt)

            if new_t[-1] > t[-1]:
                print("Warning: new_t array goes beyond generated t array.")

            return (
                new_t,
                spline_p(new_t),
                spline_e(new_t),
                spline_Phi_phi(new_t),
                spline_Phi_r(new_t),
            )

        return (t, p, e, Phi_phi, Phi_r)
