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
        dt=-1,
        T=1.0,
        new_t=None,
        spline_kwargs={},
        DENSE_STEPPING=0,
        max_init_len=1000,
        upsample=False,
    ):
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

    check = flux(
        M, mu, p0, e0, DENSE_STEPPING=DENSE_STEPPING, max_init_len=max_init_len
    )
    import pdb

    pdb.set_trace()
