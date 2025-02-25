import abc
from few.utils.citations import Citable
from few.utils.constants import YRSID_SI, MTSUN_SI
import numpy as np

from typing import Optional


class TrajectoryBase(Citable, abc.ABC):
    """Base class used for trajectory modules.

    This class provides a flexible interface to various trajectory
    implementations. Specific arguments to each trajectory can be found with
    each associated trajectory module discussed below.

    """

    def __init__(self, *args, **kwargs):
        pass

    @classmethod
    @abc.abstractmethod
    def get_inspiral(self, *args, **kwargs):
        """Inspiral Generator

        @classmethod that requires a child class to have a get_inspiral method.

        raises:
            NotImplementedError: The child class does not have this method.

        """
        raise NotImplementedError

    def __call__(
        self,
        *args,
        in_coordinate_time: bool = True,
        dt: float = 10.0,
        T: float = 1.0,
        new_t: Optional[np.ndarray] = None,
        spline_kwargs: Optional[dict] = None,
        DENSE_STEPPING: bool = False,
        buffer_length: int = 1000,
        upsample: bool = False,
        err: float = 1e-11,
        fix_t: bool = False,
        integrate_backwards: bool = False,
        **kwargs,
    ) -> tuple[np.ndarray]:
        """Call function for trajectory interface.

        This is the function for calling the creation of the
        trajectory. Inputs define the output time spacing.

        args:
            *args: Input of variable number of arguments specific to the
                inspiral model (see the trajectory class' `get_inspiral` method).
                **Important Note**: M must be the first parameter of any model
                that uses this base class.
            in_coordinate_time: If True, the trajectory will be
                outputted in coordinate time. If False, the trajectory will be
                outputted in units of M. Default is True.
            dt: Time step for output waveform in seconds. Also sets
                initial step for integrator. Default is 10.0.
            T: Total observation time in years. Sets the maximum time
                for the integrator to run. Default is 1.0.
            new_t: If given, this represents the final
                time array at which the trajectory is analyzed. This is
                performed by using a cubic spline on the integrator output.
                Default is None.
            spline_kwargs: If using upsampling, spline_kwargs
                provides the kwargs desired for scipy.interpolate.CubicSpline.
                Default is {}.
            DENSE_STEPPING: If True, the trajectory used in the
                integrator will be densely stepped at steps of :obj:`dt`. If False,
                the integrator will determine its stepping. Default is False.
            buffer_length: Sets the allocation of memory for
                trajectory parameters. This should be the maximum length
                expected for a trajectory. If it is reached, output arrays will be
                extended, but this is more expensive than allocating a larger array
                initially. Trajectories with default settings
                will be ~100 points. Default is 1000.
            upsample: If True, upsample, with a cubic spline,
                the trajectories from 0 to T in steps of dt. Default is False.
            err: Tolerance for integrator. Default is 1e-10.
                Decreasing this parameter will give more steps over the
                trajectory, but if it is too small, memory issues will occur as
                the trajectory length will blow up. We recommend not adjusting
                this parameter.
            fix_T: If upsampling, this will affect excess
                points in the t array. If True, it will shave any excess on the
                trajectories arrays where the time is greater than the overall
                time of the trajectory requested.
            **kwargs: kwargs passed to trajectory module.
                Default is {}.

        Returns:
            Tuple of (t, p, e, Phi_phi, Phi_r, flux_norm).

        Raises:
            ValueError: If input parameters are not allowed in this model.

        """
        if spline_kwargs is None:
            spline_kwargs = {}

        # add call kwargs to kwargs dictionary
        kwargs["dt"] = dt
        kwargs["T"] = T
        kwargs["buffer_length"] = buffer_length
        kwargs["err"] = err
        kwargs["DENSE_STEPPING"] = DENSE_STEPPING
        kwargs["integrate_backwards"] = integrate_backwards

        # convert from years to seconds
        T = T * YRSID_SI

        # inspiral generator that must be added to each trajectory class
        out = self.get_inspiral(*args, **kwargs)

        # get time separate from the rest of the params
        t = out[0]
        params = out[1:]

        # convert to dimensionless time
        if in_coordinate_time is False:
            # M must be the first argument
            M = args[0]
            Msec = M * MTSUN_SI
            t = t / Msec

        if not upsample:
            return (t,) + params
        else:
            if DENSE_STEPPING:
                raise NotImplementedError  # TODO: support cubic spline for dense

        if new_t is not None:
            if isinstance(new_t, np.ndarray) is False:
                raise ValueError("new_t parameter, if provided, must be numpy array.")

        else:
            # new time array for upsampling
            new_t = np.arange(int(T / dt)) * dt

        # upsample everything
        upsamp_traj = self.inspiral_generator.eval_integrator_spline(new_t).T
        if fix_t:
            if np.any(upsamp_traj[1] == 0):
                trunc_ind = np.where(upsamp_traj[1] == 0)[0][0]
                upsamp_traj = upsamp_traj[:, :trunc_ind]
                new_t = new_t[:trunc_ind]
        out = tuple(upsamp_traj)

        return (new_t,) + out
