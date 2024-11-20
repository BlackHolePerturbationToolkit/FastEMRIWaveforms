from abc import ABC
from few.utils.citations import *
from few.utils.constants import *
import numpy as np

class TrajectoryBase(ABC):
    """Base class used for trajectory modules.

    This class provides a flexible interface to various trajectory
    implementations. Specific arguments to each trajectory can be found with
    each associated trajectory module discussed below.

    """

    def __init__(self, *args, **kwargs):
        pass

    @property
    def citation(self):
        """Return citation for this class"""
        return larger_few_citation + few_citation + few_software_citation

    @classmethod
    def get_inspiral(self, *args, **kwargs):
        """Inspiral Generator

        @classmethod that requires a child class to have a get_inspiral method.

        returns:
            2D double np.ndarray: t, p, e, Phi_phi, Phi_r, flux with shape:
                (params, traj length).

        raises:
            NotImplementedError: The child class does not have this method.

        """
        raise NotImplementedError

    def __call__(
        self,
        *args,
        in_coordinate_time=True,
        dt=10.0,
        T=1.0,
        new_t=None,
        spline_kwargs={},
        DENSE_STEPPING=0,
        max_init_len=1000,
        upsample=False,
        err=1e-11,
        fix_t=False,
        integrate_backwards=False,
        **kwargs,
    ):
        """Call function for trajectory interface.

        This is the function for calling the creation of the
        trajectory. Inputs define the output time spacing.

        args:
            *args (list): Input of variable number of arguments specific to the
                inspiral model (see the trajectory class' `get_inspiral` method).
                **Important Note**: M must be the first parameter of any model
                that uses this base class.
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
            err (double, optional): Tolerance for integrator. Default is 1e-10.
                Decreasing this parameter will give more steps over the
                trajectory, but if it is too small, memory issues will occur as
                the trajectory length will blow up. We recommend not adjusting
                this parameter.
            fix_T (bool, optional): If upsampling, this will affect excess
                points in the t array. If True, it will shave any excess on the
                trajectories arrays where the time is greater than the overall
                time of the trajectory requested.
            **kwargs (dict, optional): kwargs passed to trajectory module.
                Default is {}.

        Returns:
            tuple: Tuple of (t, p, e, Phi_phi, Phi_r, flux_norm).

        Raises:
            ValueError: If input parameters are not allowed in this model.

        """

        # add call kwargs to kwargs dictionary
        kwargs["dt"] = dt
        kwargs["T"] = T
        kwargs["max_init_len"] = max_init_len
        kwargs["err"] = err
        kwargs["DENSE_STEPPING"] = DENSE_STEPPING
        kwargs["integrate_backwards"] = integrate_backwards

        # convert from years to seconds
        T = T * YRSID_SI

        # inspiral generator that must be added to each trajectory class
        out = self.get_inspiral(*args, **kwargs)

        # t = out[0]
        # params = out[1:]

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
            new_t = np.arange(int(T/dt)) * dt

        # upsample everything
        out = tuple(self.inspiral_generator.eval_integrator_spline(new_t).T)
        
        return (new_t,) + out