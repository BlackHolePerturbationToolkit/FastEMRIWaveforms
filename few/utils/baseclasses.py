from abc import ABC
import warnings

try:
    import cupy as xp
except:
    import numpy as xp

import numpy as np

# TODO: get bounds on p


class SchwarzschildEccentric(ABC):
    """Base class for Schwarzschild eccentric waveforms.

    This class creates shared traits between different implementations of the
    same model. Particularly, this class includes descriptive traits as well as
    the sanity check class method that should be used in all implementations of
    this model. This method can be overwritten if necessary. Here we describe
    the overall qualities of this base class.


    In this limit, Eq. :eq:`emri_wave_eq` is reduced to the equatortial plane
    with no spin. Therefore, we only concerned with :math:`(l,m,n)` indices and
    the parameters :math:`(p,e)` because :math:`k=a=\iota=0`. Therefore, in this
    model we calculate :math:`A_{lmn}` and :math:`\Phi_{mn}=m\Phi_\phi+n\Phi_r`.
    This also allows us to use -2 spin-weighted spherical harmonics
    (:math:`(s=-2)Y_{l,m}`) in place of the more generic angular function from
    Eq. :eq:`emri_wave_eq`.

    :math:`l` ranges from 2 to 10; :math:`m` from :math:`-l` to :math:`l`;
    and :math:`n` from -30 to 30. This is for Schwarzschild eccentric.
    The model validity ranges from (TODO: add limits).

    args:
        use_gpu (bool, optional): If True, will allocate arrays on the GPU.
            Default is False.

    attributes:
        xp (module): numpy or cupy based on hardware chosen.
        background (str): Spacetime background for this model.
        descriptor (str): Short description for model validity.
        num_modes, num_teuk_modes (int): Total number of Tuekolsky modes
            in the model.
        lmax, nmax (int): Maximum :math:`l`, :math:`n`  values
        ndim (int): Dimensionality in terms of orbital parameters and phases.

    """

    def __init__(self, use_gpu=False, **kwargs):

        if use_gpu is True:
            self.xp = xp
        else:
            self.xp = np
        self.background = "Schwarzschild"
        self.descriptor = "eccentric"

        self.num_teuk_modes = 3843
        self.num_modes = 3843

        self.lmax = 10
        self.nmax = 30

        self.ndim = 2

    def sanity_check_traj(self, p, e):
        """Sanity check on parameters output from thte trajectory module.

        Make sure parameters are within allowable ranges.

        args:
            p (1D np.ndarray): Array of semi-latus rectum values produced by
                the trajectory module.
            e (1D np.ndarray): Array of eccentricity values produced by
                the trajectory module.

        Raises:
            ValueError: If any of the trajectory points are not allowed.
            UserWarning: If any points in the trajectory are allowable,
                but outside calibration region.

        """
        if e[-1] > 0.5:
            warnings.UserWarning(
                "Plunge (or final) eccentricity value above 0.5 is outside of calibration for this model."
            )

        if self.xp.any(e < 0.0):
            raise ValueError("Members of e array are less than zero.")

        if self.xp.any(p < 0.0):
            raise ValueError("Members of p array are less than zero.")

    def sanity_check_init(self, p0, e0, M, mu, theta, phi):
        """Sanity check initial parameters.

        Make sure parameters are within allowable ranges.

        args:
            p0 (double): Initial semilatus rectum in units of M. TODO: Fix this. :math:`(\leq e0\leq0.7)`
            e0 (double): Initial eccentricity :math:`(0\leq e0\leq0.7)`
            M (double): Massive black hole mass in solar masses.
            mu (double): compact object mass in solar masses.
            theta (double): Polar viewing angle.
            phi (double): Azimuthal viewing angle.

        Raises:
            ValueError: If any of the parameters are not allowed.

        """

        # TODO: add stuff
        if e0 > 0.7:
            raise ValueError(
                "Initial eccentricity above 0.7 not allowed. (e0={})".format(e0)
            )

        if e0 < 0.0:
            raise ValueError(
                "Initial eccentricity below 0.0 not physical. (e0={})".format(e0)
            )

        if mu / M > 1e-4:
            warnings.UserWarning(
                "Mass ratio is outside of generally accepted range for an extreme mass ratio (1e-4). (q={})".format(
                    mu / M
                )
            )


class TrajectoryBase(ABC):
    """Base class used for trajectory modules.

    This class provides a flexible interface to various trajectory
    implementations. Specific arguments to each trajectory can be found with
    each associated trajectory module discussed below.

    """

    def __init__(self, *args, **kwargs):
        pass

    @classmethod
    def get_inspiral(self, *args, **kwargs):
        raise NotImplementedError

    def __call__(
        self,
        *args,
        in_coordinate_time=True,
        dt=-1,
        T=1.0,
        new_t=None,
        spline_kwargs={},
        max_init_len=1000,
        upsample=False,
        step_eps=1e-11,
    ):
        """Call function for trajectory interface.

        This is the function for calling the creation of the
        trajectory. Inputs define the output time spacing.

        args:
            *args (list): Input of variable number of arguments specific to the
                inspiral model (see the trajectory class' `get_inspiral` method).
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

        kwargs["dt"] = dt
        kwargs["T"] = (T,)
        kwargs["max_init_len"] = 1000
        kwargs["err"] = err
        kwargs["DENSE_STEPPING"] = "DENSE_STEPPING"

        out = self.get_inspiral(*args, **kwargs)

        t = out[0]
        params = out[1:]

        if in_coordinate_time is False:
            Msec = M * MTSUN_SI
            t = t / Msec

        if not upsample:
            return (t,) + params

        splines = [CubicSpline(t, temp, **spline_kwargs) for temp in list(params)]

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

        out = tuple([spl(new_t) for spl in splines])
        return (new_t,) + out


class SummationBase(ABC):
    """Base class used for summation modules.

    This class provides a common flexible interface to various summation
    implementations. Specific arguments to each summation module can be found
    with each associated module discussed below.

    args:
        pad_output (bool, optional): Add zero padding to the waveform for time
            between plunge and observation time. Default is False.

    attributes:
        waveform (1D complex128 np.ndarray): Complex waveform given by
            :math:`h_+ + i*h_x`.

    """

    def __init__(self, *args, pad_output=False, **kwargs):
        self.pad_output = pad_output

    @classmethod
    def sum(self, *args, **kwargs):
        raise NotImplementedError

    def __call__(self, t, teuk_modes, ylms, dt, T, *args, **kwargs):
        """Common call function for summation modules.

        Provides a common interface for summation modules. It can adjust for
        more dimensions in a model.

        args:
            t (1D double xp.ndarray): Array of t values.
            teuk_modes (2D double xp.array): Array of complex amplitudes.
                Shape: (len(t), num_teuk_modes).
            ylms (1D complex128 xp.ndarray): Array of ylm values for each mode,
                including m<0. Shape is (num of m==0,) + (num of m>0,)
                + (num of m<0). Number of m<0 and m>0 is the same, but they are
                ordered as (m==0 first then) m>0 then m<0.
            dt (double): Time spacing between observations in seconds (inverse of sampling
                rate).
            T (double): Maximum observing time in years.
            *args (list): This should be a tuple of phases combined with mode
                index arrays. For equatorial, :math:`(\Phi_\phi, \Phi_r, m, n)`.
                For generic, :math:`(\Phi_\phi, \Phi_\Theta, \Phi_r, m, k, n)`.
            **kwargs (dict, placeholder): Added for future flexibility.

        """

        if T < t[-1].item():
            num_pts = int(T / dt)
            num_pts_pad = 0

        else:
            num_pts = int(t[-1] / dt)
            if self.pad_output:
                num_pts_pad = int(T / dt) - num_pts
            else:
                num_pts_pad = 0

        # TODO: make sure num points adjusts for zero padding
        self.num_pts, self.num_pts_pad = num_pts, num_pts_pad
        self.dt = dt
        init_len = len(t)
        num_teuk_modes = teuk_modes.shape[1]

        self.waveform = self.xp.zeros(
            (self.num_pts + self.num_pts_pad,), dtype=self.xp.complex128
        )

        try:
            t = t.get()
        except:
            pass

        args_in = (t, teuk_modes, ylms) + args + (init_len, num_pts, num_teuk_modes, dt)
        self.sum(*args_in)

        return self.waveform
