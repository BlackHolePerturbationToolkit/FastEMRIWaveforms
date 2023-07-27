# Collection of base classes for FastEMRIWaveforms Packages

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

"""
The :code:`few.utils.baseclasses` module contains abstract base classes for the
various modules. When creating new modules, these classes should be used to maintain
a common interface and pass information related to each model.
"""


from abc import ABC
import warnings
import os

import numpy as np
from scipy.interpolate import CubicSpline
from scipy import constants as ct


# try to import cupy
try:
    import cupy as cp

    gpu_available = True

except:
    import numpy as np

    gpu_available = False

# Python imports
from few.utils.constants import *
from few.utils.citations import *


class ParallelModuleBase(ABC):
    """Base class for modules that can use GPUs.

    This class mainly handles setting GPU usage.

    args:
        use_gpu (bool, optional): If True, use GPU resources. Default is False.

    """

    def attributes_ParallelModuleBase(self):
        """
        attributes:
            use_gpu (bool): If True, use GPU.
            xp (obj): Either numpy or CuPy based on gpu preference.

        """
        pass

    def __init__(self, *args, use_gpu=False, **kwargs):
        self.use_gpu = use_gpu

        # checks if gpu capability is available if requested
        self.sanity_check_gpu(use_gpu)

    @classmethod
    @property
    def gpu_capability(self):
        """Indicator if the module has gpu capability"""
        raise NotImplementedError

    @property
    def citation(self):
        """Return citations related to this module"""
        return larger_few_citation + few_citation + few_software_citation

    @classmethod
    def __call__(*args, **kwargs):
        """Method to call waveform model"""
        raise NotImplementedError

    def sanity_check_gpu(self, use_gpu):
        """Check if this class has GPU capability

        If the user is requesting GPU usage, this will confirm the class has
        GPU capabilites.

        Args:
            use_gpu (bool): If True, the user is requesting GPU usage.

        Raises:
            ValueError: The user is requesting GPU usage, but this class does
                not have that capability.

        """
        if (self.gpu_capability is False or gpu_available is False) and use_gpu is True:
            if self.gpu_capability is False:
                raise ValueError(
                    "The use_gpu kwarg is True, but this class does not have GPU capabilites."
                )
            else:
                raise ValueError("Either a GPU and/or CuPy is not available.")

    def adjust_gpu_usage(self, use_gpu, kwargs):
        """Adjust all inputs for gpu usage

        If user wants to use gpu, it will change all :code:`kwargs` in
        so that :code:`use_gpu=True`.

        args:
            use_gpu (bool): If True, use gpu resources.
            kwargs (list of dicts or dict): List of kwargs dictionaries or
                single dictionary for each constituent class in the
                waveform generator.

        """

        if use_gpu:
            if isinstance(kwargs, list):
                for i, kwargs_i in enumerate(kwargs):
                    kwargs[i]["use_gpu"] = use_gpu
            else:
                kwargs["use_gpu"] = use_gpu

        return kwargs


class SchwarzschildEccentric(ParallelModuleBase, ABC):
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
    The model validity ranges from :math:`0.1 \leq e_0 \leq 0.7` and
    :math:`10 \leq p_0 \leq 16 + 2*e_0`. The user can start at any :math:`p` and
    :math:`e` combination that exists under the :math:`p_0=10, e_0=0.7`
    trajectory within those bounds (and is outside of the separatrix). **Important Note**: if the trajectory is
    within the bounds listed, but it is above :math:`p_0=10, e_0=0.7` trajectory.,
    the user may not receive an error. See the documentation introduction for
    more information on this.

    args:
        use_gpu (bool, optional): If True, will allocate arrays on the GPU.
            Default is False.

    """

    def attributes_SchwarzschildEccentric(self):
        """
        attributes:
            background (str): Spacetime background for this model.
            descriptor (str): Short description for model validity.
            num_modes, num_teuk_modes (int): Total number of Teukolsky modes
                in the model.
            lmax, nmax (int): Maximum :math:`l`, :math:`n`  values
            ndim (int): Dimensionality in terms of orbital parameters and phases.
            m0sort (1D int xp.ndarray): array of indices to sort accoring to
                :math:`(m=0)` parts first and then :math:`m>0` parts.
            m0mask (1D bool xp.ndarray): Masks values with :math:`m==0`.
            m_zero_up_mask (1D bool xp.ndarray): Masks values with :math:`m<1`.
            l_arr, m_arr, n_arr (1D int xp.ndarray): :math:`(l,m,n)` arrays
                containing indices for each mode.
            lmn_indices (dict): Dictionary mapping a tuple of :math:`(l,m,n)` to
                the respective index in l_arr, m_arr, and n_arr.
            num_m_zero_up (int): Number of modes with :math:`m\geq0`.
            num_m0 (int): Number of modes with :math:`m=0`.
            num_m_1_up (int): Number of modes with :math:`m\geq1`.
            unique_l, unique_m (1D int xp.ndarray): Arrays of unique :math:`l` and
                :math:`m` values.
            inverse_lm (1D int xp.ndarray): Array of indices that expands unique
                :math:`(l, m)` values to the full array of :math:`(l,m,n)` values.
            index_map (dict): Dictionary mapping the location of the `(l,m,n)`
                indices back to there spot in l_arr, m_arr, n_arr.
            special_index_map (dict): Dictionary mapping the location of the `(l,m,n)`
                indices back to there spot in l_arr, m_arr, n_arr. However, this
                maps locations of -m values to +m values.

        """
        pass

    def __init__(self, *args, use_gpu=False, **kwargs):
        ParallelModuleBase.__init__(self, *args, use_gpu=use_gpu, **kwargs)

        if self.use_gpu:
            xp = cp
        else:
            xp = np

        # some descriptive information
        self.background = "Schwarzschild"
        self.descriptor = "eccentric"
        self.frame = "source"

        # set mode index settings
        self.lmax = 10
        self.nmax = 30

        self.ndim = 2

        # fill all lmn mode values
        md = []

        for l in range(2, self.lmax + 1):
            for m in range(0, l + 1):
                for n in range(-self.nmax, self.nmax + 1):
                    md.append([l, m, n])

        # total number of modes in the model
        self.num_modes = self.num_teuk_modes = len(md)

        # mask for m == 0
        m0mask = xp.array(
            [
                m == 0
                for l in range(2, 10 + 1)
                for m in range(0, l + 1)
                for n in range(-30, 30 + 1)
            ]
        )

        # sorts so that order is m=0, m<0, m>0
        self.m0sort = m0sort = xp.concatenate(
            [
                xp.arange(self.num_teuk_modes)[m0mask],
                xp.arange(self.num_teuk_modes)[~m0mask],
            ]
        )

        # sorts the mode indexes
        md = xp.asarray(md).T[:, m0sort].astype(xp.int32)

        # store l m and n values
        self.l_arr, self.m_arr, self.n_arr = md[0], md[1], md[2]

        # adjust with .get method for cupy
        try:
            self.lmn_indices = {tuple(md_i): i for i, md_i in enumerate(md.T.get())}

        except AttributeError:
            self.lmn_indices = {tuple(md_i): i for i, md_i in enumerate(md.T)}

        # store the mask as m != 0 is True
        self.m0mask = self.m_arr != 0

        # number of m >= 0
        self.num_m_zero_up = len(self.m_arr)

        # number of m == 0
        self.num_m0 = len(xp.arange(self.num_teuk_modes)[m0mask])

        # number of m > 0
        self.num_m_1_up = self.num_m_zero_up - self.num_m0

        # create final arrays to include -m modes
        self.l_arr = xp.concatenate([self.l_arr, self.l_arr[self.m0mask]])
        self.m_arr = xp.concatenate([self.m_arr, -self.m_arr[self.m0mask]])
        self.n_arr = xp.concatenate([self.n_arr, self.n_arr[self.m0mask]])

        # mask for m >= 0
        self.m_zero_up_mask = self.m_arr >= 0

        # find unique sets of (l,m)
        # create inverse array to build full (l,m,n) from unique l and m
        # also adjust for cupy
        try:
            temp, self.inverse_lm = np.unique(
                np.asarray([self.l_arr.get(), self.m_arr.get()]).T,
                axis=0,
                return_inverse=True,
            )

        except AttributeError:
            temp, self.inverse_lm = np.unique(
                np.asarray([self.l_arr, self.m_arr]).T, axis=0, return_inverse=True
            )

        # unique values of l and m
        self.unique_l, self.unique_m = xp.asarray(temp).T

        # number of unique values
        self.num_unique_lm = len(self.unique_l)

        # creates special maps to the modes
        self.index_map = {}
        self.special_index_map = {}  # maps the minus m values to positive m
        for i, (l, m, n) in enumerate(zip(self.l_arr, self.m_arr, self.n_arr)):
            try:
                l = l.item()
                m = m.item()
                n = n.item()

            except AttributeError:
                pass

            # regular index to mode tuple
            self.index_map[(l, m, n)] = i

            # special map that gives m < 0 indices as m > 0 indices
            self.special_index_map[(l, m, n)] = (
                i if i < self.num_modes else i - self.num_m_1_up
            )

    @property
    def gpu_capability(self):
        """Confirms GPU capability"""
        return True

    @property
    def citation(self):
        """Return citations of this class"""
        return larger_few_citation + few_citation + few_software_citation

    def sanity_check_viewing_angles(self, theta, phi):
        """Sanity check on viewing angles.

        Make sure parameters are within allowable ranges.

        args:
            theta (double): Polar viewing angle.
            phi (double): Azimuthal viewing angle.

        Returns:
            tuple: (theta, phi). Phi is wrapped.

        Raises:
            ValueError: If any of the angular values are not allowed.

        """
        # if theta < 0.0 or theta > np.pi:
        #    raise ValueError("theta must be between 0 and pi.")

        phi = phi % (2 * np.pi)
        return (theta, phi)

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
            warn: If any points in the trajectory are allowable,
                but outside calibration region.

        """

        if np.any(e < 0.0):
            raise ValueError("Members of e array are less than zero.")

        if np.any(p < 0.0):
            raise ValueError("Members of p array are less than zero.")

    def sanity_check_init(self, M, mu, p0, e0):
        """Sanity check initial parameters.

        Make sure parameters are within allowable ranges.

        args:
            M (double): Massive black hole mass in solar masses.
            mu (double): compact object mass in solar masses.
            p0 (double): Initial semilatus rectum (dimensionless)
                :math:`(10\leq p_0\leq 16 + 2e_0)`. See the documentation for
                more information on :math:`p_0 \leq 10.0`.
            e0 (double): Initial eccentricity :math:`(0\leq e_0\leq0.7)`.

        Raises:
            ValueError: If any of the parameters are not allowed.

        """

        for val, key in [[M, "M"], [p0, "p0"], [e0, "e0"], [mu, "mu"]]:
            test = val < 0.0
            if test:
                raise ValueError("{} is negative. It must be positive.".format(key))

        if e0 > 0.75:
            raise ValueError(
                "Initial eccentricity above 0.75 not allowed. (e0={})".format(e0)
            )

        if e0 < 0.0:
            raise ValueError(
                "Initial eccentricity below 0.0 not physical. (e0={})".format(e0)
            )

        if mu / M > 1e-4:
            warnings.warn(
                "Mass ratio is outside of generally accepted range for an extreme mass ratio (1e-4). (q={})".format(
                    mu / M
                )
            )

        if p0 < 10.0:
            if p0 < 7 * (6.0 + 2 * e0) - 41.9:
                raise ValueError(
                    "This p0 ({}) and e0 ({}) combination is outside of our domain of validity.".format(
                        p0, e0
                    )
                )

        if p0 > 16.0 + 2 * e0:
            raise ValueError(
                "Initial p0 is too large (p0={}). Must be 10 <= p0 <= 16 + 2 * e.".format(
                    p0
                )
            )


class Pn5AAK(ABC):
    """Base class for Pn5AAK waveforms.

    This class contains some basic checks and information for AAK waveforms
    with a 5PN trajectory model. Please see :class:`few.waveform.Pn5AAKWaveform`
    for more details.

    args:
        use_gpu (bool, optional): If True, will allocate arrays on the GPU.
            Default is False.

    """

    def attributes_Pn5AAK(self):
        """
        attributes:
            xp (module): numpy or cupy based on hardware chosen.
            background (str): Spacetime background for this model.
            descriptor (str): Short description for model validity.
            needs_Y (bool): If True, indicates modules that inherit this class
                requires the inclination definition of :math:`Y\equiv\cos{\iota}=L/\sqrt{L^2 + Q}`
                rather than :math:`x_I`.

        """
        pass

    def __init__(self, use_gpu=False, **kwargs):
        # some descriptive information
        self.background = "Kerr"
        self.descriptor = "generic orbits"
        self.frame = "detector"
        self.needs_Y = True

    @property
    def citation(self):
        """Return citations of this class"""
        return larger_few_citation + few_citation + few_software_citation + Pn5_citation

    def sanity_check_angles(self, qS, phiS, qK, phiK):
        """Sanity check on viewing angles.

        Make sure parameters are within allowable ranges.

        args:
            qS (double): Sky location polar angle in ecliptic
                coordinates.
            phiS (double): Sky location azimuthal angle in
                ecliptic coordinates.
            qK (double): Initial BH spin polar angle in ecliptic
                coordinates.
            phiK (double): Initial BH spin azimuthal angle in
                ecliptic coordinates.

        Returns:
            tuple: (qS, phiS, qK, phiK). phiS and phiK are wrapped.

        Raises:
            ValueError: If any of the angular values are not allowed.

        """
        if qS < 0.0 or qS > np.pi:
            raise ValueError("qS must be between 0 and pi.")

        if qK < 0.0 or qK > np.pi:
            raise ValueError("qK must be between 0 and pi.")

        phiS = phiS % (2 * np.pi)
        phiK = phiK % (2 * np.pi)
        return (qS, phiS, qK, phiK)

    def sanity_check_traj(self, p, e, Y):
        """Sanity check on parameters output from thte trajectory module.

        Make sure parameters are within allowable ranges.

        args:
            p (1D np.ndarray): Array of semi-latus rectum values produced by
                the trajectory module.
            e (1D np.ndarray): Array of eccentricity values produced by
                the trajectory module.
            Y (1D np.ndarray): Array of cos:math:`\iota` values produced by
                the trajectory module.

        Raises:
            ValueError: If any of the trajectory points are not allowed.
            warn: If any points in the trajectory are allowable,
                but outside calibration region.

        """

        if np.any(e < 0.0):
            raise ValueError("Members of e array are less than zero.")

        if np.any(p < 0.0):
            raise ValueError("Members of p array are less than zero.")

        if np.any(Y < -1.0) or np.any(Y > 1.0):
            raise ValueError(
                "Members of Y array are greater than 1.0 or less than -1.0."
            )

    def sanity_check_init(self, M, mu, a, p0, e0, Y0):
        """Sanity check initial parameters.

        Make sure parameters are within allowable ranges.

        args:
            M (double): Massive black hole mass in solar masses.
            mu (double): compact object mass in solar masses.
            a (double): Dimensionless spin of massive black hole.
            p0 (double): Initial semilatus rectum (dimensionless)
                :math:`(10\leq p_0\leq 16 + 2e_0)`. See the documentation for
                more information on :math:`p_0 \leq 10.0`.
            e0 (double): Initial eccentricity :math:`(0\leq e_0\leq0.7)`.
            Y0 (double): Initial cos:math:`\iota` :math:`(-1.0\leq Y_0\leq1.0)`.

        Raises:
            ValueError: If any of the parameters are not allowed.

        """

        for val, key in [[M, "M"], [p0, "p0"], [e0, "e0"], [mu, "mu"], [a, "a"]]:
            test = val < 0.0
            if test:
                raise ValueError("{} is negative. It must be positive.".format(key))

        if mu / M > 1e-4:
            warnings.warn(
                "Mass ratio is outside of generally accepted range for an extreme mass ratio (1e-4). (q={})".format(
                    mu / M
                )
            )

        if Y0 > 1.0 or Y0 < -1.0:
            raise ValueError(
                "Y0 is greater than 1 or less than -1. Must be between -1 and 1."
            )


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
        err=1e-10,
        use_rk4=False,
        fix_t=False,
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
            use_rk4 (bool, optional): If True, use rk4 integrator from gsl.
                If False, use rk8. Default is False (rk8).
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
        kwargs["use_rk4"] = use_rk4

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

        # create splines for upsampling
        splines = [CubicSpline(t, temp, **spline_kwargs) for temp in list(params)]

        if new_t is not None:
            if isinstance(new_t, np.ndarray) is False:
                raise ValueError("new_t parameter, if provided, must be numpy array.")

        else:
            # new time array for upsampling
            new_t = np.arange(0.0, T + dt, dt)

        # check if new time ends before or after output array from the trajectory
        if new_t[-1] > t[-1]:
            if fix_t:
                new_t = new_t[new_t <= t[-1]]
            else:
                warnings.warn(
                    "new_t array goes beyond generated t array. If you want to cut the t array at the end of the trajectory, set fix_t to True."
                )

        # upsample everything
        out = tuple([spl(new_t) * (new_t < t[-1]) for spl in splines])
        return (new_t,) + out


class SummationBase(ABC):
    """Base class used for summation modules.

    This class provides a common flexible interface to various summation
    implementations. Specific arguments to each summation module can be found
    with each associated module discussed below.

    args:
        pad_output (bool, optional): Add zero padding to the waveform for time
            between plunge and observation time. Default is False.
        output_type (str, optional): Type of domain in which to calculate the waveform.
            Default is 'td' for time domain. Options are 'td' (time domain) or 'fd' (Fourier domain). In the future we hope to add 'tf'
            (time-frequency) and 'wd' (wavelet domain).
        odd_len (bool, optional): The waveform output will be padded to be an odd number if True.
            If ``output_type == "fd"``, odd_len will be set to ``True``. Default is False.

    """

    def __init__(
        self, *args, output_type="td", pad_output=False, odd_len=False, **kwargs
    ):
        self.pad_output = pad_output
        self.odd_len = odd_len

        if output_type not in ["td", "fd"]:
            raise ValueError(
                "{} waveform domain not available. Choices are 'td' (time domain) or 'fd' (frequency domain).".format(
                    output_type
                )
            )
        self.output_type = output_type
        if self.output_type == "fd":
            self.odd_len = True

    def attributes_SummationBase(self):
        """
        attributes:
            waveform (1D complex128 np.ndarray): Complex waveform given by
                :math:`h_+ + i*h_x`.
        """
        pass

    @property
    def citation(self):
        """Return citation for this class"""
        return larger_few_citation + few_citation + few_software_citation

    @classmethod
    def sum(self, *args, **kwargs):
        """Sum Generator

        @classmethod that requires a child class to have a sum method.

        raises:
            NotImplementedError: The child class does not have this method.

        """
        raise NotImplementedError

    def __call__(self, t, *args, T=1.0, dt=10.0, t_window=None, **kwargs):
        """Common call function for summation modules.

        Provides a common interface for summation modules. It can adjust for
        more dimensions in a model.

        args:
            t (1D double xp.ndarray): Array of t values.
            *args (list): Added for flexibility with summation modules. `args`
                tranfers directly into sum function.
            dt (double, optional): Time spacing between observations in seconds (inverse of sampling
                rate). Default is 10.0.
            T (double, optional): Maximum observing time in years. Default is 1.0.
            **kwargs (dict, placeholder): Added for future flexibility.

        """

        if self.use_gpu:
            xp = cp
        else:
            xp = np

        n_pts = int(T * YRSID_SI / dt)
        T = n_pts * dt
        # determine the output array setup

        # adjust based on if observations time is less than or more than trajectory time array
        # if the user wants zero-padding, add number of zero pad points
        if T < t[-1].item():
            num_pts = int((T - t[0]) / dt) + 1
            num_pts_pad = 0

        else:
            num_pts = int((t[-1] - t[0]) / dt) + 1
            if self.pad_output:
                num_pts_pad = int((T - t[0]) / dt) + 1 - num_pts
            else:
                num_pts_pad = 0

        self.num_pts, self.num_pts_pad = num_pts, num_pts_pad
        self.dt = dt

        # impose to be always odd
        if self.odd_len:
            if (self.num_pts + self.num_pts_pad) % 2 == 0:
                self.num_pts_pad = self.num_pts_pad + 1
                # print("n points",self.num_pts + self.num_pts_pad)

        # make sure that the FD waveform has always an odd number of points
        if self.output_type == "fd":
            if "f_arr" in kwargs:
                frequency = kwargs["f_arr"]
                dt = float(xp.max(frequency) * 2)
                Nf = len(frequency)
                # total
                self.waveform = xp.zeros(Nf, dtype=xp.complex128)
                # print("user defined frequencies Nf=", Nf)
            else:
                self.waveform = xp.zeros(
                    (self.num_pts + self.num_pts_pad,), dtype=xp.complex128
                )
            # if self.num_pts + self.num_pts_pad % 2:
            #     self.num_pts_pad = self.num_pts_pad + 1
            #     print("n points",self.num_pts + self.num_pts_pad)
        else:
            # setup waveform holder for time domain
            self.waveform = xp.zeros(
                (self.num_pts + self.num_pts_pad,), dtype=xp.complex128
            )

        # get the waveform summed in place
        self.sum(t, *args, dt=dt, **kwargs)

        return self.waveform


class AmplitudeBase(ABC):
    """Base class used for amplitude modules.

    This class provides a common flexible interface to various amplitude
    implementations. Specific arguments to each amplitude module can be found
    with each associated module discussed below.

    args:
        pad_output (bool, optional): Add zero padding to the waveform for time
            between plunge and observation time. Default is False.

    """

    def __init__(self, **kwargs):
        pass

    @classmethod
    def get_amplitudes(self, *args, **kwargs):
        """Amplitude Generator

        @classmethod that requires a child class to have a get_amplitudes method.

        raises:
            NotImplementedError: The child class does not have this method.

        """
        raise NotImplementedError

    @property
    def citation(self):
        """Return citation for this class"""
        return larger_few_citation + few_citation + few_software_citation

    def __call__(self, *args, specific_modes=None, **kwargs):
        """Common call for Teukolsky amplitudes

        This function takes the inputs the trajectory in :math:`(p,e)` as arrays
        and returns the complex amplitude of all modes to adiabatic order at
        each step of the trajectory.

        args:
            *args (tuple, placeholder): Added to create future flexibility when calling different
                amplitude modules. Transfers directly into get_amplitudes function.
            specific_modes (list, optional): List of tuples for (l, m, n) values
                desired modes. Default is None. This is not available for all waveforms.
            **kwargs (dict, placeholder): Added to create flexibility when calling different
                amplitude modules. It is not used.

        returns:
            2D array (double): If specific_modes is None, Teukolsky modes in shape (number of trajectory points, number of modes)
            dict: Dictionary with requested modes.


        """

        return self.get_amplitudes(*args, specific_modes=specific_modes, **kwargs)
