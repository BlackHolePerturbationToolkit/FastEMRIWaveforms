# Utilities to aid in FastEMRIWaveforms Packages

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


import requests
import os
import subprocess
import warnings
from rich.progress import track

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq

from ..cutils.pyUtility import (
    pyKerrGeoCoordinateFrequencies,
    pyGetSeparatrix,
    pyKerrGeoConstantsOfMotionVectorized,
    pyY_to_xI_vector,
)

# check to see if cupy is available for gpus
try:
    import cupy as cp
    from cupy.cuda.runtime import setDevice

    gpu = True

except (ImportError, ModuleNotFoundError) as e:
    import numpy as np

    setDevice = None
    gpu = False

import few
from few.utils.constants import *

# get path to this file
dir_path = os.path.dirname(os.path.realpath(__file__))


def get_overlap(time_series_1, time_series_2, use_gpu=False):
    """Calculate the overlap.

    Takes two time series and finds which one is shorter in length. It then
    shortens the longer time series if necessary. Then it performs a
    normalized correlation calulation on the two time series to give the
    overlap. The overlap of :math:`a(t)` and
    :math:`b(t)`, :math:`\gamma_{a,b}`, is given by,

    .. math:: \gamma_{a,b} = <a,b>/(<a,a><b,b>)^{(1/2)},

    where :math:`<a,b>` is the inner product of the two time series.

    args:
        time_series_1 (1D complex128 xp.ndarray): Strain time series 1.
        time_series_2 (1D complex128 xp.ndarray): Strain time series 2.
        use_gpu (bool, optional): If True use cupy. If False, use numpy. Default
            is False.

    """

    # adjust arrays based on GPU usage
    if use_gpu:
        xp = cp

        if isinstance(time_series_1, np.ndarray):
            time_series_1 = xp.asarray(time_series_1)
        if isinstance(time_series_2, np.ndarray):
            time_series_2 = xp.asarray(time_series_2)

    else:
        xp = np

        try:
            if isinstance(time_series_1, cp.ndarray):
                time_series_1 = xp.asarray(time_series_1)

        except NameError:
            pass

        try:
            if isinstance(time_series_2, cp.ndarray):
                time_series_2 = xp.asarray(time_series_2)

        except NameError:
            pass

    # get the lesser of the two lengths
    min_len = int(np.min([len(time_series_1), len(time_series_2)]))

    if len(time_series_1) != len(time_series_2):
        warnings.warn(
            "The two time series are not the same length ({} vs {}). The calculation will run with length {} starting at index 0 for both arrays.".format(
                len(time_series_1), len(time_series_2), min_len
            )
        )

    # chop off excess length on a longer array
    # take fft
    time_series_1_fft = xp.fft.fft(time_series_1[:min_len])
    time_series_2_fft = xp.fft.fft(time_series_2[:min_len])

    # autocorrelation
    ac = xp.dot(time_series_1_fft.conj(), time_series_2_fft) / xp.sqrt(
        xp.dot(time_series_1_fft.conj(), time_series_1_fft)
        * xp.dot(time_series_2_fft.conj(), time_series_2_fft)
    )

    # if using cupy, it will return a dimensionless array
    if use_gpu:
        return ac.item().real
    return ac.real


def get_mismatch(time_series_1, time_series_2, use_gpu=False):
    """Calculate the mismatch.

    The mismatch is 1 - overlap. Therefore, see documentation for
    :func:`few.utils.utility.overlap` for information on the overlap
    calculation.

    args:
        time_series_1 (1D complex128 xp.ndarray): Strain time series 1.
        time_series_2 (1D complex128 xp.ndarray): Strain time series 2.
        use_gpu (bool, optional): If True use cupy. If False, use numpy. Default
            is False.

    """
    overlap = get_overlap(time_series_1, time_series_2, use_gpu=use_gpu)
    return 1.0 - overlap


def p_to_y(p, e, use_gpu=False):
    """Convert from separation :math:`p` to :math:`y` coordinate

    Conversion from the semilatus rectum or separation :math:`p` to :math:`y`.

    arguments:
        p (double scalar or 1D double xp.ndarray): Values of separation,
            :math:`p`, to convert.
        e (double scalar or 1D double xp.ndarray): Associated eccentricity values
            of :math:`p` necessary for conversion.
        use_gpu (bool, optional): If True, use Cupy/GPUs. Default is False.

    """
    if use_gpu:
        return cp.log(-(21 / 10) - 2 * e + p)

    else:
        return np.log(-(21 / 10) - 2 * e + p)


def get_fundamental_frequencies(a, p, e, x):
    """Get dimensionless fundamental frequencies.

    Determines fundamental frequencies in generic Kerr from
    `Schmidt 2002 <https://arxiv.org/abs/gr-qc/0202090>`_.

    arguments:
        a (double scalar or 1D np.ndarray): Dimensionless spin of massive
            black hole. If other parameters are arrays and the spin is scalar,
            it will be cast to a 1D array.
        p (double scalar or 1D double np.ndarray): Values of separation,
            :math:`p`.
        e (double scalar or 1D double np.ndarray): Values of eccentricity,
            :math:`e`.
        x (double scalar or 1D double np.ndarray): Values of cosine of the
            inclination, :math:`x=\cos{I}`. Please note this is different from
            :math:`Y=\cos{\iota}`.

    returns:
        tuple: Tuple of (OmegaPhi, OmegaTheta, OmegaR).
            These are 1D arrays or scalar values depending on inputs.

    """

    # check if inputs are scalar or array
    if isinstance(p, float):
        scalar = True

    else:
        scalar = False

    p_in = np.atleast_1d(p)
    e_in = np.atleast_1d(e)
    x_in = np.atleast_1d(x)

    # cast the spin to the same size array as p
    if isinstance(a, float):
        a_in = np.full_like(p_in, a)
    else:
        a_in = np.atleast_1d(a)

    assert len(a_in) == len(p_in)

    # get frequencies
    OmegaPhi, OmegaTheta, OmegaR = pyKerrGeoCoordinateFrequencies(
        a_in, p_in, e_in, x_in
    )

    # set output to shape of input
    if scalar:
        return (OmegaPhi[0], OmegaTheta[0], OmegaR[0])

    else:
        return (OmegaPhi, OmegaTheta, OmegaR)


def get_kerr_geo_constants_of_motion(a, p, e, x):
    """Get Kerr constants of motion.

    Determines the constants of motion: :math:`(E, L, Q)` associated with a
    geodesic orbit in the generic Kerr spacetime.

    arguments:
        a (double scalar or 1D np.ndarray): Dimensionless spin of massive
            black hole. If other parameters are arrays and the spin is scalar,
            it will be cast to a 1D array.
        p (double scalar or 1D double np.ndarray): Values of separation,
            :math:`p`.
        e (double scalar or 1D double np.ndarray): Values of eccentricity,
            :math:`e`.
        x (double scalar or 1D double np.ndarray): Values of cosine of the
            inclination, :math:`x=\cos{I}`. Please note this is different from
            :math:`Y=\cos{\iota}`.

    returns:
        tuple: Tuple of (E, L, Q).
            These are 1D arrays or scalar values depending on inputs.

    """

    # check if inputs are scalar or array
    if isinstance(p, float):
        scalar = True

    else:
        scalar = False

    p_in = np.atleast_1d(p)
    e_in = np.atleast_1d(e)
    x_in = np.atleast_1d(x)

    # cast the spin to the same size array as p
    if isinstance(a, float):
        a_in = np.full_like(p_in, a)
    else:
        a_in = np.atleast_1d(a)

    assert len(a_in) == len(p_in)

    # get constants of motion
    E, L, Q = pyKerrGeoConstantsOfMotionVectorized(a_in, p_in, e_in, x_in)

    # set output to shape of input
    if scalar:
        return (E[0], L[0], Q[0])

    else:
        return (E, L, Q)


def xI_to_Y(a, p, e, x):
    """Convert from :math:`x_I=\cos{I}` to :math:`Y=\cos{\iota}`.

    Converts between the two different inclination parameters. :math:`\cos{I}\equiv x_I`,
    where :math:`I` describes the orbit's inclination from the equatorial plane.
    :math:`\cos{\iota}\equiv Y`, where :math:`\cos{\iota}=L/\sqrt{L^2 + Q}`.

    arguments:
        a (double scalar or 1D np.ndarray): Dimensionless spin of massive
            black hole. If other parameters are arrays and the spin is scalar,
            it will be cast to a 1D array.
        p (double scalar or 1D double np.ndarray): Values of separation,
            :math:`p`.
        e (double scalar or 1D double np.ndarray): Values of eccentricity,
            :math:`e`.
        x (double scalar or 1D double np.ndarray): Values of cosine of the
            inclination, :math:`x=\cos{I}`.

    returns:
        1D array or scalar: :math:`Y=\cos{\iota}` value with shape based on input shapes.

    """

    # get constants of motion
    E, L, Q = get_kerr_geo_constants_of_motion(a, p, e, x)

    Y = L / np.sqrt(L**2 + Q)
    return Y


def Y_to_xI(a, p, e, Y):
    """Convert from :math:`Y=\cos{\iota}` to :math:`x_I=\cos{I}`.

    Converts between the two different inclination parameters. :math:`\cos{I}\equiv x_I`,
    where :math:`I` describes the orbit's inclination from the equatorial plane.
    :math:`\cos{\iota}\equiv Y`, where :math:`\cos{\iota}=L/\sqrt{L^2 + Q}`.

    This computation may have issues near edge cases.

    arguments:
        a (double scalar or 1D np.ndarray): Dimensionless spin of massive
            black hole. If other parameters are arrays and the spin is scalar,
            it will be cast to a 1D array.
        p (double scalar or 1D double np.ndarray): Values of separation,
            :math:`p`.
        e (double scalar or 1D double np.ndarray): Values of eccentricity,
            :math:`e`.
        Y (double scalar or 1D double np.ndarray): Values of cosine of the
            :math:`\iota`.

    returns:
        1D array or scalar: :math:`x=\cos{I}` value with shape based on input shapes.

    """

    # TODO: check error for not c-contiguous
    # determines shape of input
    if isinstance(e, float):
        scalar = True

    else:
        scalar = False

    p_in = np.atleast_1d(p)
    e_in = np.atleast_1d(e)
    Y_in = np.atleast_1d(Y)

    # cast spin values if necessary
    if isinstance(a, float):
        a_in = np.full_like(e_in, a)
    else:
        a_in = np.atleast_1d(a)

    assert len(a_in) == len(e_in)

    x = pyY_to_xI_vector(a_in, p_in, e_in, Y_in)

    # output in same shape as input
    if scalar:
        return x[0]

    else:
        return x


def get_separatrix(a, e, x):
    """Get separatrix in generic Kerr.

    Determines separatrix in generic Kerr from
    `Stein & Warburton 2020 <https://arxiv.org/abs/1912.07609>`_.

    arguments:
        a (double scalar or 1D np.ndarray): Dimensionless spin of massive
            black hole. If other parameters are arrays and the spin is scalar,
            it will be cast to a 1D array.
        e (double scalar or 1D double np.ndarray): Values of eccentricity,
            :math:`e`.
        x (double scalar or 1D double np.ndarray): Values of cosine of the
            inclination, :math:`x=\cos{I}`. Please note this is different from
            :math:`Y=\cos{\iota}`.

    returns:
        1D array or scalar: Separatrix value with shape based on input shapes.

    """
    # determines shape of input
    if isinstance(e, float):
        scalar = True

    else:
        scalar = False

    e_in = np.atleast_1d(e)

    if isinstance(x, float):
        x_in = np.full_like(e_in, x)
    else:
        x_in = np.atleast_1d(x)

    # cast spin values if necessary
    if isinstance(a, float):
        a_in = np.full_like(e_in, a)
    else:
        a_in = np.atleast_1d(a)

    if isinstance(x, float):
        x_in = np.full_like(e_in, x)
    else:
        x_in = np.atleast_1d(x)

    assert len(a_in) == len(e_in) == len(x_in)

    separatrix = pyGetSeparatrix(a_in, e_in, x_in)

    # output in same shape as input
    if scalar:
        return separatrix[0]

    else:
        return separatrix


def get_mu_at_t(
    traj_module,
    t_out,
    traj_args,
    index_of_mu=1,
    traj_kwargs={},
    min_mu=1.0,
    max_mu=1e3,
    num_mu=100,
    logspace=True,
):
    """Find the value of mu that will give a specific length inspiral.

    If you want to generate an inspiral that is a specific length, you
    can adjust mu accordingly. This function tells you what that value of mu
    is based on the trajectory module and other input parameters at a
    desired time of observation.

    The function grids mu values and finds their associated end times. These
    end times then become the x values in a spline with the gridded mu
    values as the y values. The spline is then evaluated at the desired end time
    in order to get the desired mu value.

    arguments:
        traj_module (obj): Instantiated trajectory module. It must output
            the time array of the trajectory sparse trajectory as the first
            output value in the tuple.
        t_out (double): The desired length of time for the waveform in years.
        traj_args (list): List of arguments for the trajectory function.
            mu is removed. **Note**: It must be a list, not a tuple because the
            new mu values are inserted into the argument list.
        index_of_mu (int, optional): Index where to insert the new mu values in
            the :code:`traj_args` list. Default is 1 because mu usually comes
            after M.
        traj_kwargs (dict, optional): Keyword arguments for :code:`traj_module`.
            Default is an empty dict.
        min_mu (double, optional): The minumum value of mu for search array.
            Default is :math:`1 M_\odot`.
        max_mu (double, optional): The maximum value of mu for search array.
            Default is :math:`10^3M_\odot`.
        num_mu (int, optional): Number of mu values to search over. Default is
            100.
        logspace (bool, optional): If True, logspace the search array.
            If False, linspace search array. Default is True.

    returns:
        double: Value of mu that creates the proper length trajectory.

    """

    # setup search array
    array_creator = np.logspace if logspace else np.linspace
    start_mu = np.log10(min_mu) if logspace else min_mu
    end_mu = np.log10(max_mu) if logspace else max_mu
    mu_new = array_creator(start_mu, end_mu, num_mu)

    # set maximum time value of trajectory to be just beyond desired time
    traj_kwargs["T"] = t_out * 1.1

    # array for end time values for trajectories
    t_end = np.zeros_like(mu_new)

    for i, mu in enumerate(mu_new):
        # insert mu into args list
        args_new = traj_args.copy()
        args_new.insert(index_of_mu, mu)

        # run the trajectory
        out = traj_module(*args_new, **traj_kwargs)

        # get the last time in the trajectory
        t = out[0]
        t_end[i] = t[-1]

    # get rid of extra values beyond the maximum allowable time
    # remove repeated initital values for proper splining.
    try:
        ind_stop = np.where(np.diff(t_end) > 0.0)[0][-1] + 1
    except IndexError:
        ind_stop = len(t_end)

    try:
        ind_start = np.where(np.diff(t_end) == 0.0)[0][-1] + 1
    except IndexError:
        ind_stop = 0

    mu_new = mu_new[ind_start:ind_stop]
    t_end = t_end[ind_start:ind_stop]

    # put them in increasing order
    sort = np.argsort(t_end)
    t_end = t_end[sort]
    mu_new = mu_new[sort]

    # setup spline
    spline = CubicSpline(t_end, mu_new)

    # return proper mu value
    return spline(t_out * YRSID_SI).item()


def get_at_t(
    traj_module,
    traj_args,
    bounds,
    t_out,
    index_of_interest,
    traj_kwargs={},
    xtol=2e-12,
    rtol=8.881784197001252e-16,
):
    """Root finding wrapper using Brent's method.

    This function uses scipy's brentq routine to find root.

    arguments:
        traj_module (obj): Instantiated trajectory module. It must output
            the time array of the trajectory sparse trajectory as the first
            output value in the tuple.
        traj_args (list): List of arguments for the trajectory function.
            p is removed. **Note**: It must be a list, not a tuple because the
            new p values are inserted into the argument list.
        bounds (list): Minimum and maximum values over which brentq will search for a root.
        t_out (double): The desired length of time for the waveform.
        index_of_interest (int): Index where to insert the new values in
            the :code:`traj_args` list.
        traj_kwargs (dict, optional): Keyword arguments for :code:`traj_module`.
            Default is an empty dict.
        xtol (float, optional): Absolute tolerance of the brentq root-finding - see :code: `np.allclose()` for details.
            Defaults to 2e-12 (scipy default).
        rtol (float, optional): Relative tolerance of the brentq root-finding - see :code: `np.allclose()` for details.
            Defaults to ~8.8e-16 (scipy default).

    returns:
        double: Root value.

    """

    def get_time_root(val, traj, inj_args, traj_kwargs, t_out, ind_interest):
        """
        Function with one p root at T = t_outp, for brentq input.
        """
        inputs = inj_args.copy()
        inputs.insert(ind_interest, val)
        traj_kwargs["T"] = t_out * 2.0
        out = traj(*inputs, **traj_kwargs)
        return out[0][-1] - t_out * YRSID_SI

    root = brentq(
        get_time_root,
        bounds[0],
        bounds[1],
        xtol=xtol,
        rtol=rtol,
        args=(traj_module, traj_args, traj_kwargs, t_out, index_of_interest),
    )
    return root


def get_p_at_t(
    traj_module,
    t_out,
    traj_args,
    index_of_p=3,
    index_of_a=2,
    index_of_e=4,
    index_of_x=5,
    bounds=None,
    **kwargs,
):
    """Find the value of p that will give a specific length inspiral using Brent's method.

    If you want to generate an inspiral that is a specific length, you
    can adjust p accordingly. This function tells you what that value of p
    is based on the trajectory module and other input parameters at a
    desired time of observation.

    This function uses scipy's brentq routine to find the (presumed only)
    value of p that gives a trajectory of duration t_out.

    arguments:
        traj_module (obj): Instantiated trajectory module. It must output
            the time array of the trajectory sparse trajectory as the first
            output value in the tuple.
        t_out (double): The desired length of time for the waveform.
        traj_args (list): List of arguments for the trajectory function.
            p is removed. **Note**: It must be a list, not a tuple because the
            new p values are inserted into the argument list.
        index_of_p (int, optional): Index where to insert the new p values in
            the :code:`traj_args` list. Default is 3.
        index_of_a (int, optional): Index of a in provided :code:`traj_module` arguments. Default is 2.
        index_of_e (int, optional): Index of e0 in provided :code:`traj_module` arguments. Default is 4.
        index_of_x (int, optional): Index of x0 in provided :code:`traj_module` arguments. Default is 5.
        bounds (list, optional): Minimum and maximum values of p over which brentq will search for a root.
            If not given, will be set to [separatrix + 0.101, 50]. To supply only one of these two limits, set the
            other limit to None.
        **kwargs (dict, optional): Keyword arguments for :func:`get_at_t`.

    returns:
        double: Value of p that creates the proper length trajectory.

    """

    # fix indexes for p
    if index_of_a > index_of_p:
        index_of_a -= 1
    if index_of_e > index_of_p:
        index_of_e -= 1
    if index_of_x > index_of_p:
        index_of_x -= 1

    if "traj_kwargs" in kwargs and "enforce_schwarz_sep" in kwargs["traj_kwargs"]:
        enforce_schwarz_sep = kwargs["traj_kwargs"]["enforce_schwarz_sep"]

    else:
        enforce_schwarz_sep = False

    # fix bounds
    if bounds is None:
        if not enforce_schwarz_sep:
            p_sep = get_separatrix(
                traj_args[index_of_a], traj_args[index_of_e], traj_args[index_of_x]
            )  # should be fairly close.
        else:
            p_sep = 6 + 2 * traj_args[index_of_e]
        bounds = [p_sep + 0.1, 16.0 + 2 * traj_args[index_of_e]]

    elif bounds[0] is None:
        if not enforce_schwarz_sep:
            p_sep = get_separatrix(
                traj_args[index_of_a], traj_args[index_of_e], traj_args[index_of_x]
            )  # should be fairly close.
        else:
            p_sep = 6 + 2 * traj_args[index_of_e]
        bounds[0] = p_sep + 0.1

    elif bounds[1] is None:
        bounds[1] = 16.0 + 2 * traj_args[index_of_e]

    root = get_at_t(traj_module, traj_args, bounds, t_out, index_of_p, **kwargs)
    return root


def get_mu_at_t(
    traj_module,
    t_out,
    traj_args,
    index_of_mu=1,
    bounds=None,
    **kwargs,
):
    """Find the value of mu that will give a specific length inspiral using Brent's method.

    If you want to generate an inspiral that is a specific length, you
    can adjust mu accordingly. This function tells you what that value of mu
    is based on the trajectory module and other input parameters at a
    desired time of observation.

    This function uses scipy's brentq routine to find the (presumed only)
    value of mu that gives a trajectory of duration t_out.

    arguments:
        traj_module (obj): Instantiated trajectory module. It must output
            the time array of the trajectory sparse trajectory as the first
            output value in the tuple.
        t_out (double): The desired length of time for the waveform.
        traj_args (list): List of arguments for the trajectory function.
            p is removed. **Note**: It must be a list, not a tuple because the
            new p values are inserted into the argument list.
        index_of_mu (int, optional): Index where to insert the new p values in
            the :code:`traj_args` list. Default is 1.
        bounds (list, optional): Minimum and maximum values of p over which brentq will search for a root.
            If not given, will be set to [1e-1, 1e3]. To supply only one of these two limits, set the
            other limit to None.
        **kwargs (dict, optional): Keyword arguments for :func:`get_at_t`.

    returns:
        double: Value of mu that creates the proper length trajectory.

    """

    # fix bounds
    if bounds is None:
        bounds = [1e-1, 1e3]

    elif bounds[0] is None:
        bounds[0] = 1e-1

    elif bounds[1] is None:
        bounds[1] = 1e3

    root = get_at_t(traj_module, traj_args, bounds, t_out, index_of_mu, **kwargs)
    return root


# data history is saved here nased on version nunber
# record_by_version = {
#     "1.0.0": 3981654,
#     "1.1.0": 3981654,
#     "1.1.1": 3981654,
#     "1.1.2": 3981654,
#     "1.1.3": 3981654,
#     "1.1.4": 3981654,
#     "1.1.5": 3981654,
#     "1.2.0": 3981654,
#     "1.2.1": 3981654,
#     "1.2.2": 3981654,
#     "1.3.0": 3981654,
#     "1.3.1": 3981654,
#     "1.3.2": 3981654,
#     "1.3.3": 3981654,
#     "1.3.4": 3981654,
#     "1.3.5": 3981654,
#     "1.3.6": 3981654,
#     "1.3.7": 3981654,
#     "1.4.0": 3981654,
#     "1.4.1": 3981654,
#     "1.4.2": 3981654,
#     "1.4.3": 3981654,
#     "1.4.4": 3981654,
#     "1.4.5": 3981654,
#     "1.4.6": 3981654,
#     "1.4.7": 3981654,
#     "1.4.8": 3981654,
#     "1.4.9": 3981654,
#     "1.4.10": 3981654,
#     "1.4.11": 3981654,
#     "1.5.0": 3981654,
#     "1.5.1": 3981654,
# }


def check_for_file_download(fp, few_dir, version_string=None):
    """Download files direct from download.bhptoolkit.org.

    This function downloads the files from download.bhptoolkit.org as they are needed. They are
    downloaded based on the associated Zenodo record for each version (`record_by_version`).

    The version is determined from the `__version__` attribute of `few` unless
    a version string is provided.

    arguments:
        fp (string): File name.
        few_dir (string): absolute path to FastEMRIWaveforms directory.
        version_string (string, optional): Provide a specific version string to
            get a specific dataset. Default is None.

    raises:
        ValueError: Version string does not exist.

    """

    # make sure version_string is available
    # if version_string is not None:
    #     if version_string not in record_by_version:
    #         raise ValueError(
    #             "The version_string provided ({}) does not exist.".format(
    #                 version_string
    #             )
    #         )
    # else:
    #     version_string = few.__version__

    # check if the files directory exists
    try:
        os.listdir(few_dir + "few/files/")

    # if not, create it
    except OSError:
        os.mkdir(few_dir + "few/files/")

    # check if the file is in the files filder
    # if not, download it from download.bhptoolkit.org
    if fp not in os.listdir(few_dir + "few/files/"):
        print("Data file " + fp + " not found. Downloading now.")

        # get record number based on version
        # record = record_by_version.get(version_string)

        # temporary fix
        record = 3981654

        # url to download from with Zenodo fallback in case of failure
        url = "https://download.bhptoolkit.org/few/data/" + str(record) + "/" + fp
        zenodourl = "https://zenodo.org/record/" + str(record) + "/files/" + fp

        # download the file
        response = requests.get(url, stream=True)
        if response.ok != True:
            response = requests.get(zenodourl, stream=True)

        # Save the file to the files folder, downloading 8KB at a time
        with open(few_dir + "few/files/" + fp, mode="wb") as file:
            filesize = int(response.headers.get("content-length"))
            csize = 2**15
            for chunk in track(
                response.iter_content(chunk_size=csize),
                description="Downloading " + fp,
                total=filesize / csize,
            ):
                file.write(chunk)


def wrapper(*args, **kwargs):
    """Function to convert array and C/C++ class arguments to ptrs

    This function checks the object type. If it is a cupy or numpy array,
    it will determine its pointer by calling the proper attributes. If you design
    a Cython class to be passed through python, it must have a :code:`ptr`
    attribute.

    If you use this function, you must convert input arrays to size_t data type in Cython and
    then properly cast the pointer as it enters the c++ function. See the
    Cython codes
    `here <https://github.com/BlackHolePerturbationToolkit/FastEMRIWaveforms/tree/master/src>`_
    for examples.

    args:
        *args (list): list of the arguments for a function.
        **kwargs (dict): dictionary of keyword arguments to be converted.

    returns:
        Tuple: (targs, tkwargs) where t indicates target (with pointer values
            rather than python objects).

    """
    # declare target containers
    targs = []
    tkwargs = {}

    # args first
    for arg in args:
        if gpu:
            # cupy arrays
            if isinstance(arg, cp.ndarray):
                targs.append(arg.data.mem.ptr)
                continue

        # numpy arrays
        if isinstance(arg, np.ndarray):
            targs.append(arg.__array_interface__["data"][0])
            continue

        try:
            # cython classes
            targs.append(arg.ptr)
            continue
        except AttributeError:
            # regular argument
            targs.append(arg)

    # kwargs next
    for key, arg in kwargs.items():
        if gpu:
            # cupy arrays
            if isinstance(arg, cp.ndarray):
                tkwargs[key] = arg.data.mem.ptr
                continue

        if isinstance(arg, np.ndarray):
            # numpy arrays
            tkwargs[key] = arg.__array_interface__["data"][0]
            continue

        try:
            # cython classes
            tkwargs[key] = arg.ptr
            continue
        except AttributeError:
            # other arguments
            tkwargs[key] = arg

    return (targs, tkwargs)


def pointer_adjust(func):
    """Decorator function for cupy/numpy agnostic cython

    This decorator applies :func:`few.utils.utility.wrapper` to functions
    via the decorator construction.

    If you use this decorator, you must convert input arrays to size_t data type in Cython and
    then properly cast the pointer as it enters the c++ function. See the
    Cython codes
    `here <https://github.com/BlackHolePerturbationToolkit/FastEMRIWaveforms/tree/master/src>`_
    for examples.

    """

    def func_wrapper(*args, **kwargs):
        # get pointers
        targs, tkwargs = wrapper(*args, **kwargs)
        return func(*targs, **tkwargs)

    return func_wrapper


def cuda_set_device(dev):
    """Globally sets CUDA device

    Args:
        dev (int): CUDA device number.

    """
    if setDevice is not None:
        setDevice(dev)
    else:
        warnings.warn("Setting cuda device, but cupy/cuda not detected.")


def get_ode_function_options():
    """Get ode options.

    This includes all the subinfo for each ODE derivative
    function that is available.

    Returns:
        dict: Dictionary with all the information on available functions.

    Raises:
        ValueError: ODE files have not been built.

    """
    try:
        from few.utils.odeoptions import ode_options
    except (ImportError, ModuleNotFoundError) as e:
        raise ValueError("ODE files not built yet.")

    return ode_options
