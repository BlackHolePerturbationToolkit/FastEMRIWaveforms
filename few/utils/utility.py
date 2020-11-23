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

import numpy as np
from scipy.interpolate import CubicSpline

from pyFundamentalFrequencies import pyKerrGeoCoordinateFrequencies, pyGetSeparatrix

# check to see if cupy is available for gpus
try:
    import cupy as cp

    gpu = True

except (ImportError, ModuleNotFoundError) as e:
    import numpy as np

    gpu = False

import few
from few.utils.constants import *


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
            inclination, :math:`\cos{\iota}`.

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
            inclination, :math:`\cos{\iota}`.

    returns:
        1D array or scalar: Separatrix value with shape based on input shapes.

    """
    # determines shape of input
    if isinstance(e, float):
        scalar = True

    else:
        scalar = False

    e_in = np.atleast_1d(e)
    x_in = np.atleast_1d(x)

    # cast spin values if necessary
    if isinstance(a, float):
        a_in = np.full_like(e_in, a)
    else:
        a_in = np.atleast_1d(a)

    assert len(a_in) == len(e_in)

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
        t_out (double): The desired length of time for the waveform.
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

    # put them in increasing order
    sort = np.argsort(t_end)
    t_end = t_end[sort]
    mu_new = mu_new[sort]

    # get rid of extra values beyond the maximum allowable time
    ind_stop = np.where(np.diff(t_end) > 0.0)[0][-1] + 1
    mu_new = mu_new[:ind_stop]
    t_end = t_end[:ind_stop]

    # setup spline
    spline = CubicSpline(t_end, mu_new)

    # return proper mu value
    return spline(t_out * YRSID_SI).item()


def get_p_at_t(
    traj_module,
    t_out,
    traj_args,
    index_of_p=2,
    traj_kwargs={},
    min_p=8.0,
    max_p=16.0,
    num_p=100,
):
    """Find the value of p that will give a specific length inspiral.

    If you want to generate an inspiral that is a specific length, you
    can adjust p accordingly. This function tells you what that value of p
    is based on the trajectory module and other input parameters at a
    desired time of observation.

    The function grids p values and finds their associated end times. These
    end times then become the x values in a spline with the gridded p
    values as the y values. The spline is then evaluated at the desired end time
    in order to get the desired p value.

    arguments:
        traj_module (obj): Instantiated trajectory module. It must output
            the time array of the trajectory sparse trajectory as the first
            output value in the tuple.
        t_out (double): The desired length of time for the waveform.
        traj_args (list): List of arguments for the trajectory function.
            p is removed. **Note**: It must be a list, not a tuple because the
            new p values are inserted into the argument list.
        index_of_p (int, optional): Index where to insert the new p values in
            the :code:`traj_args` list. Default is 2 because p usually comes
            after p.
        traj_kwargs (dict, optional): Keyword arguments for :code:`traj_module`.
            Default is an empty dict.
        min_p (double, optional): The minumum value of p for search array.
            Default is :math:`1 M_\odot`.
        max_p (double, optional): The maximum value of p for search array.
            Default is :math:`10^3M_\odot`.
        num_p (int, optional): Number of p values to search over. Default is
            100.

    returns:
        double: Value of p that creates the proper length trajectory.

    """

    # setup search array
    p_new = np.linspace(min_p, max_p, num_p)

    # set maximum time value of trajectory to be just beyond desired time
    traj_kwargs["T"] = t_out * 1.1

    # array for end time values for trajectories
    t_end = np.zeros_like(p_new)

    for i, p in enumerate(p_new):

        # insert mu into args list
        args_new = traj_args.copy()
        args_new.insert(index_of_p, p)

        # run the trajectory
        out = traj_module(*args_new, **traj_kwargs)

        # get the last time in the trajectory
        t = out[0]
        t_end[i] = t[-1]

    # put them in increasing order
    sort = np.argsort(t_end)
    t_end = t_end[sort]
    p_new = p_new[sort]

    # get rid of extra values beyond the maximum allowable time

    try:
        ind_stop = np.where(np.diff(t_end) > 0.0)[0][-1] + 1

    except IndexError:
        if np.all(np.diff(t_end) == 0.0):
            warnings.warn("All trajectories hit the end point. Returning max_p.")
            return max_p

        else:
            raise IndexError

    if ind_stop == 1:
        ind_stop = 2

    p_test = p_new.copy()
    t_test = t_end.copy()
    p_new = p_new[:ind_stop]
    t_end = t_end[:ind_stop]

    # setup spline
    spline = CubicSpline(t_end, p_new)

    # return proper p value
    return spline(t_out * YRSID_SI).item()


# data history is saved here nased on version nunber
record_by_version = {
    "1.0.0": 3981654,
    "1.1.0": 3981654,
    "1.1.1": 3981654,
    "1.1.2": 3981654,
    "1.1.3": 3981654,
    "1.1.4": 3981654,
    "1.1.5": 3981654,
}


def check_for_file_download(fp, few_dir, version_string=None):
    """Download files direct from zenodo.

    This function downloads the files from zenodo as they are needed. They are
    downloaded based on the associated record for each version (`record_by_version`).

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
    if version_string is not None:
        if version_string not in record_by_version:
            raise ValueError(
                "The version_string provided ({}) does not exist.".format(
                    version_string
                )
            )
    else:
        version_string = few.__version__

    # check if the files directory exists
    try:
        os.listdir(few_dir + "few/files/")

    # if not, create it
    except OSError:
        os.mkdir(few_dir + "few/files/")

    # check if the file is in the files filder
    # if not, download it from zenodo
    if fp not in os.listdir(few_dir + "few/files/"):
        warnings.warn(
            "The file {} did not open sucessfully. It will now be downloaded to the proper location.".format(
                fp
            )
        )

        # get record number based on version
        record = record_by_version.get(version_string)

        # url to zenodo API
        url = "https://zenodo.org/record/" + str(record) + "/files/" + fp

        # run wget from terminal to get the folder
        # download to proper location
        subprocess.run(["wget", "--no-check-certificate", url])

        # move it into the files folder
        os.rename(fp, few_dir + "few/files/" + fp)


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
