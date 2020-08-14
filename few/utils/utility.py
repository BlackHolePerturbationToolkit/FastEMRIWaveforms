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

# check to see if cupy is available for gpus
try:
    import cupy as cp

except (ImportError, ModuleNotFoundError) as e:
    import numpy as np

import few


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

    min_len = int(np.min([len(time_series_1), len(time_series_2)]))
    time_series_1_fft = xp.fft.fft(time_series_1[:min_len])
    time_series_2_fft = xp.fft.fft(time_series_2[:min_len])
    ac = xp.dot(time_series_1_fft.conj(), time_series_2_fft) / xp.sqrt(
        xp.dot(time_series_1_fft.conj(), time_series_1_fft)
        * xp.dot(time_series_2_fft.conj(), time_series_2_fft)
    )

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
        return xp.log(-(21 / 10) - 2 * e + p)

    else:
        return np.log(-(21 / 10) - 2 * e + p)


# data history is saved here nased on version nunber
record_by_version = {"1.0.0": 3981654}


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
        subprocess.run(["wget", url])

        # move it into the files folder
        os.rename(fp, few_dir + "few/files/" + fp)
