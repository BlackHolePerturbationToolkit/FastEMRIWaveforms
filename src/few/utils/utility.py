# Utilities to aid in FastEMRIWaveforms Packages

from math import acos, cos, sqrt
from typing import Optional

import numpy as np
from numba import njit
from scipy.optimize import brentq

from few.utils.exceptions import TrajectoryOffGridException
from few.utils.globals import get_first_backend, get_logger

from .constants import PI, YRSID_SI


def get_overlap(
    time_series_1: np.ndarray, time_series_2: np.ndarray, use_gpu: bool = False
) -> float:
    r"""Calculate the overlap.

    Takes two time series and finds which one is shorter in length. It then
    shortens the longer time series if necessary. Then it performs a
    normalized correlation calulation on the two time series to give the
    overlap. The overlap of :math:`a(t)` and
    :math:`b(t)`, :math:`\gamma_{a,b}`, is given by,

    .. math:: \gamma_{a,b} = <a,b>/(<a,a><b,b>)^{(1/2)},

    where :math:`<a,b>` is the inner product of the two time series.

    args:
        time_series_1: Strain time series 1.
        time_series_2: Strain time series 2.
        use_gpu: If True use cupy. If False, use numpy. Default
            is False.

    """

    # adjust arrays based on GPU usage
    if use_gpu:
        import cupy as xp
    else:
        xp = np

    if not isinstance(time_series_1, xp.ndarray):
        time_series_1 = xp.asarray(time_series_1)
    if not isinstance(time_series_2, xp.ndarray):
        time_series_2 = xp.asarray(time_series_2)

    # get the lesser of the two lengths
    min_len = int(np.min([len(time_series_1), len(time_series_2)]))

    if len(time_series_1) != len(time_series_2):
        import logging

        get_logger().warning(
            "The two time series are not the same length ({} vs {}). The calculation will run with length {} starting at index 0 for both arrays.".format(
                len(time_series_1), len(time_series_2), min_len
            ),
            stack_info=get_logger().getEffectiveLevel() <= logging.DEBUG,
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
    return ac.item().real if use_gpu else ac.real


def get_mismatch(
    time_series_1: np.ndarray, time_series_2: np.ndarray, use_gpu: bool = False
) -> float:
    """Calculate the mismatch.

    The mismatch is 1 - overlap. Therefore, see documentation for
    :func:`few.utils.utility.overlap` for information on the overlap
    calculation.

    args:
        time_series_1: Strain time series 1.
        time_series_2: Strain time series 2.
        use_gpu: If True use cupy. If False, use numpy. Default
            is False.

    """
    overlap = get_overlap(time_series_1, time_series_2, use_gpu=use_gpu)
    return 1.0 - overlap


@njit(fastmath=False)
def _solveCubic(A2, A1, A0):
    # Coefficients
    a = 1.0  # coefficient of r^3
    b = A2  # coefficient of r^2
    c = A1  # coefficient of r^1
    d = A0  # coefficient of r^0

    # Calculate p and q
    p = (3.0 * a * c - b * b) / (3.0 * a * a)
    q = (2.0 * b * b * b - 9.0 * a * b * c + 27.0 * a * a * d) / (27.0 * a * a * a)

    # Calculate discriminant
    discriminant = q * q / 4.0 + p * p * p / 27.0

    if discriminant >= 0:
        # One real root and two complex conjugate roots
        u = (-q / 2.0 + sqrt(discriminant)) ** (1 / 3)
        v = (-q / 2.0 - sqrt(discriminant)) ** (1 / 3)
        root = u + v - b / (3.0 * a)
        # cout << "Real Root: " << root << endl

        # imaginaryPart(-sqrt(3.0) / 2.0 * (u - v), 0.5 * (u + v))
        imaginaryPart = 0.5 * (u + v)
        root2 = -0.5 * (u + v) - b / (3.0 * a) + imaginaryPart
        root3 = -0.5 * (u + v) - b / (3.0 * a) - imaginaryPart
        # cout << "Complex Root 1: " << root2 << endl
        # cout << "Complex Root 2: " << root3 << endl
        ra = -0.5 * (u + v) - b / (3.0 * a)
        rp = -0.5 * (u + v) - b / (3.0 * a)
        r3 = root
    # } else if (discriminant == 0) {
    #     # All roots are real and at least two are equal
    #     u = cbrt(-q/2.)
    #     v = cbrt(-q/2.)
    #     root = u + v - b/(3.*a)
    #     # cout << "Real Root: " << root << endl
    #     # cout << "Real Root (equal to above): " << root << endl
    #     # complex<double> root2 = -0.5 * (u + v) - b / (3 * a)
    #     # cout << "Complex Root: " << root2 << endl
    #     *ra = -0.5 * (u + v) - b / (3. * a)
    #     *rp = -0.5 * (u + v) - b / (3. * a)
    #     *r3 = root
    else:
        # All three roots are real and different
        r = sqrt(-p / 3.0)
        theta = acos(-q / (2.0 * r * r * r))
        root1 = 2.0 * r * cos(theta / 3.0) - b / (3.0 * a)
        root2 = 2.0 * r * cos((theta + 2.0 * PI) / 3.0) - b / (3.0 * a)
        root3 = 2.0 * r * cos((theta - 2.0 * PI) / 3.0) - b / (3.0 * a)

        ra = root1
        rp = root3
        r3 = root2

    return rp, ra, r3
    # cout << "ra: " << *ra << endl
    # cout << "rp: " << *rp << endl
    # cout << "r3: " << *r3 << endl


@njit(fastmath=False)
def _brentq_jit(f, a, b, args, tol):
    # Machine epsilon for double precision
    eps = 2.220446049250313e-16

    fa = f(a, args)
    fb = f(b, args)

    # Check that f(a) and f(b) have different signs
    if fa == 0.0 or fb == 0.0:
        return a if fa == 0.0 else b

    if fa * fb > 0.0:
        raise ValueError("f(a) and f(b) must have different signs.")

    c = a
    fc = fa
    d = b - a
    e = d

    while True:
        if abs(fc) < abs(fb):
            a = b
            b = c
            c = a
            fa = fb
            fb = fc
            fc = fa

        tol1 = 2.0 * eps * abs(b) + 0.5 * tol
        xm = 0.5 * (c - b)

        if abs(xm) <= tol1 or fb == 0.0:
            # within tolerance -> return root
            return b

        # Check if bisection is forced
        if abs(e) >= tol1 and abs(fa) > abs(fb):
            s = fb / fa
            if a == c:
                # Linear interpolation
                p = 2.0 * xm * s
                q = 1.0 - s
            else:
                # Inverse quadratic interpolation
                q = fa / fc
                r = fb / fc
                p = s * (2.0 * xm * q * (q - r) - (b - a) * (r - 1.0))
                q = (q - 1.0) * (r - 1.0) * (s - 1.0)

            if p > 0.0:
                q = -q
            else:
                p = -p

            if (2.0 * p < 3.0 * xm * q - abs(tol1 * q)) and (p < abs(0.5 * e * q)):
                # Accept interpolation
                d = p / q
            else:
                # Bisection step
                d = xm
                e = d
        else:
            # Bisection step
            d = xm
            e = d

        a = b
        fa = fb
        if abs(d) > tol1:
            b += d
        else:
            b += tol1 if xm > 0 else -tol1

        fb = f(b, args)
        if fb * fc > 0.0:
            c = a
            fc = fa
            d = b - a
            e = d


def get_at_t(
    traj_module: object,
    traj_args: list[float],
    bounds: list[float],
    t_out: float,
    index_of_interest: int,
    traj_kwargs: Optional[dict] = None,
    xtol: float = 2e-12,
    rtol: float = 8.881784197001252e-16,
) -> float:
    """Root finding wrapper using Brent's method.

    This function uses scipy's brentq routine to find root.

    arguments:
        traj_module: Instantiated trajectory module. It must output
            the time array of the trajectory sparse trajectory as the first
            output value in the tuple.
        traj_args: List of arguments for the trajectory function.
            p is removed. **Note**: It must be a list, not a tuple because the
            new p values are inserted into the argument list.
        bounds: Minimum and maximum values over which brentq will search for a root.
        t_out: The desired length of time for the waveform.
        index_of_interest: Index where to insert the new values in
            the :code:`traj_args` list.
        traj_kwargs: Keyword arguments for :code:`traj_module`.
            Default is an empty dict.
        xtol: Absolute tolerance of the brentq root-finding - see :code: `np.allclose()` for details.
            Defaults to 2e-12 (scipy default).
        rtol: Relative tolerance of the brentq root-finding - see :code: `np.allclose()` for details.
            Defaults to ~8.8e-16 (scipy default).

    returns:
        Root value.

    """
    if traj_kwargs is None:
        traj_kwargs = {}

    def get_time_root(val, traj, inj_args, traj_kwargs, t_out, ind_interest):
        """
        Function with one p root at T = t_outp, for brentq input.
        """
        inputs = inj_args.copy()
        inputs.insert(ind_interest, val)
        traj_kwargs["T"] = t_out * 2.0
        out = traj(*inputs, **traj_kwargs)
        try:
            return out[0][-1] - t_out * YRSID_SI
        except IndexError:  # trajectory must have started at p_sep
            return -t_out * YRSID_SI

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
    traj_module: object,
    t_out: float,
    traj_args: list[float],
    index_of_p: int = 3,
    index_of_a: int = 2,
    index_of_e: int = 4,
    index_of_x: int = 5,
    bounds: list[Optional[float]] = None,
    lower_bound_buffer: float = 1e-6,
    upper_bound_maximum: float = 1e6,
    **kwargs,
) -> float:
    """Find the value of p that will give a specific length inspiral using Brent's method.

    If you want to generate an inspiral that is a specific length, you
    can adjust p accordingly. This function tells you what that value of p
    is based on the trajectory module and other input parameters at a
    desired time of observation.

    This function uses scipy's brentq routine to find the (presumed only)
    value of p that gives a trajectory of duration t_out.

    arguments:
        traj_module: Instantiated trajectory module. It must output
            the time array of the trajectory sparse trajectory as the first
            output value in the tuple.
        t_out: The desired length of time for the waveform.
        traj_args: List of arguments for the trajectory function.
            p is removed. **Note**: It must be a list, not a tuple because the
            new p values are inserted into the argument list.
        index_of_p: Index where to insert the new p values in
            the :code:`traj_args` list. Default is 3.
        index_of_a: Index of a in provided :code:`traj_module` arguments. Default is 2.
        index_of_e: Index of e0 in provided :code:`traj_module` arguments. Default is 4.
        index_of_x: Index of x0 in provided :code:`traj_module` arguments. Default is 5.
        bounds: Minimum and maximum values of p over which brentq will search for a root.
            If not given, will be set to the minimum and maximum values of p for the trajectory modules.
        **kwargs: Keyword arguments for :func:`get_at_t`.
        lower_bound_buffer: A float that offsets the lower bound by a small number to prevent starting on the separatrix
            If not provided, it will default to 1e-6
        upper_bound_maximum: A float that sets the maximum value of the upper bound to a large finite number if max_p returns inf
            If not provided, it will default to 1e6
    returns:
        Value of p that creates the proper length trajectory.

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
        bounds = [None, None]

    if bounds[0] is None:
        if not enforce_schwarz_sep:
            bounds[0] = traj_module.func.min_p(
                traj_args[index_of_e], x=traj_args[index_of_x], a=traj_args[index_of_a]
            )
        else:
            bounds[0] = min(
                traj_module.func.min_p(
                    traj_args[index_of_e],
                    x=traj_args[index_of_x],
                    a=traj_args[index_of_a],
                ),
                6 + 2 * traj_args[index_of_e],
            )

    if bounds[1] is None:
        bounds[1] = traj_module.func.max_p(
            traj_args[index_of_e], x=traj_args[index_of_x], a=traj_args[index_of_a]
        )
    # Adding a buffer to prevent starting trajectories on the separatrix
    bounds[0] += lower_bound_buffer

    # Prevent the rootfinder from starting at infinity
    if bounds[1] == float("inf"):
        bounds[1] = upper_bound_maximum

    # With the varying bounds of eccentricity used for KerrEccEqFlux,
    # it is possible to have no solution within the bounds of the interpolants
    # It might also be possible for the trajectory to evolve off of the grid
    while bounds[0] < bounds[1]:
        try:
            traj_pars = [
                traj_args[0],
                traj_args[1],
                traj_args[index_of_a],
                bounds[0],
                traj_args[index_of_e],
                traj_args[index_of_x],
            ]
            t, p, e, xI, Phi_phi, Phi_theta, Phi_r = traj_module(
                *traj_pars, T=t_out * 1.001
            )
            if t[-1] >= t_out * YRSID_SI:
                raise ValueError(
                    "No solution found within the bounds of the interpolants."
                )
            break
        except TrajectoryOffGridException:
            # Trajectory is off the grid
            # Increase lower bound and try again
            bounds[0] += 1e-2

    root = get_at_t(traj_module, traj_args, bounds, t_out, index_of_p, **kwargs)
    return root


def get_m2_at_t(
    traj_module: object,
    t_out: float,
    traj_args: list[float],
    index_of_m2: int = 1,
    bounds: list[Optional[float]] = None,
    **kwargs,
) -> float:
    """Find the value of m2 that will give a specific length inspiral using Brent's method.

    If you want to generate an inspiral that is a specific length, you
    can adjust m2 accordingly. This function tells you what that value of m2
    is based on the trajectory module and other input parameters at a
    desired time of observation.

    This function uses scipy's brentq routine to find the (presumed only)
    value of m2 that gives a trajectory of duration t_out.

    arguments:
        traj_module: Instantiated trajectory module. It must output
            the time array of the trajectory sparse trajectory as the first
            output value in the tuple.
        t_out: The desired length of time for the waveform.
        traj_args: List of arguments for the trajectory function.
            p is removed. **Note**: It must be a list, not a tuple because the
            new p values are inserted into the argument list.
        index_of_m2: Index where to insert the new p values in
            the :code:`traj_args` list. Default is 1.
        bounds: Minimum and maximum values of m2 over which brentq will search for a root.
            If not given, will be set to [1e-1, m1]. To supply only one of these two limits, set the
            other limit to None.
        **kwargs: Keyword arguments for :func:`get_at_t`.

    returns:
        Value of m2 that creates the proper length trajectory.

    """

    # fix bounds
    if bounds is None:
        bounds = [1e-1, traj_args[0]]

    elif bounds[0] is None:
        bounds[0] = 1e-1

    elif bounds[1] is None:
        bounds[1] = traj_args[0]

    # Check lower bound
    traj_pars = [
        traj_args[0],
        bounds[0],
        traj_args[1],
        traj_args[2],
        traj_args[3],
        traj_args[4],
    ]
    t, p, e, xI, Phi_phi, Phi_theta, Phi_r = traj_module(*traj_pars, T=t_out * 1.001)
    if t[-1] <= t_out * YRSID_SI:
        raise ValueError("No solution found within the bounds for secondary mass.")

    # Check lower bound
    traj_pars = [
        traj_args[0],
        bounds[1],
        traj_args[1],
        traj_args[2],
        traj_args[3],
        traj_args[4],
    ]
    t, p, e, xI, Phi_phi, Phi_theta, Phi_r = traj_module(*traj_pars, T=t_out * 1.001)
    if t[-1] >= t_out * YRSID_SI:
        raise ValueError("No solution found within the bounds for secondary mass.")

    root = get_at_t(traj_module, traj_args, bounds, t_out, index_of_m2, **kwargs)
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


def wrapper(*args, **kwargs):
    """Function to convert array and C/C++ class arguments to ptrs

    This function checks the object type. If it is a cupy or numpy array,
    it will determine its pointer by calling the proper attributes. If you design
    a Cython class to be passed through python, it must have a :code:`ptr`
    attribute.

    If you use this function, you must convert input arrays to size_t data type in Cython and
    then properly cast the pointer as it enters the c++ function. See the
    Cython codes
    `here <https://github.com/BlackHolePerturbationToolkit/FastEMRIWaveforms/tree/master/few/cutils/src>`_
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

    best_backend = get_first_backend(["cuda12x", "cuda11x", "cpu"])

    # args first
    for arg in args:
        if best_backend.uses_cupy:
            # cupy arrays
            if isinstance(arg, best_backend.xp.ndarray):
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
        if best_backend.uses_cupy:
            # cupy arrays
            if isinstance(arg, best_backend.xp.ndarray):
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
    `here <https://github.com/BlackHolePerturbationToolkit/FastEMRIWaveforms/tree/master/few/cutils/src>`_
    for examples.

    """

    def func_wrapper(*args, **kwargs):
        # get pointers
        targs, tkwargs = wrapper(*args, **kwargs)
        return func(*targs, **tkwargs)

    return func_wrapper
