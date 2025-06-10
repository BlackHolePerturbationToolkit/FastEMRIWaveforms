# check to see if cupy is available for gpus
try:
    import cupy as cp

    gpu = True

except (ImportError, ModuleNotFoundError):
    gpu = False

import numpy as np


def obj_to_ptr(obj):
    if gpu and isinstance(obj, cp.ndarray):
        return obj.data.mem.ptr

    if isinstance(obj, np.ndarray):
        return obj.__array_interface__["data"][0]

    try:
        # cython classes
        return obj.ptr
    except AttributeError:
        # regular argument
        return obj


def wrapper(*args, **kwargs):
    """Function to convert array and C/C++ class arguments to ptrs

    This function checks the object type. If it is a cupy or numpy array,
    it will determine its pointer by calling the proper attributes. If you
    design a Cython class to be passed through python, it must have a :code:`ptr`
    attribute.

    If you use this function, you must convert input arrays to size_t data type
    in Cython and then properly cast the pointer as it enters the c++ function.
    See the Cython codes
    `here <https://github.com/BlackHolePerturbationToolkit/FastEMRIWaveforms/tree/master/src/few/cutils>`_
    for examples.

    args:
        *args (list): list of the arguments for a function.
        **kwargs (dict): dictionary of keyword arguments to be converted.

    returns:
        Tuple: (targs, tkwargs) where t indicates target (with pointer values
            rather than python objects).

    """
    # declare target containers
    targs = [obj_to_ptr(arg) for arg in args]
    tkwargs = {key: obj_to_ptr(arg) for key, arg in kwargs.items()}

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
