"""Pointer conversion helpers for the FEW Cython extensions.

The compiled modules (``pyAmpInterp2D``, ``pyinterp``, ``pyAAK``, ``pymatmul``; sources
``cutils/*.pyx``) receive arrays as raw addresses. :func:`wrapper` converts the arguments of
one call, and :func:`pointer_adjust` is the decorator the ``.pyx`` files apply
(``interp2D`` in ``pyampinterp2D.pyx`` is decorated; ``pyinterp.pyx``, ``gpuAAKWrap.pyx``
and ``pymatmul.pyx`` call :func:`wrapper` directly). On the production EMRI path every
amplitude evaluation (``AmpInterp2D`` -> ``backend.interp2D``) goes through
:func:`pointer_adjust`.

The module tries ``import cupy`` once at import (module flag ``gpu``); without cupy only
numpy arrays are recognised.
"""

# check to see if cupy is available for gpus
try:
    import cupy as cp

    gpu = True

except (ImportError, ModuleNotFoundError):
    gpu = False

import numpy as np


def obj_to_ptr(obj):
    """Return the address of an array or Cython object, or the object itself.

    Args:
        obj (object): A ``cupy.ndarray`` (device address ``obj.data.mem.ptr``; recognised
            only when cupy imported), a ``numpy.ndarray`` (host address from
            ``__array_interface__``), an object with a ``ptr`` attribute (Cython classes), or
            anything else (returned unchanged).

    Returns:
        int or object: The address as an integer, or ``obj`` unchanged.

    Note:
        For numpy the address is that of the first element. For cupy it is the base address
        of the underlying allocation (``data.mem.ptr``, not ``data.ptr``), which differs
        from the first element for a view with a non-zero offset. Pass contiguous arrays
        that own their memory, with the dtype the C++ side expects.
    """
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
    """Convert array and C/C++ class arguments to pointers.

    Applies :func:`obj_to_ptr` to every positional and keyword argument: cupy and numpy
    arrays become their integer addresses, objects with a ``ptr`` attribute (Cython
    classes) become that attribute, everything else passes through unchanged.

    The receiving Cython function must declare array arguments as ``size_t`` and cast them
    to the proper pointer type before calling the C++ function; see the ``.pyx`` files in
    ``src/few/cutils`` (e.g. ``pyinterp.pyx``) for examples.

    Args:
        *args: Positional arguments of the target function.
        **kwargs: Keyword arguments of the target function.

    Returns:
        tuple[list, dict]: ``(targs, tkwargs)``, the converted positional arguments (a list)
        and keyword arguments.
    """
    # declare target containers
    targs = [obj_to_ptr(arg) for arg in args]
    tkwargs = {key: obj_to_ptr(arg) for key, arg in kwargs.items()}

    return (targs, tkwargs)


def pointer_adjust(func):
    """Decorate a Cython function so it accepts numpy / cupy arrays and Cython objects.

    The returned function converts its arguments with :func:`wrapper` and calls ``func``
    with the converted values. ``pyampinterp2D.pyx`` decorates ``interp2D`` with it, so
    :meth:`few.amplitude.ampinterp2d.AmpInterp2D.__call__` can pass arrays directly.

    The decorated Cython function must declare array arguments as ``size_t`` and cast them
    to the proper pointer type before calling the C++ function; see
    ``src/few/cutils/pyampinterp2D.pyx`` for an example.

    Args:
        func (Callable): Function taking pointer (``size_t``) arguments.

    Returns:
        Callable: Wrapper with the same positional / keyword interface as ``func`` that
        returns ``func``'s return value.
    """

    def func_wrapper(*args, **kwargs):
        # get pointers
        targs, tkwargs = wrapper(*args, **kwargs)
        return func(*targs, **tkwargs)

    return func_wrapper
