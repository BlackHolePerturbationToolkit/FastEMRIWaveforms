import numpy as np

try:
    import cupy as cp

    gpu = True

except (ImportError, ModuleNotFoundError) as e:
    gpu = False


def pointer_adjust(func):
    def func_wrapper(*args, **kwargs):
        targs = []
        for arg in args:
            if gpu:
                if isinstance(arg, cp.ndarray):
                    targs.append(arg.data.mem.ptr)
                    continue

            if isinstance(arg, np.ndarray):
                targs.append(arg.__array_interface__["data"][0])
                continue

            try:
                targs.append(arg.ptr)
                continue
            except AttributeError:
                targs.append(arg)

        return func(*targs, **kwargs)

    return func_wrapper
