"""Cython binding for the Apple Metal time-domain mode sum."""

from libc.stddef cimport size_t

import numpy as np
from operator import index


def _buffer(array, dtype, size, name):
    if not isinstance(array, np.ndarray) or array.dtype != dtype:
        raise TypeError(f"{name} must be a NumPy array of dtype {np.dtype(dtype)}")
    if array.size < size:
        raise ValueError(f"{name} needs {size} elements, got {array.size}")
    return np.require(array, requirements=["C", "A"])


cdef extern from "metal_sum.hh":
    const char *few_metal_sum_last_error()
    void *few_metal_sum_context_create()
    void few_metal_sum_context_destroy(void *opaque_context)
    int few_metal_sum_evaluate(
        void *opaque_context,
        double *waveform,
        const double *interpolation,
        const double *phase_times,
        const double *phase_coefficients,
        const int *m_values,
        const int *k_values,
        const int *n_values,
        int init_length,
        int output_length,
        int mode_count,
        const double *ylms,
        double delta_t,
        const double *trajectory_times,
        double *gpu_seconds,
    )


cdef class MetalSummation:
    cdef void *_context
    cdef double _last_gpu_seconds

    def __cinit__(self):
        self._context = few_metal_sum_context_create()
        self._last_gpu_seconds = 0.0
        if self._context == NULL:
            self._raise_last_error("Metal context creation failed")

    cdef void _raise_last_error(self, str fallback) except *:
        cdef const char *message = few_metal_sum_last_error()
        if message == NULL:
            raise RuntimeError(fallback)
        raise RuntimeError((<bytes>message).decode("utf-8"))

    def close(self):
        """Release the native Metal queue and runtime-compiled pipeline."""
        if self._context != NULL:
            few_metal_sum_context_destroy(self._context)
            self._context = NULL

    def __dealloc__(self):
        if self._context != NULL:
            few_metal_sum_context_destroy(self._context)
            self._context = NULL

    @property
    def closed(self):
        return self._context == NULL

    @property
    def last_gpu_seconds(self):
        return self._last_gpu_seconds

    def get_waveform_wrap(
        self,
        waveform,
        interpolation,
        phase_times,
        phase_coefficients,
        m_values,
        k_values,
        n_values,
        init_length,
        output_length,
        mode_count,
        ylms,
        delta_t,
        trajectory_times,
        _device,
    ):
        """Evaluate FEW's existing 14-argument time-domain summation ABI."""
        cdef int knots = index(init_length)
        cdef int samples = index(output_length)
        cdef int modes = index(mode_count)
        cdef double dt = delta_t
        cdef int status

        if self._context == NULL:
            raise RuntimeError("Metal summation context is closed")
        if knots < 2 or samples < 1 or modes < 1 or not np.isfinite(dt) or dt <= 0:
            raise ValueError("Metal summation needs positive dimensions and dt, and at least two knots")
        # MSL uses uint to index all four planes of the complex amplitude spline.
        if knots > (2**32 - 1) // 8 // modes:
            raise ValueError("Metal spline exceeds the 32-bit shader index range")
        if not isinstance(waveform, np.ndarray) or waveform.ndim != 1 or not waveform.flags.writeable:
            raise ValueError("waveform must be a writable one-dimensional NumPy array")

        out = _buffer(waveform[:samples], np.complex128, samples, "waveform")
        interpolation = _buffer(interpolation, np.float64, 8 * int(knots) * modes, "interpolation")
        phase_times = _buffer(phase_times, np.float64, knots, "phase_times")
        phase_coefficients = _buffer(phase_coefficients, np.float64, 24 * int(knots - 1), "phase_coefficients")
        m_values = _buffer(m_values, np.int32, modes, "m_values")
        k_values = _buffer(k_values, np.int32, modes, "k_values")
        n_values = _buffer(n_values, np.int32, modes, "n_values")
        ylms = _buffer(ylms, np.complex128, 2 * int(modes), "ylms")
        trajectory_times = _buffer(trajectory_times, np.float64, knots, "trajectory_times")
        status = few_metal_sum_evaluate(
            self._context, <double *><size_t>out.ctypes.data,
            <const double *><size_t>interpolation.ctypes.data,
            <const double *><size_t>phase_times.ctypes.data,
            <const double *><size_t>phase_coefficients.ctypes.data,
            <const int *><size_t>m_values.ctypes.data,
            <const int *><size_t>k_values.ctypes.data,
            <const int *><size_t>n_values.ctypes.data,
            knots, samples, modes, <const double *><size_t>ylms.ctypes.data, dt,
            <const double *><size_t>trajectory_times.ctypes.data,
            &self._last_gpu_seconds,
        )
        if status != 0:
            self._raise_last_error("Metal summation failed")
        if not (waveform.flags.c_contiguous and waveform.flags.aligned):
            waveform[:samples] = out
