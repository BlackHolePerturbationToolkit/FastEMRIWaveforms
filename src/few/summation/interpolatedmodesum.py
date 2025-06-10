# Interpolated summation of modes in python for the FastEMRIWaveforms Package

import numpy as np

from few.utils.globals import get_logger

# Python imports
from ..utils.baseclasses import BackendLike, ParallelModuleBase
from .base import SummationBase


class CubicSplineInterpolant(ParallelModuleBase):
    """GPU-accelerated Multiple Cubic Splines

    This class produces multiple cubic splines on a GPU. It has a CPU option
    as well. The cubic splines are produced with "not-a-knot" boundary
    conditions.

    This class can be run out of Python similar to
    `scipy.interpolate.CubicSpline <https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.CubicSpline.html#cubicspline>`_.
    However, the most efficient way to use this method is in a customized
    cuda kernel. See the
    `source code for the interpolated summation <https://github.com/BlackHolePerturbationToolkit/FastEMRIWaveforms/blob/master/few/cutils/src/interpolate.cu>`_
    in cuda for an example of this.

    This class can be run on GPUs and CPUs.

    args:
        t: t values as input for the spline. If 2D, must have shape (ninterps, length).
        y_all: y values for the spline; shape is (length,) or (ninterps, length).
        **kwargs: Optional keyword arguments for the base class:
            :class:`few.utils.baseclasses.ParallelModuleBase`.
    """

    def __init__(self, t: np.ndarray, y_all: np.ndarray, **kwargs):
        ParallelModuleBase.__init__(self, **kwargs)

        y_all = self.xp.atleast_2d(y_all)

        # hardcoded,only cubic spline is available
        self.degree = 3
        """int: Degree of the spline. Only cubic splines are available."""

        # get quantities related to how many interpolations
        ninterps, length = y_all.shape
        self.ninterps = ninterps
        """int: Number of interpolants."""
        self.length = length
        """int: Length of the input t array."""

        # store this for reshaping flattened arrays
        self.reshape_shape = (self.degree + 1, ninterps, length)
        """tuple: Shape of the spline coefficients array."""

        # interp array is (y, c1, c2, c3)
        interp_array = self.xp.zeros(self.reshape_shape)

        # fill y
        interp_array[0] = y_all

        interp_array = interp_array.flatten()

        # arrays to store banded matrix and solution
        B = self.xp.zeros((ninterps * length,))
        upper_diag = self.xp.zeros_like(B)
        diag = self.xp.zeros_like(B)
        lower_diag = self.xp.zeros_like(B)

        if t.ndim == 1:
            if len(t) < 2:
                raise ValueError("t must have length greater than 2.")

            # could save memory by adjusting c code to treat 1D differently
            self.t = self.xp.tile(t, (ninterps, 1)).flatten().astype(self.xp.float64)

        elif t.ndim == 2:
            if t.shape[1] < 2:
                raise ValueError("t must have length greater than 2 along time axis.")

            self.t = t.flatten().copy().astype(self.xp.float64)

        else:
            raise ValueError("t must be 1 or 2 dimensions.")

        # perform interpolation
        self.interpolate_arrays(
            self.t,
            interp_array,
            ninterps,
            length,
            B,
            upper_diag,
            diag,
            lower_diag,
        )

        # set up storage of necessary arrays

        self.interp_array = self.xp.transpose(
            interp_array.reshape(self.reshape_shape), [0, 2, 1]
        ).flatten()

        self.interp_array
        """Flattened array containing all spline coefficients."""

        # update reshape_shape
        self.reshape_shape = (self.degree + 1, length, ninterps)
        self.t = self.t.reshape((ninterps, length))

    @property
    def interpolate_arrays(self) -> callable:
        """GPU or CPU waveform generation."""
        return self.backend.interpolate_arrays_wrap

    @classmethod
    def supported_backends(cls):
        return cls.GPU_RECOMMENDED()

    @property
    def y(self) -> np.ndarray:
        """y values associated with the spline"""
        return self.interp_array.reshape(self.reshape_shape)[0].T

    @property
    def c1(self) -> np.ndarray:
        """constants for the linear term"""
        return self.interp_array.reshape(self.reshape_shape)[1].T

    @property
    def c2(self) -> np.ndarray:
        """constants for the quadratic term"""
        return self.interp_array.reshape(self.reshape_shape)[2].T

    @property
    def c3(self) -> np.ndarray:
        """constants for the cubic term"""
        return self.interp_array.reshape(self.reshape_shape)[3].T

    def _get_inds(self, tnew: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        # find where in the old t array the new t values split

        inds = self.xp.zeros((self.ninterps, tnew.shape[1]), dtype=int)

        # Optional TODO: remove loop ? if speed needed
        for i, (t, tnew_i) in enumerate(zip(self.t, tnew)):
            inds[i] = self.xp.searchsorted(t, tnew_i, side="right") - 1

            # fix end value
            inds[i][tnew_i == t[-1]] = len(t) - 2

        # get values outside the edges
        inds_bad_left = tnew < self.t[:, 0][:, None]
        inds_bad_right = tnew > self.t[:, -1][:, None]

        if np.any(inds < 0) or np.any(inds >= self.t.shape[1]):
            get_logger().warning(
                "(CubicSplineInterpolant) Warning: New t array outside bounds of input t array. These points are filled with edge values."
            )
        return inds, inds_bad_left, inds_bad_right

    def __call__(self, tnew: np.ndarray, deriv_order: int = 0) -> np.ndarray:
        """Evaluation function for the spline

        Put in an array of new t values at which all interpolants will be
        evaluated. If t values are outside of the original t values, edge values
        are used to fill the new array at these points.

        args:
            tnew: Array of new t values. All of these new
                t values must be within the bounds of the input t values,
                including the beginning t and **excluding** the ending t. If tnew is 1D
                and :code:`self.t` is 2D, tnew will be cast to 2D.
            deriv_order: Order of the derivative to evaluate. Default
                is 0 meaning the basic spline is evaluated. deriv_order of 1, 2, and 3
                correspond to their respective derivatives of the spline. Unlike :code:`scipy`,
                this is purely an evaluation of the derivative values, not a new class to evaluate
                for the derivative.

        raises:
            ValueError: a new t value is not in the bounds of the input t array.

        returns:
            1D or 2D array of evaluated spline values (or derivatives).

        """

        tnew = self.xp.atleast_1d(tnew)

        if tnew.ndim == 2:
            if tnew.shape[0] != self.t.shape[0]:
                raise ValueError(
                    "If providing a 2D tnew array, must have same number of interpolants as was entered during initialization."
                )

        # copy input to all splines
        elif tnew.ndim == 1:
            tnew = self.xp.tile(tnew, (self.t.shape[0], 1))

        tnew = self.xp.atleast_2d(tnew)

        # get indices into spline
        inds, inds_bad_left, inds_bad_right = self._get_inds(tnew)

        # x value for spline

        # indexes for which spline
        inds0 = np.tile(np.arange(self.ninterps), (tnew.shape[1], 1)).T
        t_here = self.t[(inds0.flatten(), inds.flatten())].reshape(
            self.ninterps, tnew.shape[1]
        )

        x = tnew - t_here
        x2 = x * x
        x3 = x2 * x

        # get spline coefficients
        y = self.y[(inds0.flatten(), inds.flatten())].reshape(
            self.ninterps, tnew.shape[1]
        )
        c1 = self.c1[(inds0.flatten(), inds.flatten())].reshape(
            self.ninterps, tnew.shape[1]
        )
        c2 = self.c2[(inds0.flatten(), inds.flatten())].reshape(
            self.ninterps, tnew.shape[1]
        )
        c3 = self.c3[(inds0.flatten(), inds.flatten())].reshape(
            self.ninterps, tnew.shape[1]
        )

        # evaluate spline
        if deriv_order == 0:
            out = y + c1 * x + c2 * x2 + c3 * x3
            # fix bad values
            if self.xp.any(inds_bad_left):
                temp = self.xp.tile(self.y[:, 0], (tnew.shape[1], 1)).T
                out[inds_bad_left] = temp[inds_bad_left]

            if self.xp.any(inds_bad_right):
                temp = self.xp.tile(self.y[:, -1], (tnew.shape[1], 1)).T
                out[inds_bad_right] = temp[inds_bad_right]

        else:
            # derivatives
            if self.xp.any(inds_bad_right) or self.xp.any(inds_bad_left):
                raise ValueError(
                    "x points outside of the domain of the spline are not supported when taking derivatives."
                )
            if deriv_order == 1:
                out = c1 + 2 * c2 * x + 3 * c3 * x2
            elif deriv_order == 2:
                out = 2 * c2 + 6 * c3 * x
            elif deriv_order == 3:
                out = 6 * c3
            else:
                raise ValueError("deriv_order must be within 0 <= deriv_order <= 3.")

        return out.squeeze()


class InterpolatedModeSum(SummationBase):
    """Create waveform by interpolating a sparse trajectory.

    It interpolates all of the modes of interest and phases at sparse
    trajectories. Within the summation phase, the values are calculated using
    the interpolants and summed.

    This class can be run on GPUs and CPUs.

    """

    def __init__(self, force_backend: BackendLike = None, **kwargs):
        SummationBase.__init__(self, **kwargs, force_backend=force_backend)

    @property
    def get_waveform(self) -> callable:
        """GPU or CPU waveform generation."""
        return self.backend.get_waveform_wrap

    @classmethod
    def supported_backends(cls):
        return cls.GPU_RECOMMENDED()

    def sum(
        self,
        t: np.ndarray,
        teuk_modes: np.ndarray,
        ylms: np.ndarray,
        phase_interp_t: np.ndarray,
        phase_interp_coeffs: np.ndarray,
        l_arr: np.ndarray,
        m_arr: np.ndarray,
        n_arr: np.ndarray,
        *args,
        dt: float = 10.0,
        integrate_backwards: bool = False,
        **kwargs,
    ):
        r"""Interpolated summation function.

        This function interpolates the amplitude and phase information, and
        creates the final waveform with the combination of ylm values for each
        mode.

        args:
            t: Array of t values.
            teuk_modes: Array of complex amplitudes; shape is (len(t), num_teuk_modes).
            ylms: Array of ylm values for each mode,
                including m<0. Shape is (num of m==0,) + (num of m>0,)
                + (num of m<0). Number of m<0 and m>0 is the same, but they are
                ordered as (m==0 first then) m>0 then m<0.
            Phi_phi: Array of azimuthal phase values
                (:math:`\Phi_\phi`).
            Phi_r: Array of radial phase values
                 (:math:`\Phi_r`).
            l_arr: :math:`\ell` values associated with each mode.
            m_arr: :math:`m` values associated with each mode.
            n_arr: :math:`n` values associated with each mode.
            *args: Added for future flexibility.
            dt: Time spacing between observations (inverse of sampling
                rate). Default is 10.0.
            **kwargs: Added for future flexibility.
        """

        init_len = len(t)
        num_teuk_modes = teuk_modes.shape[1]
        num_pts = self.num_pts

        length = init_len
        ninterps = 2 * num_teuk_modes  # 2 for re and im
        y_all = self.xp.zeros((ninterps, length))

        y_all[:num_teuk_modes] = teuk_modes.T.real
        y_all[num_teuk_modes : 2 * num_teuk_modes] = teuk_modes.T.imag

        spline = self.build_with_same_backend(CubicSplineInterpolant, args=[t, y_all])

        h_t = t.get() if self.backend.uses_cupy else t

        if integrate_backwards:
            # For consistency with forward integration, we slightly shift the knots so that they line up at t=0
            offset = h_t[-1] - int(h_t[-1] / dt) * dt
            h_t = h_t - offset
            phase_interp_t = phase_interp_t - offset

        if not self.backend.uses_cuda:
            dev = 0
        else:
            dev = int(self.xp.cuda.runtime.getDevice())

        phase_interp_coeffs_in = self.xp.transpose(
            self.xp.asarray(phase_interp_coeffs), [2, 0, 1]
        ).flatten()

        # for ylm with negative m, need to multiply by (-1)**l as this is assumed to have happened by the kernel
        ylms = ylms.copy()
        ylms[num_teuk_modes:] *= (-1) ** l_arr

        self.get_waveform(
            self.waveform,
            spline.interp_array,
            phase_interp_t,
            phase_interp_coeffs_in,
            m_arr.astype(self.xp.int32),
            n_arr.astype(self.xp.int32),
            init_len,
            num_pts,
            num_teuk_modes,
            ylms.astype(self.xp.complex128),
            dt,
            h_t,
            dev,
        )
