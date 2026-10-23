"""Time-domain mode sum from the sparse trajectory, and a batched cubic spline.

- :class:`InterpolatedModeSum`: the time-domain summation module of the fast
  spherical-harmonic models. The mode amplitudes are fitted with cubic splines on the sparse
  trajectory knots, the phases come from the trajectory's DOP853 dense-output spline, and the
  backend kernel ``get_waveform_wrap`` evaluates and sums all modes at every output sample.
- :class:`CubicSplineInterpolant`: many not-a-knot cubic splines fitted at once with the
  backend's batched tridiagonal solve (``interpolate_arrays_wrap``).
"""

# Interpolated summation of modes in python for the FastEMRIWaveforms Package

import numpy as np

from few.utils.globals import get_logger

# Python imports
from ..utils.baseclasses import BackendLike, FewBackendConsumer
from .base import SummationBase


class CubicSplineInterpolant(FewBackendConsumer):
    """GPU-accelerated Multiple Cubic Splines

    This class produces multiple cubic splines on a GPU. It has a CPU option
    as well. The cubic splines are produced with "not-a-knot" boundary
    conditions.

    This class can be run out of Python similar to
    `scipy.interpolate.CubicSpline <https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.CubicSpline.html#cubicspline>`_.
    However, the most efficient way to use this method is in a customized
    cuda kernel. See the
    `source code for the interpolated summation <https://github.com/BlackHolePerturbationToolkit/FastEMRIWaveforms/blob/master/src/few/cutils/interpolate.cu>`_
    in cuda for an example of this.

    This class can be run on GPUs and CPUs.

    Args:
        t: Knots, shape ``(length,)`` (shared by all splines) or ``(ninterps, length)``;
            at least 3 knots. An ``xp.ndarray`` of the backend.
        y_all: Values, shape ``(length,)`` or ``(ninterps, length)``.
        **kwargs: Forwarded to :class:`~few.utils.baseclasses.FewBackendConsumer`
            (``force_backend``).

    Raises:
        ValueError: If ``t`` has fewer than 3 knots or more than 2 dimensions.

    Attributes:
        t (xp.ndarray): Knots, shape ``(ninterps, length)``.
        interp_array (xp.ndarray): Flattened coefficients ``(y, c1, c2, c3)``; reshape with
            ``reshape_shape`` = ``(4, length, ninterps)``.

    Example:
        .. code-block:: python

            import numpy as np
            from few.summation.interpolatedmodesum import CubicSplineInterpolant

            t = np.linspace(0.0, 10.0, 11)
            spl = CubicSplineInterpolant(t, np.array([np.sin(t), np.cos(t)]),
                                         force_backend="cpu")
            spl(np.array([0.5, 1.5]))  # shape (2, 2)
    """

    def __init__(self, t: np.ndarray, y_all: np.ndarray, **kwargs):
        FewBackendConsumer.__init__(self, **kwargs)

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
            # The batched tridiagonal solve (cusparseDgtsv2StridedBatch on GPU)
            # requires system size >= 3; fewer points (near-plunge trajectories)
            # would hard-exit inside the CUDA layer, so reject them here.
            if len(t) < 3:
                raise ValueError("t must have length greater than 2.")

            # could save memory by adjusting c code to treat 1D differently
            self.t = self.xp.tile(t, (ninterps, 1)).flatten().astype(self.xp.float64)

        elif t.ndim == 2:
            if t.shape[1] < 3:
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
        """The backend's batched spline-fit kernel (``interpolate_arrays_wrap``)."""
        return self.backend.interpolate_arrays_wrap

    @classmethod
    def supported_backends(cls):
        """Backend preference order: ``["cuda13x", "cuda12x", "cuda11x", "cpu"]``."""
        return cls.GPU_RECOMMENDED()

    @property
    def y(self) -> np.ndarray:
        """Values at the knots, shape ``(ninterps, length)``."""
        return self.interp_array.reshape(self.reshape_shape)[0].T

    @property
    def c1(self) -> np.ndarray:
        """Linear-term coefficients, shape ``(ninterps, length)``."""
        return self.interp_array.reshape(self.reshape_shape)[1].T

    @property
    def c2(self) -> np.ndarray:
        """Quadratic-term coefficients, shape ``(ninterps, length)``."""
        return self.interp_array.reshape(self.reshape_shape)[2].T

    @property
    def c3(self) -> np.ndarray:
        """Cubic-term coefficients, shape ``(ninterps, length)``."""
        return self.interp_array.reshape(self.reshape_shape)[3].T

    def _get_inds(self, tnew: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Segment index of each ``tnew`` point, and masks of the points left / right of the knots.

        Args:
            tnew: New times, shape ``(ninterps, M)``.

        Returns:
            tuple[xp.ndarray, xp.ndarray, xp.ndarray]: ``(inds, bad_left, bad_right)``, each of
            shape ``(ninterps, M)``.
        """
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
        """Evaluate all splines (or a derivative) at new times.

        Put in an array of new t values at which all interpolants will be
        evaluated. For ``deriv_order=0``, points outside the knot range get the edge values
        (a warning is logged for points left of the range); derivatives are not allowed
        outside the range.

        Args:
            tnew: New times, shape ``(M,)`` (used for every spline) or ``(ninterps, M)``.
            deriv_order: Order of the derivative to evaluate, 0 (default, the spline
                itself) to 3. Unlike :code:`scipy`, this is purely an evaluation of the
                derivative values, not a new class to evaluate for the derivative.

        Returns:
            xp.ndarray: Shape ``(ninterps, M)``, squeezed (size-1 axes removed).

        Raises:
            ValueError: If a 2-D ``tnew`` does not have ``ninterps`` rows, if a derivative is
                requested outside the knot range, or if ``deriv_order > 3``.
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
    """Create a time-domain waveform by interpolating the modes along a sparse trajectory.

    The real and imaginary parts of the mode amplitudes are fitted with
    :class:`CubicSplineInterpolant` on the trajectory knots; the phases are evaluated from the
    trajectory's DOP853 dense-output phase coefficients. The backend kernel
    ``get_waveform_wrap`` evaluates both at every output sample ``i * dt`` and sums the
    modes with their spherical harmonics (each :math:`m > 0` mode also with its
    :math:`-m` partner). The time-domain summation module of
    :class:`~few.waveform.waveform.FastKerrEccentricEquatorialFlux`,
    :class:`~few.waveform.waveform.FastSchwarzschildEccentricFlux` and
    :class:`~few.waveform.waveform.FastSchwarzschildEccentricFluxBicubic`.

    This class can be run on GPUs and CPUs.

    Args:
        force_backend: Backend name or object (see
            :class:`~few.utils.baseclasses.FewBackendConsumer`). Default ``None``.
        **kwargs: Forwarded to :class:`~few.summation.base.SummationBase` (``pad_output``,
            ``odd_len``, ``output_type``).

    Example:
        .. code-block:: python

            from few.waveform import FastKerrEccentricEquatorialFlux

            # built by the waveform model from its sum_kwargs
            few = FastKerrEccentricEquatorialFlux(sum_kwargs=dict(pad_output=True))
            few.create_waveform  # the InterpolatedModeSum instance
    """

    def __init__(self, force_backend: BackendLike = None, **kwargs):
        SummationBase.__init__(self, **kwargs, force_backend=force_backend)

    @property
    def get_waveform(self) -> callable:
        """The backend's mode-summation kernel (``get_waveform_wrap``)."""
        return self.backend.get_waveform_wrap

    @classmethod
    def supported_backends(cls):
        """Backend preference order: ``["cuda13x", "cuda12x", "cuda11x", "cpu"]``."""
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
        k_arr: np.ndarray,
        n_arr: np.ndarray,
        *args,
        dt: float = 10.0,
        integrate_backwards: bool = False,
        **kwargs,
    ):
        r"""Interpolate the amplitudes and phases and sum the modes into :attr:`waveform`.

        Called by :meth:`~few.summation.base.SummationBase.__call__`. The :math:`-m`
        harmonics (second half of ``ylms``) are multiplied by :math:`(-1)^l` before the
        kernel, which expects them that way.

        Args:
            t: Trajectory knot times in seconds, shape ``(N,)`` (backend array).
            teuk_modes: Complex amplitudes, shape ``(N, K)``.
            ylms: Complex spherical harmonics, shape ``(2 K,)``: :math:`Y_{lm}` of the ``K``
                modes, then :math:`Y_{l,-m}` of the same modes.
            phase_interp_t: Knot times of the phase spline in seconds
                (``integrator_spline_t``), shape ``(N_p,)``.
            phase_interp_coeffs: DOP853 phase coefficients of
                ``(Phi_phi, Phi_theta, Phi_r)`` in rad (``integrator_spline_phase_coeff``),
                shape ``(N_p - 1, 3, 8)``.
            l_arr: :math:`\ell` of each mode, shape ``(K,)``.
            m_arr: :math:`m` of each mode, shape ``(K,)``.
            k_arr: :math:`k` of each mode, shape ``(K,)``.
            n_arr: :math:`n` of each mode, shape ``(K,)``.
            *args: Accepted and ignored (the waveform base passes ``M, a, p, e, xI``).
            dt: Output sampling interval in seconds. Default ``10.0``.
            integrate_backwards: If ``True``, subtract ``t[-1] - int(t[-1] / dt) * dt`` from the
                amplitude and phase knot times, so the last knot falls on a multiple of ``dt``.
                Default ``False``.
            **kwargs: Accepted and ignored.
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
            k_arr.astype(self.xp.int32),
            n_arr.astype(self.xp.int32),
            init_len,
            num_pts,
            num_teuk_modes,
            ylms.astype(self.xp.complex128),
            dt,
            h_t,
            dev,
        )
