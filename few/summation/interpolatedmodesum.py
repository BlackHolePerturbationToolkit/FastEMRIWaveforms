from pyinterp_cpu import interpolate_arrays_wrap as interpolate_arrays_wrap_cpu
from pyinterp_cpu import get_waveform_wrap as get_waveform_wrap_cpu

from few.utils.baseclasses import SummationBase, SchwarzschildEccentric

try:
    import cupy as xp
    from pyinterp import interpolate_arrays_wrap, get_waveform_wrap

except (ImportError, ModuleNotFoundError) as e:
    import numpy as xp
import numpy as np


class CubicSplineInterpolant:
    """GPU-accelerated Multiple Cubic Splines

    This class produces multiple cubic splines on a GPU. It has a CPU option
    as well. The cubic splines are produced with "not-a-knot" boundary
    conditions.

    This class can be run on GPUs and CPUs.

    args:
        t (1D double xp.ndarray): t values as input for the spline.
        y_all (2D double xp.ndarray): y values for the spline.
            Shape: (ninterps, length).
        use_gpu (bool, optional): If True, prepare arrays for a GPU. Default is
            False.

    attributes:
        interpolate_arrays (func): CPU or GPU function for mode interpolation.


    """

    def __init__(self, t, y_all, use_gpu=False):

        if use_gpu:
            self.xp = xp
            self.interpolate_arrays = interpolate_arrays_wrap

        else:
            self.xp = np
            self.interpolate_arrays = interpolate_arrays_wrap_cpu

        self.degree = 3

        ninterps, length = y_all.shape

        self.reshape_shape = (self.degree + 1, ninterps, length)

        interp_array = self.xp.zeros(self.reshape_shape)

        interp_array[0] = y_all

        interp_array = interp_array.flatten()

        B = self.xp.zeros((ninterps * length,))
        upper_diag = self.xp.zeros_like(B)
        diag = self.xp.zeros_like(B)
        lower_diag = self.xp.zeros_like(B)

        self.interpolate_arrays(
            t, interp_array, ninterps, length, B, upper_diag, diag, lower_diag
        )

        self.t = t
        self.interp_array = self.xp.transpose(
            interp_array.reshape(self.reshape_shape), [0, 2, 1]
        ).flatten()
        self.reshape_shape = (self.degree + 1, length, ninterps)

    @property
    def y(self):
        return self.interp_array.reshape(self.reshape_shape)[0].T

    @property
    def c1(self):
        return self.interp_array.reshape(self.reshape_shape)[1].T

    @property
    def c2(self):
        return self.interp_array.reshape(self.reshape_shape)[2].T

    @property
    def c3(self):
        return self.interp_array.reshape(self.reshape_shape)[3].T

    def __call__(self, tnew):

        inds = self.xp.searchsorted(self.t, tnew)

        x = tnew - self.t[inds]
        x2 = x * x
        x3 = x2 * x

        out = (
            self.y[:, inds]
            + self.c1[:, inds] * x
            + self.c2[:, inds] * x2
            + self.c3[:, inds] * x3
        )
        return out

    def d1(self, tnew):
        inds = self.xp.searchsorted(self.t, tnew)

        x = tnew - self.t[inds]
        x2 = x * x

        out = (
            self.c1[:, inds] + 2.0 * self.c2[:, inds] * x + 3.0 * self.c3[:, inds] * x2
        )
        return out

    def d2(self, tnew):
        inds = self.xp.searchsorted(self.t, tnew)

        x = tnew - self.t[inds]

        out = 2.0 * self.c2[:, inds] * x + 6.0 * self.c3[:, inds] * x
        return out

    def d3(self, tnew):
        inds = self.xp.searchsorted(self.t, tnew)
        out = 6.0 * self.c3[:, inds]
        return out


class InterpolatedModeSum(SummationBase, SchwarzschildEccentric):
    """Create waveform by interpolating sparse trajectory.

    It interpolates all of the modes of interest and phases at sparse
    trajectories. Within the summation phase, the values are calculated using
    the interpolants and summed.

    This class can be run on GPUs and CPUs.

    attributes:
        get_waveform (func): CPU or GPU function for waveform creation.
        ninterps (int): number of interpolants. It is the (number of phases) +
            2*(number of modes).
        y_all (2D double xp.ndarray): All of the y values for the
            interpolants. Real and imaginary values from the complex amplitudes
            are separated giving all real numbers.
        c1, c2, c3 (2D double xp.ndarray): (1st, 2nd, 3rd) constants for each
            cubic spline.
        waveform (1D complex128 np.ndarray): Complex waveform given by
            :math:`h_+ + i*h_x`.
        interp (obj): Interpolant. See
            :class:`few.summation.interpolated_mode_sum.CubicSplineInterpolant`.

    """

    def __init__(self, *args, **kwargs):

        SchwarzschildEccentric.__init__(self, *args, **kwargs)
        SummationBase.__init__(self, *args, **kwargs)

        self.kwargs = kwargs

        if self.use_gpu:
            self.xp = xp
            self.get_waveform = get_waveform_wrap

        else:
            self.xp = np
            self.get_waveform = get_waveform_wrap_cpu

    def sum(
        self,
        t,
        teuk_modes,
        ylms,
        Phi_phi,
        Phi_r,
        m_arr,
        n_arr,
        init_len,
        num_pts,
        num_teuk_modes,
        dt,
        *args,
        **kwargs
    ):
        """Interpolated summation function.

        This function interpolates the amplitude and phase information, and
        creates the final waveform with the combination of ylm values for each
        mode.

        args:
            t (1D double xp.ndarray): Array of t values.
            teuk_modes (2D double xp.array): Array of complex amplitudes.
                Shape: (len(t), num_teuk_modes).
            ylms (1D complex128 xp.ndarray): Array of ylm values for each mode,
                including m<0. Shape is (num of m==0,) + (num of m>0,)
                + (num of m<0). Number of m<0 and m>0 is the same, but they are
                ordered as (m==0 first then) m>0 then m<0.
            Phi_phi (1D double xp.ndarray): Array of azimuthal phase values
                (:math:`\Phi_\phi`).
            Phi_r (1D double xp.ndarray): Array of radial phase values
                 (:math:`\Phi_r`).
            m_arr (1D int xp.ndarray): :math:`m` values associated with each mode.
            n_arr (1D int xp.ndarray): :math:`n` values associated with each mode.
            init_len (int): len(t).
            num_pts (int): len(self.waveform).
            num_teuk_modes (int): Number of amplitude modes included.
            dt (double): Time spacing between observations (inverse of sampling
                rate).
            *args (list, placeholder): Added for future flexibility.
            **kwargs (dict, placeholder): Added for future flexibility.

        """
        length = init_len
        ninterps = self.ndim + 2 * num_teuk_modes  # 2 for re and im
        y_all = self.xp.zeros((ninterps, length))

        y_all[:num_teuk_modes] = teuk_modes.T.real
        y_all[num_teuk_modes : 2 * num_teuk_modes] = teuk_modes.T.imag

        y_all[-2] = Phi_phi
        y_all[-1] = Phi_r

        spline = CubicSplineInterpolant(t, y_all, use_gpu=self.use_gpu)

        try:
            h_t = t.get()
        except:
            h_t = t

        self.get_waveform(
            self.waveform,
            spline.interp_array,
            m_arr,
            n_arr,
            init_len,
            num_pts,
            num_teuk_modes,
            ylms,
            dt,
            h_t,
        )
