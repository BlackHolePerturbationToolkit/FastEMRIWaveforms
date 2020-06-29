from pyinterp_cpu import interpolate_arrays_wrap as interpolate_arrays_wrap_cpu
from pyinterp_cpu import get_waveform_wrap as get_waveform_wrap_cpu

try:
    import cupy as xp
    from pyinterp import interpolate_arrays_wrap, get_waveform_wrap

except ImportError:
    import numpy as xp
import numpy as np


class InterpolatedModeSum:
    def __init__(self, pad_output=False, use_gpu=False):

        self.use_gpu = use_gpu
        if use_gpu:
            self.xp = xp
            self.interpolate_arrays = interpolate_arrays_wrap
            self.get_waveform = get_waveform_wrap

        else:
            self.xp = np
            self.interpolate_arrays = interpolate_arrays_wrap_cpu
            self.get_waveform = get_waveform_wrap_cpu

        self.num_phases = 2
        self.pad_output = pad_output

    def _interp(self, t, p, e, Phi_phi, Phi_r, teuk_modes):
        self.length, num_modes_keep = teuk_modes.shape

        self.ninterps = self.num_phases + 2 * num_modes_keep  # 2 for re and im
        self.y_all = self.xp.zeros((self.ninterps, self.length))

        self.y_all[:num_modes_keep] = teuk_modes.T.real
        self.y_all[num_modes_keep : 2 * num_modes_keep] = teuk_modes.T.imag

        self.y_all[-2] = Phi_phi
        self.y_all[-1] = Phi_r

        self.y_all = self.y_all.flatten()
        self.c1 = self.xp.zeros((self.ninterps, self.length - 1)).flatten()
        self.c2 = self.xp.zeros_like(self.c1).flatten()
        self.c3 = self.xp.zeros_like(self.c1).flatten()

        B = self.xp.zeros((self.ninterps * self.length,))
        upper_diag = self.xp.zeros_like(B)
        diag = self.xp.zeros_like(B)
        lower_diag = self.xp.zeros_like(B)

        self.interpolate_arrays(
            t,
            self.y_all,
            self.c1,
            self.c2,
            self.c3,
            self.ninterps,
            self.length,
            B,
            upper_diag,
            diag,
            lower_diag,
        )

        self.y_all = self.y_all.reshape(self.ninterps, self.length).T.flatten()
        self.c1 = self.c1.reshape(self.ninterps, self.length - 1).T.flatten()
        self.c2 = self.c2.reshape(self.ninterps, self.length - 1).T.flatten()
        self.c3 = self.c3.reshape(self.ninterps, self.length - 1).T.flatten()

    def _sum(self, m_arr, n_arr, init_len, num_pts, num_teuk_modes, ylms, dt, h_t):

        self.get_waveform(
            self.waveform,
            self.y_all,
            self.c1,
            self.c2,
            self.c3,
            m_arr,
            n_arr,
            init_len,
            num_pts,
            num_teuk_modes,
            ylms,
            dt,
            h_t,
        )

    def __call__(self, t, p, e, Phi_phi, Phi_r, teuk_modes, m_arr, n_arr, ylms, dt, T):

        if T < t[-1].item():
            num_pts = int(T / dt)
            num_pts_pad = 0

        else:
            num_pts = int(t[-1] / dt)
            if self.pad_output:
                num_pts_pad = int(T / dt) - num_pts
            else:
                num_pts_pad = 0

        # TODO: make sure num points adjusts for zero padding
        self.num_pts, self.num_pts_pad = num_pts, num_pts_pad
        self.dt = dt
        init_len = len(t)
        num_teuk_modes = teuk_modes.shape[1]

        self.waveform = self.xp.zeros(
            (self.num_pts + self.num_pts_pad,), dtype=self.xp.complex128
        )

        self._interp(t, p, e, Phi_phi, Phi_r, teuk_modes)

        """
        from scipy.interpolate import CubicSpline
        import numpy as np

        Phi_phi_spl = CubicSpline(t.get(), Phi_phi.get())
        Phi_r_spl = CubicSpline(t.get(), Phi_r.get())

        mode_re_spl = CubicSpline(t.get(), teuk_modes[:, 5].get().real)
        mode_im_spl = CubicSpline(t.get(), teuk_modes[:, 5].get().imag)

        t_nn = np.arange(0, 10000, dt)
        Phi_phi_nn = Phi_phi_spl(t_nn)
        Phi_r_nn = Phi_r_spl(t_nn)
        mode_re_nn = mode_re_spl(t_nn)
        mode_im_nn = mode_im_spl(t_nn)
        """

        try:
            t = t.get()
        except:
            pass

        self._sum(m_arr, n_arr, init_len, num_pts, num_teuk_modes, ylms, dt, t)

        return self.waveform
