import numpy as np
import time

try:
    import cupy as xp
except ImportError:
    import numpy as xp

from .nn import NN
from .ylm import get_ylms

from pyNIT import NIT
from scipy.interpolate import CubicSpline


class GetZlmnk:
    def __init__(
        self,
        batch_size,
        transform_file="few/files/reduced_basis.dat",
        nn_kwargs={},
        **kwargs,
    ):
        self.neural_net = NN(**nn_kwargs)

        self.transform_matrix = xp.asarray(
            np.genfromtxt(transform_file, dtype=xp.complex64)
        )

        self.transform_factor = 1000.0

        self.buffer = xp.zeros((batch_size, 2))

    def __call__(self, p, e):
        if len(p) != len(self.buffer):
            self.buffer[: len(p), 0] = p
            self.buffer[: len(p), 1] = e

            self.buffer[len(p) :, 0] = 0.0
            self.buffer[len(p) :, 1] = 0.0

        else:
            self.buffer[:, 0] = p
            self.buffer[:, 1] = e

        output = self.neural_net(self.buffer)

        re = output[:, :97]
        im = output[:, 97:]

        temp = (re + 1j * im).astype(xp.complex64)

        import pdb

        pdb.set_trace()

        Zlmkn = xp.matmul(temp / self.transform_factor, self.transform_matrix)

        return Zlmkn


class CreateWaveform:
    def __init__(self, num_n, **kwargs):
        batch_size = kwargs["batch_size"]
        self.num_modes = 3843
        self.get_zlmnk = GetZlmnk(**kwargs)
        self.expiphases = xp.zeros((batch_size, self.num_modes), dtype=xp.complex64)
        self.zlmnk = xp.zeros((batch_size, self.num_modes), dtype=xp.complex64)
        self.ylms = xp.zeros(self.num_modes, dtype=xp.complex64)
        self.buffer = xp.zeros(int(self.num_modes / num_n), dtype=xp.complex64)
        self.mode_inds = kwargs["mode_inds"]
        self.batch_size = kwargs["batch_size"]

    def __call__(
        self,
        M,
        mu,
        p,
        e,
        l,
        m,
        n,
        theta,
        phi,
        dt,
        get_modes=None,
        Phi_phi=None,
        Phi_r=None,
        nit_err=1e-10,
        spline_modes=True,
    ):

        if Phi_phi is None or Phi_r is None:
            if isinstance(p, np.ndarray) or isinstance(e, np.ndarray):
                raise ValueError(
                    "if not providing Phi_phi, please provide scalar p and e"
                )

            t_, p_, e_, Phi_phi_, Phi_r_ = NIT(M, mu, p, e, err=nit_err)

            t = np.arange(0.0, t_[-1] + dt, dt)

            Phi_phi = CubicSpline(t_, Phi_phi_)(t)
            Phi_r = CubicSpline(t_, Phi_r_)(t)

            if spline_modes is False:
                p = CubicSpline(t_, p_)(t)
                e = CubicSpline(t_, e_)(t)

        if len(Phi_phi) > self.batch_size:
            print(
                "Raise the batch size. Only running first {} points.".format(
                    self.batch_size
                )
            )

            Phi_phi = Phi_phi[: self.batch_size]
            Phi_r = Phi_r[: self.batch_size]
            t = t[: self.batch_size]

            if spline_modes is False:
                p = p[: self.batch_size]
                e = e[: self.batch_size]

        if get_modes is not None:
            mode_out = {}

        self.ylms[:] = xp.repeat(
            get_ylms(l[0::61], m[0::61], theta, phi, self.buffer), 61
        )

        if spline_modes:
            # only :len(p_) are good
            self.zlmnk[:] = self.get_zlmnk(p_, e_)
            import pdb

            pdb.set_trace()
            for mode_i in range(self.num_modes):
                self.zlmnk[: len(t), mode_i] = CubicSpline(
                    t_, self.zlmnk[: len(p_), mode_i]
                )(t)
            print("interpolated modes: init shape {}".format(p_.shape))

        else:
            self.zlmnk[: len(Phi_phi)] = self.get_zlmnk(p, e)

            print("direct solve: init shape {}".format(p.shape))

        self.expiphases[: len(Phi_phi)] = xp.exp(
            -1j
            * (
                m[xp.newaxis, :] * Phi_phi[:, xp.newaxis]
                + n[xp.newaxis, :] * Phi_r[:, xp.newaxis]
            )
        )

        if get_modes is not None:
            temp = (
                self.zlmnk[: len(Phi_phi)]
                * self.ylms[xp.newaxis, :]
                * self.expiphases[: len(Phi_phi)]
            )

            for i, mode in enumerate(get_modes):
                l_here, m_here, n_here = mode
                if m_here < 0:
                    continue
                ind = self.mode_inds[mode]
                mode_out[mode] = temp[: len(Phi_phi), ind]

        else:
            waveform = xp.sum(
                self.zlmnk[: len(Phi_phi)]
                * self.ylms[xp.newaxis, :]
                * self.expiphases[: len(Phi_phi)],
                axis=1,
            )

        self.ylms[:] = xp.repeat(
            get_ylms(l[0::61], m[0::61], theta, phi, self.buffer), 61
        )
        self.expiphases[: len(Phi_phi)] = xp.exp(
            -1j
            * (
                -m[xp.newaxis, :] * Phi_phi[:, xp.newaxis]
                + -n[xp.newaxis, :] * Phi_r[:, xp.newaxis]
            )
        )

        """
        # I think this pairs the wrong Zlmnk for the conjugate transformation to -m and -n.
        # I think all that needs to happen is take -m and -n and then conj(Zlmnk)
        inds = np.arange(len(l))[::61]
        for start_ind, end_ind in zip(inds[:-1], inds[1:]):
            self.zlmnk[start_ind:end_ind] = self.zlmnk[start_ind:end_ind][::-1]
        """

        if get_modes is not None:
            temp = (
                self.zlmnk[: len(Phi_phi)].conj()
                * self.ylms[xp.newaxis, :]
                * self.expiphases[: len(Phi_phi)]
            )

            for i, mode in enumerate(get_modes):
                l_here, m_here, n_here = mode
                if m_here >= 0:
                    continue
                ind = self.mode_inds[mode]
                mode_out[mode] = temp[: len(Phi_phi), ind]

        else:
            waveform = waveform + xp.sum(
                self.zlmnk[: len(Phi_phi)].conj()
                * self.ylms[xp.newaxis, :]
                * self.expiphases[: len(Phi_phi)],
                axis=1,
            )

        if get_modes is not None:
            return mode_out

        return waveform


if __name__ == "__main__":
    nn_kwargs = dict(
        input_str="SE_n30_", folder="few/files/weights/", activation_kwargs={}
    )

    kwargs = dict(transform_file="few/files/reduced_basis_n30.dat", nn_kwargs=nn_kwargs)

    traj = np.genfromtxt("insp_p12.5_e0.7_tspacing_1M.dat")[0::3][:100000]

    batch_size = kwargs["batch_size"] = len(traj)

    p = xp.asarray(traj[:, 0], dtype=xp.float32)
    e = xp.asarray(traj[:, 1], dtype=xp.float32)
    Phi_phi = xp.asarray(traj[:, 2], dtype=xp.float32)
    Phi_r = xp.asarray(traj[:, 3], dtype=xp.float32)

    l = xp.zeros(3843, dtype=int)
    m = xp.zeros(3843, dtype=int)
    n = xp.zeros(3843, dtype=int)

    ind = 0
    for l_i in range(2, 10 + 1):
        for m_i in range(1, l_i + 1):
            for n_i in range(-20, 20 + 1):
                l[ind] = l_i
                m[ind] = m_i
                n[ind] = n_i
                ind += 1

    cw = CreateWaveform(**kwargs)
    theta = np.pi / 4
    phi = np.pi / 3

    num = 10
    out = cw(p, e, Phi_r, Phi_phi, l, m, n, theta, phi)
    check = []
    for _ in range(num):
        st = time.perf_counter()
        out = cw(p, e, Phi_r, Phi_phi, l, m, n, theta, phi)
        et = time.perf_counter()
        print("Timing:", (et - st))
    import pdb

    pdb.set_trace()
