import numpy as np

try:
    import cupy as xp

except ImportError:
    import numpy as xp

from flux import RunFluxInspiral
from amplitude import Amplitude
from interpolated_mode_sum import InterpolatedModeSum
from ylm import GetYlms

# TODO: make sure constants are same
from scipy import constants as ct


class FEW:
    def __init__(
        self, inspiral_kwargs={}, amplitude_kwargs={}, Ylm_kwargs={}, sum_kwargs={}
    ):
        """
        Carrier class for FEW
        """
        self.inspiral_gen = RunFluxInspiral()
        self.inspiral_kwargs = inspiral_kwargs

        self.amplitude_gen = Amplitude(**amplitude_kwargs)
        self.sum = InterpolatedModeSum(**sum_kwargs)

        m_arr = xp.zeros((3843,), dtype=int)
        n_arr = xp.zeros_like(m_arr)

        md = []

        for l in range(2, 10 + 1):
            for m in range(0, l + 1):
                for n in range(-30, 30 + 1):
                    md.append([l, m, n])

        self.num_teuk_modes = len(md)

        m0mask = xp.array(
            [
                m == 0
                for l in range(2, 10 + 1)
                for m in range(0, l + 1)
                for n in range(-30, 30 + 1)
            ]
        )
        m0sort = xp.concatenate(
            [
                xp.arange(self.num_teuk_modes)[m0mask],
                xp.arange(self.num_teuk_modes)[~m0mask],
            ]
        )

        md = xp.asarray(md).T[:, m0sort]

        self.l_arr, self.m_arr, self.n_arr = md[0], md[1], md[2]

        self.ylm_gen = GetYlms(self.num_teuk_modes, **Ylm_kwargs)

    def __call__(self, M, mu, p0, e0, theta, phi, dt=10.0, T=1.0, eps=2e-4):

        T = 1.0 * ct.Julian_year
        # get trajectory
        (t, p, e, Phi_phi, Phi_r) = self.inspiral_gen(
            M, mu, p0, e0, **self.inspiral_kwargs
        )

        # convert for gpu
        t = xp.asarray(t)
        p = xp.asarray(p)
        e = xp.asarray(e)
        Phi_phi = xp.asarray(Phi_phi)
        Phi_r = xp.asarray(Phi_r)

        # amplitudes
        teuk_modes = self.amplitude_gen(p, e)

        # TODO: implement normalization to flux
        power = xp.abs(teuk_modes) ** 2

        inds_sort = xp.argsort(power, axis=1)[:, ::-1]
        power = xp.sort(power, axis=1)[:, ::-1]
        cumsum = xp.cumsum(power, axis=1)

        inds_keep = xp.full(cumsum.shape, True)

        inds_keep[:, 1:] = cumsum[:, :-1] < cumsum[:, -1][:, xp.newaxis] * (1 - eps)

        keep_modes = xp.unique(inds_sort[inds_keep])

        self.ls = self.l_arr[keep_modes]
        self.ms = self.m_arr[keep_modes]
        self.ns = self.n_arr[keep_modes]

        ylms = self.ylm_gen(self.ls, self.ms, theta, phi)

        waveform = self.sum(
            t,
            p,
            e,
            Phi_phi,
            Phi_r,
            teuk_modes[:, keep_modes],
            self.ms,
            self.ns,
            ylms,
            dt,
            T,
        )
        return waveform


if __name__ == "__main__":
    import time

    few = FEW(inspiral_kwargs={}, amplitude_kwargs={"max_input_len": 3000})
    M = 5e5
    mu = 1e1
    p0 = 12.8
    e0 = 0.2
    theta = np.pi / 3.0
    phi = np.pi / 4.0
    dt = 15.0
    T = 1.0
    eps = 2e-4

    check = few(M, mu, p0, e0, theta, phi, dt=dt, T=T)
    num = 100

    st = time.perf_counter()
    for _ in range(num):
        check = few(M, mu, p0, e0, theta, phi, dt=dt, T=T, eps=eps)
    et = time.perf_counter()

    import pdb

    pdb.set_trace()
    print(check.shape)
    print((et - st) / num)
