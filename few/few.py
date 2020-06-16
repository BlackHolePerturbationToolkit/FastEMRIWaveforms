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
        self.m0sort = m0sort = xp.concatenate(
            [
                xp.arange(self.num_teuk_modes)[m0mask],
                xp.arange(self.num_teuk_modes)[~m0mask],
            ]
        )

        md = xp.asarray(md).T[:, m0sort].astype(xp.int32)

        self.l_arr, self.m_arr, self.n_arr = md[0], md[1], md[2]

        self.ylm_gen = GetYlms(self.num_teuk_modes, **Ylm_kwargs)

    def __call__(
        self, M, mu, p0, e0, theta, phi, dt=10.0, T=1.0, eps=2e-4, all_modes=False
    ):

        T = 1.0 * ct.Julian_year
        # get trajectory
        (t, p, e, Phi_phi, Phi_r, amp_norm) = self.inspiral_gen(
            M, mu, p0, e0, **self.inspiral_kwargs
        )

        # convert for gpu
        t = xp.asarray(t)
        p = xp.asarray(p)
        e = xp.asarray(e)
        Phi_phi = xp.asarray(Phi_phi)
        Phi_r = xp.asarray(Phi_r)
        amp_norm = xp.asarray(amp_norm)

        """
        insp = np.loadtxt("inspiral_new.txt")[45000:55000]
        t, p, e = xp.asarray(insp[:, :3].T)

        Phi_phi, Phi_r = xp.asarray(insp[:, 3:5]).T

        Ylms_check = np.tile(
            np.loadtxt("few/files/Ylm_pi2_0.dat"), (61, 1)
        ).T.flatten()[self.m0sort.get()]
        t = xp.arange(len(p)) * dt
        """

        # amplitudes
        teuk_modes = self.amplitude_gen(p, e)

        # TODO: implement normalization to flux
        power = xp.abs(teuk_modes) ** 2

        power = power + (self.m_arr != 0.0) * power

        inds_sort = xp.argsort(power, axis=1)[:, ::-1]
        power = xp.sort(power, axis=1)[:, ::-1]
        cumsum = xp.cumsum(power, axis=1)

        factor = amp_norm / cumsum[:, -1]
        teuk_modes = teuk_modes * factor[:, np.newaxis]

        inds_keep = xp.full(cumsum.shape, True)

        inds_keep[:, 1:] = cumsum[:, :-1] < cumsum[:, -1][:, xp.newaxis] * (1 - eps)

        keep_modes = xp.unique(inds_sort[inds_keep])

        self.num_modes_kept = len(keep_modes)
        # keep_modes = xp.arange(3843)
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

    few = FEW(inspiral_kwargs={}, amplitude_kwargs={"max_input_len": 11000})
    M = 1e5
    mu = 1e1
    p0 = 10.0
    e0 = 0.1
    theta = np.pi / 2
    phi = 0.0
    dt = 10.0
    T = 1.0
    eps = 1e-6

    """
    mismatch = []
    num_modes = []
    eps_all = np.logspace(-10, -3)

    eps_all = np.concatenate([np.array([1e-25]), eps_all])[:1]
    fullwave = np.load("control.npy")[:57684]

    for i, eps in enumerate(eps_all):
        all_modes = False if i > 0 else True
        wc = few(
            M, mu, p0, e0, theta, phi, dt=dt, T=T, eps=eps, all_modes=all_modes
        ).get()

        mm = (
            1.0
            - (
                np.dot(wc.conj(), fullwave)
                / np.sqrt(np.dot(wc.conj(), wc) * np.dot(fullwave.conj(), fullwave))
            ).real
        )
        mismatch.append(mm)
        num_modes.append(few.num_modes_kept)

    import pdb

    pdb.set_trace()
    np.save("info_check", np.asarray([eps_all, mismatch, num_modes]).T)
    """

    num = 20
    st = time.perf_counter()
    for _ in range(num):
        check = few(M, mu, p0, e0, theta, phi, dt=dt, T=T, eps=eps)
    et = time.perf_counter()

    import pdb

    pdb.set_trace()

    # print(check.shape)
    print((et - st) / num)
