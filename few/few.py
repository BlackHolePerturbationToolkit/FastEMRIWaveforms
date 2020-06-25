import numpy as np

try:
    import cupy as xp

except ImportError:
    import numpy as xp

from flux import RunFluxInspiral
from amplitude import Amplitude
from interpolated_mode_sum import InterpolatedModeSum
from ylm import GetYlms
from direct_mode_sum import DirectModeSum

# TODO: make sure constants are same
from scipy import constants as ct


# TODO: free memory in trajectory
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
        # self.sum = DirectModeSum(**sum_kwargs)

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

        self.m0mask = self.m_arr != 0
        self.num_m_zero_up = len(self.m_arr)
        self.num_m0 = len(xp.arange(self.num_teuk_modes)[m0mask])

        self.num_m_1_up = self.num_m_zero_up - self.num_m0
        self.l_arr = xp.concatenate([self.l_arr, self.l_arr[self.m0mask]])
        self.m_arr = xp.concatenate([self.m_arr, -self.m_arr[self.m0mask]])
        self.n_arr = xp.concatenate([self.n_arr, self.n_arr[self.m0mask]])

        temp, self.inverse_lm = np.unique(
            np.asarray([self.l_arr.get(), self.m_arr.get()]).T,
            axis=0,
            return_inverse=True,
        )

        self.unique_l, self.unique_m = xp.asarray(temp).T
        self.num_unique_lm = len(self.unique_l)

        self.ylm_gen = GetYlms(self.num_teuk_modes, **Ylm_kwargs)

    def __call__(
        self, M, mu, p0, e0, theta, phi, dt=10.0, T=1.0, eps=2e-4, all_modes=False
    ):

        T = T * ct.Julian_year
        # get trajectory
        (t, p, e, Phi_phi, Phi_r, amp_norm) = self.inspiral_gen(
            M, mu, p0, e0, **self.inspiral_kwargs
        )

        # convert for gpu
        t = xp.asarray(t)[: int(1e4)]
        p = xp.asarray(p)[: int(1e4)]
        e = xp.asarray(e)[: int(1e4)]
        Phi_phi = xp.asarray(Phi_phi)[: int(1e4)]
        Phi_r = xp.asarray(Phi_r)[: int(1e4)]
        amp_norm = xp.asarray(amp_norm)[: int(1e4)]

        """
        insp = np.loadtxt("inspiral_new.txt")[45000:55000]
        t, p, e = xp.asarray(insp[:, :3].T)

        Phi_phi, Phi_r = xp.asarray(insp[:, 3:5]).T

        Ylms_check = np.tile(
            np.loadtxt("few/files/Ylm_pi2_0.dat"), (61, 1)
        ).T.flatten()[self.m0sort.get()]
        t = xp.arange(len(p)) * dt
        """

        ylms = self.ylm_gen(self.unique_l, self.unique_m, theta, phi).copy()[
            self.inverse_lm
        ]

        # amplitudes
        teuk_modes = self.amplitude_gen(p, e)

        # TODO: check normalization of flux
        power = (
            xp.abs(
                xp.concatenate(
                    [teuk_modes, xp.conj(teuk_modes[:, self.m0mask])], axis=1
                )
                * ylms
            )
            ** 2
        )

        inds_sort = xp.argsort(power, axis=1)[:, ::-1]
        power = xp.sort(power, axis=1)[:, ::-1]
        cumsum = xp.cumsum(power, axis=1)

        # factor = amp_norm / cumsum[:, -1] ** (1 / 2)

        # teuk_modes = teuk_modes * factor[:, np.newaxis]
        # cumsum = cumsum * factor[:, np.newaxis] ** 2

        inds_keep = xp.full(cumsum.shape, True)

        inds_keep[:, 1:] = cumsum[:, :-1] < cumsum[:, -1][:, xp.newaxis] * (1 - eps)

        if all_modes:
            keep_modes = xp.arange(3843)
        else:
            temp = inds_sort[inds_keep]

            temp = temp * (temp < self.num_m_zero_up) + (temp - self.num_m_1_up) * (
                temp >= self.num_m_zero_up
            )

            keep_modes = xp.unique(temp)

            temp2 = keep_modes * (keep_modes < self.num_m0) + (
                keep_modes + self.num_m_1_up
            ) * (keep_modes >= self.num_m0)

            ylmkeep = xp.concatenate([keep_modes, temp2])

        self.num_modes_kept = len(keep_modes)

        # keep_modes = xp.array([646])

        # keep_modes = xp.arange(3843)
        self.ls = self.l_arr[keep_modes]
        self.ms = self.m_arr[keep_modes]
        self.ns = self.n_arr[keep_modes]

        waveform = self.sum(
            t,
            p,
            e,
            Phi_phi,
            Phi_r,
            teuk_modes[:, keep_modes],
            self.ms,
            self.ns,
            ylms[ylmkeep],
            dt,
            T,
        )

        return waveform


if __name__ == "__main__":
    import time

    few = FEW(
        inspiral_kwargs={"DENSE_STEPPING": 0, "max_init_len": int(1e3)},
        amplitude_kwargs={"max_input_len": int(1e3)},
        Ylm_kwargs={"assume_positive_m": False},
    )

    M = 1e6
    mu = 1e1
    p0 = 12.0
    e0 = 0.5
    theta = np.pi / 2
    phi = 0.0
    dt = 10.0
    T = 1.0  # 1124936.040602 / ct.Julian_year
    eps = 1e-2
    all_modes = False

    mismatch = []
    num_modes = []
    timing = []
    eps_all = 10.0 ** np.arange(-10, -2)

    eps_all = np.concatenate([np.array([1e-25]), eps_all])
    fullwave = np.genfromtxt("/projects/b1095/mkatz/emri/slow_1e6_1e1_12_05.txt")
    fullwave = fullwave[:, 5] + 1j * fullwave[:, 6]

    for i, eps in enumerate(eps_all):
        # all_modes = False if i > 0 else True
        num = 30
        st = time.perf_counter()
        for jjj in range(num):

            # print(jjj, "\n")
            wc = few(
                M, mu, p0, e0, theta, phi, dt=dt, T=T, eps=eps, all_modes=all_modes
            ).get()
        et = time.perf_counter()

        # if i == 0:
        #    np.save("dircheck", wc)

        min_len = np.min([len(wc), len(fullwave)])

        wc_fft = np.fft.fft(wc[:min_len])
        fullwave_fft = np.fft.fft(fullwave[:min_len])
        mm = (
            1.0
            - (
                np.dot(wc_fft.conj(), fullwave_fft)
                / np.sqrt(
                    np.dot(wc_fft.conj(), wc_fft)
                    * np.dot(fullwave_fft.conj(), fullwave_fft)
                )
            ).real
        )
        mismatch.append(mm)
        num_modes.append(few.num_modes_kept)
        timing.append((et - st) / num)
        print(
            "eps:",
            eps,
            "Mismatch:",
            mm,
            "Num modes:",
            few.num_modes_kept,
            "timing:",
            (et - st) / num,
        )

    # np.save(
    #    "info_check_1e6_1e1_14_05", np.asarray([eps_all, mismatch, num_modes, timing]).T
    # )

    """
    num = 20
    st = time.perf_counter()
    for _ in range(num):
        check = few(M, mu, p0, e0, theta, phi, dt=dt, T=T, eps=eps, all_modes=all_modes)
    et = time.perf_counter()

    import pdb

    pdb.set_trace()
    """
    # print(check.shape)
    print((et - st) / num)
