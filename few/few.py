import numpy as np
from abc import ABC

try:
    import cupy as xp

except ImportError:
    import numpy as xp

from flux import RunFluxInspiral
from amplitude import ROMANAmplitude, Interp2DAmplitude
from interpolated_mode_sum import InterpolatedModeSum
from ylm import GetYlms
from direct_mode_sum import DirectModeSum
from mode_filter import ModeFilter
from tqdm import tqdm

# TODO: make sure constants are same
# TODO: batching and mode selection
from scipy import constants as ct


class SchwarzschildEccentricBase:
    def __init__(
        self,
        inspiral_module,
        amplitude_module,
        sum_module,
        inspiral_kwargs={},
        amplitude_kwargs={},
        sum_kwargs={},
        Ylm_kwargs={},
        use_gpu=False,
    ):
        """
        Carrier class for FEW
        """

        if use_gpu:
            self.xp = xp

        else:
            self.xp = np

        self.inspiral_kwargs = inspiral_kwargs
        self.inspiral_generator = inspiral_module()

        self.amplitude_generator = amplitude_module(**amplitude_kwargs)
        self.sum = sum_module(**sum_kwargs)
        # self.sum = DirectModeSum(**sum_kwargs)

        m_arr = self.xp.zeros((3843,), dtype=int)
        n_arr = self.xp.zeros_like(m_arr)

        md = []

        for l in range(2, 10 + 1):
            for m in range(0, l + 1):
                for n in range(-30, 30 + 1):
                    md.append([l, m, n])

        self.num_teuk_modes = len(md)

        m0mask = self.xp.array(
            [
                m == 0
                for l in range(2, 10 + 1)
                for m in range(0, l + 1)
                for n in range(-30, 30 + 1)
            ]
        )

        self.m0sort = m0sort = self.xp.concatenate(
            [
                self.xp.arange(self.num_teuk_modes)[m0mask],
                self.xp.arange(self.num_teuk_modes)[~m0mask],
            ]
        )

        md = self.xp.asarray(md).T[:, m0sort].astype(self.xp.int32)

        self.l_arr, self.m_arr, self.n_arr = md[0], md[1], md[2]

        self.m0mask = self.m_arr != 0
        self.num_m_zero_up = len(self.m_arr)
        self.num_m0 = len(self.xp.arange(self.num_teuk_modes)[m0mask])

        self.num_m_1_up = self.num_m_zero_up - self.num_m0
        self.l_arr = self.xp.concatenate([self.l_arr, self.l_arr[self.m0mask]])
        self.m_arr = self.xp.concatenate([self.m_arr, -self.m_arr[self.m0mask]])
        self.n_arr = self.xp.concatenate([self.n_arr, self.n_arr[self.m0mask]])

        try:
            temp, self.inverse_lm = np.unique(
                np.asarray([self.l_arr.get(), self.m_arr.get()]).T,
                axis=0,
                return_inverse=True,
            )

        except AttributeError:
            temp, self.inverse_lm = np.unique(
                np.asarray([self.l_arr, self.m_arr]).T, axis=0, return_inverse=True
            )

        self.unique_l, self.unique_m = self.xp.asarray(temp).T
        self.num_unique_lm = len(self.unique_l)

        self.ylm_gen = GetYlms(self.num_teuk_modes, use_gpu=use_gpu, **Ylm_kwargs)

        self.mode_filter = ModeFilter(
            self.m0mask, self.num_m_zero_up, self.num_m_1_up, self.num_m0
        )

    def __call__(
        self,
        M,
        mu,
        p0,
        e0,
        theta,
        phi,
        dt=10.0,
        T=1.0,
        eps=2e-4,
        all_modes=False,
        show_progress=False,
        batch_size=-1,
    ):
        T = T * ct.Julian_year
        # get trajectory
        (t, p, e, Phi_phi, Phi_r, amp_norm) = self.inspiral_generator(
            M, mu, p0, e0, T=T, dt=dt, **self.inspiral_kwargs
        )

        # convert for gpu
        t = self.xp.asarray(t)
        p = self.xp.asarray(p)
        e = self.xp.asarray(e)
        Phi_phi = self.xp.asarray(Phi_phi)
        Phi_r = self.xp.asarray(Phi_r)
        amp_norm = self.xp.asarray(amp_norm)

        ylms = self.ylm_gen(self.unique_l, self.unique_m, theta, phi).copy()[
            self.inverse_lm
        ]

        # split into batches

        if batch_size == -1 or self.allow_batching is False:
            inds_split_all = [self.xp.arange(len(t))]
        else:
            split_inds = []
            i = 0
            while i < len(t):
                i += batch_size
                if i >= len(t):
                    break
                split_inds.append(i)

            inds_split_all = self.xp.split(self.xp.arange(len(t)), split_inds)

        iterator = enumerate(inds_split_all)
        iterator = tqdm(iterator, desc="time batch") if show_progress else iterator

        for i, inds_in in iterator:

            t_temp = t[inds_in]
            p_temp = p[inds_in]
            e_temp = e[inds_in]
            Phi_phi_temp = Phi_phi[inds_in]
            Phi_r_temp = Phi_r[inds_in]
            amp_norm_temp = amp_norm[inds_in]

            # amplitudes
            teuk_modes = self.amplitude_generator(
                p_temp, e_temp, self.l_arr, self.m_arr, self.n_arr
            )

            amp_for_norm = self.xp.sum(
                self.xp.abs(
                    self.xp.concatenate(
                        [teuk_modes, self.xp.conj(teuk_modes[:, self.m0mask])], axis=1
                    )
                )
                ** 2,
                axis=1,
            ) ** (1 / 2)

            factor = amp_norm_temp / amp_for_norm
            teuk_modes = teuk_modes * factor[:, np.newaxis]

            # TODO: check normalization of flux

            if all_modes:
                self.ls = self.l_arr[: teuk_modes.shape[1]]
                self.ms = self.m_arr[: teuk_modes.shape[1]]
                self.ns = self.n_arr[: teuk_modes.shape[1]]

                keep_modes = self.xp.arange(teuk_modes.shape[1])
                temp2 = keep_modes * (keep_modes < self.num_m0) + (
                    keep_modes + self.num_m_1_up
                ) * (keep_modes >= self.num_m0)

                ylmkeep = self.xp.concatenate([keep_modes, temp2])
                ylms_in = ylms[ylmkeep]
                teuk_modes_in = teuk_modes

            else:
                (teuk_modes_in, ylms_in, self.ls, self.ms, self.ns) = self.mode_filter(
                    eps, teuk_modes, ylms, self.l_arr, self.m_arr, self.n_arr
                )

            self.num_modes_kept = teuk_modes.shape[1]

            waveform_temp = self.sum(
                t_temp,
                p_temp,
                e_temp,
                Phi_phi_temp,
                Phi_r_temp,
                teuk_modes_in,
                self.ms,
                self.ns,
                ylms_in,
                dt,
                T,
            )

            if i > 0:
                waveform = self.xp.concatenate([waveform, waveform_temp])

            else:
                waveform = waveform_temp

        return waveform

    """
    @classmethod
    def inspiral_generator(self, *args, **kwargs):
        raise NotImplementedError

    @classmethod
    def amplitude_genertor(self, *args, **kwargs):
        raise NotImplementedError

    @classmethod
    def create_waveform(self, *args, **kwargs):
        raise NotImplementedError
    """


# TODO: free memory in trajectory
class FastSchwarzschildEccentricFlux(SchwarzschildEccentricBase):
    def __init__(self, *args, **kwargs):

        self.gpu_capability = True
        self.allow_batching = False

        SchwarzschildEccentricBase.__init__(
            self,
            RunFluxInspiral,
            ROMANAmplitude,
            InterpolatedModeSum,
            *args,
            use_gpu=self.gpu_capability,
            **kwargs
        )


class SlowSchwarzschildEccentricFlux(SchwarzschildEccentricBase):
    def __init__(self, *args, **kwargs):

        # declare specific properties
        if "inspiral_kwargs" not in kwargs:
            kwargs["inspiral_kwargs"] = {}
        kwargs["inspiral_kwargs"]["DENSE_STEPPING"] = 1

        self.gpu_capability = False
        self.allow_batching = True

        SchwarzschildEccentricBase.__init__(
            self,
            RunFluxInspiral,
            Interp2DAmplitude,
            DirectModeSum,
            *args,
            use_gpu=self.gpu_capability,
            **kwargs
        )


if __name__ == "__main__":
    import time

    few = SlowSchwarzschildEccentricFlux(
        inspiral_kwargs={
            "DENSE_STEPPING": 1,
            "max_init_len": int(1e7),
            "step_eps": 1e-10,
        },
        # amplitude_kwargs={"max_input_len": int(1e5)},
        amplitude_kwargs=dict(num_teuk_modes=3843, lmax=10, nmax=30),
        Ylm_kwargs={"assume_positive_m": False},
    )

    M = 1e6
    mu = 1e1
    p0 = 14.0
    e0 = 0.5
    theta = np.pi / 2
    phi = 0.0
    dt = 10.0
    T = 1.0  # 1124936.040602 / ct.Julian_year
    eps = 1e-2
    all_modes = False
    step_eps = 1e-11
    show_progress = True
    batch_size = 10000

    mismatch = []
    num_modes = []
    timing = []
    eps_all = 10.0 ** np.arange(-10, -2)

    eps_all = np.concatenate([np.array([1e-25]), eps_all])
    fullwave = np.genfromtxt("/projects/b1095/mkatz/emri/slow_1e6_1e1_14_05.txt")
    fullwave = fullwave[:, 5] + 1j * fullwave[:, 6]

    for i, eps in enumerate(eps_all):
        all_modes = False if i > 0 else True
        num = 1
        st = time.perf_counter()
        for jjj in range(num):

            # print(jjj, "\n")
            wc = few(
                M,
                mu,
                p0,
                e0,
                theta,
                phi,
                dt=dt,
                T=T,
                eps=eps,
                all_modes=all_modes,
                show_progress=show_progress,
                batch_size=batch_size,
            )

            try:
                wc = wc.get()
            except AttributeError:
                pass

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
