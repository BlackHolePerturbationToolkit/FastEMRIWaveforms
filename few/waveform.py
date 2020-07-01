import sys
import os

sys.path.insert(0, os.path.abspath("/home/mlk667/FastEMRIWaveforms/"))
sys.path.insert(0, os.path.abspath("/Users/michaelkatz/Research/FastEMRIWaveforms/"))


import numpy as np
from tqdm import tqdm

try:
    import cupy as xp

except (ImportError, ModuleNotFoundError) as e:
    import numpy as xp

from few.utils.baseclasses import SchwarzschildEccentric

from few.trajectory.flux import RunSchwarzEccFluxInspiral

from few.amplitude.interp2dcubicspline import Interp2DAmplitude

from few.utils.overlap import get_mismatch

try:
    from few.amplitude.romannet import ROMANAmplitude
    from few.summation.interpolated_mode_sum import InterpolatedModeSum

except (ModuleNotFoundError, ImportError) as e:
    pass

from few.utils.mode_filter import ModeFilter
from few.utils.ylm import GetYlms
from few.summation.direct_mode_sum import DirectModeSum

from abc import ABC


# work out imports with sphinx
# TODO: unit tests
# TODO: deal with libs and includes
# TODO: make sure constants are same
# TODO: Allow for use of gpu in one module but not another (?)
# TODO: add omp to CPU modules
# TODO: associated files for install
# TODO: highest level waveform that uses kwargs to pick waveform.
# TODO: shared memory based on CUDA_ARCH
# TODO: adjust into packages
# TODO: choice of integrator
# TODO: remove step_eps in flux.py
# TODO: free memory in trajectory
# TODO: deal with attributes
# TODO: ABC for specific classes
# TODO: Add more safeguards on settings.
from scipy import constants as ct


class SchwarzschildEccentricWaveformBase(SchwarzschildEccentric, ABC):
    """Base class for the actual Schwarzschild eccentric waveforms.

    This class carries information and methods that are common to any
    implementation of Schwarzschild eccentric waveforms. These include
    initialization and the actual base code for building a waveform. This base
    code calls the various modules chosen by the user or according to the
    predefined waveform classes available. See
    :class:`few.utils.baseclasses.SchwarzschildEccentric` for information
    high level information on these waveform models.

    args:
        inspiral_module (obj): Class object representing the module
            for creating the inspiral. This returns the phases and orbital
            parameters. See :ref:`trajectory-label`.
        amplitude_module (obj): Class object representing the module for
            generating amplitudes. See :ref:`amplitude-label` for more
            information.
        sum_module (obj): Class object representing the module for summing the
            final waveform from the amplitude and phase information. See
            :ref:`summation-label`.
        inspiral_kwargs (dict, optional): Optional kwargs to pass to the
            inspiral generator. **Important Note**: These kwargs are passed
            online, not during instantiation like other kwargs here. Default is
            {}.
        amplitude_kwargs (dict, optional): Optional kwargs to pass to the
            amplitude generator during instantiation. Default is {}.
        sum_kwargs (dict, optional): Optional kwargs to pass to the
            sum module during instantiation. Default is {}.
        Ylm_kwargs (dict, optional): Optional kwargs to pass to the
            Ylm generator during instantiation. Default is {}.
        use_gpu (bool, optional): If True, use GPU resources. Default is False.

    """

    def attributes_SchwarzschildEccentricWaveformBase(self):
        """
        attributes:
            xp (obj): cupy or numpy based on GPU usage.
            inspiral_generator (obj): instantiated trajectory module.
            amplitude_generator (obj): instantiated amplitude module.
            create_waveform (obj): instantiated summation module.
            ylm_gen (obj): instantiated Ylm module.
            num_teuk_modes (int): number of Teukolsky modes in the model.
            m0sort (1D int xp.ndarray): array of indices to sort accoring to
                :math:`(m=0)` parts first and then :math:`m>0` parts.
            l_arr, m_arr, n_arr (1D int xp.ndarray): :math:`(l,m,n)` arrays
                containing indices for each mode.
            lmn_indices (dict): Dictionary mapping a tuple of :math:`(l,m,n)` to
                the respective index in l_arr, m_arr, and n_arr.
            num_m_zero_up (int): Number of modes with :math:`m\geq0`.
            num_m0 (int): Number of modes with :math:`m=0`.
            num_m_1_up (int): Number of modes with :math:`m\geq1`.
            unique_l, unique_m (1D int xp.ndarray): Arrays of unique :math:`l` and
                :math:`m` values.
            inverse_lm (1D int xp.ndarray): Array of indices that expands unique
                :math:`(l, m)` values to the full array of :math:`(l,m,n)` values.
            ls, ms, ns (1D int xp.ndarray): Arrays of mode indices :math:`(l,m,n)`
                after filtering operation. If no filtering, these are equivalent
                to l_arr, m_arr, n_arr.
        """
        pass

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

        self.sanity_check_gpu(use_gpu)

        if use_gpu:
            self.xp = xp

        else:
            self.xp = np

        self.inspiral_kwargs = inspiral_kwargs
        self.inspiral_generator = inspiral_module()

        self.amplitude_generator = amplitude_module(**amplitude_kwargs)
        self.create_waveform = sum_module(**sum_kwargs)
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

        try:
            self.lmn_indices = {tuple(md_i): i for i, md_i in enumerate(md.T.get())}

        except AttributeError:
            self.lmn_indices = {tuple(md_i): i for i, md_i in enumerate(md.T)}

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
            self.m0mask,
            self.num_m_zero_up,
            self.num_m_1_up,
            self.num_m0,
            use_gpu=use_gpu,
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
        eps=1e-5,
        show_progress=False,
        batch_size=-1,
        mode_selection=None,
    ):
        """Call function for SchwarzschildEccentric models.

        This function will take input parameters and produce Schwarzschild
        eccentric waveforms. It will use all of the modules preloaded to
        compute desired outputs.

        TODO: add limits on p0

        args:
            M (double): Mass of larger black hole in solar masses.
            mu (double): Mass of compact object in solar masses.
            p0 (double): Initial semilatus rectum (:math:`6.0\leq p_0\leq18.0`).
            e0 (double): Initial eccentricity (:math:`0.0\leq e_0\leq0.7`).
            theta (double): Polar viewing angle (:math:`-\pi/2\leq\Theta\leq\pi/2`).
            phi (double): Azimuthal viewing angle.
            dt (double, optional): Time between samples in seconds (inverse of
                sampling frequency). Default is 10.0.
            T (double, optional): Total observation time in years.
                Default is 1.0.
            eps (double, optional): Controls the fractional accuracy during mode
                filtering. Raising this parameter will remove modes. Lowering
                this parameter will add modes. Default that gives a good overalp
                is 1e-5.
            show_progress (bool, optional): If True, show progress through
                amplitude/waveform batches using
                `tqdm <https://tqdm.github.io/>`_. Default is False.
            batch_size (int, optional): If less than 0, create the waveform
                without batching. If greater than zero, create the waveform
                batching in sizes of batch_size. Default is -1.
            mode_selection (str or list or None): Determines the type of mode
                filtering to perform. If None, perform our base mode filtering
                with eps as the fractional accuracy on the total power.
                If 'all', it will run all modes without filtering. If a list of
                tuples (or lists) of mode indices
                (e.g. [(:math:`l_1,m_1,n_1`), (:math:`l_2,m_2,n_2`)]) is
                provided, it will return those modes combined into a
                single waveform.


        Returns:
            1D complex128 xp.ndarray: The output waveform.

        """

        theta, phi = self.sanity_check_viewing_angles(theta, phi)
        self.sanity_check_init(M, mu, p0, e0)
        T = T * ct.Julian_year
        # get trajectory
        (t, p, e, Phi_phi, Phi_r, amp_norm) = self.inspiral_generator(
            M, mu, p0, e0, T=T, dt=dt, **self.inspiral_kwargs
        )
        self.sanity_check_traj(p, e)

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

            if isinstance(mode_selection, str):
                if mode_selection == "all":
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
                    raise ValueError("If mode selection is a string, must be `all`.")

            elif isinstance(mode_selection, list):
                if mode_selection == []:
                    raise ValueError("If mode selection is a list, cannot be empty.")

                keep_modes = self.xp.zeros(len(mode_selection), dtype=self.xp.int32)
                for jj, lmn in enumerate(mode_selection):
                    keep_modes[jj] = self.xp.int32(self.lmn_indices[tuple(lmn)])

                self.ls = self.l_arr[keep_modes]
                self.ms = self.m_arr[keep_modes]
                self.ns = self.n_arr[keep_modes]

                temp2 = keep_modes * (keep_modes < self.num_m0) + (
                    keep_modes + self.num_m_1_up
                ) * (keep_modes >= self.num_m0)

                ylmkeep = self.xp.concatenate([keep_modes, temp2])
                ylms_in = ylms[ylmkeep]
                teuk_modes_in = teuk_modes[:, keep_modes]

            else:
                modeinds = [self.l_arr, self.m_arr, self.n_arr]
                (teuk_modes_in, ylms_in, self.ls, self.ms, self.ns) = self.mode_filter(
                    teuk_modes, ylms, modeinds, eps=eps
                )

            self.num_modes_kept = teuk_modes_in.shape[1]

            waveform_temp = self.create_waveform(
                t_temp,
                teuk_modes_in,
                ylms_in,
                dt,
                T,
                Phi_phi_temp,
                Phi_r_temp,
                self.ms,
                self.ns,
            )

            if i > 0:
                waveform = self.xp.concatenate([waveform, waveform_temp])

            else:
                waveform = waveform_temp

        return waveform

    def sanity_check_gpu(self, use_gpu):
        if self.gpu_capability is False and use_gpu is True:
            raise Exception(
                "The use_gpu kwarg is True, but this class does not have GPU capabilites."
            )


class FastSchwarzschildEccentricFlux(SchwarzschildEccentricWaveformBase):
    """Prebuilt model for fast Schwarzschild eccentric flux-based waveforms.

    This model combines the most efficient modules to produce the fastest
    accurate EMRI waveforms. It leverages GPU hardware for maximal acceleration,
    but is also available on for CPUs. Please see
    :class:`few.utils.baseclasses.SchwarzschildEccentric` for general
    information on this class of models.

    The trajectory module used here is :class:`few.trajectory.flux` for a
    flux-based, sparse trajectory. This returns approximately 100 points.

    The amplitudes are then determined with
    :class:`few.amplitude.romannet.ROMANAmplitude` along these sparse
    trajectories. This gives complex amplitudes for all modes in this model at
    each point in the trajectory. These are then filtered with
    :class:`few.utils.mode_filter.ModeFilter`.

    The modes that make it through the filter are then summed by
    :class:`few.summation.interpolated_mode_sum.InterpolatedModeSum`.

    See :class:`few.waveform.SchwarzschildEccentricWaveformBase` for information
    on inputs. See examples as well.

    args:
        inspiral_kwargs (dict, optional): Optional kwargs to pass to the
            inspiral generator. **Important Note**: These kwargs are passed
            online, not during instantiation like other kwargs here. Default is
            {}.
        amplitude_kwargs (dict, optional): Optional kwargs to pass to the
            amplitude generator during instantiation. Default is {}.
        sum_kwargs (dict, optional): Optional kwargs to pass to the
            sum module during instantiation. Default is {}.
        Ylm_kwargs (dict, optional): Optional kwargs to pass to the
            Ylm generator during instantiation. Default is {}.
        use_gpu (bool, optional): If True, use GPU resources. Default is False.
        *args (list, placeholder): args for waveform model.
        **kwargs (dict, placeholder): kwargs for waveform model.

    attributes:
        gpu_capability (bool): If True, this wavefrom can leverage gpu
            resources. For this class it is True.
        allow_batching (bool): If True, this waveform can use the batch_size
            kwarg. For this class it is False.

    """

    def __init__(
        self,
        inspiral_kwargs={},
        amplitude_kwargs={},
        sum_kwargs={},
        Ylm_kwargs={},
        use_gpu=False,
        *args,
        **kwargs
    ):

        self.gpu_capability = True
        self.allow_batching = False

        SchwarzschildEccentricWaveformBase.__init__(
            self,
            RunSchwarzEccFluxInspiral,
            ROMANAmplitude,
            InterpolatedModeSum,
            inspiral_kwargs=inspiral_kwargs,
            amplitude_kwargs=amplitude_kwargs,
            sum_kwargs=sum_kwargs,
            Ylm_kwargs=Ylm_kwargs,
            use_gpu=use_gpu,
            *args,
            **kwargs
        )


class SlowSchwarzschildEccentricFlux(SchwarzschildEccentricWaveformBase):
    """Prebuilt model for slow Schwarzschild eccentric flux-based waveforms.

    This model combines the various modules to produce the a reference waveform
    against which we test our fast models. Please see
    :class:`few.utils.baseclasses.SchwarzschildEccentric` for general
    information on this class of models.

    The trajectory module used here is :class:`few.trajectory.flux` for a
    flux-based trajectory. For this slow waveform, the DENSE_SAMPLING parameter
    from :class:`few.utils.baseclasses.TrajectoryBase` is fixed to 1 to create
    a densely sampled trajectory.

    The amplitudes are then determined with
    :class:`few.amplitude.interp2dcubicspline.Interp2DAmplitude`
    along a densely sampled trajectory. This gives complex amplitudes
    for all modes in this model at each point in the trajectory. These, can be
    chosent to be filtered, but for reference waveforms, they should not be.

    The modes that make it through the filter are then summed by
    :class:`few.summation.direct_mode_sum.DirectModeSum`.

    See :class:`few.waveform.SchwarzschildEccentricWaveformBase` for information
    on inputs. See examples as well.

    args:
        inspiral_kwargs (dict, optional): Optional kwargs to pass to the
            inspiral generator. **Important Note**: These kwargs are passed
            online, not during instantiation like other kwargs here. Default is
            {}.
        amplitude_kwargs (dict, optional): Optional kwargs to pass to the
            amplitude generator during instantiation. Default is {}.
        sum_kwargs (dict, optional): Optional kwargs to pass to the
            sum module during instantiation. Default is {}.
        Ylm_kwargs (dict, optional): Optional kwargs to pass to the
            Ylm generator during instantiation. Default is {}.
        use_gpu (bool, optional): If True, use GPU resources. Default is False.
        *args (list, placeholder): args for waveform model.
        **kwargs (dict, placeholder): kwargs for waveform model.

    """

    def attributes_SlowSchwarzschildEccentricFlux(self):
        """
        attributes:
            gpu_capability (bool): If True, this wavefrom can leverage gpu
                resources. For this class it is False.
            allow_batching (bool): If True, this waveform can use the batch_size
                kwarg. For this class it is True.
        """
        pass

    def __init__(
        self,
        inspiral_kwargs={},
        amplitude_kwargs={},
        sum_kwargs={},
        Ylm_kwargs={},
        use_gpu=False,
        *args,
        **kwargs
    ):

        # declare specific properties
        inspiral_kwargs["DENSE_STEPPING"] = 1

        self.gpu_capability = False
        self.allow_batching = True

        SchwarzschildEccentricWaveformBase.__init__(
            self,
            RunSchwarzEccFluxInspiral,
            Interp2DAmplitude,
            DirectModeSum,
            inspiral_kwargs=inspiral_kwargs,
            amplitude_kwargs=amplitude_kwargs,
            sum_kwargs=sum_kwargs,
            Ylm_kwargs=Ylm_kwargs,
            use_gpu=use_gpu,
            *args,
            **kwargs
        )


if __name__ == "__main__":
    import time

    use_gpu = False
    few = FastSchwarzschildEccentricFlux(
        inspiral_kwargs={
            "DENSE_STEPPING": 0,
            "max_init_len": int(1e3),
            "step_eps": 1e-10,
        },
        amplitude_kwargs={"max_input_len": int(1e3), "use_gpu": use_gpu},
        # amplitude_kwargs=dict(),
        Ylm_kwargs={"assume_positive_m": False},
        sum_kwargs={"use_gpu": use_gpu},
        use_gpu=use_gpu,
    )

    few2 = SlowSchwarzschildEccentricFlux(
        inspiral_kwargs={
            "DENSE_STEPPING": 1,
            "max_init_len": int(1e7),
            "step_eps": 1e-10,
        },
        # amplitude_kwargs={"max_input_len": int(1e3), "use_gpu": use_gpu},
        amplitude_kwargs=dict(),
        Ylm_kwargs={"assume_positive_m": False},
        sum_kwargs={"use_gpu": use_gpu},
        use_gpu=use_gpu,
    )

    M = 1e6
    mu = 1e1
    p0 = 14.0
    e0 = 0.5
    theta = np.pi / 2
    phi = 0.0
    dt = 10.0
    T = 1.0 / 100.0  # 1124936.040602 / ct.Julian_year
    eps = 1e-2
    mode_selection = None
    step_eps = 1e-11
    show_progress = True
    batch_size = 10000

    mismatch_out = []
    num_modes = []
    timing = []
    eps_all = 10.0 ** np.arange(-10, -2)

    eps_all = np.concatenate([np.array([1e-25]), eps_all])

    try:
        fullwave = np.genfromtxt("/projects/b1095/mkatz/emri/slow_1e6_1e1_14_05.txt")
    except OSError:
        fullwave = np.genfromtxt("slow_1e6_1e1_14_05.txt")

    if use_gpu:
        fullwave = xp.asarray(fullwave[:, 5] + 1j * fullwave[:, 6])
    else:
        fullwave = np.asarray(fullwave[:, 5] + 1j * fullwave[:, 6])

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
                mode_selection=mode_selection,
                show_progress=show_progress,
                batch_size=batch_size,
            )

            wc2 = few2(
                M,
                mu,
                p0,
                e0,
                theta,
                phi,
                dt=dt,
                T=T,
                eps=eps,
                mode_selection=mode_selection,
                show_progress=show_progress,
                batch_size=batch_size,
            )

            # try:
            #    wc = wc.get()
            # except AttributeError:
            #    pass

        et = time.perf_counter()

        mm = get_mismatch(fullwave, wc2, use_gpu=use_gpu)
        mismatch_out.append(mm)
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
    #    "info_check_1e6_1e1_14_05", np.asarray([eps_all, mismatch_out, num_modes, timing]).T
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
