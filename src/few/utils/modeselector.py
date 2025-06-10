# Online mode selection for FastEMRIWaveforms Packages

import os
import sys

import numpy as np

from .baseclasses import BackendLike, ParallelModuleBase
from .constants import MTSUN_SI, PI
from .geodesic import get_fundamental_frequencies
from .globals import get_logger

# pytorch
try:
    import torch
except (ImportError, ModuleNotFoundError):
    pass  #  we can catch this later

from typing import Optional, Union

dir_path = os.path.dirname(os.path.realpath(__file__))


# fmt: off
@np.vectorize
def SPAFunc(x, th=7.0):
    II = 0.0 + 1.0j
    Gamp13 = 2.67893853470774763  # Gamma(1/3)
    Gamm13 = -4.06235381827920125  # Gamma(-1/3)

    if np.abs(x) <= th:
        xx = complex(x)
        pref1 = np.exp(-2. * np.pi * II / 3.) * pow(xx, 5. / 6.) * Gamm13 / pow(2., 1. / 3.)
        pref2 = np.exp(-np.pi * II / 3.) * pow(xx, 1. / 6.) * Gamp13 / pow(2., 2. / 3.)
        x2 = x * x

        c1_0, c1_2, c1_4, c1_6, c1_8, c1_10, c1_12, c1_14, c1_16, c1_18, c1_20, c1_22, c1_24, c1_26 = (
            0.5, -0.09375, 0.0050223214285714285714, -0.00012555803571428571429, 1.8109332074175824176e-6,
            -1.6977498819539835165e-8, 1.1169407118118312608e-10, -5.4396463237589184781e-13,
            2.0398673714095944293e-15, -6.0710338434809358015e-18, 1.4687985105195812423e-20,
            -2.9454515585285720100e-23, 4.9754249299469121790e-26, -7.1760936489618925658e-29
        )

        ser1 = c1_0 + x2*(c1_2 + x2*(c1_4 + x2*(c1_6 + x2*(c1_8 + x2*(c1_10 + x2*(c1_12 + x2*(c1_14 + x2*(c1_16 + x2*(c1_18 + x2*(c1_20 + x2*(c1_22 + x2*(c1_24 + x2*c1_26))))))))))))

        c2_0, c2_2, c2_4, c2_6, c2_8, c2_10, c2_12, c2_14, c2_16, c2_18, c2_20, c2_22, c2_24, c2_26 = (
            1., -0.375, 0.028125, -0.00087890625, 0.000014981356534090909091, -1.6051453429383116883e-7,
            1.1802539286311115355e-9, -6.3227889033809546546e-12, 2.5772237377911499951e-14,
            -8.2603324929203525483e-17, 2.1362928861000911763e-19, -4.5517604107246260858e-22,
            8.1281435905796894390e-25, -1.2340298973552160079e-27
        )

        ser2 = c2_0 + x2*(c2_2 + x2*(c2_4 + x2*(c2_6 + x2*(c2_8 + x2*(c2_10 + x2*(c2_12 + x2*(c2_14 + x2*(c2_16 + x2*(c2_18 + x2*(c2_20 + x2*(c2_22 + x2*(c2_24 + x2*c2_26))))))))))))

        ans = np.exp(-II * x) * (pref1 * ser1 + pref2 * ser2)
    else:
        y = 1. / x
        pref = np.exp(-0.75 * II * np.pi) * np.sqrt(0.5 * np.pi)

        c_0, c_1, c_2, c_3, c_4, c_5, c_6, c_7, c_8 = (
            II, 0.069444444444444444444, -0.037133487654320987654 * II, -0.037993059127800640146,
            0.057649190412669721333 * II, 0.11609906402551541102, -0.29159139923075051147 * II,
            -0.87766696951001691647, 3.0794530301731669934 * II
        )

        ser = c_0 + y * (c_1 + y * (c_2 + y * (c_3 + y * (c_4 + y * (c_5 + y * (c_6 + y * (c_7 + y * c_8)))))))

        ans = pref * ser

    return ans
# fmt: on


class ModeSelector(ParallelModuleBase):
    r"""Filter teukolsky amplitudes based on power contribution.

    This module takes teukolsky modes, combines them with their associated ylms,
    and determines the power contribution from each mode. It then filters the
    modes bases on the fractional accuracy on the total power (eps) parameter.
    Additionally, if a sensitivity curve is provided, the mode power is also
    weighted according to the PSD of the sensitivity.

    The mode filtering is a major contributing factor to the speed of these
    waveforms as it removes large numbers of useles modes from the final
    summation calculation.

    Be careful as this is built based on the construction that input mode arrays
    will in order of :math:`m=0`, :math:`m>0`, and then :math:`m<0`.

    args:
        l_arr: The l-mode indices for each mode index.
        m_arr: The m-mode indices for each mode index. Requires all :math:`m \geq 0`.
        n_arr: The n-mode indices for each mode index.
        mode_selection: Determines the type of mode
            filtering to perform. If None, use default mode filtering provided
            by :code:`mode_selector`. If 'all', it will run all modes without
            filtering. If 'threshold' it will override other options to filter by the
            threshold value set by :code:`mode_selection_threshold`. If a list of tuples (or lists) of
            mode indices (e.g. [(:math:`l_1,m_1,n_1`), (:math:`l_2,m_2,n_2`)]) is
            provided, it will return those modes combined into a
            single waveform.
            Default is None.
        include_minus_mkn: If True, then include :math:`(-m, -k, -n)` mode when
            computing a :math:`(m, k, n)` mode. This only affects modes if :code:`mode_selection`
            is a list of specific modes when the class is called. If True, this list of modes
            provided at call time must only contain :math:`m\geq 0`. Default is True.
        mode_selection_threshold: Fractional accuracy of the total power used
            to determine the contributing modes. Lowering this value will
            calculate more modes slower the waveform down, but generally
            improving accuracy. Increasing this value removes modes from
            consideration and can have a considerable affect on the speed of
            the waveform, albeit at the cost of some accuracy (usually an
            acceptable loss). Default that gives good mismatch qualities is
            1e-5.
        sensitivity_fn: Sensitivity curve function that takes
            a frequency (Hz) array as input and returns the Power Spectral Density (PSD)
            of the sensitivity curve. Default is None. If this is not none, this
            sennsitivity is used to weight the mode values when determining which
            modes to keep. **Note**: if the sensitivity function is provided,
            and GPUs are used, then this function must accept CuPy arrays as input.
        **kwargs: Optional keyword arguments for the base class:
            :class:`few.utils.baseclasses.ParallelModuleBase`.

    """

    def __init__(
        self,
        l_arr: np.ndarray,
        m_arr: np.ndarray,
        n_arr: np.ndarray,
        mode_selection: Optional[Union[str, list, np.ndarray]] = None,
        include_minus_mkn: Optional[bool] = True,
        mode_selection_threshold: float = 1e-5,
        sensitivity_fn: Optional[object] = None,
        force_backend: BackendLike = None,
        **kwargs,
    ):
        ParallelModuleBase.__init__(self, force_backend=force_backend, **kwargs)

        assert self.xp.all(m_arr >= 0), "ModeSelector only supports m >= 0."

        # store information releated to m values
        # the order is m = 0, m > 0, m < 0
        m0mask = m_arr != 0
        self.m0mask = m0mask
        """array: Mask for m=0 modes."""
        self.num_m_zero_up = len(m0mask)
        r"""int: Number of modes with :math:`m\geq0`."""
        self.num_m_1_up = len(self.xp.arange(len(m0mask))[m0mask])
        r"""int: Number of modes with :math:`m\geq1`."""
        self.num_m0 = len(self.xp.arange(len(m0mask))[~m0mask])
        """int: Number of modes with :math:`m=0`."""

        self.l_arr = l_arr
        """array: l-mode indices for each mode index."""
        self.m_arr = m_arr
        """array: m-mode indices for each mode index."""
        self.n_arr = n_arr
        """array: n-mode indices for each mode index."""

        self.sensitivity_fn = sensitivity_fn
        """object: sensitivity generating function for power-weighting."""

        self.include_minus_mkn = include_minus_mkn
        """bool: Whether to include modes with m < 0 in the output."""
        """Note that the module always outputs m>=0 Teukolsky amplitudes, but +m and -m Ylms.
        Instead m<0 Teukolsky amplitudes are included in the mode sum via mode symmetry.
        Therefore, to include m<0 Teukolsky amplitudes in the waveform
        but NOT m>0, we set the m>0 Ylm to zero."""

        self.mode_selection = mode_selection
        self.negative_m_flag = False  # flag to check if m < 0 modes are included
        """bool: Specifies whether there are negative m-modes in mode_selection."""

        if isinstance(self.mode_selection, str) and self.mode_selection not in [
            "all",
            "threshold",
        ]:
            raise ValueError(
                "If mode_selection is a string, it must be either 'all', 'threshold', or None."
            )
        elif isinstance(self.mode_selection, list):
            # Do not allow empty lists
            if len(self.mode_selection) == 0:
                raise ValueError("If mode selection is a list, cannot be empty.")

            # Do not allow duplicate modes
            if self.xp.any(
                self.xp.unique(
                    self.xp.asarray(mode_selection), return_counts=True, axis=0
                )[1]
                > 1
            ):
                raise ValueError("Mode selection has duplicate modes.")

            # Check if m < 0 modes are included. Warn user of potential errors this might cause if true.
            self.mode_arr = self.xp.asarray(self.mode_selection)
            if self.xp.any(self.mode_arr[:, 1] < 0):
                get_logger().warning(
                    "(ModeSelector) Warning: Only supports mode_selection with m < 0 if include_minus_mkn = False. May lead to error during evaluation."
                )
                self.negative_m_flag = True
            if self.xp.any(self.xp.abs(self.mode_arr[:, 1]) > self.mode_arr[:, 0]):
                raise ValueError("Mode selection has unphysical |m| > l mode(s).")
            if self.xp.any(self.mode_arr[:, 0] < 2):
                raise ValueError("Mode selection has unphysical l < 2 mode(s).")

        else:
            self.mode_arr = None

        self.mode_selection_threshold = mode_selection_threshold
        """float: Default threshold."""

    @classmethod
    def supported_backends(cls):
        return cls.GPU_RECOMMENDED()

    @property
    def is_predictive(self):
        """Whether this mode selector should be used before or after amplitude generation"""
        return False

    def __call__(
        self,
        teuk_modes: np.ndarray,
        ylms: np.ndarray,
        modeinds: list[np.ndarray],
        fund_freq_args: Optional[tuple] = None,
        mode_selection: Optional[Union[str, list, np.ndarray]] = None,
        modeinds_map: Optional[np.ndarray] = None,
        include_minus_mkn: Optional[bool] = None,
        mode_selection_threshold: float = None,
    ) -> np.ndarray:
        r"""Call to sort and filer teukolsky modes.

        This is the call function that takes the teukolsky modes, ylms,
        mode indices and fractional accuracy of the total power and returns
        filtered teukolsky modes and ylms.

        args:
            teuk_modes: Complex teukolsky amplitudes
                from the amplitude modules.
                Shape: (number of trajectory points, number of modes).
            ylms: Array of ylm values for each mode,
                including m<0. Shape is (num of m==0,) + (num of m>0,)
                + (num of m<0). Number of m<0 and m>0 is the same, but they are
                ordered as (m==0) first then m>0 then m<0.
            modeinds: List containing the mode index arrays. If in an
                equatorial model, need :math:`(l,m,n)` arrays. If generic,
                :math:`(l,m,k,n)` arrays. e.g. [l_arr, m_arr, n_arr].
            fund_freq_args: Args necessary to determine
                fundamental frequencies along trajectory. The tuple will represent
                :math:`(m1, m2, a, p, e, \cos\iota)` where the primary mass (:math:`m_1`),
                secondary mass (:math:`m_2`), and dimensionless spin (:math:`a`),
                are scalar and the other three quantities are self.xp.ndarrays.
                This must be provided if sensitivity weighting is used. Default is None.
            mode_selection: Determines the type of mode
                filtering to perform. If None, use default mode filtering provided
                by :code:`mode_selector`. If 'all', it will run all modes without
                filtering. If 'threshold' it will override other options to filter by the
                threshold value set by :code:`mode_selection_threshold`. If a list of tuples (or lists) of
                mode indices (e.g. [(:math:`l_1,m_1,n_1`), (:math:`l_2,m_2,n_2`)]) is
                provided, it will return those modes combined into a
                single waveform. If :code:`include_minus_mkn = True`, we require that :math:`m \geq 0` for this list.
                Default is None.
            modeinds_map: Map of mode indices to Teukolsky amplitude data. Only required if :code:`mode_selection` is a list of specific mode.
                Default is None.
            include_minus_mkn: If True, then include :math:`(-m, -k, -n)` mode when
                computing a :math:`(m, k, n)` mode. This only affects modes if :code:`mode_selection`
                is a list of specific modes. Default is True.
            mode_selection_threshold: Fractional accuracy of the total power used
                to determine the contributing modes. Lowering this value will
                calculate more modes slower the waveform down, but generally
                improving accuracy. Increasing this value removes modes from
                consideration and can have a considerable affect on the speed of
                the waveform, albeit at the cost of some accuracy (usually an
                acceptable loss). Default that gives good mismatch qualities is
                1e-5.

        """
        if include_minus_mkn is None:
            include_minus_mkn = self.include_minus_mkn

        # If mode_selection is None, default to values specified at instantiation
        if mode_selection is None:
            mode_selection = self.mode_selection
            mode_arr = self.mode_arr
            if self.negative_m_flag and include_minus_mkn:
                raise ValueError(
                    "Only supports mode_selection with m >= 0 when include_minus_mkn = True."
                )
        # if it is a string, check if it is 'all' or 'eps'. If so, return all modes
        elif isinstance(mode_selection, str) and mode_selection not in [
            "all",
            "threshold",
        ]:
            raise ValueError(
                "If mode_selection is a string, it must be either 'all', 'threshold', or None."
            )
        # if it is a list of modes, make sure it passes checks
        elif isinstance(mode_selection, list):
            if len(mode_selection) == 0:
                raise ValueError("If mode selection is a list, cannot be empty.")

            # warn if mode selection is large and user provides a list of tuples
            if len(mode_selection) > 50:
                # warnings.warn("Mode selection is large. Instantiate class with mode selection rather than providing it at call time for better performance.")
                get_logger().warning(
                    "(ModeSelector) Warning: Mode selection is large. Instantiate class with mode selection rather than providing it at call time for better performance."
                )

            if self.xp.any(
                self.xp.unique(
                    self.xp.asarray(mode_selection), return_counts=True, axis=0
                )[1]
                > 1
            ):
                raise ValueError("Mode selection has duplicate modes.")

            mode_arr = self.xp.asarray(mode_selection)
            if include_minus_mkn:
                if self.xp.any(mode_arr[:, 1] < 0):
                    raise ValueError(
                        "Only supports mode_selection with m >= 0 when include_minus_mkn = True."
                    )
            if self.xp.any(self.xp.abs(mode_arr[:, 1]) > mode_arr[:, 0]):
                raise ValueError("Mode selection has unphysical |m| > l mode(s).")
            if self.xp.any(mode_arr[:, 0] < 2):
                raise ValueError("Mode selection has unphysical l < 2 mode(s).")

            if modeinds_map is None:
                raise ValueError(
                    "If mode_selection is a list, modeinds_map must be provided."
                )
            elif len(modeinds_map.shape) != len(modeinds):
                raise ValueError("modeinds_map must have the same length as modeinds.")

        if mode_selection_threshold is None:
            mode_selection_threshold = self.mode_selection_threshold

        if not include_minus_mkn and not isinstance(mode_selection, list):
            get_logger().warning(
                "(ModeSelector) Warning: Overriding include_minus_mkn to True as mode_selection is not a list."
            )

        if mode_selection == "all":
            keep_modes = self.xp.arange(teuk_modes.shape[1])
            temp2 = keep_modes * (keep_modes < self.num_m0) + (
                keep_modes + self.num_m_1_up
            ) * (keep_modes >= self.num_m0)

            ylmkeep = self.xp.concatenate([keep_modes, temp2])
            ylms_out = ylms[ylmkeep]
            teuk_modes_out = teuk_modes
            return (
                teuk_modes_out,
                ylms_out,
                modeinds[0][: teuk_modes_out.shape[1]],
                modeinds[1][: teuk_modes_out.shape[1]],
                modeinds[2][: teuk_modes_out.shape[1]],
            )

        elif isinstance(mode_selection, list):
            try:
                temp = modeinds_map[mode_arr[:, 0], mode_arr[:, 1], mode_arr[:, 2]]
            except IndexError:
                raise ValueError("Mode selection indices are out of bounds.")

        else:
            # get the power contribution of each mode including m < 0
            # if self.sensitivity_fn is None:
            power = (
                self.xp.abs(
                    self.xp.concatenate(
                        [teuk_modes, self.xp.conj(teuk_modes[:, self.m0mask])], axis=1
                    )
                    * ylms
                )
                ** 2
            )

            # if noise weighting
            if self.sensitivity_fn is not None:
                if fund_freq_args is None:
                    raise ValueError(
                        "If sensitivity weighting is desired, the fund_freq_args kwarg must be provided."
                    )

                m1, m2 = fund_freq_args[0:2]
                M = m1 + m2
                Msec = M * MTSUN_SI

                a_fr, p_fr, e_fr, x_fr = fund_freq_args[2:-1]

                if (
                    self.backend.uses_cupy
                ):  # fundamental frequencies only defined on CPU
                    p_fr = p_fr.get()
                    e_fr = e_fr.get()
                    x_fr = x_fr.get()

                # get dimensionless fundamental frequency
                OmegaPhi, OmegaTheta, OmegaR = get_fundamental_frequencies(
                    a_fr, p_fr, e_fr, x_fr
                )
                # NOTE: These frequencies may differ from waveform frequencies at 1PA order

                # get frequencies in Hz
                f_Phi, _f_omega, f_r = OmegaPhi, OmegaTheta, OmegaR = (
                    self.xp.asarray(OmegaPhi) / (Msec * 2 * PI),
                    self.xp.asarray(OmegaTheta) / (Msec * 2 * PI),
                    self.xp.asarray(OmegaR) / (Msec * 2 * PI),
                )

                # TODO: update when in kerr
                freqs = (
                    modeinds[1][self.xp.newaxis, :] * f_Phi[:, self.xp.newaxis]
                    + modeinds[2][self.xp.newaxis, :] * f_r[:, self.xp.newaxis]
                )

                freqs_shape = freqs.shape

                # make all frequencies positive
                freqs_in = self.xp.abs(freqs)
                PSD = self.sensitivity_fn(freqs_in.flatten()).reshape(freqs_shape)

                power /= PSD

            # sort the power for a cumulative summation
            inds_sort = self.xp.argsort(power, axis=1)[:, ::-1]
            power = self.xp.sort(power, axis=1)[:, ::-1]
            cumsum = self.xp.cumsum(power, axis=1)

            # initialize and indices array for keeping modes
            inds_keep = self.xp.full(cumsum.shape, True)

            # keep modes that add to within the fractional power (1 - eps)
            inds_keep[:, 1:] = cumsum[:, :-1] < cumsum[:, -1][:, self.xp.newaxis] * (
                1 - mode_selection_threshold
            )

            # finds indices of each mode to be kept
            temp = inds_sort[inds_keep]

            # adjust the index arrays to make -m indices equal to +m indices
            # if +m or -m contributes, we keep both because of structure of CUDA kernel
            temp = temp * (temp < self.num_m_zero_up) + (temp - self.num_m_1_up) * (
                temp >= self.num_m_zero_up
            )

        # if +m or -m contributes, we keep both because of structure of CUDA kernel
        keep_modes, indices, counts = self.xp.unique(
            temp, return_index=True, return_counts=True
        )

        # find minus mkn modes that need to be removed
        if include_minus_mkn or not isinstance(
            mode_selection, list
        ):  # if true then we do not want to exclude any modes
            exclude_minus_mkn = []
        else:
            # check if the mode_selection array includes both (m,k,n) and (-m,-k,-n) modes
            # if it does, we keep all of these modes and only search for (m,k,n) modes without
            # their negative counterparts
            indices_count_1 = indices[counts == 1]

            # exclude m > 0 modes if m < 0 is selected
            exclude_positive = self.xp.where((mode_arr[:, 1] < 0)[indices_count_1])[0]
            # exclude m < 0 modes if m > 0 is selected
            exclude_negative = self.xp.where((mode_arr[:, 1] > 0)[indices_count_1])[
                0
            ] + len(keep_modes)
            # concatenate the two arrays
            exclude_minus_mkn = self.xp.concatenate(
                [exclude_positive, exclude_negative]
            )

        # set ylms
        # adust temp arrays specific to ylm setup
        temp2 = keep_modes * (keep_modes < self.num_m0) + (
            keep_modes + self.num_m_1_up
        ) * (keep_modes >= self.num_m0)

        # ylm duplicates the m = 0 unlike teuk_modes
        ylmkeep = self.xp.concatenate([keep_modes, temp2])
        ylms_out = ylms[ylmkeep]

        # throw out minus mkn modes if required
        ylms_out[exclude_minus_mkn] = 0.0 + 0.0j

        # setup up teuk mode and ylm returns
        out1 = (teuk_modes[:, keep_modes], ylms_out)

        # setup up mode values that have been kept
        out2 = tuple([arr[keep_modes] for arr in modeinds])

        return out1 + out2


class NeuralModeSelector(ParallelModuleBase):
    """Filter teukolsky amplitudes based on power contribution.

    This module uses a combination of a pre-computed mask and a feed-forward neural
    network to predict the mode content of the waveform given its parameters. Therefore,
    the results will vary compared to manually computing the mode selection, but it should be
    very accurate especially for the stronger modes.

    The mode filtering is a major contributing factor to the speed of these
    waveforms as it removes large numbers of useles modes from the final
    summation calculation.

    args:
        l_arr (1D int self.xp.ndarray): The l-mode indices for each mode index.
        m_arr (1D int self.xp.ndarray): The m-mode indices for each mode index.
        n_arr (1D int self.xp.ndarray): The n-mode indices for each mode index.
        threshold (double): The network threshold value for mode retention. Decrease to keep more modes,
            minimising missed modes but slowing down the waveform computation. Defaults to 0.5 (the optimal value for accuracy).

        **kwargs (dict, optional): Keyword arguments for the base classes:
            :class:`few.utils.baseclasses.ParallelModuleBase`.
            Default is {}.
    """

    def __init__(
        self,
        l_arr,
        m_arr,
        n_arr,
        threshold=0.5,
        mode_selector_location=None,
        return_type="tuples",
        keep_inds=None,
        maximise_over_theta=False,
        force_backend: BackendLike = None,
    ):
        if mode_selector_location is None:
            raise ValueError("mode_selector_location kwarg cannot be none.")
        elif not os.path.isdir(mode_selector_location):
            raise ValueError(
                f"mode_selector location path ({mode_selector_location}) does not point to an existing directory."
            )

        ParallelModuleBase.__init__(self, force_backend=force_backend)

        # we set the pytorch device here for use with the neural network
        if self.backend.uses_cuda:
            import cupy as cp

            self.device = f"cuda:{cp.cuda.runtime.getDevice()}"
            self.neural_mode_list = [
                (lh, mh, nh)
                for lh, mh, nh in zip(l_arr.get(), m_arr.get(), n_arr.get())
            ]

            self.device = f"cuda:{cp.cuda.runtime.getDevice()}"
            self.neural_mode_list = [
                (lh, mh, nh)
                for lh, mh, nh in zip(l_arr.get(), m_arr.get(), n_arr.get())
            ]
        else:
            self.device = "cpu"
            self.neural_mode_list = [
                (lh, mh, nh) for lh, mh, nh in zip(l_arr, m_arr, n_arr)
            ]
            self.device = "cpu"
            self.neural_mode_list = [
                (lh, mh, nh) for lh, mh, nh in zip(l_arr, m_arr, n_arr)
            ]

        # TODO include waveform name in paths
        self.precomputed_mask = np.load(
            os.path.join(mode_selector_location, "precomputed_mode_mask.npy")
        ).astype(np.int32)
        self.masked_mode_list = [
            self.neural_mode_list[maskind] for maskind in self.precomputed_mask
        ]
        self.precomputed_mask = np.load(
            os.path.join(mode_selector_location, "precomputed_mode_mask.npy")
        ).astype(np.int32)
        self.masked_mode_list = [
            self.neural_mode_list[maskind] for maskind in self.precomputed_mask
        ]

        self.precomputed_mask = self.xp.asarray(
            self.precomputed_mask
        )  # for "array" return type compatibility
        self.precomputed_mask = self.xp.asarray(
            self.precomputed_mask
        )  # for "array" return type compatibility

        if "torch" not in sys.modules:
            raise ModuleNotFoundError("Pytorch not installed.")
        # if torch wasn't installed with GPU capability, raise
        if self.backend.uses_cuda and not torch.cuda.is_available():
            raise RuntimeError(
                "pytorch has not been installed with CUDA capability. Fix installation or force 'cpu' backend."
            )

        self.model_loc = os.path.join(
            mode_selector_location, "neural_mode_selector.tjm"
        )
        self.model_loc = os.path.join(
            mode_selector_location, "neural_mode_selector.tjm"
        )
        try:
            self.load_model(self.model_loc)
        except FileNotFoundError:
            raise FileNotFoundError("Neural mode predictor model file not found.")

        self.threshold = threshold
        self.vector_min, self.vector_max = np.load(
            os.path.join(mode_selector_location, "network_norm.npy")
        )
        self.vector_min, self.vector_max = np.load(
            os.path.join(mode_selector_location, "network_norm.npy")
        )

        self.return_type = return_type

        self.keep_inds = keep_inds

        self.maximise_over_theta = maximise_over_theta

    @classmethod
    def supported_backends(cls):
        return cls.GPU_RECOMMENDED()

    @property
    def is_predictive(self):
        """Whether this mode selector should be used before or after amplitude generation"""
        return True

    def __getstate__(self):
        # needs to be modified for pickleability
        if hasattr(self, "model"):
            del self.model
        return self.__dict__

    def __setstate__(self, d):
        self.__dict__ = d
        # we now load the model as we've unpickled again
        self.load_model(self.model_loc)

    def load_model(self, fp):
        self.model = torch.jit.load(fp)  # assume the model has been jit compiled
        self.model.to(self.device)
        self.model.eval()

    def __call__(self, m1, m2, a, p0, e0, xI, theta, phi, T, mode_selection_threshold):
        """Call to predict the mode content of the waveform.

        This is the call function that takes the waveform parameters, applies a
        precomputed mask and then evaluates the remaining mode content with a
        neural network classifier.

        args:
            m1 (double): Mass of larger black hole in solar masses.
            m2 (double): Mass of compact object in solar masses.
            p0 (double): Initial semilatus rectum (Must be greater than
                the separatrix at the the given e0 and x0).
                See documentation for more information on :math:`p_0<10`.
            e0 (double): Initial eccentricity.
            theta (double): Polar viewing angle.
            phi (double): Azimuthal viewing angle.
            T (double): Duration of waveform in years.
            mode_selection_threshold (double): Mode selection threshold power.
        """

        if not hasattr(self, "model"):
            self.load_model(self.model_loc)

        # wrap angles to training bounds
        phi = phi % (2 * np.pi)
        # wrap angles to training bounds
        phi = phi % (2 * np.pi)

        # maximise mode count over theta (stabilises the likelihood but often requires more modes)
        if self.maximise_over_theta:
            theta = 1.39

        inputs = np.asarray(
            [
                [
                    np.log(m1),
                    m2,
                    a,
                    p0,
                    e0,
                    xI,
                    T,
                    theta,
                    phi,
                    np.log10(mode_selection_threshold),
                ]
            ]
        )[:, np.asarray(self.keep_inds)]  # throw away the params we dont need
        inputs = np.asarray(
            [
                [
                    np.log(m1),
                    m2,
                    a,
                    p0,
                    e0,
                    xI,
                    T,
                    theta,
                    phi,
                    np.log10(mode_selection_threshold),
                ]
            ]
        )[:, np.asarray(self.keep_inds)]  # throw away the params we dont need
        # rescale network input from pre-computed
        inputs = (
            2 * (inputs - self.vector_min) / (self.vector_max - self.vector_min) - 1
        )
        inputs = (
            2 * (inputs - self.vector_min) / (self.vector_max - self.vector_min) - 1
        )
        inputs = torch.as_tensor(inputs, device=self.device).float()
        self.check_bounds(inputs)
        # get network output and threshold it based on the defined value
        with torch.inference_mode():
            mode_predictions = torch.nn.functional.sigmoid(self.model(inputs))
            keep_inds = (
                torch.where(mode_predictions > self.threshold)[0].int().cpu().numpy()
            )  # cpu() works for cpu and gpu
            keep_inds = (
                torch.where(mode_predictions > self.threshold)[0].int().cpu().numpy()
            )  # cpu() works for cpu and gpu
        # return list of modes for kept indices
        if self.return_type == "tuples":
            return [self.masked_mode_list[ind] for ind in keep_inds]
        elif self.return_type == "array":
            return self.precomputed_mask[keep_inds]

    def check_bounds(self, inputs):
        # Enforce bounds. Normalising to [-1,1] so it's easy to check (space for numerical errors)
        if torch.any(torch.abs(inputs) > 1 + 1e-3):
            breakpoint()
            raise ValueError(
                f"One of the inputs to the neural mode selector is out of bounds. Normalised inputs: {inputs}"
            )
            raise ValueError(
                f"One of the inputs to the neural mode selector is out of bounds. Normalised inputs: {inputs}"
            )
