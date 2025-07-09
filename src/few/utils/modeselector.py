# Online mode selection for FastEMRIWaveforms Packages

import numpy as np

from .baseclasses import BackendLike, ParallelModuleBase
from .globals import get_logger

from .ylm import GetYlms

from typing import Optional, Union

def get_mode_frequencies(f_phi, f_theta, f_r, m, k, n):
    return f_phi[:,None] * m[None,:] + f_theta[:,None] * k[None,:] + f_r[:,None] * n[None,:]

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
        k_arr: The k-mode indices for each mode index.
        n_arr: The n-mode indices for each mode index.
        mode_selection: Determines the type of mode
            filtering to perform. If None, use default mode filtering provided
            by :code:`mode_selector`. If 'all', it will run all modes without
            filtering. If 'threshold' it will override other options to filter by the
            threshold value set by :code:`mode_selection_threshold`. If a list of tuples (or lists) of
            mode indices (e.g. [(:math:`l_1,m_1,k_1,n_1`), (:math:`l_2,m_2,k_2,n_2`)]) is
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
        amplitude_generator: object,
        ylm_generator: Optional[object] = None,
        mode_selection: Optional[Union[str, list, np.ndarray]] = None,
        include_minus_mkn: Optional[bool] = True,
        mode_selection_threshold: float = 1e-5,
        sensitivity_fn: Optional[object] = None,
        modeinds_map: Optional[np.ndarray] = None,
        force_backend: BackendLike = None,
        **kwargs,
    ):
        ParallelModuleBase.__init__(self, force_backend=force_backend, **kwargs)
        
        self.amplitude_generator = amplitude_generator

        if ylm_generator is None:
            self.ylm_generator = GetYlms()
        else:
            self.ylm_generator = ylm_generator

        self.l_arr = self.amplitude_generator.l_arr_no_mask
        """array: l-mode indices for each mode index, given by amplitude module."""
        self.m_arr = self.amplitude_generator.m_arr_no_mask
        """array: m-mode indices for each mode index, given by amplitude module."""
        self.k_arr = self.amplitude_generator.k_arr_no_mask
        """array: k-mode indices for each mode index, given by amplitude module."""
        self.n_arr = self.amplitude_generator.n_arr_no_mask
        """array: n-mode indices for each mode index, given by amplitude module."""

        self.unique_l = self.amplitude_generator.unique_l
        self.unique_m = self.amplitude_generator.unique_m
        self.inverse_lm = self.amplitude_generator.inverse_lm

        assert self.xp.all(self.m_arr >= 0), "ModeSelector only supports m >= 0."

        # store information releated to m values
        # the order is m = 0, m > 0, m < 0
        m0mask = self.m_arr != 0
        self.m0mask = m0mask
        """array: Mask for m=0 modes."""
        self.num_m_zero_up = len(m0mask)
        r"""int: Number of modes with :math:`m\geq0`."""
        self.num_m_1_up = len(self.xp.arange(len(m0mask))[m0mask])
        r"""int: Number of modes with :math:`m\geq1`."""
        self.num_m0 = len(self.xp.arange(len(m0mask))[~m0mask])
        """int: Number of modes with :math:`m=0`."""
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
        """float: Target mismatch threshold for mode selection. Modes will be selected such that the mismatch is approximately equal to this value."""

        if modeinds_map is None:
            self.modeinds_map = self.amplitude_generator.special_index_map_arr
        else:
            self.modeinds_map = self.xp.asarray(modeinds_map)
            assert self.modeinds_map.ndim == 4, "Modeinds map must be a 4D array."

    @classmethod
    def supported_backends(cls):
        return cls.GPU_RECOMMENDED()

    def _set_defaults_and_check_inputs(self, mode_selection, mode_selection_threshold, include_minus_mkn,):
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
        else:
            mode_arr = None

        if mode_selection_threshold is None:
            mode_selection_threshold = self.mode_selection_threshold

        if not include_minus_mkn and not isinstance(mode_selection, list):
            get_logger().warning(
                "(ModeSelector) Warning: Overriding include_minus_mkn to True as mode_selection is not a list."
            )

        return mode_selection, mode_selection_threshold, include_minus_mkn, mode_arr

    def __call__(
        self,
        t,
        a,
        p,
        e,
        xI,
        theta,
        phi,
        online_mode_selection_args: Optional[dict] = None,
        mode_selection: Optional[Union[str, list, np.ndarray]] = None,
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
            modeinds: List containing the mode index :math:`(l,m,k,n)` arrays,
                e.g. [l_arr, m_arr, k_arr, n_arr].
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
                mode indices (e.g. [(:math:`l_1,m_1,k_1,n_1`), (:math:`l_2,m_2,k_2,n_2`)]) is
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

        # set defaults, check inputs are consistent, etc.
        mode_selection, mode_selection_threshold, include_minus_mkn, mode_arr = self._set_defaults_and_check_inputs(
            mode_selection, mode_selection_threshold, include_minus_mkn
        )
        
        if mode_selection == "all":
            # get teuk modes
            teuk_modes = self.amplitude_generator(a, p, e, xI)
            
            # get ylms
            ylms = self.ylm_generator(self.unique_l, self.unique_m, theta, phi)[self.inverse_lm]

            # get mode indices
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
                self.l_arr,
                self.m_arr,
                self.k_arr,
                self.n_arr,
            )

        # if mode selection is a list, compute only these modes for efficiency
        elif isinstance(mode_selection, list):
            try:
                keep_modes = self.modeinds_map[mode_arr[:, 0], mode_arr[:, 1], mode_arr[:, 2], mode_arr[:,3]]
            except IndexError:
                raise ValueError("Mode selection indices are out of bounds.")

            # pass array of mode indexes (most efficient, returns an array)
            teuk_modes = self.amplitude_generator(a, p, e, xI, specific_modes=keep_modes)

            # get ylms, only for the desired modes (needs to include -m modes in general, so we pass the kwarg)
            ylms_out = self.ylm_generator(mode_arr[:, 0], mode_arr[:,1], theta, phi, include_minus_m=True)
            
            # if include_minus_mkn is False, we need to zero out the latter half of this array
            if not include_minus_mkn:
                # zero out the -m modes
                # ylms[keep_modes >= self.num_m_zero_up] = 0.0 + 0.0j
                ylms_out[teuk_modes.shape[1]:] = 0.0 + 0.0j

            return (
                teuk_modes, 
                ylms_out,
                self.l_arr[keep_modes],
                self.m_arr[keep_modes],
                self.k_arr[keep_modes],
                self.n_arr[keep_modes],
            )

        else:
            # get teuk modes
            teuk_modes = self.amplitude_generator(a, p, e, xI)

            # get ylms
            ylms = self.ylm_generator(self.unique_l, self.unique_m, theta, phi)[self.inverse_lm]

            # get the power contribution of each mode including m < 0 --- make more efficient?
            power = (
                self.xp.abs(
                    teuk_modes * ylms[:teuk_modes.shape[1]]
                )
                ** 2

            )
            power[:,self.m0mask] += (
                self.xp.abs(
                    self.xp.conj(teuk_modes[:, self.m0mask]) * ylms[teuk_modes.shape[1]:]
                )** 2
            )

            if self.sensitivity_fn is None:
                mode_psds = 1.  # no weights applied
            else:
                # obtain the PSD for each mode in each time segment
                
                mode_freqs = self.xp.abs(
                    get_mode_frequencies(
                        online_mode_selection_args["f_phi"],
                        online_mode_selection_args["f_theta"],
                        online_mode_selection_args["f_r"],
                        self.m_arr,
                        self.k_arr,
                        self.n_arr
                    )
                )
                mode_psds = self.sensitivity_fn(
                    mode_freqs.flatten()
                ).reshape(mode_freqs.shape)
            
            # duration of each trajectory segment, used to estimate the SNR per segment
            node_times = self.xp.diff(t)

            mode_snr2_ests = ((power / mode_psds)[:-1]*node_times[:,None]).sum(0)

            # sort the power for a cumulative summation
            inds_sort = self.xp.argsort(mode_snr2_ests)[::-1]
            mode_snr2_ests = self.xp.sort(mode_snr2_ests)[::-1]
            cumsum = self.xp.cumsum(mode_snr2_ests)

            # initialize and indices array for keeping modes
            inds_keep = self.xp.full(cumsum.shape, True)

            # keep modes that add to within the fractional square SNR (1 - kappa)^2
            inds_keep[1:] = cumsum[:-1] < cumsum[-1] * (
                1 - mode_selection_threshold
            )**2

            # finds indices of each mode to be kept
            keep_modes_temp = inds_sort[inds_keep]

            # adjust the index arrays to make -m indices equal to +m indices
            # if +m or -m contributes, we keep both because of structure of CUDA kernel
            keep_modes_temp = keep_modes_temp * (keep_modes_temp < self.num_m_zero_up) + (keep_modes_temp - self.num_m_1_up) * (
                keep_modes_temp >= self.num_m_zero_up
            )

            # if +m or -m contributes, we keep both because of structure of CUDA kernel
            keep_modes, indices, counts = self.xp.unique(
                keep_modes_temp, return_index=True, return_counts=True
            )

            # find minus mkn modes that need to be removed
            if include_minus_mkn:  # if true then we do not want to exclude any modes
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
            out2 = tuple([
                self.l_arr[keep_modes],
                self.m_arr[keep_modes],
                self.k_arr[keep_modes],
                self.n_arr[keep_modes],
            ])

            return out1 + out2


def get_selected_modes_from_initial_conditions(mode_selector_module, traj_module, m1, m2, a, p0, e0, xI0, theta, phi, traj_args=None,traj_kwargs=None, mode_selector_kwargs=None):
    if traj_args is None:
        traj_args = []
    if traj_kwargs is None:
        traj_kwargs = {}
    if mode_selector_kwargs is None:
        mode_selector_kwargs = {}

    t, p, e, x, _, _, _ = traj_module(m1, m2, a, p0, e0, xI0, **traj_kwargs)
    teuk_modes_out, ylms_out, ls, ms, ks, ns = mode_selector_module(
        t, a, p, e, x, theta, phi, **mode_selector_kwargs
    )
    return teuk_modes_out, ylms_out, ls, ms, ks, ns