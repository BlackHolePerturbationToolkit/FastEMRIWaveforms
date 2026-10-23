# Online mode selection for FastEMRIWaveforms Packages

r"""Online selection of the Teukolsky modes of a trajectory.

:class:`ModeSelector` evaluates the amplitudes and :math:`{}_{-2}Y_{lm}` of a trajectory and
returns every mode, a requested list, or (the default) the smallest set of modes whose
estimated power reaches a threshold fraction of the total; the spherical-harmonic waveforms
(:class:`~few.waveform.base.SphericalHarmonicWaveformBase`, e.g. the production
``FastKerrEccentricEquatorialFlux``) call it once per waveform. Also here:
:func:`get_mode_frequencies` (mode frequencies :math:`m f_\phi + k f_\theta + n f_r`) and
:func:`get_selected_modes_from_initial_conditions` (trajectory plus selection in one call).
"""

import numpy as np

from .baseclasses import BackendLike, FewBackendConsumer
from .globals import get_logger

from .ylm import GetYlms
from ..trajectory.inspiral import EMRIInspiral

from typing import Optional, Union

def get_mode_frequencies(f_phi, f_theta, f_r, m, k, n):
    r"""Return the mode frequencies :math:`m f_\phi + k f_\theta + n f_r` along a trajectory.

    Args:
        f_phi (xp.ndarray): Azimuthal orbital frequency at each trajectory point, shape
            ``(N,)``.
        f_theta (xp.ndarray): Polar orbital frequency, shape ``(N,)``.
        f_r (xp.ndarray): Radial orbital frequency, shape ``(N,)``.
        m (xp.ndarray): ``m`` index of each mode, shape ``(n_modes,)``.
        k (xp.ndarray): ``k`` index of each mode, shape ``(n_modes,)``.
        n (xp.ndarray): ``n`` index of each mode, shape ``(n_modes,)``.

    Returns:
        xp.ndarray: Signed frequencies of shape ``(N, n_modes)``, in the unit of the inputs
        (Hz for :class:`ModeSelector`, which takes the absolute value).
    """
    return f_phi[:,None] * m[None,:] + f_theta[:,None] * k[None,:] + f_r[:,None] * n[None,:]

class ModeSelector(FewBackendConsumer):
    r"""Select the Teukolsky modes of a trajectory by their estimated power.

    Evaluates the amplitudes of an amplitude module and the :math:`{}_{-2}Y_{lm}` of a Ylm
    generator along a trajectory and returns every mode (``"all"``), a requested list of
    modes, or (default, threshold selection) the smallest set of modes that carries the
    requested fraction of the estimated power. The selection is a major contributor to the
    speed of the FEW waveforms. Called once per waveform by
    :meth:`~few.waveform.base.SphericalHarmonicWaveformBase._generate_waveform`.

    Threshold selection: for each :math:`m \ge 0` mode the power at each trajectory point
    is :math:`|A_{lmkn} Y_{lm}|^2`, plus :math:`|A^*_{lmkn} Y_{l,-m}|^2` (the
    :math:`(-m, -k, -n)` partner) for :math:`m > 0`. With ``sensitivity_fn`` it is divided
    by the PSD at the mode frequency :math:`|m f_\phi + k f_\theta + n f_r|`. The power is
    summed over the trajectory segments weighted by their durations (left Riemann sum).
    The modes are sorted by decreasing power, and each is kept while the summed power of
    the stronger modes is below :math:`(1 - \epsilon)^2` of the total, with
    :math:`\epsilon` the ``mode_selection_threshold``. The strongest mode is always kept.

    Args:
        amplitude_generator: Amplitude module, e.g.
            :class:`~few.amplitude.ampinterp2d.AmpInterpKerrEccEq`, called as
            ``amplitude_generator(a, p, e, xI)`` (with ``specific_modes`` for mode lists). It
            must carry the mode bookkeeping of :class:`~few.utils.baseclasses.SphericalHarmonic`
            (``l_arr_no_mask`` ..., ``unique_l``, ``unique_m``, ``inverse_lm``,
            ``index_map_arr``) for :math:`m \ge 0` modes only.
        ylm_generator: Ylm generator called as ``ylm_generator(l, m, theta, phi)`` (with
            ``include_minus_m=True`` for mode lists). Default ``None`` builds
            :class:`~few.utils.ylm.GetYlms` (on the default backend).
        mode_selection: Default selection of the calls: ``None`` (default; threshold
            selection), ``"threshold"``, ``"all"``, or a ``list`` of ``(l, m, k, n)`` tuples
            or lists (only a ``list`` is treated as a mode list). A list must be non-empty,
            without duplicates, with ``l >= 2`` and ``|m| <= l``; entries with ``m < 0`` log a
            warning, and a later call with ``include_minus_mkn=True`` then raises.
        include_minus_mkn: For mode lists: ``True`` (default) also sums each listed mode's
            :math:`(-m, -k, -n)` partner (through the second half of the Ylm array; the list
            must then have :math:`m \ge 0`), ``False`` zeroes that half. ``"all"`` ignores
            it; threshold selection requires ``True`` (see the Warning).
        mode_selection_threshold: Default :math:`\epsilon` of the threshold selection, used
            when a call passes ``None``. Default ``1e-5``. Larger values keep fewer modes.
            The spherical-harmonic waveforms always pass their own call value (default
            ``1e-5``), so this constructor value has no effect there.
        sensitivity_fn: PSD function of frequency: called with a flat array of the mode
            frequencies in Hz (cupy on GPU backends), it must return the PSD at those
            frequencies. Default ``None`` (no weighting). Requires
            ``online_mode_selection_args`` at call time.
        modeinds_map: 4-D integer array indexed ``[l, m, k, n]`` giving the amplitude column
            of each mode, used for mode lists. Default ``None`` uses
            ``amplitude_generator.index_map_arr``.
        force_backend: See :class:`~few.utils.baseclasses.FewBackendConsumer`.
        **kwargs: Forwarded to :class:`~few.utils.baseclasses.FewBackendConsumer`, which
            accepts no other keyword, so this must be empty.

    Attributes:
        amplitude_generator: The amplitude module.
        ylm_generator: The Ylm generator.
        unique_l (xp.ndarray): Unique ``l`` of the amplitude module's full mode arrays.
        unique_m (xp.ndarray): Matching unique ``m``.
        inverse_lm (np.ndarray): Index into ``unique_l`` / ``unique_m`` of every entry of the
            amplitude module's full mode arrays.
        mode_selection: The constructor's ``mode_selection``.
        mode_arr (xp.ndarray or None): The constructor's mode list as an ``(n, 4)`` array,
            ``None`` unless ``mode_selection`` is a list.
        modeinds_map (xp.ndarray): The 4-D mode-index map.

    Raises:
        ValueError: If ``mode_selection`` is an unknown string or an invalid list.
        AssertionError: If the amplitude module has :math:`m < 0` modes, or
            ``modeinds_map`` is not 4-D.

    Warning:
        For a selection that is not a mode list, ``include_minus_mkn=False`` logs that it is
        overridden to ``True`` but it is not; threshold selection then fails with a
        ``TypeError`` (it reads the mode list, which is ``None``).

    Example:
        .. code-block:: python

            from few.amplitude.ampinterp2d import AmpInterpSchwarzEcc
            from few.trajectory.inspiral import EMRIInspiral
            from few.trajectory.ode import SchwarzEccFlux
            from few.utils.modeselector import ModeSelector
            from few.utils.ylm import GetYlms

            traj = EMRIInspiral(func=SchwarzEccFlux)
            t, p, e, x, Phi_phi, Phi_theta, Phi_r = traj(1e5, 1e1, 0.0, 10.0, 0.7, 1.0)

            # build all three on the same backend
            selector = ModeSelector(
                AmpInterpSchwarzEcc(force_backend="cpu"),
                ylm_generator=GetYlms(force_backend="cpu"),
                force_backend="cpu",
            )
            teuk_modes, ylms, ls, ms, ks, ns = selector(
                t, 0.0, p, e, x, 1.0, 0.5, mode_selection_threshold=1e-2
            )
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
        FewBackendConsumer.__init__(self, force_backend=force_backend, **kwargs)
        
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
        """array: True for the m > 0 modes (m != 0) of the amplitude module's m >= 0 list."""
        self.num_m_zero_up = len(m0mask)
        r"""int: Number of modes with :math:`m\geq0`."""
        self.num_m_1_up = len(self.xp.arange(len(m0mask))[m0mask])
        r"""int: Number of modes with :math:`m\geq1`."""
        self.num_m0 = len(self.xp.arange(len(m0mask))[~m0mask])
        """int: Number of modes with :math:`m=0`."""
        self.sensitivity_fn = sensitivity_fn
        """object: sensitivity generating function for power-weighting."""

        self.include_minus_mkn = include_minus_mkn
        """bool: Default of ``include_minus_mkn``: whether a mode list also sums each mode's
        (-m, -k, -n) partner."""
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
        """float: Default threshold of the power-based selection (modes are kept up to
        (1 - threshold)**2 of the estimated power)."""

        if modeinds_map is None:
            self.modeinds_map = self.amplitude_generator.index_map_arr
        else:
            self.modeinds_map = self.xp.asarray(modeinds_map)
            assert self.modeinds_map.ndim == 4, "Modeinds map must be a 4D array."

    @classmethod
    def supported_backends(cls):
        """Backend preference: ``GPU_RECOMMENDED()`` (CUDA 13/12/11, then CPU)."""
        return cls.GPU_RECOMMENDED()

    def _set_defaults_and_check_inputs(self, mode_selection, mode_selection_threshold, include_minus_mkn,):
        """Fill the call's ``None`` options from the constructor and validate a mode list.

        Returns:
            tuple: ``(mode_selection, mode_selection_threshold, include_minus_mkn, mode_arr)``
            with ``mode_arr`` the ``(n, 4)`` array of a mode list, else ``None``.

        Raises:
            ValueError: For an unknown string, an empty / duplicated / unphysical mode list,
                or :math:`m < 0` entries together with ``include_minus_mkn=True``.
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
        return_sort_inds: bool = False
    ) -> tuple[np.ndarray]:
        r"""Select the modes of a trajectory and return their amplitudes and Ylms.

        Args:
            t (xp.ndarray): Trajectory times in s, shape ``(N,)``; threshold selection uses
                the segment durations ``diff(t)`` as weights.
            a (float): Dimensionless spin of the primary.
            p (xp.ndarray): Semi-latus rectum along the trajectory, shape ``(N,)``.
            e (xp.ndarray): Eccentricity along the trajectory, shape ``(N,)``.
            xI (xp.ndarray): :math:`x_I` along the trajectory, shape ``(N,)``.
            theta (float): Source-frame polar viewing angle in rad.
            phi (float): Source-frame azimuthal viewing angle in rad.
            online_mode_selection_args: Dict with the keys ``"f_phi"``, ``"f_theta"``,
                ``"f_r"``: the orbital frequencies in Hz at the trajectory points, shape
                ``(N,)``. Needed only for threshold selection with ``sensitivity_fn``.
                Default ``None``.
            mode_selection: Overrides the constructor's selection: ``"all"``,
                ``"threshold"``, or a ``list`` of ``(l, m, k, n)`` (:math:`m \ge 0` when
                ``include_minus_mkn`` is ``True``; a list of more than 50 modes logs a
                warning). ``None`` (default) uses the constructor value.
            include_minus_mkn: For mode lists: sum the :math:`(-m, -k, -n)` partners too.
                ``None`` (default) uses the constructor value.
            mode_selection_threshold: :math:`\epsilon` of the threshold selection (see the
                class docstring). ``None`` (default) uses the constructor value.
            return_sort_inds: Threshold selection only: also return the power ordering of
                the kept modes. Default ``False``.

        Returns:
            tuple: ``(teuk_modes, ylms, l, m, k, n)``:

            - ``teuk_modes``: complex, shape ``(N, n_modes)``, the :math:`m \ge 0` amplitudes
              of the selected modes;
            - ``ylms``: complex, shape ``(2 n_modes,)``: :math:`Y_{lm}` of the selected modes,
              then :math:`Y_{l,-m}` of their partners (the same :math:`Y_{l0}` again for
              :math:`m = 0`; zeroed where a partner is excluded);
            - ``l, m, k, n``: integer index arrays of shape ``(n_modes,)``.

            With ``return_sort_inds=True`` (threshold selection) a seventh entry, an integer
            array of shape ``(n_modes,)``: its ``i``-th element is the position, in the
            returned mode arrays, of the ``i``-th strongest kept mode. All arrays are on the
            backend.

        Raises:
            ValueError: For an invalid ``mode_selection`` (see
                :meth:`_set_defaults_and_check_inputs`), or "Mode selection indices are out
                of bounds." if a listed mode is outside ``modeinds_map`` (negative indices
                wrap around instead).
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

            out_tuple = (
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

            out_tuple = (
                teuk_modes, 
                ylms_out,
                mode_arr[:,0],
                mode_arr[:,1],
                mode_arr[:,2],
                mode_arr[:,3],
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
            keep_modes, indices, inv_inds, counts = self.xp.unique(
                keep_modes_temp, return_index=True, return_counts=True, return_inverse=True,
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

            out_tuple = out1 + out2
        
            if return_sort_inds:
                out_tuple += (inv_inds,)

        return out_tuple

def get_selected_modes_from_initial_conditions(
        mode_selector_module: ModeSelector, 
        traj_module: EMRIInspiral, 
        m1: float, 
        m2: float, 
        a: float, 
        p0: float, 
        e0: float, 
        xI0: float, 
        theta: float, 
        phi: float, 
        traj_args: Optional[list]=None,
        traj_kwargs: Optional[dict]=None, 
        mode_selector_kwargs: Optional[dict]=None
    ) -> tuple[np.ndarray]:
    r"""Run a trajectory from initial conditions and select its modes.

    Runs ``traj_module(m1, m2, a, p0, e0, xI0, **traj_kwargs)``, evaluates the orbital
    frequencies in Hz at the trajectory times (first derivative of the integrator's phase
    spline divided by :math:`2\pi`), and calls ``mode_selector_module`` with them as
    ``online_mode_selection_args`` (so a ``sensitivity_fn`` can be used).

    Args:
        mode_selector_module: The :class:`ModeSelector`.
        traj_module: The :class:`~few.trajectory.inspiral.EMRIInspiral` (its
            ``inspiral_generator`` provides the phase-derivative spline).
        m1: Primary mass in solar masses.
        m2: Secondary mass in solar masses.
        a: Dimensionless spin of the primary.
        p0: Initial semi-latus rectum in units of :math:`M`.
        e0: Initial eccentricity.
        xI0: Initial :math:`x_I`.
        theta: Source-frame polar viewing angle in rad.
        phi: Source-frame azimuthal viewing angle in rad.
        traj_args: Accepted but not used (not passed to the trajectory). Default ``None``.
        traj_kwargs: Keyword arguments of the trajectory call. Default ``None`` (none).
        mode_selector_kwargs: Keyword arguments of the selector call, e.g.
            ``mode_selection_threshold`` or ``return_sort_inds``. Default ``None`` (none).

    Returns:
        dict: Keys ``"teuk_modes"``, ``"ylms"``, ``"ls"``, ``"ms"``, ``"ks"``, ``"ns"`` (the
        selector outputs, see :meth:`ModeSelector.__call__`), ``"trajectory"`` (the full
        trajectory tuple ``(t, p, e, xI, Phi_phi, Phi_theta, Phi_r)``),
        ``"online_mode_selection_args"`` (the frequency dict) and ``"inds_sort"`` (the
        power ordering with ``return_sort_inds=True``, else ``None``).

    Example:
        .. code-block:: python

            from few.utils.modeselector import get_selected_modes_from_initial_conditions

            # selector: a ModeSelector; traj: an EMRIInspiral (see ModeSelector)
            out = get_selected_modes_from_initial_conditions(
                selector, traj, 1e5, 1e1, 0.0, 10.0, 0.7, 1.0, 1.0, 0.5,
                mode_selector_kwargs=dict(mode_selection_threshold=1e-2),
            )
            out["ls"], out["ms"], out["ns"]
    """
    if traj_args is None:
        traj_args = []
    if traj_kwargs is None:
        traj_kwargs = {}
    if mode_selector_kwargs is None:
        mode_selector_kwargs = {}

    traj = traj_module(m1, m2, a, p0, e0, xI0, **traj_kwargs)

    freqs = traj_module.inspiral_generator.eval_integrator_derivative_spline(traj[0], order=1)[:,3:6] / 2 / np.pi

    online_mode_selection_args = dict(
        f_phi = freqs[:,0],
        f_theta = freqs[:,1],
        f_r = freqs[:,2],
    )

    if mode_selector_kwargs.get("return_sort_inds", False):
        teuk_modes_out, ylms_out, ls, ms, ks, ns, inds_sort = mode_selector_module(
            traj[0], a, traj[1], traj[2], traj[3], theta, phi, online_mode_selection_args=online_mode_selection_args, **mode_selector_kwargs
        )
    else:
        teuk_modes_out, ylms_out, ls, ms, ks, ns = mode_selector_module(
            traj[0], a, traj[1], traj[2], traj[3], theta, phi, online_mode_selection_args=online_mode_selection_args, **mode_selector_kwargs
        )    
        inds_sort = None

    return_dict = dict(
        teuk_modes = teuk_modes_out,
        ylms = ylms_out,
        ls = ls,
        ms = ms,
        ks = ks,
        ns = ns,
        trajectory=traj,
        online_mode_selection_args=online_mode_selection_args,
        inds_sort = inds_sort,
    )

    return return_dict