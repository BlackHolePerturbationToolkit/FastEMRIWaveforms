"""Base classes of the FEW waveform models and the sparse-output container.

- :class:`SphericalHarmonicWaveformBase`: the trajectory -> amplitudes -> mode selection ->
  mode sum pipeline shared by the spherical-harmonic (flux-based) models of
  :mod:`few.waveform.waveform`; its :meth:`~SphericalHarmonicWaveformBase._generate_waveform`
  is the call body of those models.
- :class:`SparseInfoHolder`: what ``_generate_waveform`` returns with
  ``return_sparse_holder=True`` (mode-selected amplitudes, harmonics, phases and frequencies
  at the trajectory knots, no mode sum). LISAanalysistools' direct EMRI likelihood reads it.
- :class:`AAKWaveformBase`: trajectory + AAK summation for the kludge models
  (:class:`~few.waveform.waveform.Pn5AAKWaveform`).
"""

from typing import Generic, Optional, TypeVar, Union
import dataclasses

import numpy as np
from tqdm import tqdm

from ..utils.baseclasses import BackendLike, FewBackendConsumer, Pn5AAK
from ..utils.citations import REFERENCE
from ..utils.constants import MRSUN_SI, Gpc
from ..utils.globals import get_logger
from ..utils.mappings.schwarzecc import (
    schwarzecc_p_to_y,
)
from ..utils.ylm import GetYlms

InspiralModule = TypeVar("InspiralModule", bound=FewBackendConsumer)
"""Used for type hinting the Inspiral generator classes."""

AmplitudeModule = TypeVar("AmplitudeModule", bound=FewBackendConsumer)
"""Used for type hinting the Amplitude generator classes."""

SumModule = TypeVar("SumModule", bound=FewBackendConsumer)
"""Used for type hinting the Sum classes."""

ModeSelectorModule = TypeVar("ModeSelectorModule", bound=FewBackendConsumer)
"""Used for type hinting the Mode selector classes."""

WaveformModule = TypeVar("WaveformModule", bound=FewBackendConsumer)
"""Used for type hinting Waveform Generator classes"""

@dataclasses.dataclass
class SparseInfoHolder:
    r"""Mode-selected waveform ingredients at the sparse trajectory knots.

    Returned by :meth:`SphericalHarmonicWaveformBase._generate_waveform` with
    ``return_sparse_holder=True`` in place of the summed waveform: the inputs of the mode sum,
    at the ``N`` knots of the adaptive trajectory, after mode selection. The arrays live on the
    waveform backend (``xp.ndarray``: cupy on a GPU backend). ``K`` is the number of kept
    modes (all with :math:`m \ge 0` for threshold and ``"all"`` selection).

    The source-frame waveform these ingredients define (the sum of
    :class:`~few.summation.directmodesum.DirectModeSum`) is

    .. math::

        h_+ - i h_\times = \sum_{K} \left[ Y_{lm} A_{lmkn}\, e^{-i \Phi_{mkn}}
            + (-1)^l Y_{l,-m} A^*_{lmkn}\, e^{+i \Phi_{mkn}} \right],
        \qquad \Phi_{mkn} = m \Phi_\phi + k \Phi_\theta + n \Phi_r,

    where the second term is summed over the :math:`m > 0` modes only (and vanishes where the
    second half of ``ylms`` is zero). Neither the ``-1`` factor nor the SSB polarization
    rotation that :class:`~few.waveform.waveform.GenerateEMRIWaveform` applies to
    source-frame models is included.

    Attributes:
        t_arr: Knot times in seconds from the start of the trajectory, shape ``(N,)``.
        teuk_modes: Complex Teukolsky amplitudes :math:`A_{lmkn}` of the kept modes, shape
            ``(N, K)``, divided by ``dist * Gpc / (mu * MRSUN_SI)`` when the call passed
            ``dist`` (no distance scaling otherwise).
        phases: Trajectory phases ``(Phi_phi, Phi_theta, Phi_r)`` at the knots in rad, shape
            ``(N, 3)``; the backwards-integration offset of the mode sum is not applied.
        freqs: Orbital frequencies ``(f_phi, f_theta, f_r)`` in Hz at the knots, shape
            ``(N, 3)``: the first time derivative of the integrator's dense output divided by
            :math:`2\pi`.
        ylms: Complex spin-weighted spherical harmonics, shape ``(2 K,)``:
            :math:`Y_{lm}(\theta, \phi)` of the kept modes, then :math:`Y_{l,-m}(\theta, \phi)`
            of the same modes (the second half is zero for a mode list with
            ``include_minus_mkn=False``).
        ls: :math:`l` of the kept modes, shape ``(K,)``.
        ms: :math:`m` of the kept modes, shape ``(K,)``.
        ks: :math:`k` of the kept modes, shape ``(K,)`` (0 for equatorial models).
        ns: :math:`n` of the kept modes, shape ``(K,)``.
        integrate_backwards: Whether the trajectory was integrated backwards in time.
    """

    t_arr: np.ndarray
    teuk_modes: np.ndarray
    phases: np.ndarray
    freqs: np.ndarray
    ylms: np.ndarray
    ls: np.ndarray
    ms: np.ndarray
    ks: np.ndarray
    ns: np.ndarray
    integrate_backwards: bool


class SphericalHarmonicWaveformBase(
    FewBackendConsumer,
    Generic[InspiralModule, AmplitudeModule, SumModule, ModeSelectorModule],
):
    """Base class of the waveform models built from amplitudes in a spherical-harmonic basis.

    Builds the trajectory, amplitude, Ylm, mode-selector and summation modules and provides
    their combined call, :meth:`_generate_waveform`. The stock models
    (:class:`~few.waveform.waveform.FastKerrEccentricEquatorialFlux` and the Schwarzschild
    models) subclass it together with a model base class
    (:class:`~few.utils.baseclasses.KerrEccentricEquatorial` or
    :class:`~few.utils.baseclasses.SchwarzschildEccentric`, which provide the
    ``sanity_check_*`` methods and the mode bookkeeping) and implement ``__call__`` by
    filling in their fixed parameters.

    Construction: the trajectory is built as ``inspiral_module(**inspiral_kwargs)`` with no
    backend (it runs on the CPU); the amplitude, summation and Ylm modules are built on this
    object's backend with
    :meth:`~gpubackendtools.parallelbase.ParallelModuleBase.build_with_same_backend` (which
    writes ``force_backend`` into the given kwargs dicts); the mode selector is built the same
    way with ``(amplitude_generator, ylm_gen)`` as positional arguments.

    Args:
        inspiral_module: Trajectory class (e.g. :class:`~few.trajectory.inspiral.EMRIInspiral`).
            Its call returns ``(t, p, e, xI, Phi_phi, Phi_theta, Phi_r)``. See
            :ref:`trajectory-label`.
        amplitude_module: Amplitude class (an :class:`~few.amplitude.base.AmplitudeBase`
            subclass) returning the complex mode amplitudes. See :ref:`amplitude-label`.
        sum_module: Summation class (a :class:`~few.summation.base.SummationBase` subclass).
            See :ref:`summation-label`.
        mode_selector_module: Mode-selector class
            (:class:`~few.utils.modeselector.ModeSelector`). See :ref:`utilities-label`.
        inspiral_kwargs: Keywords for the trajectory, used for its construction AND merged into
            every trajectory call. Kept by reference (not copied) as :attr:`inspiral_kwargs`.
            Default ``None`` means ``{}``.
        amplitude_kwargs: Constructor keywords for the amplitude module. Default ``None``.
        sum_kwargs: Constructor keywords for the summation module. Default ``None``.
        Ylm_kwargs: Constructor keywords for :class:`~few.utils.ylm.GetYlms`. Default ``None``.
        mode_selector_kwargs: Constructor keywords for the mode selector. Default ``None``.
        force_backend: Backend of the non-trajectory modules (``"cpu"``, ``"cuda12x"``, ...;
            see :class:`~few.utils.baseclasses.FewBackendConsumer`). Default ``None`` picks the
            first available backend of ``supported_backends()``.

    Attributes:
        end_time (float): Last trajectory time of the last call, in seconds (set by a call).
        num_modes_kept (int): Number of modes kept by the last mode selection (set by a call).
        ls (xp.ndarray): :math:`l` of the modes kept by the last call (also ``ms``, ``ks``,
            ``ns``).
    """

    inspiral_kwargs: dict
    """Keyword arguments passed to the inspiral generator call function"""

    inspiral_generator: InspiralModule
    """Instance of the trajectory module"""

    amplitude_generator: AmplitudeModule
    """Instance of the amplitude module"""

    create_waveform: SumModule
    """Instance of the summation module"""

    ylm_gen: GetYlms
    """Instance of the Ylm module"""

    mode_selector: ModeSelectorModule
    """Instance of the mode selector module"""

    def __init__(
        self,
        /,  # force use of keyword arguments for readability
        inspiral_module: type[InspiralModule],
        amplitude_module: type[AmplitudeModule],
        sum_module: type[SumModule],
        mode_selector_module: type[ModeSelectorModule],
        inspiral_kwargs: Optional[dict] = None,
        amplitude_kwargs: Optional[dict] = None,
        sum_kwargs: Optional[dict] = None,
        Ylm_kwargs: Optional[dict] = None,
        mode_selector_kwargs: Optional[dict] = None,
        force_backend: BackendLike = None,
    ):
        FewBackendConsumer.__init__(self, force_backend=force_backend)

        self.inspiral_kwargs = {} if inspiral_kwargs is None else inspiral_kwargs
        self.inspiral_generator = inspiral_module(
            **self.inspiral_kwargs
        )  # The inspiral generator does not rely on backend adjustement

        self.amplitude_generator = self.build_with_same_backend(
            amplitude_module, kwargs=amplitude_kwargs
        )
        self.create_waveform = self.build_with_same_backend(
            sum_module, kwargs=sum_kwargs
        )
        self.ylm_gen = self.build_with_same_backend(GetYlms, kwargs=Ylm_kwargs)

        # selecting modes that contribute at threshold to the waveform
        self.mode_selector = self.build_with_same_backend(
            mode_selector_module,
            args=[self.amplitude_generator, self.ylm_gen],
            kwargs=mode_selector_kwargs,
        )

    def _generate_waveform(
        self,
        m1: float,
        m2: float,
        a: float,
        p0: float,
        e0: float,
        xI0: float,
        theta: float,
        phi: float,
        *args: Optional[tuple],
        dist: Optional[float] = None,
        Phi_phi0: float = 0.0,
        Phi_theta0: float = 0.0,
        Phi_r0: float = 0.0,
        dt: float = 10.0,
        T: float = 1.0,
        mode_selection_threshold: float = 1e-5,
        show_progress: bool = False,
        batch_size: int = -1,
        mode_selection: Optional[Union[str, list, np.ndarray]] = None,
        include_minus_mkn: bool = None,
        return_sparse_holder: Optional[bool] = False,
        **kwargs: Optional[dict],
    ) -> np.ndarray:
        r"""Generate a source-frame waveform (the call body of the spherical-harmonic models).

        Steps:

        1. ``xI0 < 0`` is mapped to the internal convention
           ``(a, xI0, theta, phi) -> (-a, -xI0, pi - theta, -phi)``; ``phi`` is wrapped to
           :math:`[0, 2\pi)` and the initial parameters are checked by the model's
           ``sanity_check_init``.
        2. Call-time ``inspiral_kwargs`` are merged into :attr:`inspiral_kwargs` (permanently),
           ``err`` defaults to ``1e-11``, and the trajectory is integrated (CPU) from the
           initial phases for at most ``T`` years; the output arrays are moved to the backend.
        3. For each time batch (one unless batching is allowed and requested): the orbital
           frequencies at the knots (needed unless the selector's constructor
           ``mode_selection`` is ``"all"``) are evaluated on the host from the integrator's
           derivative spline; the mode selector returns the amplitudes, harmonics and mode
           indices; the phase input of the sum is the DOP853 phase spline
           (``integrator_spline_phase_coeff`` on the knots ``integrator_spline_t``), or the
           phase samples with ``DENSE_STEPPING``.
        4. ``return_sparse_holder=True`` returns a :class:`SparseInfoHolder` here. Otherwise
           the summation module is called with the batch's ``t``, amplitudes, harmonics,
           phase knots and coefficients, ``l, m, k, n``, then ``M = m1 + m2``, ``a`` and the
           full ``p, e, xI`` arrays, with ``dt``, ``T``, ``integrate_backwards`` and
           ``**kwargs``; the batches are concatenated.

        Args:
            m1: Primary (massive black hole) mass in solar masses.
            m2: Secondary mass in solar masses.
            a: Dimensionless spin of the primary.
            p0: Initial semi-latus rectum in units of :math:`M = m_1 + m_2`.
            e0: Initial eccentricity.
            xI0: Initial :math:`x_I = \cos I` (``+1`` / ``-1`` for equatorial models).
            theta: Source-frame polar viewing angle in rad, in :math:`[0, \pi]`.
            phi: Source-frame azimuthal viewing angle in rad.
            *args: Extra positional trajectory parameters (appended after ``xI0``).
            dist: Luminosity distance in Gpc. ``None`` (default) means no distance scaling:
                the output is :math:`r h / \mu` (dimensionless, :math:`\mu` the reduced mass).
            Phi_phi0: Initial azimuthal phase in rad. Default ``0.0``.
            Phi_theta0: Initial polar phase in rad. Default ``0.0``.
            Phi_r0: Initial radial phase in rad. Default ``0.0``.
            dt: Output sampling interval in seconds (also the fixed trajectory step with
                ``DENSE_STEPPING``). Default ``10.0``.
            T: Maximum duration in years; the trajectory stops at ``T`` or at the plunge
                (separatrix buffer). Default ``1.0``.
            mode_selection_threshold: Threshold of the power-based mode selection: modes are
                kept in decreasing order of their estimated power until the kept fraction
                reaches ``(1 - mode_selection_threshold)**2`` of the total. Larger values keep
                fewer modes. ``None`` uses the selector's constructor value. Default ``1e-5``.
            show_progress: Show a `tqdm <https://tqdm.github.io/>`_ bar over the time batches.
                Default ``False``.
            batch_size: Number of trajectory points per time batch; ``-1`` (default) means one
                batch. Used only when the model's ``allow_batching`` is ``True``.
            mode_selection: ``None`` (default) uses the selector's constructor setting (itself
                ``None``, i.e. threshold selection, by default); ``"threshold"`` selects by
                ``mode_selection_threshold``; ``"all"`` keeps every mode; a list of
                ``(l, m, k, n)`` tuples evaluates exactly those modes (``m >= 0`` required when
                ``include_minus_mkn`` is ``True``).
            include_minus_mkn: For a mode list: ``True`` adds each mode's :math:`(-m, -k, -n)`
                partner to the sum, ``False`` sums only the listed modes. ``None`` (default)
                uses the selector's constructor value (``True`` by default).
            return_sparse_holder: If ``True``, return a :class:`SparseInfoHolder` after the mode
                selection instead of summing; requires a single time batch. Default ``False``.
            **kwargs: ``inspiral_kwargs`` (dict): trajectory call keywords, merged permanently
                into :attr:`inspiral_kwargs`. All keywords (``inspiral_kwargs`` included) are
                also forwarded to the summation call: e.g. ``f_arr``, ``mask_positive``,
                ``include_minus_m``, ``separate_modes`` for
                :class:`~few.summation.fdinterp.FDInterpolatedModeSum`; the time-domain sums
                ignore them.

        Returns:
            xp.ndarray or SparseInfoHolder: On the backend. Time domain: complex
            :math:`h_+ - i h_\times`, shape ``(n,)`` sampled every ``dt`` (``n`` set by the
            summation module from ``T``, ``dt``, the trajectory length and ``pad_output`` /
            ``odd_len``). Frequency domain: shape ``(2, n_f)``, rows :math:`\tilde h_+` and
            :math:`\tilde h_\times`. Divided by ``dist * Gpc / (mu * MRSUN_SI)`` when
            ``dist`` is given. With ``return_sparse_holder=True``: the
            :class:`SparseInfoHolder` (its amplitudes scaled the same way).

        Raises:
            ValueError: If ``dist <= 0``; if ``sanity_check_init`` or ``sanity_check_traj``
                rejects the parameters; if ``mode_selection`` is invalid; if a trajectory
                point maps outside the amplitude grid (``AmpInterpKerrEccEq``).
            AssertionError: From the trajectory's start-point check (``isvalid_pex``) or the
                amplitude module's input checks.
            few.utils.exceptions.TrajectoryOffGridException: If a ``KerrEccEqFlux``
                trajectory leaves the flux grid.
        """

        # switch to internal convention of xI > 0
        if xI0 < 0.0:
            a = -a
            xI0 = -xI0
            theta = np.pi - theta
            phi = -phi

        # define total mass and reduced mass
        M = m1 + m2
        mu = m1 * m2 / (m1 + m2)

        if dist is not None:
            if dist <= 0.0:
                raise ValueError("Luminosity distance must be greater than zero.")

            dist_dimensionless = (dist * Gpc) / (mu * MRSUN_SI)

        else:
            dist_dimensionless = 1.0

        # makes sure viewing angles are allowable
        theta, phi = self.sanity_check_viewing_angles(theta, phi)

        a, xI0 = self.sanity_check_init(m1, m2, a, p0, e0, xI0)

        # Ensure kwargs['inspiral_kwargs'] exists and is a dictionary
        # Essential if inspiral_kwargs passed into waveform generator
        kwargs_inspiral = kwargs.get("inspiral_kwargs", {})
        # Merge kwargs_inspiral into self.inspiral_kwargs
        self.inspiral_kwargs.update(kwargs_inspiral)
        # get trajectory
        self.inspiral_kwargs.setdefault(
            "err", 1e-11
        )  # Will only set default if "err" is not supplied

        (t, p, e, xI, Phi_phi, Phi_theta, Phi_r) = self.inspiral_generator(
            m1,
            m2,
            a,
            p0,
            e0,
            xI0,
            *args,
            Phi_phi0=Phi_phi0,
            Phi_theta0=Phi_theta0,
            Phi_r0=Phi_r0,
            T=T,
            dt=dt,
            **self.inspiral_kwargs,
        )
        # makes sure p and e are generally within the model
        self.sanity_check_traj(a, p, e, xI)

        self.end_time = t[-1]

        # convert for gpu
        t = self.xp.asarray(t)
        p = self.xp.asarray(p)
        e = self.xp.asarray(e)
        xI = self.xp.asarray(xI)
        Phi_phi = self.xp.asarray(Phi_phi)
        Phi_theta = self.xp.asarray(Phi_theta)
        Phi_r = self.xp.asarray(Phi_r)

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

        # select tqdm if user wants to see progress
        iterator = enumerate(inds_split_all)
        iterator = (
            tqdm(iterator, desc="time batch", total=len(inds_split_all))
            if show_progress
            else iterator
        )

        for i, inds_in in iterator:
            # get subsections of the arrays for each batch
            t_temp = t[inds_in]
            p_temp = p[inds_in]
            e_temp = e[inds_in]
            xI_temp = xI[inds_in]
            Phi_phi_temp = Phi_phi[inds_in]
            Phi_theta_temp = Phi_theta[inds_in]
            Phi_r_temp = Phi_r[inds_in]

            # get frequencies to pass to mode selection
            # TODO: write a method that just returns the derivatives at each spline knot (vectorises easier).
            if self.mode_selector.mode_selection != "all":
                # eval_integrator_derivative_spline runs on the integrator's
                # cached knots/coeffs, which are always host (numpy) arrays --
                # the DOPR integrator is CPU-only. On the hybrid CPU-trajectory
                # / GPU-sum path t_temp is a device array, so evaluate on host
                # and move the frequencies back to the waveform backend for the
                # (GPU) mode selector.
                t_temp_host = t_temp.get() if hasattr(t_temp, "get") else t_temp
                freqs = self.xp.asarray(
                    self.inspiral_generator.inspiral_generator.eval_integrator_derivative_spline(t_temp_host, order=1)[:, 3:6]
                ) / 2 / np.pi

                online_mode_selection_args = dict(
                    f_phi = freqs[:,0],
                    f_theta = freqs[:,1],
                    f_r = freqs[:,2],
                )
            else:
                # handles slow waveform
                online_mode_selection_args = None

            # get amplitudes that have been selected / sorted to user requirements
            (
                teuk_modes_in,
                ylms_in,
                self.ls,
                self.ms,
                self.ks,
                self.ns,
            ) = self.mode_selector(
                t_temp,
                a,
                p_temp,
                e_temp,
                xI_temp,
                theta,
                phi,
                online_mode_selection_args=online_mode_selection_args,
                mode_selection=mode_selection,
                include_minus_mkn=include_minus_mkn,
                mode_selection_threshold=mode_selection_threshold,
            )
            # store number of modes for external information
            self.num_modes_kept = teuk_modes_in.shape[1]

            # prepare phases for summation modules
            if not self.inspiral_generator.dense_stepping:
                # prepare phase spline coefficients
                phase_information_in = self.xp.asarray(
                    self.inspiral_generator.integrator_spline_phase_coeff
                )

                # flip azimuthal phase for retrograde inspirals
                if a > 0:
                    phase_information_in[:, 0] *= self.xp.sign(xI0)

                if self.inspiral_generator.integrate_backwards:
                    phase_information_in[:, :, 0] += self.xp.array(
                        [Phi_phi[-1] + Phi_phi[0], Phi_theta[-1] + Phi_theta[0], Phi_r[-1] + Phi_r[0]]
                    )
                phase_t_in = self.inspiral_generator.integrator_spline_t
            else:
                phase_information_in = self.xp.asarray(
                    [Phi_phi_temp, Phi_theta_temp, Phi_r_temp]
                )
                if self.inspiral_generator.integrate_backwards:
                    phase_information_in[0] += self.xp.array([Phi_phi[-1] + Phi_phi[0]])
                    phase_information_in[1] += self.xp.array([Phi_theta[-1] + Phi_theta[0]])
                    phase_information_in[2] += self.xp.array([Phi_r[-1] + Phi_r[0]])

                # flip azimuthal phase for retrograde inspirals
                if a > 0:
                    phase_information_in[0] *= self.xp.sign(xI0)

                phase_t_in = None

            if return_sparse_holder:
                try:
                    freqs
                except NameError:
                    # Host eval on the CPU integrator, result on the waveform
                    # backend (see the mode-selection branch above).
                    t_temp_host = t_temp.get() if hasattr(t_temp, "get") else t_temp
                    freqs = self.xp.asarray(
                        self.inspiral_generator.inspiral_generator.eval_integrator_derivative_spline(t_temp_host, order=1)[:, 3:6]
                    ) / 2 / np.pi

                assert len(inds_split_all) == 1
                phases = self.xp.array([Phi_phi, Phi_theta, Phi_r]).T

                return SparseInfoHolder(
                    t_temp,
                    teuk_modes_in / dist_dimensionless,
                    phases,
                    freqs,
                    ylms_in,
                    self.ls,
                    self.ms,
                    self.ks,
                    self.ns,
                    self.inspiral_generator.integrate_backwards
                )

            # create waveform
            waveform_temp = self.create_waveform(
                t_temp,
                teuk_modes_in,
                ylms_in,
                phase_t_in,
                phase_information_in,
                self.ls,
                self.ms,
                self.ks,
                self.ns,
                M,  # waveform generation will also be done with respect to total mass
                a,
                p,
                e,
                xI,
                dt=dt,
                T=T,
                integrate_backwards=self.inspiral_generator.integrate_backwards,
                **kwargs,
            )

            # if batching, need to add the waveform
            if i > 0:
                waveform = self.xp.concatenate([waveform, waveform_temp])  # noqa: F821

            # return entire waveform
            else:
                waveform = waveform_temp

        return waveform / dist_dimensionless


class AAKWaveformBase(Pn5AAK, FewBackendConsumer, Generic[InspiralModule, SumModule]):
    r"""Waveform generation class for AAK with arbitrary trajectory.

    This class generates waveforms based on the Augmented Analytic Kludge
    given in the
    `EMRI Kludge Suite <https://github.com/alvincjk/EMRI_Kludge_Suite/>`_.
    The trajectory is chosen by user or by default in child classes.

    The trajectory is calculated until the orbit reaches the separatrix (determined from
    `arXiv:1912.07609 <https://arxiv.org/abs/1912.07609/>`_) plus the trajectory model's
    ``separatrix_buffer_dist`` (0.1 in ``p`` for :class:`~few.trajectory.ode.pn5.PN5`). The
    fundamental frequencies along the trajectory at each point are then
    calculated from the orbital parameters and the spin value given by
    (`Schmidt 2002 <https://arxiv.org/abs/gr-qc/0202090>`_).

    These frequencies along the trajectory are then used to map to the
    frequency basis of the `Analytic Kludge <https://arxiv.org/abs/gr-qc/0310125>`_. This mapping
    takes the form of time evolving large mass and spin parameters, as
    well as the use of phases and frequencies in
    :math:`(\alpha, \Phi, \gamma)`:

    .. math:: \Phi = \Phi_\phi,

    .. math:: \gamma = \Phi_\phi + \Phi_\Theta,

    .. math:: \alpha = \Phi_\phi + \Phi_\Theta + \Phi_r.

    The frequencies in that basis are found by taking the time derivatives
    of each equation above.

    This class has GPU capabilities and works from the sparse trajectory
    methodology with cubic spline interpolation of the smoothly varying
    waveform quantities. The trajectory runs on the CPU; the summation module on this
    object's backend. The model is built in the detector (SSB) frame.

    **Please note:** the AAK waveform takes the parameter
    :math:`Y\equiv\cos{\iota}=L/\sqrt{L^2 + Q}` rather than :math:`x_I` as is accepted
    for relativistic waveforms (see the note in
    :meth:`~few.waveform.waveform.GenerateEMRIWaveform.__call__` on how the generic interface
    passes ``x0``).

    Args:
        inspiral_module: Trajectory class (e.g. :class:`~few.trajectory.inspiral.EMRIInspiral`).
            See :ref:`trajectory-label`.
        sum_module: Summation class (e.g. :class:`~few.summation.aakwave.AAKSummation`). See
            :ref:`summation-label`.
        inspiral_kwargs: Keywords for the trajectory, used for its construction and at every
            call; kept by reference as :attr:`inspiral_kwargs`. Default ``None`` means ``{}``.
        sum_kwargs: Constructor keywords for the summation module. Default ``None``.
        force_backend: Backend of the summation module (see
            :class:`~few.utils.baseclasses.FewBackendConsumer`). Default ``None``.

    Attributes:
        nmodes (int): Number of harmonics used by the last call (equal to
            :attr:`num_modes_kept`).
        end_time (float): Last trajectory time of the last call, in seconds.
    """

    inspiral_kwargs: dict
    """Keyword arguments passed to the inspiral generator call function"""

    inspiral_generator: InspiralModule
    """Instance of the trajectory module"""

    create_waveform: SumModule
    """Instance of the summation module"""

    num_modes_kept: int
    """Number of harmonics of the last call (``None`` before it); see ``nmodes`` of the call."""

    def __init__(
        self,
        inspiral_module: type[InspiralModule],
        sum_module: type[SumModule],
        inspiral_kwargs: Optional[dict] = None,
        sum_kwargs: Optional[dict] = None,
        force_backend: BackendLike = None,
    ):
        Pn5AAK.__init__(self)
        FewBackendConsumer.__init__(self, force_backend=force_backend)

        self.inspiral_kwargs = {} if inspiral_kwargs is None else inspiral_kwargs
        self.inspiral_generator = inspiral_module(
            **self.inspiral_kwargs
        )  # The inspiral generator does not rely on backend adjustement
        self.create_waveform = self.build_with_same_backend(
            sum_module, kwargs=sum_kwargs
        )

        self.num_modes_kept = None

    @classmethod
    def module_references(cls) -> list[REFERENCE]:
        """Return the citations of the AAK models.

        Returns:
            list[REFERENCE]: ``FD``, ``AAK1``, ``AAK2``, ``AK``, ``KERR_SEPARATRIX``, then the
            parent classes' references.
        """
        return [
            REFERENCE.FD,
            REFERENCE.AAK1,
            REFERENCE.AAK2,
            REFERENCE.AK,
            REFERENCE.KERR_SEPARATRIX,
        ] + super().module_references()

    @classmethod
    def supported_backends(cls):
        """Backend preference order: ``["cuda13x", "cuda12x", "cuda11x", "cpu"]``."""
        return cls.GPU_RECOMMENDED()

    @property
    def allow_batching(self):
        """bool: ``False`` (the AAK call has no batching)."""
        return False

    def __call__(
        self,
        m1: float,
        m2: float,
        a: float,
        p0: float,
        e0: float,
        Y0: float,
        dist: float,
        qS: float,
        phiS: float,
        qK: float,
        phiK: float,
        *args: Optional[tuple],
        Phi_phi0: float = 0.0,
        Phi_theta0: float = 0.0,
        Phi_r0: float = 0.0,
        mich: bool = False,
        dt: float = 10.0,
        T: float = 1.0,
        nmodes: Optional[int] = None,
    ) -> np.ndarray:
        r"""Generate an AAK waveform in the detector (SSB) frame.

        Checks the angles (``Pn5AAK.sanity_check_angles``: ``qS``, ``qK`` in :math:`[0, \pi]`,
        ``phiS``, ``phiK`` wrapped) and the initial parameters, integrates the trajectory
        (CPU), sets the number of harmonics, divides the phase rows of the trajectory spline by
        the symmetric mass ratio :math:`\mu / M` and calls the AAK summation module with the
        total mass :math:`M` and the reduced mass :math:`\mu` in the places of its ``m1`` and
        ``m2`` arguments.

        Args:
            m1: Primary (massive black hole) mass in solar masses.
            m2: Secondary mass in solar masses (``m2 <= m1``).
            a: Dimensionless spin of the primary.
            p0: Initial semi-latus rectum in units of :math:`M = m_1 + m_2`; must be above the
                separatrix (plus the trajectory buffer) at ``e0`` and ``Y0``.
            e0: Initial eccentricity.
            Y0: Initial :math:`Y = \cos\iota \equiv L_z / \sqrt{L_z^2 + Q}`, in ``[-1, 1]``.
            dist: Luminosity distance in Gpc.
            qS: Sky polar angle (colatitude) in rad, SSB ecliptic frame by FEW convention.
            phiS: Sky azimuthal angle in rad, same frame.
            qK: Polar angle of the primary's spin direction in rad, same frame.
            phiK: Azimuthal angle of the primary's spin direction in rad, same frame.
            *args: Extra positional trajectory parameters.
            Phi_phi0: Initial phase :math:`\Phi_\phi` in rad. Default ``0.0``.
            Phi_theta0: Initial phase :math:`\Phi_\theta` in rad. Default ``0.0``.
            Phi_r0: Initial phase :math:`\Phi_r` in rad. Default ``0.0``.
            mich: If ``True``, return the long-wavelength response channels
                :math:`h_I, h_{II}` (not TDI); if ``False`` (default), :math:`h_+, h_\times`.
                Through :class:`~few.waveform.waveform.GenerateEMRIWaveform` with
                ``frame="detector"`` either output is rotated by :math:`2\psi` afterwards.
            dt: Sampling interval in seconds. Default ``10.0``.
            T: Maximum duration in years. Default ``1.0``.
            nmodes: Number of harmonics. ``None`` (default) means ``int(30 * e0)`` (``e`` at the
                last trajectory point for a backwards integration), at least 4.

        Returns:
            xp.ndarray: The summation module's complex output on the backend, shape ``(n,)``
            sampled every ``dt``; for :class:`~few.summation.aakwave.AAKSummation` the kernel
            writes ``h_I - i h_II`` (:math:`h_+ - i h_\times` with ``mich=False``).

        Raises:
            ValueError: If an angle, an initial parameter (negative values, ``m1 < m2``,
                ``|Y0| > 1``) or a trajectory point is out of range.
        """

        # makes sure angular extrinsic parameters are allowable
        qS, phiS, qK, phiK = self.sanity_check_angles(qS, phiS, qK, phiK)

        a, Y0 = self.sanity_check_init(m1, m2, a, p0, e0, Y0)

        # define total mass and reduced mass
        M = m1 + m2
        mu = m1 * m2 / (m1 + m2)

        # get trajectory
        t, p, e, Y, Phi_phi, Phi_theta, Phi_r = self.inspiral_generator(
            m1,
            m2,
            a,
            p0,
            e0,
            Y0,
            *args,
            Phi_phi0=Phi_phi0,
            Phi_theta0=Phi_theta0,
            Phi_r0=Phi_r0,
            T=T,
            dt=dt,
            **self.inspiral_kwargs,
        )

        if nmodes is None:
            if (p[0] - p[1]) < 0:  # Integrating backwards
                # Need to keep the number of modes equivalent
                initial_e = e[-1]
                self.num_modes_kept = self.nmodes = int(30 * initial_e)
            else:
                # number of modes to use (from original AAK model)
                self.num_modes_kept = self.nmodes = int(30 * e0)

            if self.num_modes_kept < 4:
                self.num_modes_kept = self.nmodes = 4
        else:
            self.num_modes_kept = self.nmodes = nmodes

        # makes sure p, Y, and e are generally within the model
        self.sanity_check_traj(p, e, Y)

        self.end_time = t[-1]

        # prepare phase spline coefficients
        traj_spline_coeff = self.xp.asarray(
            self.inspiral_generator.integrator_spline_coeff
        )

        # scale coefficients here by the (symmetric) mass ratio
        traj_spline_coeff_in = traj_spline_coeff.copy()
        traj_spline_coeff_in[:, 3:, :] /= mu / M

        if self.inspiral_generator.integrate_backwards:
            traj_spline_coeff_in[:, 3:, 0] += self.xp.array(
                [
                    Phi_phi[-1] + Phi_phi[0],
                    Phi_theta[-1] + Phi_theta[0],
                    Phi_r[-1] + Phi_r[0],
                ]
            )

        # TODO: Check that the mass conventions here are consistent with adiabatic model
        waveform = self.create_waveform(
            t,
            M,  # for the AAK waveform this parameter sets the dimensionless frequency with which to scale the amplitude
            a,
            dist,
            mu,  # this is also used to scale the amplitude, so we want to scale with reduced mass
            qS,
            phiS,
            qK,
            phiK,
            self.nmodes,
            self.inspiral_generator.integrator_spline_t,
            traj_spline_coeff_in,
            mich=mich,
            dt=dt,
            T=T,
            integrate_backwards=self.inspiral_generator.integrate_backwards,
        )

        return waveform
