r"""FEW's stock waveform models and the generic interface :class:`GenerateEMRIWaveform`.

Stock models (also selectable by name through :class:`GenerateEMRIWaveform`):

- :class:`FastKerrEccentricEquatorialFlux`: Kerr eccentric equatorial, flux-based adiabatic
  inspiral (:class:`~few.trajectory.ode.flux.KerrEccEqFlux`), spline-interpolated Teukolsky
  amplitudes (:class:`~few.amplitude.ampinterp2d.AmpInterpKerrEccEq`), mode selection and the
  interpolated mode sum. GPU recommended.
- :class:`FastSchwarzschildEccentricFlux`: Schwarzschild eccentric, ROMAN neural-network
  amplitudes (:class:`~few.amplitude.romannet.RomanAmplitude`).
- :class:`FastSchwarzschildEccentricFluxBicubic`: Schwarzschild eccentric, bicubic-spline
  amplitudes (:class:`~few.amplitude.ampinterp2d.AmpInterpSchwarzEcc`).
- :class:`SlowSchwarzschildEccentricFlux`: CPU reference model (dense trajectory, all modes,
  :class:`~few.summation.directmodesum.DirectModeSum`).
- :class:`Pn5AAKWaveform`: 5PN trajectory with the Augmented Analytic Kludge, built in the
  detector frame.

The spherical-harmonic models (all but the AAK) derive from
:class:`~few.waveform.base.SphericalHarmonicWaveformBase` and produce the source-frame waveform
:math:`h_+ - i h_\times`; :class:`GenerateEMRIWaveform` adds the luminosity-distance scaling and
the rotation to the solar-system-barycentre (SSB) polarization basis.

Conventions: masses in solar masses, ``p0`` in units of :math:`M = m_1 + m_2`, distance in Gpc,
angles and phases in radians, ``dt`` in seconds, ``T`` in years.
"""

# Main waveform class location

import os
from typing import Generic, Optional, Union

import numpy as np

from ..amplitude.ampinterp2d import AmpInterpKerrEccEq, AmpInterpSchwarzEcc
from ..amplitude.romannet import RomanAmplitude
from ..summation.aakwave import AAKSummation
from ..summation.directmodesum import DirectModeSum
from ..summation.fdinterp import FDInterpolatedModeSum
from ..summation.interpolatedmodesum import InterpolatedModeSum
from ..trajectory.inspiral import EMRIInspiral
from ..trajectory.ode import PN5, KerrEccEqFlux, SchwarzEccFlux
from ..utils.baseclasses import (
    BackendLike,
    KerrEccentricEquatorial,
    SchwarzschildEccentric,
)
from ..utils.constants import MRSUN_SI, Gpc
from ..utils.mappings.pn import xI_to_Y
from ..utils.modeselector import ModeSelector
from .base import AAKWaveformBase, SphericalHarmonicWaveformBase, WaveformModule
from ..utils.utility import get_viewing_angles, get_polarization_angle


# get path to this file
dir_path = os.path.dirname(os.path.realpath(__file__))


class GenerateEMRIWaveform(Generic[WaveformModule]):
    r"""Generic waveform interface: one call signature for every FEW model.

    Builds one waveform model (:attr:`waveform_generator`) and exposes it through the full
    14-parameter signature of :meth:`__call__`:
    ``(m1, m2, a, p0, e0, x0, dist, qS, phiS, qK, phiK, Phi_phi0, Phi_theta0, Phi_r0)``.
    Parameters a model does not use are dropped before the model call:

    - Schwarzschild models: the spin ``a`` and ``x0``;
    - ``"eccentric"`` models (the Schwarzschild ones): ``Phi_theta0``;
    - source-frame models (all spherical-harmonic models): ``dist, qS, phiS, qK, phiK``. They
      receive instead the source-frame viewing angles of
      :func:`~few.utils.utility.get_viewing_angles`.

    The output of a source-frame model is divided by the dimensionless distance
    ``dist * Gpc / (mu * MRSUN_SI)`` (``mu = m1 m2 / (m1 + m2)``) and multiplied by ``-1``; a
    detector-frame model (:class:`Pn5AAKWaveform`) receives ``dist`` and the four angles
    itself. With ``frame="detector"`` the time-domain output (``hp = Re h``, ``hc = -Im h``) or
    the frequency-domain rows of either kind of model are then rotated into the SSB
    polarization basis by twice the polarization angle of
    :func:`~few.utils.utility.get_polarization_angle` (:meth:`_to_SSB_frame`).

    Args:
        waveform_class: Name of a stock model (see :attr:`stock_waveform_options`:
            ``"FastKerrEccentricEquatorialFlux"``, ``"FastSchwarzschildEccentricFlux"``,
            ``"FastSchwarzschildEccentricFluxBicubic"``, ``"SlowSchwarzschildEccentricFlux"``,
            ``"Pn5AAKWaveform"``), or a waveform class (not an instance) with the same
            interface.
        *args: Positional arguments for the model constructor.
        frame: Output frame. ``"detector"`` (default): :math:`h_+, h_\times` in the SSB frame.
            ``"source"``: a source-frame model is returned without distance scaling or rotation
            (still multiplied by ``-1``; ``dist`` is ignored and ``qS, phiS, qK, phiK`` enter
            only through the viewing angles); for a detector-frame model only the distance
            factor is undone (multiplied by ``dist * Gpc / (mu * MRSUN_SI)``), no rotation is
            applied.
        return_list: If ``True``, return ``[hp, hc]``; if ``False`` (default), return
            ``hp - 1j * hc``.
        flip_output: If ``True``, reverse the time order of ``hp`` and ``hc`` (time-domain
            output only). Default ``False``.
        **kwargs: Keyword arguments for the model constructor (``inspiral_kwargs``,
            ``amplitude_kwargs``, ``sum_kwargs``, ``Ylm_kwargs``, ``mode_selector_kwargs``,
            ``force_backend``, ...; see each model).

    Raises:
        ValueError: If ``waveform_class`` is a string that is not a stock model. A ``KeyError``
            raised inside the model constructor is reported the same way.

    Attributes:
        args_keep (np.ndarray): Indices, into the first 11 parameters of :meth:`__call__`, of
            the positional parameters passed to the model.

    Example:
        .. code-block:: python

            from few.waveform import GenerateEMRIWaveform

            gen = GenerateEMRIWaveform(
                "FastKerrEccentricEquatorialFlux",
                sum_kwargs=dict(pad_output=True),
                return_list=True,
            )
            # m1, m2 [Msun], a, p0 [M], e0, x0, dist [Gpc], qS, phiS, qK, phiK [rad],
            # Phi_phi0, Phi_theta0, Phi_r0 [rad]
            hp, hc = gen(
                1e6, 10.0, 0.9, 12.0, 0.1, 1.0, 1.0,
                0.8, 2.0, 1.2, 0.5, 0.0, 0.0, 0.0,
                T=1.0, dt=10.0, mode_selection_threshold=1e-5,
            )
    """

    waveform_generator: WaveformModule
    """Instance of the waveform module"""

    frame: str
    """Frame in which waveform is generated."""

    return_list: bool
    """Whether to return :math:`h_p` and :math:`h_x` as list, otherwise returned as :math:`h_p - i h_x`"""

    flip_output: bool
    """Whether :math:`h_p` and :math:`h_x` output time series (if time-domain) should be reversed"""

    args_remove: list[int]
    """List of arguments to remove based on the specific waveform"""

    phases_needed: dict[str, int]
    """Phases needed based on specific waveform"""

    def __init__(
        self,
        waveform_class: Union[str, type[WaveformModule]],
        *args: Optional[Union[list, tuple]],
        frame: str = "detector",
        return_list: bool = False,
        flip_output: bool = False,
        **kwargs: Optional[dict],
    ):
        # instantiate the class
        if isinstance(waveform_class, str):
            try:
                waveform = self._stock_waveform_definitions[waveform_class]
                self.waveform_generator = waveform(*args, **kwargs)
            except KeyError:
                raise ValueError(
                    "{} waveform class is not available.".format(waveform_class)
                )
        else:
            self.waveform_generator = waveform_class(*args, **kwargs)

        self.frame = frame
        self.return_list = return_list
        self.flip_output = flip_output

        # setup arguments to remove based on the specific waveform
        # also get proper phases
        self.args_remove = []
        if self.waveform_generator.descriptor == "eccentric":
            if self.waveform_generator.background == "Schwarzschild":
                self.args_remove.append(5)  # prograde vs retrograde 1/-1

            self.phases_needed = {"Phi_phi0": 11, "Phi_r0": 13}

        else:
            self.phases_needed = {"Phi_phi0": 11, "Phi_theta0": 12, "Phi_r0": 13}

        if self.waveform_generator.background == "Schwarzschild":
            # remove spin
            self.args_remove.append(2)

        # remove sky and orientation parameters
        if self.waveform_generator.frame == "source":
            for i in range(6, 11):
                self.args_remove.append(i)

        # these are the arguments that go in to the generator
        self.args_keep = np.delete(np.arange(11), self.args_remove)

    @property
    def _stock_waveform_definitions(self):
        """dict[str, type]: Stock model name -> waveform class."""
        return {
            "FastSchwarzschildEccentricFlux": FastSchwarzschildEccentricFlux,
            "FastSchwarzschildEccentricFluxBicubic": FastSchwarzschildEccentricFluxBicubic,
            "SlowSchwarzschildEccentricFlux": SlowSchwarzschildEccentricFlux,
            "FastKerrEccentricEquatorialFlux": FastKerrEccentricEquatorialFlux,
            "Pn5AAKWaveform": Pn5AAKWaveform,
        }

    @property
    def stock_waveform_options(self) -> list[str]:
        """Names of the stock models accepted as ``waveform_class``."""
        return list(self._stock_waveform_definitions.keys())

    def _to_SSB_frame(self, hp, hc, qS, phiS, qK, phiK):
        r"""Rotate source-frame polarizations into the SSB polarization basis.

        With :math:`\psi` from :func:`~few.utils.utility.get_polarization_angle`:
        ``hp' = cos(2 psi) hp - sin(2 psi) hc`` and ``hc' = sin(2 psi) hp + cos(2 psi) hc``.

        Args:
            hp (xp.ndarray): Plus polarization (source frame).
            hc (xp.ndarray): Cross polarization (source frame).
            qS (float): Sky polar angle in rad.
            phiS (float): Sky azimuthal angle in rad.
            qK (float): Spin polar angle in rad.
            phiK (float): Spin azimuthal angle in rad.

        Returns:
            tuple[xp.ndarray, xp.ndarray]: ``(hp', hc')``, same shape as the inputs.
        """

        psi_ldc = get_polarization_angle(qS, phiS, qK, phiK)
        c2psi_ldc = np.cos(2.0 * psi_ldc)
        s2psi_ldc = np.sin(2.0 * psi_ldc)

        # rotate
        FplusI = c2psi_ldc
        FcrosI = -s2psi_ldc
        FplusII = s2psi_ldc
        FcrosII = c2psi_ldc

        hp_new = FplusI * hp + FcrosI * hc
        hc_new = FplusII * hp + FcrosII * hc

        return hp_new, hc_new

    def __call__(
        self,
        m1: float,
        m2: float,
        a: float,
        p0: float,
        e0: float,
        x0: float,
        dist: float,
        qS: float,
        phiS: float,
        qK: float,
        phiK: float,
        Phi_phi0: float,
        Phi_theta0: float,
        Phi_r0: float,
        *add_args: Optional[tuple],
        **kwargs: Optional[dict],
    ) -> Union[np.ndarray, list]:
        r"""Generate the waveform of one source.

        If ``x0 < 0`` the call is first mapped to the internal prograde convention:
        ``(a, x0) -> (-a, -x0)``, ``qK -> pi - qK``, ``phiK -> phiK + pi`` and
        ``Phi_phi0 -> Phi_phi0 + pi``. The parameters the model uses (see the class docstring)
        are then passed positionally, the initial phases it uses as keywords, together with
        ``**kwargs``.

        Args:
            m1: Primary (massive black hole) mass in solar masses.
            m2: Secondary (compact object) mass in solar masses.
            a: Dimensionless spin of the primary. Ignored by Schwarzschild models.
            p0: Initial semi-latus rectum in units of :math:`M = m_1 + m_2`; must lie inside the
                model's domain (above the separatrix plus its buffer).
            e0: Initial eccentricity.
            x0: Initial cosine of the inclination, :math:`x_I = \cos I` (``+1`` prograde, ``-1``
                retrograde for equatorial models). Ignored by Schwarzschild models. This differs
                from :math:`Y = \cos\iota \equiv L_z / \sqrt{L_z^2 + Q}` used by the kludge
                models (see the note below).
            dist: Luminosity distance in Gpc.
            qS: Sky polar angle (colatitude) in rad, SSB ecliptic frame by FEW convention.
            phiS: Sky azimuthal angle in rad, same frame.
            qK: Polar angle of the primary's spin direction in rad, same frame.
            phiK: Azimuthal angle of the primary's spin direction in rad, same frame.
            Phi_phi0: Initial azimuthal phase :math:`\Phi_{\phi,0}` in rad.
            Phi_theta0: Initial polar phase :math:`\Phi_{\theta,0}` in rad. Not passed to
                Schwarzschild models.
            Phi_r0: Initial radial phase :math:`\Phi_{r,0}` in rad.
            *add_args: Extra positional model parameters, appended after the model's own
                positional arguments (after the viewing angles for source-frame models).
            **kwargs: Call keywords of the model. For the spherical-harmonic models see
                :meth:`~few.waveform.base.SphericalHarmonicWaveformBase._generate_waveform`
                (``T``, ``dt``, ``mode_selection_threshold``, ``mode_selection``,
                ``include_minus_mkn``, ``inspiral_kwargs``, summation keywords such as ``f_arr``
                and ``mask_positive`` for frequency-domain output); for :class:`Pn5AAKWaveform`
                see :meth:`~few.waveform.base.AAKWaveformBase.__call__` (``T``, ``dt``, ``mich``,
                ``nmodes``). Do not repeat ``dist`` or the initial phases here (a ``dist``
                keyword would scale a source-frame model a second time).

        Returns:
            xp.ndarray or list[xp.ndarray]: ``hp - 1j * hc`` (``return_list=False``) or
            ``[hp, hc]`` (``return_list=True``), on the model's backend (cupy on a GPU backend).
            Time domain: real ``hp``, ``hc`` of shape ``(n,)`` sampled every ``dt`` seconds, ``n``
            set by the summation module from ``T``, ``dt``, the trajectory length and
            ``pad_output`` / ``odd_len``. Frequency domain
            (``sum_kwargs["output_type"] == "fd"``): complex ``hp``, ``hc`` on the summation
            module's frequency grid (``waveform_generator.create_waveform.frequency``; only
            its ``f >= 0`` part with ``mask_positive=True``).

        Note:
            For a model with ``needs_Y`` (:class:`Pn5AAKWaveform`) the code computes
            ``x0 = xI_to_Y(a, p0, e0, x0)``, but after the argument tuple has been built, so
            the model receives ``x0`` unchanged and interprets it as :math:`Y_0`.
        """

        if x0 < 0.0:
            a = -a
            x0 = -x0
            qK = np.pi - qK
            phiK = phiK + np.pi
            Phi_phi0 = Phi_phi0 + np.pi

        args_all = (
            m1,
            m2,
            a,
            p0,
            e0,
            x0,
            dist,
            qS,
            phiS,
            qK,
            phiK,
            Phi_phi0,
            Phi_theta0,
            Phi_r0,
        )

        # if Y is needed rather than x (inclination definition)
        if (
            hasattr(self.waveform_generator, "needs_Y")
            and self.waveform_generator.background.lower() == "kerr"
            and self.waveform_generator.needs_Y
        ):
            x0 = xI_to_Y(a, p0, e0, x0)

        # remove the arguments that are not used in this waveform
        args = tuple([args_all[i] for i in self.args_keep])

        # pick out the phases to be used
        initial_phases = {key: args_all[i] for key, i in self.phases_needed.items()}

        # define reduced mass for scaling waveform amplitude
        mu = m1 * m2 / (m1 + m2)

        # generate waveform in source frame
        if self.waveform_generator.frame == "source":
            # distance factor
            dist_dimensionless = (dist * Gpc) / (mu * MRSUN_SI)

            # get viewing angles in the source frame
            (
                theta_source,
                phi_source,
            ) = get_viewing_angles(qS, phiS, qK, phiK)

            args += (theta_source, phi_source)

        else:
            dist_dimensionless = 1.0

        # if output is to be in the source frame, need to properly scale with distance
        if self.frame == "source":
            if self.waveform_generator.frame == "detector":
                dist_dimensionless = 1.0 / ((dist * Gpc) / (mu * MRSUN_SI))
            else:
                dist_dimensionless = 1.0

        # add additional arguments to waveform interface
        args += add_args

        # get waveform
        h = (
            self.waveform_generator(*args, **{**initial_phases, **kwargs})
            / dist_dimensionless
        )

        # by definition of the source frame, need to rotate by pi
        if self.waveform_generator.frame == "source":
            h *= -1

        # transform to SSB frame if desired
        if self.waveform_generator.create_waveform.output_type == "td":
            if self.frame == "detector":
                hp, hc = self._to_SSB_frame(h.real, -h.imag, qS, phiS, qK, phiK)
            elif self.frame == "source":
                hp, hc = h.real, -h.imag

            if self.flip_output:
                hp = hp[::-1]
                hc = hc[::-1]

        # if FD, h is of length 2 rather than h+ - ihx
        if self.waveform_generator.create_waveform.output_type == "fd":
            if self.frame == "detector":
                hp, hc = self._to_SSB_frame(h[0], h[1], qS, phiS, qK, phiK)
            elif self.frame == "source":
                hp, hc = h[0], h[1]

        if self.return_list is False:
            return hp - 1j * hc
        else:
            return [hp, hc]


class FastKerrEccentricEquatorialFlux(
    SphericalHarmonicWaveformBase, KerrEccentricEquatorial
):
    r"""Fast Kerr eccentric equatorial flux-based waveform model (source frame).

    The adiabatic EMRI model for equatorial (:math:`x_I = \pm 1`) eccentric orbits in Kerr,
    built from these modules (see :class:`~few.waveform.base.SphericalHarmonicWaveformBase`):

    - trajectory: :class:`~few.trajectory.inspiral.EMRIInspiral` with the
      :class:`~few.trajectory.ode.flux.KerrEccEqFlux` ODE (inserted as
      ``inspiral_kwargs["func"]`` unless given; data file ``KerrEccEqFluxData.h5``),
      integrated adaptively with DOP853 on the CPU; it returns a sparse set of adaptive knots
      plus a dense-output phase spline;
    - amplitudes: :class:`~few.amplitude.ampinterp2d.AmpInterpKerrEccEq` (bicubic splines per
      spin knot, linear in the spin coordinate; data file ``ZNAmps_l10_m10_n55_DS2Outer.h5``)
      at the knots, for the :math:`m \ge 0` modes with :math:`2 \le l \le 10`,
      :math:`|n| \le 55`, :math:`k = 0` (6993 modes; :math:`-m` modes follow by symmetry);
    - spherical harmonics :class:`~few.utils.ylm.GetYlms` and the power-based
      :class:`~few.utils.modeselector.ModeSelector`;
    - summation: :class:`~few.summation.interpolatedmodesum.InterpolatedModeSum` (time domain),
      or :class:`~few.summation.fdinterp.FDInterpolatedModeSum` when
      ``sum_kwargs["output_type"] == "fd"``.

    GPU recommended (amplitudes, mode selection and summation run on the backend; the
    trajectory always runs on the CPU). Calls return the source-frame waveform; use
    :class:`GenerateEMRIWaveform` for the SSB-frame :math:`h_+, h_\times` from sky angles.
    LISAanalysistools' direct EMRI likelihood calls this class with
    ``return_sparse_holder=True`` and builds its own response from the returned
    :class:`~few.waveform.base.SparseInfoHolder`.

    Args:
        inspiral_kwargs: Keywords for the trajectory module. The dict is used both to build
            the trajectory (``EMRIInspiral(**inspiral_kwargs)``) and at every call; it is kept
            by reference as :attr:`inspiral_kwargs` and modified in place (``func`` inserted
            here, ``err`` set to ``1e-11`` at the first call unless given, call-time
            ``inspiral_kwargs`` merged permanently). Default ``None`` means ``{}``.
        amplitude_kwargs: Constructor keywords for
            :class:`~few.amplitude.ampinterp2d.AmpInterpKerrEccEq` (``filename``,
            ``downsample_Z``, ``lmax``, ``nmax``). Default ``None`` means ``{}``.
        sum_kwargs: Constructor keywords for the summation module (``output_type``,
            ``pad_output``, ``odd_len``; see :class:`~few.summation.base.SummationBase`).
            Default ``None`` means ``{}``.
        Ylm_kwargs: Constructor keywords for :class:`~few.utils.ylm.GetYlms`. Default ``None``.
        mode_selector_kwargs: Constructor keywords for
            :class:`~few.utils.modeselector.ModeSelector` (``mode_selection``,
            ``include_minus_mkn``, ``mode_selection_threshold``, ``sensitivity_fn``, ...). The
            constructor ``mode_selection_threshold`` is used only by calls that pass
            ``mode_selection_threshold=None`` (the call default is ``1e-5``). Default ``None``.
        force_backend: Backend of the amplitude, Ylm, mode-selector and summation modules:
            ``"cpu"``, ``"cuda11x"``, ``"cuda12x"``, ``"cuda13x"`` (optionally ``few_``-prefixed)
            or a backend object. ``None`` (default) picks the first available of
            :meth:`supported_backends`. The ``*_kwargs`` dicts of these modules receive the
            resolved ``force_backend`` entry in place.
        **kwargs: ``lmax`` and ``nmax`` (default 10 and 55) are forwarded to
            :class:`~few.utils.baseclasses.KerrEccentricEquatorial` and set this object's own
            mode arrays (the call path uses the amplitude module's mode content); any other key
            is ignored, except ``ndim``, which that constructor does not accept (``TypeError``).

    Example:
        .. code-block:: python

            from few.waveform import FastKerrEccentricEquatorialFlux

            few = FastKerrEccentricEquatorialFlux(sum_kwargs=dict(pad_output=True))
            # m1, m2 [Msun], a, p0 [M], e0, xI0, theta, phi [rad]; dist [Gpc], T [yr], dt [s]
            h = few(1e6, 10.0, 0.9, 12.0, 0.1, 1.0, 0.5, 0.0, dist=1.0, T=1.0, dt=10.0)

            # mode-selected knots only, no mode sum
            H = few(1e6, 10.0, 0.9, 12.0, 0.1, 1.0, 0.5, 0.0, dist=1.0, T=1.0, dt=10.0,
                    return_sparse_holder=True, include_minus_mkn=True)
            H.teuk_modes.shape  # (number of knots, number of kept modes)
    """

    def __init__(
        self,
        /,
        inspiral_kwargs: Optional[dict] = None,
        amplitude_kwargs: Optional[dict] = None,
        sum_kwargs: Optional[dict] = None,
        Ylm_kwargs: Optional[dict] = None,
        mode_selector_kwargs: Optional[dict] = None,
        force_backend: BackendLike = None,
        **kwargs: dict,
    ):
        if inspiral_kwargs is None:
            inspiral_kwargs = {}

        if "func" not in inspiral_kwargs.keys():
            inspiral_kwargs["func"] = KerrEccEqFlux

        # inspiral_kwargs = augment_ODE_func_name(inspiral_kwargs)

        if sum_kwargs is None:
            sum_kwargs = {}
        mode_summation_module = InterpolatedModeSum
        if "output_type" in sum_kwargs:
            if sum_kwargs["output_type"] == "fd":
                mode_summation_module = FDInterpolatedModeSum

        if mode_selector_kwargs is None:
            mode_selector_kwargs = {}
        mode_selection_module = ModeSelector

        KerrEccentricEquatorial.__init__(
            self,
            **{
                key: value
                for key, value in kwargs.items()
                if key in ["lmax", "nmax", "ndim"]
            },
            force_backend=force_backend,
        )
        SphericalHarmonicWaveformBase.__init__(
            self,
            inspiral_module=EMRIInspiral,
            amplitude_module=AmpInterpKerrEccEq,
            sum_module=mode_summation_module,
            mode_selector_module=mode_selection_module,
            inspiral_kwargs=inspiral_kwargs,
            amplitude_kwargs=amplitude_kwargs,
            sum_kwargs=sum_kwargs,
            Ylm_kwargs=Ylm_kwargs,
            mode_selector_kwargs=mode_selector_kwargs,
            force_backend=force_backend,
        )

    @classmethod
    def supported_backends(cls):
        """Backend preference order: ``["cuda13x", "cuda12x", "cuda11x", "cpu"]``."""
        return cls.GPU_RECOMMENDED()

    @property
    def allow_batching(self):
        """bool: ``False``; the ``batch_size`` call keyword is ignored."""
        return False

    def __call__(
        self,
        m1: float,
        m2: float,
        a: float,
        p0: float,
        e0: float,
        xI: float,
        theta: float,
        phi: float,
        *args: Optional[tuple],
        **kwargs: Optional[dict],
    ) -> np.ndarray:
        r"""Generate the source-frame waveform (or its sparse ingredients).

        Runs :meth:`~few.waveform.base.SphericalHarmonicWaveformBase._generate_waveform`:
        trajectory, amplitudes at the trajectory knots, mode selection, then either the mode
        sum or (``return_sparse_holder=True``) an early return of the knots.

        Args:
            m1: Primary (massive black hole) mass in solar masses.
            m2: Secondary mass in solar masses (``m2 <= m1``).
            a: Dimensionless spin of the primary (the domain check rejects ``a > 0.999``).
            p0: Initial semi-latus rectum in units of :math:`M = m_1 + m_2`, inside the
                ``KerrEccEqFlux`` domain (above the separatrix buffer).
            e0: Initial eccentricity, inside the model domain.
            xI: Initial :math:`x_I = \cos I`, ``+1`` (prograde) or ``-1`` (retrograde); ``-1`` is
                mapped to ``(a, xI, theta, phi) -> (-a, 1, pi - theta, -phi)``.
            theta: Source-frame polar viewing angle in rad, in :math:`[0, \pi]` (e.g. from
                :func:`~few.utils.utility.get_viewing_angles`).
            phi: Source-frame azimuthal viewing angle in rad.
            *args: Extra positional trajectory parameters (none for ``KerrEccEqFlux``).
            **kwargs: Keywords of ``_generate_waveform``:

                - ``dist`` (float): luminosity distance in Gpc; ``None`` (default) means no
                  distance scaling (the output is :math:`r h / \mu`).
                - ``Phi_phi0``, ``Phi_theta0``, ``Phi_r0`` (float): initial phases in rad,
                  default 0.
                - ``dt`` (float): output sampling interval in s, default 10.
                - ``T`` (float): maximum duration in years, default 1; the trajectory stops
                  at ``T`` or at the plunge.
                - ``mode_selection_threshold`` (float): power threshold of the mode selection,
                  default ``1e-5``.
                - ``mode_selection`` (str or list): ``None`` (default; the selector's
                  constructor setting), ``"threshold"``, ``"all"``, or a list of
                  ``(l, m, k, n)`` tuples.
                - ``include_minus_mkn`` (bool): with a mode list, also sum each mode's
                  :math:`(-m, -k, -n)` partner; ``None`` (default) keeps the selector's
                  setting (``True``).
                - ``return_sparse_holder`` (bool): return a
                  :class:`~few.waveform.base.SparseInfoHolder` instead of the waveform.
                - ``inspiral_kwargs`` (dict): trajectory keywords, merged permanently into
                  :attr:`inspiral_kwargs`.
                - summation keywords, e.g. ``f_arr`` and ``mask_positive`` for
                  frequency-domain output.

        Returns:
            xp.ndarray or SparseInfoHolder: Time domain: complex :math:`h_+ - i h_\times`,
            shape ``(n,)`` sampled every ``dt``; frequency domain: shape ``(2, n_f)``, rows
            :math:`\tilde h_+, \tilde h_\times`; divided by
            ``dist * Gpc / (mu * MRSUN_SI)`` when ``dist`` is given. With
            ``return_sparse_holder=True``: the :class:`~few.waveform.base.SparseInfoHolder`
            (amplitudes scaled the same way).

        Raises:
            ValueError: If the parameters are outside the model domain (spin, ``p0`` near the
                separatrix or off the grid, ``(a, p0, e0)`` combination, ``m1 < m2``), if
                ``dist <= 0``, if ``mode_selection`` is invalid, or if a trajectory point maps
                outside the amplitude grid.
            AssertionError: From the trajectory's start-point check (``isvalid_pex``) or the
                amplitude module's input checks.
            few.utils.exceptions.TrajectoryOffGridException: If the trajectory leaves the flux
                grid during the integration.
        """
        return self._generate_waveform(
            m1,
            m2,
            a,
            p0,
            e0,
            xI,
            theta,
            phi,
            *args,
            **kwargs,
        )


class FastSchwarzschildEccentricFlux(
    SphericalHarmonicWaveformBase, SchwarzschildEccentric
):
    """Fast Schwarzschild eccentric flux-based waveform model with ROMAN amplitudes (source frame).

    Modules: :class:`~few.trajectory.inspiral.EMRIInspiral` with the
    :class:`~few.trajectory.ode.flux.SchwarzEccFlux` ODE (inserted as
    ``inspiral_kwargs["func"]`` unless given), adaptive and sparse on the CPU; amplitudes from
    the ROMAN neural network :class:`~few.amplitude.romannet.RomanAmplitude`;
    :class:`~few.utils.ylm.GetYlms`; :class:`~few.utils.modeselector.ModeSelector`; the
    summation :class:`~few.summation.interpolatedmodesum.InterpolatedModeSum`, or
    :class:`~few.summation.fdinterp.FDInterpolatedModeSum` when
    ``sum_kwargs["output_type"] == "fd"``. The call fixes ``a = 0`` and ``xI = 1``. GPU
    recommended.

    Args:
        inspiral_kwargs: Keywords for the trajectory module, used at construction and at every
            call and modified in place (see :class:`FastKerrEccentricEquatorialFlux`). Default
            ``None`` means ``{}``.
        amplitude_kwargs: Constructor keywords for
            :class:`~few.amplitude.romannet.RomanAmplitude` (``buffer_length``, ``lmax``,
            ``nmax``). Default ``None``.
        sum_kwargs: Constructor keywords for the summation module (``output_type``,
            ``pad_output``, ``odd_len``). Default ``None``.
        Ylm_kwargs: Constructor keywords for :class:`~few.utils.ylm.GetYlms`. Default ``None``.
        mode_selector_kwargs: Constructor keywords for
            :class:`~few.utils.modeselector.ModeSelector`. Default ``None``.
        force_backend: Backend of the non-trajectory modules (see
            :class:`FastKerrEccentricEquatorialFlux`). Default ``None``.
        **kwargs: ``lmax`` (default 10) and ``nmax`` (default 30) set this object's own mode
            arrays (:class:`~few.utils.baseclasses.SchwarzschildEccentric`); ``ndim`` raises
            ``TypeError`` there; other keys are ignored.

    Example:
        .. code-block:: python

            from few.waveform import FastSchwarzschildEccentricFlux

            few = FastSchwarzschildEccentricFlux()
            # m1, m2 [Msun], p0 [M], e0, theta, phi [rad]
            h = few(1e6, 10.0, 12.0, 0.4, 0.5, 0.0, dist=1.0, T=1.0, dt=10.0)
    """

    def __init__(
        self,
        /,
        inspiral_kwargs: Optional[dict] = None,
        amplitude_kwargs: Optional[dict] = None,
        sum_kwargs: Optional[dict] = None,
        Ylm_kwargs: Optional[dict] = None,
        mode_selector_kwargs: Optional[dict] = None,
        force_backend: BackendLike = None,
        **kwargs: dict,
    ):
        if inspiral_kwargs is None:
            inspiral_kwargs = {}
        if "func" not in inspiral_kwargs.keys():
            inspiral_kwargs["func"] = SchwarzEccFlux

        if sum_kwargs is None:
            sum_kwargs = {}
        mode_summation_module = InterpolatedModeSum
        if "output_type" in sum_kwargs:
            if sum_kwargs["output_type"] == "fd":
                mode_summation_module = FDInterpolatedModeSum

        if mode_selector_kwargs is None:
            mode_selector_kwargs = {}
        mode_selection_module = ModeSelector

        SchwarzschildEccentric.__init__(
            self,
            **{k: v for k, v in kwargs.items() if k in ["lmax", "ndim"]},
            nmax=kwargs["nmax"] if "nmax" in kwargs else 30,
            force_backend=force_backend,
        )
        SphericalHarmonicWaveformBase.__init__(
            self,
            inspiral_module=EMRIInspiral,
            amplitude_module=RomanAmplitude,
            sum_module=mode_summation_module,
            mode_selector_module=mode_selection_module,
            inspiral_kwargs=inspiral_kwargs,
            amplitude_kwargs=amplitude_kwargs,
            sum_kwargs=sum_kwargs,
            Ylm_kwargs=Ylm_kwargs,
            mode_selector_kwargs=mode_selector_kwargs,
            force_backend=force_backend,
        )

    @classmethod
    def supported_backends(cls):
        """Backend preference order: ``["cuda13x", "cuda12x", "cuda11x", "cpu"]``."""
        return cls.GPU_RECOMMENDED()

    @property
    def allow_batching(self):
        """bool: ``False``; the ``batch_size`` call keyword is ignored."""
        return False

    def __call__(
        self,
        m1: float,
        m2: float,
        p0: float,
        e0: float,
        theta: float,
        phi: float,
        *args: Optional[tuple],
        **kwargs: Optional[dict],
    ) -> np.ndarray:
        r"""Generate the source-frame waveform (``a = 0``, ``xI = 1`` are inserted).

        Args:
            m1: Primary mass in solar masses.
            m2: Secondary mass in solar masses (``m2 <= m1``).
            p0: Initial semi-latus rectum in units of :math:`M = m_1 + m_2`; the domain check
                requires ``p0 <= 16 + 2 e0`` (and ``p0 >= 7 (6 + 2 e0) - 41.9`` below 10).
            e0: Initial eccentricity, ``0 <= e0 <= 0.75``.
            theta: Source-frame polar viewing angle in rad, in :math:`[0, \pi]`.
            phi: Source-frame azimuthal viewing angle in rad.
            *args: Extra positional trajectory parameters.
            **kwargs: Keywords of
                :meth:`~few.waveform.base.SphericalHarmonicWaveformBase._generate_waveform`
                (``dist``, ``Phi_phi0``, ``Phi_r0``, ``dt``, ``T``,
                ``mode_selection_threshold``, ``mode_selection``, ...).

        Returns:
            xp.ndarray: Complex :math:`h_+ - i h_\times` (time domain, shape ``(n,)``) or the
            ``(2, n_f)`` frequency-domain array, or a
            :class:`~few.waveform.base.SparseInfoHolder` with ``return_sparse_holder=True``.
        """
        # insert missing arguments for this waveform class
        return self._generate_waveform(
            m1,
            m2,
            0.0,
            p0,
            e0,
            1.0,
            theta,
            phi,
            *args,
            **kwargs,
        )


class FastSchwarzschildEccentricFluxBicubic(
    SphericalHarmonicWaveformBase, SchwarzschildEccentric
):
    """Fast Schwarzschild eccentric flux-based waveform model with bicubic-spline amplitudes.

    Same as :class:`FastSchwarzschildEccentricFlux` except for the amplitude module:
    :class:`~few.amplitude.ampinterp2d.AmpInterpSchwarzEcc` (bicubic splines of the Teukolsky
    amplitudes over the Schwarzschild ``(p, e)`` grid, data file
    ``Teuk_amps_a0.0_lmax_10_nmax_30_new.h5``). Source frame; the call fixes ``a = 0`` and
    ``xI = 1``. GPU recommended.

    Args:
        inspiral_kwargs: Keywords for the trajectory module (``func`` defaults to
            :class:`~few.trajectory.ode.flux.SchwarzEccFlux`), used at construction and at
            every call and modified in place. Default ``None`` means ``{}``.
        amplitude_kwargs: Constructor keywords for
            :class:`~few.amplitude.ampinterp2d.AmpInterpSchwarzEcc` (``filenames``, ``lmax``,
            ``nmax``). Default ``None``.
        sum_kwargs: Constructor keywords for the summation module (``output_type``,
            ``pad_output``, ``odd_len``). Default ``None``.
        Ylm_kwargs: Constructor keywords for :class:`~few.utils.ylm.GetYlms`. Default ``None``.
        mode_selector_kwargs: Constructor keywords for
            :class:`~few.utils.modeselector.ModeSelector`. Default ``None``.
        force_backend: Backend of the non-trajectory modules (see
            :class:`FastKerrEccentricEquatorialFlux`). Default ``None``.
        **kwargs: ``lmax`` (default 10) and ``nmax`` (default 30) set this object's own mode
            arrays; ``ndim`` raises ``TypeError``; other keys are ignored.

    Example:
        .. code-block:: python

            from few.waveform import FastSchwarzschildEccentricFluxBicubic

            few = FastSchwarzschildEccentricFluxBicubic()
            h = few(1e6, 10.0, 12.0, 0.4, 0.5, 0.0, dist=1.0, T=1.0, dt=10.0)
    """

    def __init__(
        self,
        /,
        inspiral_kwargs: Optional[dict] = None,
        amplitude_kwargs: Optional[dict] = None,
        sum_kwargs: Optional[dict] = None,
        Ylm_kwargs: Optional[dict] = None,
        mode_selector_kwargs: Optional[dict] = None,
        force_backend: BackendLike = None,
        **kwargs: Optional[dict],
    ):
        if inspiral_kwargs is None:
            inspiral_kwargs = {}
        if "func" not in inspiral_kwargs.keys():
            inspiral_kwargs["func"] = SchwarzEccFlux

        if sum_kwargs is None:
            sum_kwargs = {}
        mode_summation_module = InterpolatedModeSum
        if "output_type" in sum_kwargs:
            if sum_kwargs["output_type"] == "fd":
                mode_summation_module = FDInterpolatedModeSum

        if mode_selector_kwargs is None:
            mode_selector_kwargs = {}
        mode_selection_module = ModeSelector

        SchwarzschildEccentric.__init__(
            self,
            **{k: v for k, v in kwargs.items() if k in ["lmax", "ndim", "nmax"]},
            force_backend=force_backend,
        )
        SphericalHarmonicWaveformBase.__init__(
            self,
            inspiral_module=EMRIInspiral,
            amplitude_module=AmpInterpSchwarzEcc,
            sum_module=mode_summation_module,
            mode_selector_module=mode_selection_module,
            inspiral_kwargs=inspiral_kwargs,
            amplitude_kwargs=amplitude_kwargs,
            sum_kwargs=sum_kwargs,
            Ylm_kwargs=Ylm_kwargs,
            mode_selector_kwargs=mode_selector_kwargs,
            force_backend=force_backend,
        )

    @classmethod
    def supported_backends(cls):
        """Backend preference order: ``["cuda13x", "cuda12x", "cuda11x", "cpu"]``."""
        return cls.GPU_RECOMMENDED()

    @property
    def allow_batching(self):
        """bool: ``False``; the ``batch_size`` call keyword is ignored."""
        return False

    def __call__(
        self,
        m1: float,
        m2: float,
        p0: float,
        e0: float,
        theta: float,
        phi: float,
        *args: Optional[tuple],
        **kwargs: Optional[dict],
    ) -> np.ndarray:
        r"""Generate the source-frame waveform (``a = 0``, ``xI = 1`` are inserted).

        Args:
            m1: Primary mass in solar masses.
            m2: Secondary mass in solar masses (``m2 <= m1``).
            p0: Initial semi-latus rectum in units of :math:`M = m_1 + m_2` (domain as in
                :meth:`FastSchwarzschildEccentricFlux.__call__`).
            e0: Initial eccentricity, ``0 <= e0 <= 0.75``.
            theta: Source-frame polar viewing angle in rad, in :math:`[0, \pi]`.
            phi: Source-frame azimuthal viewing angle in rad.
            *args: Extra positional trajectory parameters.
            **kwargs: Keywords of
                :meth:`~few.waveform.base.SphericalHarmonicWaveformBase._generate_waveform`.

        Returns:
            xp.ndarray: Complex :math:`h_+ - i h_\times` (time domain, shape ``(n,)``) or the
            ``(2, n_f)`` frequency-domain array, or a
            :class:`~few.waveform.base.SparseInfoHolder` with ``return_sparse_holder=True``.
        """
        # insert missing arguments for this waveform class
        return self._generate_waveform(
            m1,
            m2,
            0.0,
            p0,
            e0,
            1.0,
            theta,
            phi,
            *args,
            **kwargs,
        )


class SlowSchwarzschildEccentricFlux(
    SphericalHarmonicWaveformBase, SchwarzschildEccentric
):
    """Slow Schwarzschild eccentric reference model: dense trajectory, all modes, direct sum.

    The reference against which the fast Schwarzschild models are tested. The trajectory
    (:class:`~few.trajectory.inspiral.EMRIInspiral` with
    :class:`~few.trajectory.ode.flux.SchwarzEccFlux`) is forced to fixed steps of ``dt``
    (``inspiral_kwargs["DENSE_STEPPING"] = 1``), the bicubic-spline amplitudes
    (:class:`~few.amplitude.ampinterp2d.AmpInterpSchwarzEcc`) are evaluated at every step,
    all modes are kept (``mode_selection="all"``, forced at construction and at every call)
    and :class:`~few.summation.directmodesum.DirectModeSum` sums them directly. CPU only;
    batching over time (``batch_size``, ``show_progress``) is allowed. Source frame; the call
    fixes ``a = 0`` and ``xI = 1``.

    Args:
        inspiral_kwargs: Keywords for the trajectory module; ``DENSE_STEPPING`` is set to 1 and
            ``func`` defaults to ``SchwarzEccFlux`` (the dict is modified in place). Default
            ``None`` means ``{}``.
        amplitude_kwargs: Constructor keywords for
            :class:`~few.amplitude.ampinterp2d.AmpInterpSchwarzEcc`. Default ``None``.
        sum_kwargs: Constructor keywords for :class:`~few.summation.directmodesum.DirectModeSum`
            (``pad_output``, ``odd_len``). Default ``None``.
        Ylm_kwargs: Constructor keywords for :class:`~few.utils.ylm.GetYlms`. Default ``None``.
        mode_selector_kwargs: Constructor keywords for
            :class:`~few.utils.modeselector.ModeSelector`; ``mode_selection`` is set to
            ``"all"`` (the dict is modified in place). Default ``None``.
        force_backend: Backend (``supported_backends`` is CPU only). Default ``None``.
        **kwargs: ``lmax``, ``nmax`` (defaults 10, 30) set this object's own mode arrays;
            ``ndim`` raises ``TypeError``; other keys are ignored.

    Raises:
        ValueError: If ``mode_selector_kwargs["mode_selection"]`` is given and is not
            ``"all"``.

    Example:
        .. code-block:: python

            from few.waveform import SlowSchwarzschildEccentricFlux

            slow = SlowSchwarzschildEccentricFlux()
            h = slow(1e6, 10.0, 12.0, 0.4, 0.5, 0.0, dist=1.0, T=0.01, dt=10.0,
                     batch_size=10000)
    """

    @classmethod
    def supported_backends(cls):
        """Backend preference order: ``["cpu"]``."""
        return cls.CPU_ONLY()

    @property
    def allow_batching(self):
        """bool: ``True``; ``batch_size > 0`` splits the trajectory into time batches."""
        return True

    def __init__(
        self,
        /,
        inspiral_kwargs: Optional[dict] = None,
        amplitude_kwargs: Optional[dict] = None,
        sum_kwargs: Optional[dict] = None,
        Ylm_kwargs: Optional[dict] = None,
        mode_selector_kwargs: Optional[dict] = None,
        force_backend: BackendLike = None,
        **kwargs: dict,
    ):
        if inspiral_kwargs is None:
            inspiral_kwargs = {}
        # declare specific properties
        inspiral_kwargs["DENSE_STEPPING"] = 1
        if "func" not in inspiral_kwargs.keys():
            inspiral_kwargs["func"] = SchwarzEccFlux

        SchwarzschildEccentric.__init__(
            self,
            **{k: v for k, v in kwargs.items() if k in ["lmax", "ndim", "nmax"]},
            force_backend=force_backend,
        )
    
        if mode_selector_kwargs is None:
            mode_selector_kwargs = {}
        if "mode_selection" in mode_selector_kwargs.keys():
            if mode_selector_kwargs["mode_selection"] != "all":
                raise ValueError("Mode selection must be 'all' for slow waveform.")
        mode_selector_kwargs.update(dict(mode_selection="all"))
        SphericalHarmonicWaveformBase.__init__(
            self,
            inspiral_module=EMRIInspiral,
            amplitude_module=AmpInterpSchwarzEcc,
            sum_module=DirectModeSum,
            mode_selector_module=ModeSelector,
            inspiral_kwargs=inspiral_kwargs,
            amplitude_kwargs=amplitude_kwargs,
            sum_kwargs=sum_kwargs,
            Ylm_kwargs=Ylm_kwargs,
            mode_selector_kwargs=mode_selector_kwargs,
            force_backend=force_backend,
        )

    def __call__(
        self,
        m1: float,
        m2: float,
        p0: float,
        e0: float,
        theta: float,
        phi: float,
        *args: Optional[tuple],
        **kwargs: Optional[dict],
    ) -> np.ndarray:
        r"""Generate the source-frame reference waveform with all modes.

        ``mode_selection="all"`` is written into ``kwargs``, and ``a = 0``, ``xI = 1`` are
        inserted.

        Args:
            m1: Primary mass in solar masses.
            m2: Secondary mass in solar masses (``m2 <= m1``).
            p0: Initial semi-latus rectum in units of :math:`M = m_1 + m_2` (domain as in
                :meth:`FastSchwarzschildEccentricFlux.__call__`).
            e0: Initial eccentricity, ``0 <= e0 <= 0.75``.
            theta: Source-frame polar viewing angle in rad, in :math:`[0, \pi]`.
            phi: Source-frame azimuthal viewing angle in rad.
            *args: Extra positional trajectory parameters.
            **kwargs: Keywords of
                :meth:`~few.waveform.base.SphericalHarmonicWaveformBase._generate_waveform`
                (``dist``, ``Phi_phi0``, ``Phi_r0``, ``dt``, ``T``, ``batch_size``,
                ``show_progress``, ...); a ``mode_selection`` entry is overwritten.

        Returns:
            xp.ndarray: Complex :math:`h_+ - i h_\times`, one sample per (fixed-``dt``)
            trajectory step, batches concatenated.
        """
        # insert missing arguments for this waveform class
        if kwargs is None:
            kwargs = {}
        kwargs.update(dict(mode_selection="all"))
        return self._generate_waveform(
            m1,
            m2,
            0.0,
            p0,
            e0,
            1.0,
            theta,
            phi,
            *args,
            **kwargs,
        )


class Pn5AAKWaveform(AAKWaveformBase):
    r"""Waveform generation class for AAK with 5PN trajectory.

    This class generates waveforms based on the Augmented Analytic Kludge
    given in the
    `EMRI Kludge Suite <https://github.com/alvincjk/EMRI_Kludge_Suite/>`_.
    However, here the trajectory is vastly improved by employing the 5PN
    fluxes for generic Kerr orbits from
    `Fujita & Shibata 2020 <https://arxiv.org/abs/2008.13554>`_.

    The 5PN trajectory (:class:`~few.trajectory.ode.pn5.PN5`, inserted as
    ``inspiral_kwargs["func"]`` unless given) produces orbital and phase trajectories.
    The trajectory is calculated until the orbit reaches
    within 0.1 (in ``p``) of the separatrix, determined from
    `arXiv:1912.07609 <https://arxiv.org/abs/1912.07609/>`_. The
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
    waveform quantities (:class:`~few.summation.aakwave.AAKSummation`). This waveform does
    not have the freedom in terms of user-chosen quantities that
    :class:`few.waveform.base.SphericalHarmonicWaveformBase` contains.
    This is mainly due to the specific waveform constructions particular
    to the AAK/AK. It is built in the detector frame (SSB-frame :math:`h_+, h_\times`, or
    the long-wavelength :math:`h_I, h_{II}` with ``mich=True``); the call signature is
    :meth:`~few.waveform.base.AAKWaveformBase.__call__`.

    **Please note:** the 5PN trajectory and AAK waveform take the parameter
    :math:`Y\equiv\cos{\iota}=L/\sqrt{L^2 + Q}` rather than :math:`x_I` as is accepted
    for relativistic waveforms. :class:`GenerateEMRIWaveform` passes its ``x0`` to this model
    unchanged (its :math:`x_I \to Y` conversion does not reach the model, see
    :meth:`GenerateEMRIWaveform.__call__`), so ``x0`` is used as :math:`Y_0`.

    Args:
        inspiral_kwargs: Keywords for the trajectory module, used at construction and at every
            call (stored as :attr:`inspiral_kwargs`; ``func`` is inserted in place). Default
            ``None`` means ``{}``.
        sum_kwargs: Constructor keywords for :class:`~few.summation.aakwave.AAKSummation`
            (``pad_output``, ``odd_len``). Default ``None``.
        force_backend: Backend of the summation module (see
            :class:`FastKerrEccentricEquatorialFlux`). Default ``None``.

    Example:
        .. code-block:: python

            from few.waveform import Pn5AAKWaveform

            aak = Pn5AAKWaveform()
            # m1, m2 [Msun], a, p0 [M], e0, Y0, dist [Gpc], qS, phiS, qK, phiK [rad]
            h = aak(1e6, 10.0, 0.8, 12.0, 0.2, 0.9, 1.0, 0.8, 2.0, 1.2, 0.5, T=1.0, dt=10.0)
    """

    def __init__(
        self,
        inspiral_kwargs: Optional[dict] = None,
        sum_kwargs: Optional[dict] = None,
        force_backend: BackendLike = None,
    ):
        if inspiral_kwargs is None:
            inspiral_kwargs = {}
        if "func" not in inspiral_kwargs.keys():
            inspiral_kwargs["func"] = PN5

        AAKWaveformBase.__init__(
            self,
            inspiral_module=EMRIInspiral,
            sum_module=AAKSummation,
            inspiral_kwargs=inspiral_kwargs,
            sum_kwargs=sum_kwargs,
            force_backend=force_backend,
        )

    @classmethod
    def supported_backends(cls):
        """Backend preference order: ``["cuda13x", "cuda12x", "cuda11x", "cpu"]``."""
        return cls.GPU_RECOMMENDED()
