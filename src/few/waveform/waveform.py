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
from ..utils.modeselector import ModeSelector, NeuralModeSelector
from .base import AAKWaveformBase, SphericalHarmonicWaveformBase, WaveformModule

# get path to this file
dir_path = os.path.dirname(os.path.realpath(__file__))


class GenerateEMRIWaveform(Generic[WaveformModule]):
    r"""Generic waveform generator for data analysis

    This class allows the user interface to be the exact same between any
    waveform in the FEW package. For waveforms built in the source frame,
    like :class:`few.waveform.FastSchwarzschildEccentricFlux`, the waveforms
    are transformed to the detector frame. Waveforms like
    :class:`few.waveform.Pn5AAKWaveform` that are built in the detector frame
    are left alone effectively.

    For waveforms that are less than Kerr generic (i.e. certain parameters are
    unnecessary), this interface automatically removes the waveforms dependence
    on those parameters.

    Args:
        waveform_class: String with the name of the waveform class to use.
            See the `pre-built waveform models
            <https://bhptoolkit.org/FastEMRIWaveforms/user/main.html#prebuilt-waveform-models>`_.
            If an object is provided, must be a waveform class.
        *args: Arguments for the instantiation of
            the waveform generation class.
        frame: Which frame to produce the output waveform in.
            Default is "detector." Right now, the source frame is not implemented
            for waveforms that are built in the detector frame.
        return_list: If True, return :math:`h_p` and
            :math:`h_x` as a list. If False, return :math:`hp - ihx`. Default
            is False.
        **kwargs: Dictionary with kwargs for the instantiation of
            the waveform generator.

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
        return {
            "FastSchwarzschildEccentricFlux": FastSchwarzschildEccentricFlux,
            "FastSchwarzschildEccentricFluxBicubic": FastSchwarzschildEccentricFluxBicubic,
            "SlowSchwarzschildEccentricFlux": SlowSchwarzschildEccentricFlux,
            "FastKerrEccentricEquatorialFlux": FastKerrEccentricEquatorialFlux,
            "Pn5AAKWaveform": Pn5AAKWaveform,
        }

    @property
    def stock_waveform_options(self) -> list[str]:
        return list(self._stock_waveform_definitions.keys())

    def _get_viewing_angles(self, qS, phiS, qK, phiK):
        """Transform from the detector frame to the source frame"""

        cqS = np.cos(qS)
        sqS = np.sin(qS)

        cphiS = np.cos(phiS)
        sphiS = np.sin(phiS)

        cqK = np.cos(qK)
        sqK = np.sin(qK)

        cphiK = np.cos(phiK)
        sphiK = np.sin(phiK)

        # sky location vector
        R = np.array([sqS * cphiS, sqS * sphiS, cqS])

        # spin vector
        S = np.array([sqK * cphiK, sqK * sphiK, cqK])

        # get viewing angles
        phi = -np.pi / 2.0  # by definition of source frame

        theta = np.arccos(-np.dot(R, S))  # normalized vector

        return (theta, phi)

    def _to_SSB_frame(self, hp, hc, qS, phiS, qK, phiK):
        """Transform to SSB frame"""

        cqS = np.cos(qS)
        sqS = np.sin(qS)

        # cphiS = np.cos(phiS)
        # sphiS = np.sin(phiS)

        cqK = np.cos(qK)
        sqK = np.sin(qK)

        # cphiK = np.cos(phiK)
        # sphiK = np.sin(phiK)

        # get polarization angle

        up_ldc = cqS * sqK * np.cos(phiS - phiK) - cqK * sqS
        dw_ldc = sqK * np.sin(phiS - phiK)

        if dw_ldc != 0.0:
            psi_ldc = -np.arctan2(up_ldc, dw_ldc)

        else:
            psi_ldc = 0.5 * np.pi

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
        r"""Generate the waveform with the given parameters.

        Args:
            m1: Mass of larger black hole in solar masses.
            m2: Mass of compact object in solar masses.
            a: Dimensionless spin of massive black hole.
            p0: Initial semilatus rectum (Must be greater than
                the separatrix at the the given e0 and x0).
                See documentation for more information on :math:`p_0<10`.
            e0: Initial eccentricity.
            x0: Initial cosine of the inclination angle.
                (:math:`x_I=\cos{I}`). This differs from :math:`Y=\cos{\iota}\equiv L_z/\sqrt{L_z^2 + Q}`
                used in the semi-relativistic formulation. When running kludge waveforms,
                :math:`x_{I,0}` will be converted to :math:`Y_0`.
            dist: Luminosity distance in Gpc.
            qS: Sky location polar angle in ecliptic
                coordinates.
            phiS: Sky location azimuthal angle in
                ecliptic coordinates.
            qK: Initial BH spin polar angle in ecliptic
                coordinates.
            phiK: Initial BH spin azimuthal angle in
                ecliptic coordinates.
            Phi_phi0: Initial phase for :math:`\Phi_\phi`.
                Default is 0.0.
            Phi_theta0: Initial phase for :math:`\Phi_\Theta`.
                Default is 0.0.
            Phi_r0: Initial phase for :math:`\Phi_r`.
                Default is 0.0.
            *args: Tuple of any extra parameters that go into the model.
            **kwargs: Dictionary with kwargs for online waveform
                generation.

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
            ) = self._get_viewing_angles(qS, phiS, qK, phiK)

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
    """Prebuilt model for fast Kerr eccentric equatorial flux-based waveforms.

    This model combines the most efficient modules to produce the fastest
    accurate EMRI waveforms. It leverages GPU hardware for maximal acceleration,
    but is also available on for CPUs.

    The trajectory module used here is :class:`few.trajectory.inspiral` for a
    flux-based, sparse trajectory. This returns approximately 100 points.

    The amplitudes are then determined with
    :class:`few.amplitude.ampinterp2d.AmpInterp2D` along these sparse
    trajectories. This gives complex amplitudes for all modes in this model at
    each point in the trajectory. These are then filtered with
    :class:`few.utils.modeselector.ModeSelector`.

    The modes that make it through the filter are then summed by
    :class:`few.summation.interpolatedmodesum.InterpolatedModeSum`.

    See :class:`few.waveform.base.SphericalHarmonicWaveformBase` for information
    on inputs. See examples as well.

    args:
        inspiral_kwargs : Optional kwargs to pass to the
            inspiral generator. **Important Note**: These kwargs are passed
            online, not during instantiation like other kwargs here. Default is
            {}.
        amplitude_kwargs: Optional kwargs to pass to the
            amplitude generator during instantiation. Default is {}.
        sum_kwargs: Optional kwargs to pass to the
            sum module during instantiation. Default is {}.
        Ylm_kwargs: Optional kwargs to pass to the
            Ylm generator during instantiation. Default is {}.
        *args: args for waveform model.
        **kwargs: kwargs for waveform model.

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
        if "mode_selection_type" in mode_selector_kwargs:
            if mode_selector_kwargs["mode_selection_type"] == "neural":
                mode_selection_module = NeuralModeSelector
                if "mode_selector_location" not in mode_selector_kwargs:
                    mode_selector_kwargs["mode_selector_location"] = os.path.join(
                        dir_path,
                        "./files/modeselector_files/KerrEccentricEquatorialFlux/",
                    )
                mode_selector_kwargs["keep_inds"] = np.array(
                    [0, 1, 2, 3, 4, 6, 7, 8, 9]
                )

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
            **{
                key: value for key, value in kwargs.items() if key in ["normalize_amps"]
            },
            force_backend=force_backend,
        )

    @classmethod
    def supported_backends(cls):
        return cls.GPU_RECOMMENDED()

    @property
    def allow_batching(self):
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
        """
        Generate the waveform.

        Args:
            m1: Mass of larger black hole in solar masses.
            m2: Mass of compact object in solar masses.
            a: Dimensionless spin of massive black hole.
            p0: Initial semilatus rectum of inspiral trajectory.
            e0: Initial eccentricity of inspiral trajectory.
            xI: Initial cosine of the inclination angle.
            theta: Polar angle of observer.
            phi: Azimuthal angle of observer.
            *args: Placeholder for additional arguments.
            **kwargs: Placeholder for additional keyword arguments.

        Returns:
            Complex array containing generated waveform.

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
    """Prebuilt model for fast Schwarzschild eccentric flux-based waveforms.

    This model combines the most efficient modules to produce the fastest
    accurate EMRI waveforms. It leverages GPU hardware for maximal acceleration,
    but is also available on for CPUs.

    The trajectory module used here is :class:`few.trajectory.inspiral` for a
    flux-based, sparse trajectory. This returns approximately 100 points.

    The amplitudes are then determined with
    :class:`few.amplitude.romannet` along these sparse
    trajectories. This gives complex amplitudes for all modes in this model at
    each point in the trajectory. These are then filtered with
    :class:`few.utils.modeselector.ModeSelector`.

    The modes that make it through the filter are then summed by
    :class:`few.summation.interpolatedmodesum.InterpolatedModeSum`.

    See :class:`few.waveform.base.SphericalHarmonicWaveformBase` for information
    on inputs. See examples as well.

    args:
        inspiral_kwargs: Optional kwargs to pass to the
            inspiral generator. **Important Note**: These kwargs are passed
            online, not during instantiation like other kwargs here. Default is
            {}.
        amplitude_kwargs: Optional kwargs to pass to the
            amplitude generator during instantiation. Default is {}.
        sum_kwargs: Optional kwargs to pass to the
            sum module during instantiation. Default is {}.
        Ylm_kwargs: Optional kwargs to pass to the
            Ylm generator during instantiation. Default is {}.
        *args: args for waveform model.
        **kwargs: kwargs for waveform model.

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
        if "mode_selection_type" in mode_selector_kwargs:
            if mode_selector_kwargs["mode_selection_type"] == "neural":
                mode_selection_module = NeuralModeSelector
                if "mode_selector_location" not in mode_selector_kwargs:
                    mode_selector_kwargs["mode_selector_location"] = os.path.join(
                        dir_path,
                        "./files/modeselector_files/KerrEccentricEquatorialFlux/",
                    )
                mode_selector_kwargs["keep_inds"] = np.array([0, 1, 3, 4, 6, 7, 8, 9])

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
            normalize_amps=True,
            force_backend=force_backend,
        )

    @classmethod
    def supported_backends(cls):
        return cls.GPU_RECOMMENDED()

    @property
    def allow_batching(self):
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
        """
        Generate the waveform.

        Args:
            m1: Mass of larger black hole in solar masses.
            m2: Mass of compact object in solar masses.
            p0: Initial semilatus rectum of inspiral trajectory.
            e0: Initial eccentricity of inspiral trajectory.
            theta: Polar angle of observer.
            phi: Azimuthal angle of observer.
            *args: Placeholder for additional arguments.
            **kwargs: Placeholder for additional keyword arguments.

        Returns:
            Complex array containing generated waveform.

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
    """Prebuilt model for fast Schwarzschild eccentric flux-based waveforms.

    This model combines the most efficient modules to produce the fastest
    accurate EMRI waveforms. It leverages GPU hardware for maximal acceleration,
    but is also available on for CPUs.

    The trajectory module used here is :class:`few.trajectory.inspiral` for a
    flux-based, sparse trajectory. This returns approximately 100 points.

    The amplitudes are then determined with
    :class:`few.amplitude.interp2dcubicspline.AmpInterp2D` along these sparse
    trajectories. This gives complex amplitudes for all modes in this model at
    each point in the trajectory. These are then filtered with
    :class:`few.utils.modeselector.ModeSelector`.

    The modes that make it through the filter are then summed by
    :class:`few.summation.interpolatedmodesum.InterpolatedModeSum`.

    See :class:`few.waveform.base.SphericalHarmonicWaveformBase` for information
    on inputs. See examples as well.

    args:
        inspiral_kwargs: Optional kwargs to pass to the
            inspiral generator. **Important Note**: These kwargs are passed
            online, not during instantiation like other kwargs here. Default is
            {}.
        amplitude_kwargs: Optional kwargs to pass to the
            amplitude generator during instantiation. Default is {}.
        sum_kwargs: Optional kwargs to pass to the
            sum module during instantiation. Default is {}.
        Ylm_kwargs: Optional kwargs to pass to the
            Ylm generator during instantiation. Default is {}.
        *args: args for waveform model.
        **kwargs: kwargs for waveform model.

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
        if "mode_selection_type" in mode_selector_kwargs:
            if mode_selector_kwargs["mode_selection_type"] == "neural":
                mode_selection_module = NeuralModeSelector
                if "mode_selector_location" not in mode_selector_kwargs:
                    mode_selector_kwargs["mode_selector_location"] = os.path.join(
                        dir_path,
                        "./files/modeselector_files/KerrEccentricEquatorialFlux/",
                    )
                mode_selector_kwargs["keep_inds"] = np.array([0, 1, 3, 4, 6, 7, 8, 9])

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
            **{
                key: value for key, value in kwargs.items() if key in ["normalize_amps"]
            },
            force_backend=force_backend,
        )

    @classmethod
    def supported_backends(cls):
        return cls.GPU_RECOMMENDED()

    @property
    def allow_batching(self):
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
        """
        Generate the waveform.

        Args:
            m1: Mass of larger black hole in solar masses.
            m2: Mass of compact object in solar masses.
            p0: Initial semilatus rectum of inspiral trajectory.
            e0: Initial eccentricity of inspiral trajectory.
            theta: Polar angle of observer.
            phi: Azimuthal angle of observer.
            *args: Placeholder for additional arguments.
            **kwargs: Placeholder for additional keyword arguments.

        Returns:
            Complex array containing generated waveform.
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
    """Prebuilt model for slow Schwarzschild eccentric flux-based waveforms.

    This model combines the various modules to produce the a reference waveform
    against which we test our fast models. Please see
    :class:`few.utils.baseclasses.SchwarzschildEccentric` for general
    information on this class of models.

    The trajectory module used here is :class:`few.trajectory.inspiral` for a
    flux-based trajectory. For this slow waveform, the DENSE_STEPPING parameter
    from :class:`few.trajectory.base.TrajectoryBase` is fixed to 1 to create
    a densely sampled trajectory.

    The amplitudes are then determined with
    :class:`few.amplitude.interp2dcubicspline.AmpInterp2D`
    along a densely sampled trajectory. This gives complex amplitudes
    for all modes in this model at each point in the trajectory.

    As this class is meant to be a reference waveform class, all waveform mode amplitudes
    are used. The modes are summed by :class:`few.summation.directmodesum.DirectModeSum`.

    See :class:`few.waveform.base.SphericalHarmonicWaveformBase` for information
    on inputs. See examples as well.

    args:
        inspiral_kwargs: Optional kwargs to pass to the
            inspiral generator. **Important Note**: These kwargs are passed
            online, not during instantiation like other kwargs here. Default is
            {}.
        amplitude_kwargs: Optional kwargs to pass to the
            amplitude generator during instantiation. Default is {}.
        sum_kwargs: Optional kwargs to pass to the
            sum module during instantiation. Default is {}.
        Ylm_kwargs: Optional kwargs to pass to the
            Ylm generator during instantiation. Default is {}.
        *args: args for waveform model.
        **kwargs: kwargs for waveform model.

    """

    @classmethod
    def supported_backends(cls):
        return cls.CPU_ONLY()

    @property
    def allow_batching(self):
        return True

    def __init__(
        self,
        /,
        inspiral_kwargs: Optional[dict] = None,
        amplitude_kwargs: Optional[dict] = None,
        sum_kwargs: Optional[dict] = None,
        Ylm_kwargs: Optional[dict] = None,
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
            **{
                key: value
                for key, value in kwargs.items()
                if key in ["mode_selector_kwargs", "normalize_amps"]
            },
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
        """
        Generate the waveform.

        Args:
            m1: Mass of larger black hole in solar masses.
            m2: Mass of compact object in solar masses.
            p0: Initial semilatus rectum of inspiral trajectory.
            e0: Initial eccentricity of inspiral trajectory.
            theta: Polar angle of observer.
            phi: Azimuthal angle of observer.
            *args: Placeholder for additional arguments.
            **kwargs: Placeholder for additional keyword arguments.

        Returns:
            Complex array containing generated waveform.
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

    The 5PN trajectory produces orbital and phase trajectories.
    The trajectory is calculated until the orbit reaches
    within 0.2 of the separatrix, determined from
    `arXiv:1912.07609 <https://arxiv.org/abs/1912.07609/>`_. The
    fundamental frequencies along the trajectory at each point are then
    calculated from the orbital parameters and the spin value given by (`Schmidt 2002 <https://arxiv.org/abs/gr-qc/0202090>`_).

    These frequencies along the trajectory are then used to map to the
    frequency basis of the `Analytic Kludge <https://arxiv.org/abs/gr-qc/0310125>`_. This mapping
    takes the form of time evolving large mass and spin parameters, as
    well as the use of phases and frequencies in
    :math:`(alpha, \Phi, \gamma)`:

    .. math:: \Phi = \Phi_\phi,

    .. math:: \gamma = \Phi_\phi + \Phi_\Theta,

    .. math:: alpha = \Phi_\phi + \Phi_\Theta + \Phi_r.

    The frequencies in that basis are found by taking the time derivatives
    of each equation above.

    This class has GPU capabilities and works from the sparse trajectory
    methodoligy with cubic spine interpolation of the smoothly varying
    waveform quantities. This waveform does not have the freedom in terms
    of user-chosen quantitites that
    :class:`few.waveform.base.SphericalHarmonicWaveformBase` contains.
    This is mainly due to the specific waveform constructions particular
    to the AAK/AK.

    **Please note:** the 5PN trajectory and AAK waveform take the parameter
    :math:`Y\equiv\cos{\iota}=L/\sqrt{L^2 + Q}` rather than :math:`x_I` as is accepted
    for relativistic waveforms and in the generic waveform interface discussed above.
    The generic waveform interface directly converts :math:`x_I` to :math:`Y`.

    args:
        inspiral_kwargs: Optional kwargs to pass to the
            inspiral generator. **Important Note**: These kwargs are passed
            online, not during instantiation like other kwargs here. Default is
            {}. This is stored as an attribute.
        sum_kwargs: Optional kwargs to pass to the
            sum module during instantiation. Default is {}.
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
        return cls.GPU_RECOMMENDED()
