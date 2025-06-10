# AAK summation module for Fast EMRI Waveforms

import numpy as np

from few.utils.globals import get_logger

# Python imports
from ..utils.baseclasses import BackendLike, Pn5AAK
from .base import SummationBase


class AAKSummation(Pn5AAK, SummationBase):
    """Calculate an AAK waveform from an input trajectory.

    Please see the documentations for
    :class:`few.waveform.Pn5AAKWaveform`
    for overall aspects of this model.

    Given an input trajectory and other parameters, this module maps that
    trajectory to the Analytic Kludge basis as performed for the Augmented
    Analytic Kludge model. Please see
    `the AAK paper <https://arxiv.org/abs/1510.06245>`_ for more information.

    args:
        **kwargs: Optional keyword arguments for the base classes:
            :class:`few.utils.baseclasses.Pn5AAK`.
            :class:`few.utils.baseclasses.SummationBase`.
    """

    def __init__(self, force_backend: BackendLike = None, **kwargs):
        Pn5AAK.__init__(self)
        SummationBase.__init__(self, **kwargs, force_backend=force_backend)

    @property
    def waveform_generator(self):
        """obj: Compiled CPU/GPU that performs the AAK waveform generation."""
        return self.backend.pyWaveform

    @classmethod
    def supported_backends(cls):
        return cls.GPU_RECOMMENDED()

    def sum(
        self,
        tvec: np.ndarray,
        m1: float,
        a: float,
        dist: float,
        m2: float,
        qS: float,
        phiS: float,
        qK: float,
        phiK: float,
        nmodes: int,
        interp_t: np.ndarray,
        interp_coeffs: np.ndarray,
        *args,
        mich: bool = False,
        dt: float = 10.0,
        integrate_backwards: bool = False,
        **kwargs,
    ) -> None:
        r"""Compute an AAK waveform from an input trajectory.

        This function performs the AAK waveform summation and fills the waveform array in-place.

        **Please note:** the 5PN trajectory and AAK waveform take the parameter
        :math:`Y\equiv\cos{\iota}=L/\sqrt{L^2 + Q}` rather than :math:`x_I` as is accepted
        for relativistic waveforms and in the generic waveform interface discussed above.
        The generic waveform interface directly converts :math:`x_I` to :math:`Y`.

        args:
            tvec: Array containing the time values
                associated with the sparse trajectory.
            m1: Mass of massive black hole in solar masses.
            a: Dimensionless spin of massive black hole.
            p: Array containing the trajectory for values of
                the semi-latus rectum.
            e: Array containing the trajectory for values of
                the eccentricity.
            Y: Array containing the trajectory for values of
                :math:`\cos{\iota}`. **Note**: This value is different from :math:`x_I`
                used in the relativistic waveforms.
            dist: Luminosity distance in Gpc.
            Phi_phi: Array containing the trajectory for
                :math:`\Phi_\phi`.
            Phi_theta: Array containing the trajectory for
                :math:`\Phi_\theta`.
            Phi_r: Array containing the trajectory for
                :math:`\Phi_r`.
            m2: Mass of compact object in solar masses.
            qS: Sky location polar angle in ecliptic
                coordinates.
            phiS: Sky location azimuthal angle in
                ecliptic coordinates.
            qK: Initial BH spin polar angle in ecliptic
                coordinates.
            phiK: Initial BH spin azimuthal angle in
                ecliptic coordinates.
            nmodes: Number of modes to analyze. This is determined by
                the eccentricity.
            *args (tuple, placeholder): Added to create flexibility when calling different
                amplitude modules. It is not used.
            mich: If True, produce waveform with
                long-wavelength response approximation (hI, hII). Please
                note this is not TDI. If False, return hplus and hcross.
                Default is False.
            dt: Time between samples in seconds
                (inverse of sampling frequency). Default is 10.0.
            **kwargs: Added to create flexibility when calling different
                amplitude modules. It is not used.

        """

        fill_val = 1e-6
        if qK < fill_val or qK > np.pi - fill_val:
            get_logger().warning(
                "qK is within 1e-6 of the poles. We shift this value automatically away from poles by 1e-6."
            )
            if qK < fill_val:
                qK = fill_val
            else:
                qK = np.pi - fill_val

        if qS < fill_val or qS > np.pi - fill_val:
            get_logger().warning(
                "qS is within 1e-6 of the poles. We shift this value automatically away from poles by 1e-6."
            )
            if qS < fill_val:
                qS = fill_val
            else:
                qS = np.pi - fill_val

        init_len = len(interp_t)

        if integrate_backwards:
            # For consistency with forward integration, we slightly shift the knots so that they line up at t=0
            offset = tvec[-1] - int(tvec[-1] / dt) * dt
            interp_t = interp_t - offset

        # if equatorial, set theta phase coefficients equal to phi counterparts
        # we check this by inspecting the first coefficient of Y
        # as equatorial goes to equatorial, this is a necessary and sufficient condition
        if abs(interp_coeffs[0, 2, 0]) == 1.0:
            interp_coeffs[:, 4] = interp_coeffs[:, 3]

        # convert to gpu if desired
        interp_coeffs_in = self.xp.transpose(
            self.xp.asarray(interp_coeffs), [2, 0, 1]
        ).flatten()

        # generate the waveform
        self.waveform_generator(
            self.waveform,
            interp_coeffs_in,
            m1,
            a,
            m2,
            qS,
            phiS,
            qK,
            phiK,
            dist,
            nmodes,
            mich,
            init_len,
            self.num_pts,
            dt,
            interp_t,
        )

        return
