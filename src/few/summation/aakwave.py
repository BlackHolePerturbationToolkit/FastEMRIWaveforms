"""AAK (Augmented Analytic Kludge) summation module (:class:`AAKSummation`)."""

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
    The backend kernel ``pyWaveform`` builds the waveform at every output sample from the
    trajectory's DOP853 dense-output coefficients.

    Args:
        force_backend: Backend name or object (see
            :class:`~few.utils.baseclasses.FewBackendConsumer`). Default ``None``.
        **kwargs: Forwarded to :class:`~few.summation.base.SummationBase` (``pad_output``,
            ``odd_len``, ``output_type``).
    """

    def __init__(self, force_backend: BackendLike = None, **kwargs):
        Pn5AAK.__init__(self)
        SummationBase.__init__(self, **kwargs, force_backend=force_backend)

    @property
    def waveform_generator(self):
        """callable: The backend's compiled AAK kernel (``pyWaveform`` of ``pyAAK``)."""
        return self.backend.pyWaveform

    @classmethod
    def supported_backends(cls):
        """Backend preference order: ``["cuda13x", "cuda12x", "cuda11x", "cpu"]``."""
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

        This function performs the AAK waveform summation and fills :attr:`waveform`
        in place. ``qS`` and ``qK`` within ``1e-6`` of a pole are moved ``1e-6`` away (with a
        warning). For an equatorial trajectory (:math:`|Y| = 1` at the first knot) the
        :math:`\Phi_\theta` coefficients are replaced by the :math:`\Phi_\phi` ones, in place
        in ``interp_coeffs``.

        **Please note:** the 5PN trajectory and AAK waveform take the parameter
        :math:`Y\equiv\cos{\iota}=L/\sqrt{L^2 + Q}` rather than :math:`x_I` as is accepted
        for relativistic waveforms.

        Args:
            tvec: Trajectory knot times in seconds.
            m1: Mass in solar masses that sets the frequency scale of the kludge;
                :class:`~few.waveform.base.AAKWaveformBase` passes the total mass
                :math:`M = m_1 + m_2`.
            a: Dimensionless spin of the primary.
            dist: Luminosity distance in Gpc.
            m2: Mass in solar masses that sets the amplitude scale;
                :class:`~few.waveform.base.AAKWaveformBase` passes the reduced mass
                :math:`\mu = m_1 m_2 / (m_1 + m_2)`.
            qS: Sky polar angle in rad, SSB ecliptic frame by FEW convention.
            phiS: Sky azimuthal angle in rad.
            qK: Polar angle of the primary's spin direction in rad.
            phiK: Azimuthal angle of the primary's spin direction in rad.
            nmodes: Number of harmonics to sum (set from the eccentricity by the caller).
            interp_t: Knot times of the trajectory spline in seconds
                (``integrator_spline_t``), shape ``(N_p,)``.
            interp_coeffs: DOP853 dense-output coefficients of
                ``(p, e, Y, Phi_phi, Phi_theta, Phi_r)`` with the phase rows in rad, shape
                ``(N_p - 1, 6, 8)``; modified in place for equatorial trajectories.
            *args: Accepted and ignored.
            mich: If ``True``, produce the long-wavelength response channels
                :math:`h_I, h_{II}` (not TDI); if ``False`` (default), :math:`h_+, h_\times`.
            dt: Sampling interval in seconds. Default ``10.0``.
            integrate_backwards: If ``True``, shift ``interp_t`` by
                ``tvec[-1] - int(tvec[-1] / dt) * dt``. Default ``False``.
            **kwargs: Accepted and ignored.
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
