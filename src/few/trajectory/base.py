"""Call interface shared by the trajectory modules.

:class:`TrajectoryBase` turns a call ``traj(m1, m2, ..., T=..., dt=...)`` into a
``get_inspiral`` call of the concrete module (:class:`few.trajectory.inspiral.EMRIInspiral`),
handles the units of the output time and of ``max_step_size``, and optionally resamples the
result on a uniform or user grid through the integrator's dense output.
"""

import abc
from typing import Optional

import numpy as np

from few.utils.citations import Citable
from few.utils.constants import MTSUN_SI, YRSID_SI


class TrajectoryBase(Citable, abc.ABC):
    """Base class of the trajectory modules.

    Subclasses implement :meth:`get_inspiral`; :meth:`__call__` provides the common call
    interface (output time units, step-size limit, optional upsampling). The only concrete
    subclass in FEW is :class:`few.trajectory.inspiral.EMRIInspiral`, whose docstring lists
    the model arguments.

    Args:
        *args: Accepted and ignored.
        **kwargs: Accepted and ignored.
    """

    def __init__(self, *args, **kwargs):
        pass

    @classmethod
    @abc.abstractmethod
    def get_inspiral(self, *args, **kwargs):
        """Integrate the inspiral (implemented by subclasses).

        Called by :meth:`__call__` with the call's positional arguments (``m1``, ``m2`` in
        solar masses first) and keyword arguments, plus the keys ``dt`` (s), ``T``
        (years), ``err``, ``DENSE_STEPPING``, ``buffer_length``, ``integrate_backwards`` and
        ``max_step_size`` (in the integrator's time unit) that :meth:`__call__` adds.

        Args:
            *args: Model parameters.
            **kwargs: Call keywords (see above).

        Returns:
            tuple[np.ndarray, ...]: The time array in seconds first, then the trajectory
            arrays.

        Raises:
            NotImplementedError: If the subclass does not implement it.
        """
        raise NotImplementedError

    def __call__(
        self,
        *args,
        in_coordinate_time: bool = True,
        dt: float = 10.0,
        T: float = 1.0,
        new_t: Optional[np.ndarray] = None,
        spline_kwargs: Optional[dict] = None,
        DENSE_STEPPING: bool = False,
        buffer_length: int = 1000,
        upsample: bool = False,
        err: float = 1e-11,
        fix_t: bool = False,
        integrate_backwards: bool = False,
        max_step_size: Optional[float] = None,
        **kwargs,
    ) -> tuple[np.ndarray]:
        """Integrate a trajectory and return its time and parameter arrays.

        Packs the call keywords for :meth:`get_inspiral`, converts ``max_step_size`` to the
        integrator's time unit, runs the inspiral and converts the output time. With
        ``upsample=True`` the integrator's dense output is evaluated on ``new_t`` instead of
        returning the integrator's own (adaptive) knots. Exceptions of :meth:`get_inspiral`
        propagate (see :meth:`few.trajectory.inspiral.EMRIInspiral.get_inspiral`).

        Args:
            *args: Model parameters, forwarded to :meth:`get_inspiral`. The first two must
                be the masses ``m1``, ``m2`` in solar masses (they set the time units); for
                :class:`~few.trajectory.inspiral.EMRIInspiral` the full list is
                ``(m1, m2, a, p0, e0, xI0, *ode_args)``.
            in_coordinate_time: If ``True``, return the time in seconds and read
                ``max_step_size`` in seconds; if ``False``, return the time in units of
                :math:`M` (``t / (M * MTSUN_SI)`` with ``M = m1 + m2``) and read
                ``max_step_size`` in units of :math:`M`. Ignored by the ``upsample`` output,
                which is always in seconds. Default ``True``.
            dt: Time step in seconds: the fixed integrator step when
                ``DENSE_STEPPING=True`` and the spacing of the default ``upsample`` grid.
                With adaptive stepping (the default) the initial step is tuned automatically
                and ``dt`` does not affect the knots. Default ``10.0``.
            T: Maximum duration in years (``YRSID_SI``); the integration stops at ``T`` or
                at the separatrix buffer, whichever comes first. Default ``1.0``.
            new_t: Times in seconds at which to evaluate the trajectory when
                ``upsample=True`` (must be a ``numpy.ndarray``). Default ``None`` means
                ``np.arange(int(T / dt)) * dt`` with ``T`` in seconds. Ignored otherwise.
            spline_kwargs: Accepted and ignored (upsampling uses the integrator's dense
                output, not scipy). Default ``None``.
            DENSE_STEPPING: If ``True``, take fixed steps of ``dt`` without error control
                (no dense output is built). Default ``False``.
            buffer_length: Forwarded to :meth:`get_inspiral`. For
                :class:`~few.trajectory.inspiral.EMRIInspiral` it has no effect: the
                integrator uses its own buffer length (the
                :class:`~few.trajectory.integrate.Integrate` constructor value, default 10000
                rows, grown by 100 rows whenever an integration fills it). Default ``1000``.
            upsample: If ``True``, return the dense-output trajectory on ``new_t``; samples
                after the end of the trajectory are zero. Not supported with
                ``DENSE_STEPPING``. Default ``False``.
            err: Absolute tolerance of the DOP853 error control. Smaller values give more
                knots. Default ``1e-11``.
            fix_t: With ``upsample``, truncate the output at the first sample past the end
                of the trajectory, detected as the first sample whose second output row
                (``e`` for :class:`~few.trajectory.inspiral.EMRIInspiral`) is exactly zero;
                a trajectory whose ``e`` stays exactly zero (circular) is therefore
                truncated to nothing. Default ``False``.
            integrate_backwards: If ``True``, integrate backwards in time from the given
                (final) orbital parameters. Default ``False``.
            max_step_size: Largest integrator step, in seconds or in units of :math:`M`
                (see ``in_coordinate_time``). Default ``None`` means no limit.
            **kwargs: Forwarded to :meth:`get_inspiral` (for
                :class:`~few.trajectory.inspiral.EMRIInspiral`: ``Phi_phi0``,
                ``Phi_theta0``, ``Phi_r0``; any other key is ignored there).

        Returns:
            tuple[np.ndarray, ...]: ``(t,)`` followed by the :meth:`get_inspiral` outputs
            after its time array. For :class:`~few.trajectory.inspiral.EMRIInspiral`:
            ``(t, p, e, x, Phi_phi, Phi_theta, Phi_r)``, 1-D arrays of the same length
            (``t`` in s or :math:`M`, ``p`` in units of :math:`M`, phases in rad); with
            ``upsample`` the arrays are sampled on ``new_t``.

        Raises:
            ValueError: With ``upsample``, if ``new_t`` is given and is not a
                ``numpy.ndarray``.
            NotImplementedError: If ``upsample`` and ``DENSE_STEPPING`` are both set.
        """
        if spline_kwargs is None:
            spline_kwargs = {}

        # add call kwargs to kwargs dictionary
        kwargs["dt"] = dt
        kwargs["T"] = T
        kwargs["buffer_length"] = buffer_length
        kwargs["err"] = err
        kwargs["DENSE_STEPPING"] = DENSE_STEPPING
        kwargs["integrate_backwards"] = integrate_backwards

        # convert from years to seconds
        T = T * YRSID_SI

        # m1, m2 must be the first arguments
        m1, m2 = args[:2]
        mu = m1 * m2 / (m1 + m2)
        M = m1 + m2
        Msec = M * MTSUN_SI

        if max_step_size is None:
            max_step_size = np.inf

        if in_coordinate_time:
            kwargs["max_step_size"] = max_step_size * (mu / M) / Msec
        else:
            # convert max_step_size to coordinate time
            if max_step_size is not None:
                kwargs["max_step_size"] = max_step_size * (mu / M)

        # inspiral generator that must be added to each trajectory class
        out = self.get_inspiral(*args, **kwargs)

        # get time separate from the rest of the params
        t = out[0]
        params = out[1:]

        # convert to dimensionless time
        if in_coordinate_time is False:
            t = t / Msec

        if not upsample:
            return (t,) + params
        else:
            if DENSE_STEPPING:
                raise NotImplementedError  # TODO: support cubic spline for dense

        if new_t is not None:
            if isinstance(new_t, np.ndarray) is False:
                raise ValueError("new_t parameter, if provided, must be numpy array.")

        else:
            # new time array for upsampling
            new_t = np.arange(int(T / dt)) * dt

        # upsample everything
        upsamp_traj = self.inspiral_generator.eval_integrator_spline(new_t).T
        if fix_t:
            if np.any(upsamp_traj[1] == 0):
                trunc_ind = np.where(upsamp_traj[1] == 0)[0][0]
                upsamp_traj = upsamp_traj[:, :trunc_ind]
                new_t = new_t[:trunc_ind]
        out = tuple(upsamp_traj)

        return (new_t,) + out
