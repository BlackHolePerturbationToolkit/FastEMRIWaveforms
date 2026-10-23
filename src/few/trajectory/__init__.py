r"""EMRI inspiral trajectories: initial orbital parameters to orbital-element and phase tracks.

A trajectory integrates the orbit-averaged equations of motion of the compact object,
:math:`(p(t), e(t), x_I(t))` and the three orbital phases
:math:`(\Phi_\phi, \Phi_\theta, \Phi_r)`, with an adaptive (or fixed-step) DOP853
Runge-Kutta integrator whose dense output is a continuous solution.

Modules:

- :mod:`few.trajectory.inspiral`: :class:`~few.trajectory.inspiral.EMRIInspiral`, the
  user-facing trajectory module (``traj(m1, m2, a, p0, e0, xI0, T=..., dt=...)``).
- :mod:`few.trajectory.integrate`: the integrator classes
  (:class:`~few.trajectory.integrate.Integrate` and its pex / ELQ subclasses) that step the
  ODE, stop it at the separatrix buffer or at ``T``, and evaluate the dense output.
- :mod:`few.trajectory.dopr853`: the DOP853 stepper and dense-output polynomials.
- :mod:`few.trajectory.ode`: the right-hand sides (:class:`~few.trajectory.ode.base.ODEBase`
  and the stock models ``KerrEccEqFlux``, ``SchwarzEccFlux``, ``PN5``).
- :mod:`few.trajectory.pool`: fan-out of independent trajectories to worker processes and
  a cache that serves them to a waveform generator.
- :mod:`few.trajectory.base`: :class:`~few.trajectory.base.TrajectoryBase`, the call
  interface (unit handling, optional upsampling).

The waveform generators of :mod:`few.waveform` build an
:class:`~few.trajectory.inspiral.EMRIInspiral` from their ``inspiral_kwargs`` (for
:class:`~few.waveform.waveform.FastKerrEccentricEquatorialFlux` the ODE is
``KerrEccEqFlux``). All trajectory code runs on the CPU (numpy / numba); no backend is used.
"""
