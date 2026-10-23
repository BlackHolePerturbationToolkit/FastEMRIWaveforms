"""
The summation package houses codes used for combining amplitude and phase
information over the course of an inspiral. It does contain GPU accelerated
options. It deals with the overall summation of quantities seen in :eq:`emri_wave_eq`.

Modules:

- :mod:`few.summation.base`: :class:`~few.summation.base.SummationBase`, the common call
  (output length, zero padding, output buffer).
- :mod:`few.summation.interpolatedmodesum`:
  :class:`~few.summation.interpolatedmodesum.InterpolatedModeSum`, the time-domain sum from
  the sparse trajectory (amplitude splines plus the trajectory's phase spline), and the
  batched :class:`~few.summation.interpolatedmodesum.CubicSplineInterpolant`.
- :mod:`few.summation.fdinterp`: :class:`~few.summation.fdinterp.FDInterpolatedModeSum`, the
  frequency-domain (stationary-phase) sum.
- :mod:`few.summation.directmodesum`: :class:`~few.summation.directmodesum.DirectModeSum`,
  the direct sum over a dense trajectory.
- :mod:`few.summation.aakwave`: :class:`~few.summation.aakwave.AAKSummation`, the AAK
  kludge waveform.
"""
