r"""
The amplitude package houses modules for calculating Teukolsky amplitudes from
orbital parameter trajectories. The complex amplitudes are determined with
various methods documented below. Some amplitude modules have gpu capabilities.
It deals with :math:`A_{lmkn}` in :eq:`emri_wave_eq`.

Modules:

- :mod:`few.amplitude.base`: :class:`~few.amplitude.base.AmplitudeBase`, the common call
  interface (mode-index handling and the :math:`\pm m` / :math:`x_I` symmetries).
- :mod:`few.amplitude.ampinterp2d`: spline interpolation of precomputed amplitude grids;
  :class:`~few.amplitude.ampinterp2d.AmpInterpKerrEccEq` (Kerr eccentric equatorial, used by
  :class:`~few.waveform.waveform.FastKerrEccentricEquatorialFlux`) and
  :class:`~few.amplitude.ampinterp2d.AmpInterpSchwarzEcc` (Schwarzschild eccentric).
- :mod:`few.amplitude.romannet`: :class:`~few.amplitude.romannet.RomanAmplitude`, the ROMAN
  neural-network amplitudes for Schwarzschild eccentric orbits.
"""
