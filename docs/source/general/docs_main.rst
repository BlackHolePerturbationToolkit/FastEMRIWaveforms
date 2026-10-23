
FastEMRIWaveforms Publications
================================

The Fast EMRI Waveforms papers can be found at `arxiv.org/2104.04582 <https://arxiv.org/abs/2104.04582>`_ and `arxiv.org/2008.06071 <https://arxiv.org/abs/2008.06071>`_. Below is a brief description of the model. Please refer to the paper for plots and more detailed information.

Fast EMRI Waveform models are built for fast evaluation of the EMRI waveform summation equation, given by (`Drasco & Hughes 2006 <https://arxiv.org/abs/gr-qc/0509101>`_),

.. math:: h_+ - ih_\times=r^{-1}\sum_{lmkn}A_{lmkn}(t)e^{-i\Phi_{mkn}(t)}V_{lmkn}(\theta,\phi),
    :label: emri_wave_eq

where :math:`h_+ - ih_\times` combines the two polarizations of the gravitational wave (the convention of the FEW waveform output: real part :math:`h_+`, imaginary part :math:`-h_\times`); :math:`r` is the distance; :math:`l,m,k,n` are orbital angular momentum, azimuthal, polar, and radial harmonic indices, respectively; :math:`A_{lmkn}(t)` is a complex amplitude function at time :math:`t`; :math:`\Phi_{mkn}(t)=m\Phi_\phi(t) + k\Phi_\theta(t) + n\Phi_r(t)` where :math:`\Phi_\phi, \Phi_\theta, \Phi_r` are the orbital phases (the time integrals of the fundamental frequencies); and :math:`V_{lmkn}(\theta,\phi)` is the harmonic projection functions based on viewing angles :math:`\theta,\phi`. The various modules in this package deal with separate pieces of this equation. The amplitude (:mod:`few.amplitude`) module deals with :math:`A_{lmkn}(t)`. The trajectory (:mod:`few.trajectory`) module finds the orbital trajectories (:math:`p(t), e(t), x_I(t)`) which are fed into the amplitude module and phase trajectories (:math:`\Phi_\phi, \Phi_\theta, \Phi_r`) that are fed into the final summation. Utility modules aid this process from the utilities package (:mod:`few.utils`). Once such utility module is the calculation of :math:`V_{lmkn}(\theta,\phi)`: the spherical-harmonic models (Kerr eccentric equatorial and Schwarzschild eccentric) use the :math:`-2` spin-weighted spherical harmonics (:class:`~few.utils.ylm.GetYlms`), their amplitudes being given in that basis. For completeness, :math:`A_{lmkn}` is given by,

.. math:: A_{lmkn} = -2Z_{lmkn}(a, p, e, x_I)/\omega_{mkn}^2,

where :math:`Z_{lmkn}(a, p, e, x_I)` is the Teukolsky amplitudes associated with a geodesic with spin, :math:`a`; semilatus rectum, :math:`p`; eccentricity, :math:`e`; and inclination, :math:`x_I`. :math:`\omega_{mkn}` is the summation of fundamental frequencies of the geodesic with integer weights equivalent to the mode indices. The angular function, :math:`V_{lmkn}` is given by,

.. math:: V_{lmkn}(\theta, \phi) = (s=-2)S_{lmkn}(\theta)e^{im\phi},

where :math:`S_{lmkn}(\theta)` is the general angular function from `Drasco & Hughes 2006 <https://arxiv.org/abs/gr-qc/0509101>`_ (spin-weighted spheroidal harmonics).

When these modules complete their calculations, the summation (:mod:`few.summation`) module takes all of this information and creates an EMRI waveform.

Package TODOs
===============

- add SNR calculator
- zero out modes
- shared memory based on CUDA_ARCH / upping shared allocation
- deal with file locations and removing files from git history
- add benchmarking function

Change Log
===========

This log stops at 1.5.5; for later releases (2.x) see the
`tags of the repository <https://github.com/BlackHolePerturbationToolkit/FastEMRIWaveforms/tags>`_.

- 1.5.5: pip installable.
- 1.5.4: Slight bug fix on creating ``few`` file directory.
- 1.5.3: openmp removed from change file. Temporary removal of zenodo record dictionary.
- 1.5.1: Added FD waveform. Removed an ``exp`` computation. Made module pickeable. Updated install.
- 1.4.10: M1 installation and small bug fixes.
- 1.4.9: Fixed omp issue.
- 1.4.8: Throwing python errors from C++. Separatrix bug fix.
- 1.4.7: Updates to readme and small fixes.
- 1.4.6: A quick bug fix for GPU device issues.
- 1.4.5: Separatrix c function generalized for Schwarzschild, generic Kerr, and circular, equatorial Kerr.
- 1.4.4: Bug fix at zero eccentricity. Frequencies back in ODE. Fix for git pull issue with ode_base files.
- 1.4.3: Bug fixes for additional arguments in SchEcc waveform base.
- 1.4.2: Bug fixes for additional arguments in AAK waveform.
- 1.4.1: Bug fixes.
- 1.4.0: Ability to access OMP threads. Set CUDA device. Change fundamental frequency files to "utility". Initial error handler. Trajectory overhaul. get_at_t function updates.
- 1.3.7: Fixed get_at_t functions. Added new GPU architecture. Removed oldest architecture. Fixed issue #30 & #32.
- 1.3.6: Fixed Y0 < 0.0 in new AAK.
- 1.3.5: Interpolation updated for 2d and bug fixes on derivatives.
- 1.3.4: ccbin option added to setup.py.
- 1.3.3: More bug fixes in Pn5 stepping over separatrix.
- 1.3.2: 2 bug fixes in Pn5. Integrator stepping past separatrix. Randomly finding Nan returns from fundamental frequencies.
- 1.3.1: Bug fix for root finding in xI. Y_to_xI in pn5 codes.
- 1.3.0: x implemented instead of Y for cosine of inclination angle (includes conversion functions now). Python interface for KerrGeoConstantsOfMotion. Bug fix in examples. Updated citations. Changed constants to match LDC. More freedom in selecting specific modes.
- 1.2.2: Bug fix for p(t), lapacke. Added e=0 for FastSchwarzschildEccentricFlux. Fixed major bug in 5PN trajectory.
- 1.2.1: Bug fix for time at end of orbital evolution. :math:`\Omega_\theta` fixed.
- 1.2.0: Generic waveform interface added. Angular protections added to AAK.
- 1.1.5: Distance bug fixed. Wrong mass scale.
- 1.1.4: Distance added to FEW.
- 1.1.3: Schwarzschild eccentric fundamental frequencies added. Flux inspiral structure adjusted to this change. Change log added.
- 1.1.2: Memory leak on GPU corrected.
- 1.1.1: wget and lapack issues fixed.
- 1.1.0: New AAK was added.
