
FastEMRIWaveforms Publication
==============================

Fast EMRI Waveform models are built for fast evaluation of the EMRI waveform summation equation, given by (`Drasco & Hughes 2006 <https://arxiv.org/abs/gr-qc/0509101>`_),

.. math:: h_+ + ih_x=r^{-1}\sum_{lmkn}A_{lmkn}(t)e^{-i\Phi_{mkn}(t)}V_{lmkn}(\theta,\phi),
    :label: emri_wave_eq

where :math:`h_+ + ih_x` is the two polarizations of the gravitational wave; :math:`r` is the distance; :math:`l,m,k,n` are orbital angular momentum, azimuthal, polar, and radial harmonic indices, respectively; :math:`A_{lmkn}(t)` is a complex amplitude function at time :math:`t`; :math:`A(t)` is a complex amplitude function at time :math:`t`; :math:`\Phi_{mkn}(t)=m\omega_\phi + k\omega_\theta + n\omega_r` where :math:`\omega` is the fundamental frequency for a given direction; and :math:`V_{lmkn}(\theta,\phi`) is the harmonic projection functions based on viewing angles :math:`\theta,phi`. The various modules in this package deal with separate pieces of this equation. The amplitude (:mod:`few.amplitude`) module deals with :math:`A_{lmkn}(t)`. The trajectory (:mod:`few.trajectory`) module finds the orbital trajectories (:math:`p(t), e(t), \iota(t)`) which are fed into the amplitude module and phase trajectories (:math:`\Phi_\phi, \Phi_\theta, \Phi_r`) that are fed into the final summation. Utility modules aid this process from the utilities package (:mod:`few.utils`). Once such utility module is the calculation of :math:`V_{lmkn}(\theta,\phi`) (this is currently limited to the spin-weighted spherical harmonics for Schwarzschild eccentric). For completeness, :math:`A_{lmkn}` is given by,

.. math:: A_{lmkn} = -2Z_{lmkn}(a, p, e, \iota)/\omega_{mkn}^2,

where :math:`Z_{lmkn}(a, p, e, \iota)` is the teukolsky amplitudes associated with a geodesic with spin, :math:`a`; semilatus rectum, :math:`p`p; eccentricity, :math:`e`; and inclination, :math:`\iota`. :math:`\omega_{mkn}` is the summation of fundamental frequencies of the geodesic with integer weights equivalent to the mode indices. The angular function, :math:`V_{lmkn}` is given by,

.. math:: V_{lmkn}(\theta, \phi) = (s=-2)S_{lmkn}(\theta)e^{im\phi},

where :math:`S_{lmkn}(\theta)` is the general angular function from `Drasco & Hughes 2006 <https://arxiv.org/abs/gr-qc/0509101>`_.

When these modules complete their calculations, the summation (:mod:`few.summation`) module takes all of this information and creates an EMRI waveform.

Our model is valid at any :math:`(p_0,e_0)` that falls outside the separatrix, below the :math:`(p_0=10,e_0=0.7)` trajectory (see below trjectory figure), and within the outer limits (:math:`p_0 < 16 + 2e_0` and :math:`e_0 < 0.7`).

Below are figures from our paper related to this package.

.. figure:: img/EMRI_diagram.jpg
    :width: 500px
    :align: center
    :height: 500px
    :alt: alternate text
    :figclass: align-center

    A general schematic diagram for the creation of an EMRI waveform.


.. figure:: img/traj.jpg
    :width: 500px
    :align: center
    :height: 400px
    :alt: alternate text
    :figclass: align-center


    Evolution of mismatch between fast and fiducial waveforms from :math:`(p_0,e_0)` to :math:`(p,e)`, for 12 EMRIs with :math:`M=10^6M_\odot`, :math:`\mu\in[15,304]M_\odot`, and :math:`(p_0,e_0)` along the model domain boundary. Each small mass is chosen such that the EMRI plunges after a year. These results are for :math:`(\theta,\varphi)=(\pi/2,0)`, but do not depend strongly on the viewing angle. In the worst case (top-left curve), the final 0.01% of the waveform causes the mismatch to increase from under :math:`4\times10^{-4}` to :math:`5\times10^{-4}`.


.. figure:: img/waveform_example.jpg
    :width: 500px
    :align: center
    :height: 350px
    :alt: alternate text
    :figclass: align-center

    Six-hour snapshots of fast (orange) and fiducial (blue) waveforms, one year before plunge (top) and just before plunge (bottom). Waveforms are for the worst-case EMRI :math:`(M,\mu,p_0,e_0)=(10^{6}M_\odot, 15 M_\odot, 10, 0.7)`, with a one-year mismatch of :math:`5\times 10^{-4}`. Small amplitude deviations are visible just before plunge at :math:`(p,e)\approx(7,0.5)`, where the mode-distribution error approaches its maximum across the domain of validity.


.. figure:: img/timing_plot_3.jpg
    :width: 400px
    :align: center
    :height: 400px
    :alt: alternate text
    :figclass: align-center

    Computational wall time for fast and fiducial waveforms, broken down into individual modules. All times are averaged over :math:`\geq5` evaluations of the worst-case waveform on a single CPU core (and GPU), where the CPU is an Intel Xeon Gold 6132 and the GPU is an NVIDIA Tesla V100.

Package TODOs
===============

- run trajectory backward
- add initial phases
- zero out modes
- shared memory based on CUDA_ARCH / upping shared allocation
- deal with file locations and removing files from git history
- add tutorials to documentation
- general waveform base class
- more automatic/generic download from zenodo based on versioning
- add benchmarking function
- add function to determine parameter that will give waveform for :math:`t` years
