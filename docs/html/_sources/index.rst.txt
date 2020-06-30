.. few documentation master file, created by
   sphinx-quickstart on Sun Jun 28 21:23:34 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. toctree::
   :maxdepth: 4
   :caption: Contents:


.. include:: README.rst

Documentation
=============

Fast EMRI Waveform models are built for fast evaluation of the EMRI waveform summation equation, given by (`Drasco & Hughes 2006 <https://arxiv.org/abs/gr-qc/0509101>`_),

.. math:: h_+ + ih_x=r^{-1}\sum_{lmkn}A_{lmkn}(t)e^{-i\Phi_{mkn}(t)}V_{lmkn}(\theta,\phi),
    :label: emri_wave_eq

where :math:`h_+ + ih_x` is the two polarizations of the gravitational wave; :math:`r` is the distance; :math:`l,m,k,n` are orbital angular momentum, azimuthal, polar, and radial harmonic indices, respectively; :math:`A_{lmkn}(t)` is a complex amplitude function at time :math:`t`; :math:`A(t)` is a complex amplitude function at time :math:`t`; :math:`\Phi_{mkn}(t)=m\omega_\phi + k\omega_\theta + n\omega_r` where :math:`\omega` is the fundamental frequency for a given direction; and :math:`V_{lmkn}(\theta,\phi`) is the harmonic projection functions based on viewing angles :math:`\theta,phi`. The various modules in this package deal with separate pieces of this equation. The amplitude (:mod:`few.amplitude`) module deals with :math:`A_{lmkn}(t)`. The trajectory (:mod:`few.trajectory`) module finds the orbital trajectories (:math:`p(t), e(t), \iota(t)`) which are fed into the amplitude module and phase trajectories (:math:`\Phi_\phi, \Phi_\theta, \Phi_r`) that are fed into the final summation. Utility modules aid this process from the utilities package (:mod:`few.utils`). Once such utility module is the calculation of :math:`V_{lmkn}(\theta,\phi`) (this is currently limited to the spin-weighted spherical harmonics for Schwarzschild eccentric). For completeness, :math:`A_{lmkn}` is given by,

.. math:: A_{lmkn} = -2Z_{lmkn}(a, p, e, \iota)/\omega_{mkn}^2,

where :math:`Z_{lmkn}(a, p, e, \iota)` is the teukolsky amplitudes associated with a geodesic with spin, :math:`a`; semilatus rectum, :math:`p`p; eccentricity, :math:`e`; and inclination, :math:`\iota`. :math:`\omega_{mkn}` is the summation of fundamental frequencies of the geodesic with integer weights equivalent to the mode indices. The angular function, :math:`V_{lmkn}` is given by,

.. math:: V_{lmkn}(\theta, \phi) = (s=-2)S_{lmkn}(\theta)e^{im\phi},

where :math:`S_{lmkn}(\theta)` is the general angular function from `Drasco & Hughes 2006 <https://arxiv.org/abs/gr-qc/0509101>`_.

When these modules complete their calculations, the summation (:mod:`few.summation`) module takes all of this information and creates an EMRI waveform.



Overall Waveform Models
------------------------

The Fast EMRI Waveform (few) package provides multiple complete models/waveforms to generate waveforms from start to finish. These are detailed in this section. Please note there are other modules available in each subpackage that may not be listed as a part of complete model here.

Schwarzschild Eccentric
~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: few.utils.baseclasses.SchwarzschildEccentric
    :members:
    :show-inheritance:

.. autoclass:: few.waveform.SchwarzschildEccentricWaveformBase
    :members:
    :show-inheritance:

.. autoclass:: few.waveform.FastSchwarzschildEccentricFlux
    :members:
    :show-inheritance:

.. autoclass:: few.waveform.SlowSchwarzschildEccentricFlux
    :members:
    :show-inheritance:


Trajectory Package
------------------

.. automodule:: few.trajectory

.. autoclass:: few.utils.baseclasses.TrajectoryBase
    :members:
    :show-inheritance:

Flux Inspiral
~~~~~~~~~~~~~~

.. automodule:: few.trajectory.flux
    :members:
    :show-inheritance:


Amplitude Package
-----------------

.. automodule:: few.amplitude

Schwarzschild Eccentric Amplitudes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

ROMAN Network
^^^^^^^^^^^^^^

.. automodule:: few.amplitude.romannet
    :members:
    :show-inheritance:

2D Cubic Spline Interpolation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: few.amplitude.interp2dcubicspline
    :members:
    :show-inheritance:

Summation Package
------------------

.. automodule:: few.summation

.. autoclass:: few.utils.baseclasses.SummationBase
    :members:
    :show-inheritance:

Interpolated Summation
~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: few.summation.interpolated_mode_sum
    :members:
    :show-inheritance:

Direct Summation
~~~~~~~~~~~~~~~~~~

.. automodule:: few.summation.direct_mode_sum
    :members:
    :show-inheritance:






Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
