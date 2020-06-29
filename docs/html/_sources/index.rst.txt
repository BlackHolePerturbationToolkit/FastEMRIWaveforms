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





Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
