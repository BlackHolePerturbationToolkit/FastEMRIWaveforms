Overall Waveform Models
------------------------

The Fast EMRI Waveform (few) package provides multiple complete models/waveforms to generate waveforms from start to finish. These are detailed in this section. Please note there are other modules available in each subpackage that may not be listed as a part of a complete model here. We will also provide documentation in this section for the base classes that help standardize and build complete waveform models.

Generic Waveform Generator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: few.waveform.waveform.GenerateEMRIWaveform
    :members:
    :show-inheritance:
    :inherited-members:

Prebuilt Waveform Models
~~~~~~~~~~~~~~~~~~~~~~~~~~

Fast Kerr Eccentric Equatorial Flux-based Waveform
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: few.waveform.waveform.FastKerrEccentricEquatorialFlux
    :members:
    :show-inheritance:
    :inherited-members:

Fast Schwarzschild Eccentric Flux-based Waveform
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: few.waveform.waveform.FastSchwarzschildEccentricFlux
    :members:
    :show-inheritance:
    :inherited-members:

.. autoclass:: few.waveform.waveform.FastSchwarzschildEccentricFluxBicubic
    :members:
    :show-inheritance:
    :inherited-members:

Slow Schwarzschild Eccentric Flux-based Waveform
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: few.waveform.waveform.SlowSchwarzschildEccentricFlux
    :members:
    :show-inheritance:
    :inherited-members:

Generic Kerr AAK with 5PN Trajectory
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: few.waveform.waveform.Pn5AAKWaveform
    :members:
    :show-inheritance:
    :inherited-members:

Base Classes
~~~~~~~~~~~~~

Spherical Harmonic (Flux-based) Waveforms
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: few.utils.baseclasses.SphericalHarmonic
    :members:
    :show-inheritance:
    :inherited-members:

.. autoclass:: few.waveform.base.SphericalHarmonicWaveformBase
    :members:
    :show-inheritance:
    :inherited-members:

Schwarzschild Eccentric
^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: few.utils.baseclasses.SchwarzschildEccentric
    :members:
    :show-inheritance:
    :inherited-members:

Kerr Eccentric Equatorial
^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: few.utils.baseclasses.KerrEccentricEquatorial
    :members:
    :show-inheritance:
    :inherited-members:

5PN + AAK Waveform for Generic Kerr
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: few.utils.baseclasses.Pn5AAK
    :members:
    :show-inheritance:
    :inherited-members:

.. autoclass:: few.waveform.waveform.AAKWaveformBase
    :members:
    :show-inheritance:
    :inherited-members:

GPU Module Base Class
^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: few.utils.baseclasses.ParallelModuleBase
    :members:
    :show-inheritance:
    :inherited-members:
