Overall Waveform Models
------------------------

The Fast EMRI Waveform (few) package provides multiple complete models/waveforms to generate waveforms from start to finish. These are detailed in this section. Please note there are other modules available in each subpackage that may not be listed as a part of a complete model here. We will also provide documentation in this section for the base classes that help standardize and build complete waveform models.

Prebuilt Waveform Models
~~~~~~~~~~~~~~~~~~~~~~~~~~

Fast Schwarzschild Eccentric Flux-based Waveform
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: few.waveform.FastSchwarzschildEccentricFlux
    :members:
    :show-inheritance:
    :inherited-members:

Slow Schwarzschild Eccentric Flux-based Waveform
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: few.waveform.SlowSchwarzschildEccentricFlux
    :members:
    :show-inheritance:
    :inherited-members:

Generic Kerr AAK with 5PN Trajectory
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: few.waveform.Pn5AAKWaveform
    :members:
    :show-inheritance:
    :inherited-members:

Base Classes
~~~~~~~~~~~~~

General Waveform Base
^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: few.utils.baseclasses.WaveformBase
    :members:
    :show-inheritance:
    :inherited-members:

Schwarzschild Eccentric
^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: few.utils.baseclasses.SchwarzschildEccentric
    :members:
    :show-inheritance:
    :inherited-members:

.. autoclass:: few.waveform.SchwarzschildEccentricWaveformBase
    :members:
    :show-inheritance:
    :inherited-members:

5PN + AAK Waveform for Generic Kerr
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: few.utils.baseclasses.Pn5AAK
    :members:
    :show-inheritance:
    :inherited-members:
