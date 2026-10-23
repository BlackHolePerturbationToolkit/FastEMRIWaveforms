Overall Waveform Models
------------------------

The Fast EMRI Waveform (few) package provides multiple complete models/waveforms to generate waveforms from start to finish. These are detailed in this section. Please note there are other modules available in each subpackage that may not be listed as a part of a complete model here. We will also provide documentation in this section for the base classes that help standardize and build complete waveform models.

.. automodule:: few.waveform

.. automodule:: few.waveform.waveform

Generic Waveform Generator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: few.waveform.waveform.GenerateEMRIWaveform
    :members:
    :private-members: _to_SSB_frame
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

With ``return_sparse_holder=True`` the spherical-harmonic models return the mode-selected
ingredients of the mode sum at the trajectory knots instead of the summed waveform:

.. autoclass:: few.waveform.base.SparseInfoHolder

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

.. automodule:: few.waveform.base

.. automodule:: few.utils.baseclasses

Spherical Harmonic (Flux-based) Waveforms
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: few.utils.baseclasses.SphericalHarmonic
    :members:
    :show-inheritance:
    :inherited-members:

The call body of the spherical-harmonic models is the private
:meth:`~few.waveform.base.SphericalHarmonicWaveformBase._generate_waveform`, documented with
its class:

.. autoclass:: few.waveform.base.SphericalHarmonicWaveformBase
    :members:
    :private-members: _generate_waveform
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

.. autoclass:: few.waveform.base.AAKWaveformBase
    :members:
    :show-inheritance:
    :inherited-members:

Backend Base Class
^^^^^^^^^^^^^^^^^^^^

Every FEW module that runs on a compute backend (CPU or CUDA) derives from this class; its
``force_backend`` argument selects the backend (see the :doc:`configuration guide <cfg>`).

.. autoclass:: few.utils.baseclasses.FewBackendConsumer
    :members:
    :show-inheritance:
    :inherited-members:
