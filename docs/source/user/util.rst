
.. _utilities-label:

Utilities
----------

.. automodule:: few.utils

:mod:`few.utils` is a namespace package (it has no ``__init__`` module): its modules are
documented below. The configuration, global state, citation and exception modules of
``few.utils`` are documented in the :doc:`API reference <../dev/api>`.

Mode Filtering
~~~~~~~~~~~~~~~

.. automodule:: few.utils.modeselector
    :show-inheritance:
    :members:
    :inherited-members:

(-2) Spin-Weighted Spherical Harmonics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: few.utils.ylm
    :show-inheritance:
    :members:
    :inherited-members:

Kerr Geodesics
~~~~~~~~~~~~~~~

.. automodule:: few.utils.geodesic
    :members:
    :show-inheritance:

Parameter Mappings
~~~~~~~~~~~~~~~~~~~

.. automodule:: few.utils.mappings

Kerr eccentric equatorial grid coordinates
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The superseded ``kerrecceq_legacy_p_to_u`` of this module is not documented here; use
:func:`~few.utils.mappings.kerrecceq.kerrecceq_forward_map`.

.. automodule:: few.utils.mappings.kerrecceq
    :members:
    :show-inheritance:
    :exclude-members: kerrecceq_legacy_p_to_u

Flux Jacobian
^^^^^^^^^^^^^^

.. automodule:: few.utils.mappings.jacobian
    :members:
    :show-inheritance:

PN inclination mappings
^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: few.utils.mappings.pn
    :members:
    :show-inheritance:
    :inherited-members:

Schwarzschild eccentric grid coordinate
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: few.utils.mappings.schwarzecc
    :members:
    :show-inheritance:

Mass conversions
^^^^^^^^^^^^^^^^^

.. automodule:: few.utils.mappings.common
    :members:
    :show-inheritance:

Frequency Domain Utilities
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: few.utils.fdutils
    :show-inheritance:
    :members:
    :inherited-members:

Analysis and Other Tools
~~~~~~~~~~~~~~~~~~~~~~~~~

The legacy ``wrapper`` and ``pointer_adjust`` of this module duplicate
:func:`few.cutils.wrappers.wrapper` and :func:`few.cutils.wrappers.pointer_adjust` (which
the compiled modules use) and are not documented here.

.. automodule:: few.utils.utility
    :members:
    :show-inheritance:
    :inherited-members:
    :exclude-members: wrapper, pointer_adjust

Elliptic integrals (Legendre and Carlson symmetric forms)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: few.utils.elliptic
    :members:
    :show-inheritance:
    :inherited-members:

Physical Constants
~~~~~~~~~~~~~~~~~~~

.. automodule:: few.utils.constants
