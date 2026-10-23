

.. _trajectory-label:

Trajectory Package
------------------

.. automodule:: few.trajectory

Trajectory Call Interface
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: few.trajectory.base
    :members:
    :show-inheritance:
    :inherited-members:

Generic Inspiral Generator
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: few.trajectory.inspiral
    :members:
    :show-inheritance:
    :inherited-members:
    :exclude-members: get_rhs_ode

ODE Integrators
~~~~~~~~~~~~~~~~~

.. automodule:: few.trajectory.integrate
    :members:
    :show-inheritance:
    :inherited-members:

Dormand-Prince 8(5,3) Stepper
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The legacy self-test pieces of this module (``ODE_pendulum``, ``stopping_criterion`` and the
``DOPR853.integrate`` batch loop) do not run as written and are not documented here.

.. automodule:: few.trajectory.dopr853
    :members:
    :show-inheritance:
    :exclude-members: ODE_pendulum, stopping_criterion, integrate

Trajectory Pool and Cache
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: few.trajectory.pool
    :members:
    :show-inheritance:

ODE Classes
~~~~~~~~~~~~~~~

.. automodule:: few.trajectory.ode

.. automodule:: few.trajectory.ode.base
    :members:
    :show-inheritance:
    :inherited-members:

The superseded ``KerrEccEqFluxLegacy`` class of this module (not constructible: its
read-only ``flux_output_convention`` property clashes with the
:class:`~few.trajectory.ode.base.ODEBase` constructor; data files tagged ``deprecated``) is not
documented here; :class:`~few.trajectory.ode.flux.KerrEccEqFlux` replaces it.

.. automodule:: few.trajectory.ode.flux
    :members:
    :show-inheritance:
    :exclude-members: KerrEccEqFluxLegacy

.. automodule:: few.trajectory.ode.pn5
    :members:
    :show-inheritance:
