# API Reference

This page documents the package infrastructure: the package entry points, global state,
configuration, file manager, compute backends, exceptions, citations and command-line tools.
The physics modules are documented in the user guide: waveform models
({doc}`../user/main`), trajectories ({doc}`../user/traj`), amplitudes ({doc}`../user/amp`),
mode summation ({doc}`../user/sum`) and utilities ({doc}`../user/util`).

## Package

The four accessors re-exported by `few` (defined in {mod}`few.utils.globals`) are documented
here.

```{eval-rst}
.. automodule:: few
    :members: get_logger, get_config, get_config_setter, get_file_manager
```

## Global state

The `__all__` of this module also lists `get_backend`, `get_first_backend` and `has_backend`,
which it does not define; backends are looked up through `gpubackendtools` (see the
"Backends management" section of {doc}`../user/cfg`).

```{eval-rst}
.. automodule:: few.utils.globals
    :members:
    :show-inheritance:
    :ignore-module-all:
    :exclude-members: get_logger, get_config, get_config_setter, get_file_manager
```

## Configuration

```{eval-rst}
.. automodule:: few.utils.config
    :members:
    :show-inheritance:
    :inherited-members:
```

## File manager

```{eval-rst}
.. automodule:: few.files
    :members:
    :show-inheritance:

.. automodule:: few.files.manager
    :members:
    :show-inheritance:
    :exclude-members: FileManager

.. automodule:: few.files.registry
    :members:
    :show-inheritance:
    :exclude-members: FileRegistry
```

## Compute backends

```{eval-rst}
.. automodule:: few.cutils
    :members:
    :show-inheritance:

.. automodule:: few.cutils.wrappers
    :members:
```

## Exceptions

The never-raised `CudaException`, `CuPyException` and `MissingDependency` of this module are
not documented here (backend errors come from `gpubackendtools`).

```{eval-rst}
.. automodule:: few.utils.exceptions
    :members:
    :show-inheritance:
    :exclude-members: CudaException, CuPyException, MissingDependency
```

## Citations

```{eval-rst}
.. automodule:: few.utils.citations
    :members:
    :show-inheritance:
```

## Command-line tools

The console scripts `few_files` and `few_citations` (declared in `pyproject.toml`).

```{eval-rst}
.. automodule:: few.cmd.files
    :members:

.. automodule:: few.cmd.citations
    :members:
```
