# Configuration options

The `few` package can be tuned through the use of optional configuration options which can be defined by either:

- A `few.ini` configuration file
- Environment variables
- Command-line arguments when using a compatible command-line utility
- A Python object named the `ConfigurationSetter` which can be used right after importing the `few` module for the first time

The `few.ini` file is searched, in priority order:

- In your current working directory
- In your [`platformdirs.user_config_path()`](https://github.com/tox-dev/platformdirs/blob/main/README.rst) (e.g. `~/.config/few.ini`)
- In your [`platformdirs.site_config_path()`](https://github.com/tox-dev/platformdirs/blob/main/README.rst) (e.g. `/etc/xdg/few/v1.5/few.ini`)

The path to `few.ini` can be enforced by the `FEW_CONFIG_FILE` environment variable, or in command-line contexts, by the `-C` or `--config-file` argument.

## Tunable utilities

These configuration options act on

### Log level

`few` handles logging of operations using the python `logging` standard module.
Its logger can be accessed either directly through a call to

```py3
import logging
logger = logging.getLogger("few")
```

or by accessing `few` global states with

```py3
import few.utils.globals

logger = few.utils.globals.get_logger()
```

The default log level is `logging.WARNING` but this can be modified through the `log_level` configuration option.

### Backends management

Backends are plugin-like entities which can be used to delegate heavy computations to specific hardware like GPUs.
By default, FEW will try to use the best available backend for each class that the user instanciates. It is however
possible to enforce or prevent the use of specific backends through the `enabled_backends` options.

The list of available backends is:

- `cpu`: Use the CPU itself for accelerated computations
- `cuda11x`: Use a NVIDIA GPU with CUDA 11.x drivers
- `cuda12x`: Use a NVIDIA GPU with CUDA 12.x drivers

By default, all these backends can be used provided they are installed and have required sotware and hardware available.
A class that supports only CPU will only attempt to use he `cpu` backend, whilst a class with hybrid GPU/CPU support will
(usually) first attempt to use the `cuda12x` backend, in case of failure it will attempt using the `cuda11x` and finally
fallback to the `cpu` one.

When setting the `enabled_backends` option, all items present in that list will be loaded (and FEW will fail initializing
if any in not loadable) while other items will be strictly disabled.

Example:

- On a computer with only a CPU, setting `enabled_backends = ['cpu']` will speed-up FEW loading process by disabling
  the `cuda11x` and `cuda12x` backends and not even try loading them
- On a computer with a NVIDIA GPU and CUDA 12.x drivers, setting `enabled_backends = ['cuda12x', 'cpu']` ensures that
  the CUDA 12.x backend will be loaded and that CPU only classes are still supported. If for any reason, the CUDA 12.x backend
  cannot be loaded, FEW will fail to run (and the message error should explain the failure and suggest mitigation strategies).


### File manager

The `few` package requires external files of pre-computed coefficients which are too large to be bundled with the source code.
These files are accessed through the `FileManager` global entity which takes care of locating these files, checking their integrity,
and downloading missing files on request.

The file manager is accessed through `few.utils.globals.get_file_manager()`

This `FileManager` is highly tunable and propose the following options:

#### `file_storage_path`

The *storage path* is the directory where the `FileManager` will first look for files.

Its default value is built relative to [`platformdirs.user_data_dir()`](https://github.com/tox-dev/platformdirs/blob/main/README.rst)
(e.g. `~/.local/share/few/v1.5.1` on Linux systems)

#### `file_download_path`

The *download path* is the directory where the `FileManager` will download missing files.
By default, it is a subdirectory of the *storage path* named `download`.

#### `file_extra_paths`

Extra paths are supplementary read-only directories where the file manager will search for requested files.
They are provided as a ";"-separated list of paths.

#### `file_registry_path`

The *file registry* is a YAML file which lists known files which can be downloaded from FEW data repositories.
It defines these data repositories and then, for each file, lists the repositories they can be found in, their tags and their checksums.

A default `registry.yml` file is embedded with each version of `few` but one can provide a custom registry by setting its path
through the `file_registry_path` option.

#### `file_integrity_check`

By default, the first time a file is requested to the `FileManager`, its integrity is checked and cached so that future request to that
file are fast.

This behaviour can be tuned to disable entirely integrity checks, or to perform these checks each and everytime a given file is requested.

The `file_integrity_check` option can thus take on the following values:

- `always`: perform the integrity check each and every time
- `once` (default): perform the integrity check once for the lifetime of the file manager
- `never`: never check for integrity (even on download)

#### `file_allow_download`

If a file is requested but not found locally, the file manager can (and by default will) attempt to download it
from data repositories defined in the file registry.

This behavior can be switched off if FEW needs to run offline (for instance, on compute node with filtered network access).

## Summary of configuration options

General configuration options:

| Option name | Config file entry | Environment variable | Command-line option | Authorized values | Comment |
|---|---|---|---|---|---|
| `log_level` | `log-level` | `FEW_LOG_LEVEL` | `--log-level` | `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL` |  |
| `log_format` | `log-format` | `FEW_LOG_FORMAT` | `--log-format` | Any format string supported by [`logging.Formatter`](https://docs.python.org/3/library/logging.html#logging.Formatter) |  |
| `enabled_backends` | `enbaled-backends` | `FEW_ENABLED_BACKENDS` | `--enable-backend` | `cpu`, `cuda11x`, `cuda12x` | ";"-separated list of values. The CLI parameter can be used multiple time for each enabled backend. |
| `file_registry_path` | `file-registry` | `FEW_FILE_REGISTRY` | `--file-registry` | Path to a `registry.yml` file |  |
| `file_storage_path` | `file-storage-dir` | `FEW_FILE_STORAGE_DIR` | `--storage-dir` | Absolute path, or relative to current working directory | Directory must already exist |
| `file_download_path` | `file-download-dir` | `FEW_FILE_DOWNLOAD_DIR` | `--download-dir` | Absolute path, or relative to the storage directory |  |
| `file_allow_download` | `file-allow-download` | `FEW_FILE_ALLOW_DOWNLOAD` | `--(no-)file-download` | Truthy (`yes`, `true`, `on`, `1`) or Falsy (`no`, `false`, `off`, `0`) value |  |
| `file_integrity_check` | `file-integrity-check` | `FEW_FILE_INTEGRITY_CHECK` | `--file-integrity-check` | `never`, `once`, `always`  |  |
| `file_extra_paths` | `file-extra-paths` | `FEW_FILE_EXTRA_PATHS` | `--extra-path` | ";"-separated list of directories | (1) |

(1) Extra paths are cumulated between option sources. *i.e.* the final options contains all paths from the config file, the environment variable and the CLI option.

In command line contexts, the help message will usually only contain the options specific to the utility you are currently running.
To check if a utility is compatible with the previous CLI options, use the `-H` or `--config-help` argument to obtain the help message
corresponding to those options.

One may also disable reading options from the config file either by setting the `FEW_IGNORE_CFG_FILE` to a *Truthy* value (yes/on/true/1) or by using the `--ignore-config-file` argument.
Similarly, one may disable options from the environment using the `--ignore-env` CLI option (in command line contexts only).

In your program, you may access the values of these configuration option by accessing the global configuration:

```py3
>>> import few
>>> cfg = few.get_config()
>>> cfg.log_level
30
>>> cfg.file_integrity_check
'once'
```

Unless stated otherwise, command-line arguments take precedence over environment variable which in turn take precedence over configuration file entries.

## Configuration setter

If you need to update configuration settings in a given script, or in an interactive Python context (terminal, notebook),
this can be done using the [*configuration setter*](few.utils.globals.ConfigurationSetter).

```{important}
The Configuration Setter can only be used right after importing `few`.
As soon as any `few` entity makes use of a configuration option (for instance, when initializing
the first backend-accelerated object), the setter cannot be used anymore to change an option.
```

The *configuration setter* is accessed by `few.get_config_setter`and offers multiple methods to
customize configuration options. Note that these methods can be chained directly:

```{eval-rst}
.. testcode::

    import few
    # Access the setter
    setter = few.get_config_setter()

    # Use a single method
    setter.disable_file_download()

    # Chain methods
    setter.set_log_level(
        "debug"
    ).add_file_extra_paths("/tmp/few_data")
```
