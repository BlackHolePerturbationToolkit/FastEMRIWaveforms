# Configuration options

The `few` package can be tuned through the use of optional configuration options which can be defined by either:

- A `few.ini` configuration file
- Environment variables
- Command-line arguments when using a compatible command-line utility
- A Python object named the [`ConfigurationSetter`](few.utils.globals.ConfigurationSetter) which can be used right after importing the `few` module for the first time

The `few.ini` file is searched, in priority order:

- In your current working directory
- In your [`platformdirs.user_config_path()`](https://github.com/tox-dev/platformdirs/blob/main/README.rst) (e.g. `~/.config/few.ini`)
- In your [`platformdirs.site_config_path()`](https://github.com/tox-dev/platformdirs/blob/main/README.rst), in a `few/v<major>.<minor>` subdirectory named after the FEW version (e.g. `/etc/xdg/few/v2.0/few.ini`)

The path to `few.ini` can be enforced by the `FEW_CONFIG_FILE` environment variable, or in command-line contexts, by the `-C` or `--config-file` argument.
Options are read from the `[few]` section of the file.

## Tunable utilities

These configuration options act on the following utilities.

### Log level

`few` handles logging of operations using the python `logging` standard module.
Its logger can be accessed either directly through a call to

```py3
import logging
logger = logging.getLogger("few")
```

or by accessing `few` global states with

```py3
import few

logger = few.get_logger()
```

The default log level is `logging.WARNING` but this can be modified through the `log_level` configuration option.

### Backends management

Backends are plugin-like entities which can be used to delegate heavy computations to specific hardware like GPUs.
FEW's backends are managed by the `gpubackendtools` backend registry: `import few` registers them under the names

- `few_cpu`: Use the CPU itself for accelerated computations (plugin module `few_backend_cpu`)
- `few_cuda11x`: Use a NVIDIA GPU with CUDA 11.x drivers (plugin module `few_backend_cuda11x`)
- `few_cuda12x`: Use a NVIDIA GPU with CUDA 12.x drivers (plugin module `few_backend_cuda12x`)
- `few_cuda13x`: Use a NVIDIA GPU with CUDA 13.x drivers (plugin module `few_backend_cuda13x`,
  available from source builds only: no `cuda13x` wheel is published)

A backend is loaded the first time it is requested. Each backend-aware class (a
[`FewBackendConsumer`](few.utils.baseclasses.FewBackendConsumer)) selects its backend when it is built:

- with `force_backend=None` (the default), it uses the first available backend of its `supported_backends()` list.
  A class with hybrid GPU/CPU support usually tries `cuda13x`, then `cuda12x`, then `cuda11x`, and falls back to `cpu`;
  a CPU-only class uses `cpu`;
- with `force_backend="cpu"` (or `"cuda12x"`, `"few_cuda13x"`, ...), it uses that backend, and fails if it is unavailable.

The availability of a backend can be checked with the `gpubackendtools` functions:

```py3
import few
from gpubackendtools import get_backend, has_backend

has_backend("few_cpu")       # True if the CPU backend loads
get_backend("few_cuda12x")   # raises, with the reason chained, if the backend is unavailable
get_backend("few_cuda")      # first available of few_cuda13x, few_cuda12x, few_cuda11x
```

```{note}
The `enabled_backends` option (`FEW_ENABLED_BACKENDS`, `--enable-backend`, `enabled-backends`,
`ConfigurationSetter.enable_backends`) is still accepted and validated (values `cpu`, `cuda11x`, `cuda12x`,
`cuda13x`), but no FEW code reads it: it neither restricts nor forces the backends. Use `force_backend=` to
select a backend.
```


### File manager

The `few` package requires external files of pre-computed coefficients which are too large to be bundled with the source code.
These files are accessed through the `FileManager` global entity which takes care of locating these files, checking their integrity,
and downloading missing files on request.

The file manager is accessed through `few.get_file_manager()`

This `FileManager` is highly tunable and propose the following options:

#### `file_storage_path`

The *storage path* is the base directory of the `FileManager`: files opened for writing by
bare name with `few.get_file_manager().open(...)` go there, and it is searched for requested files.

Its default value is built with [`platformdirs.user_data_path()`](https://github.com/tox-dev/platformdirs/blob/main/README.rst)
from the full FEW version (e.g. `~/.local/share/few/v2.0.0` on Linux systems) and is created if missing.
In an editable (development) install, the default is the package data directory `src/few/data`.

#### `file_download_path`

The *download path* is the directory where the `FileManager` will download missing files.
By default, it is a subdirectory of the *storage path* named `download` (in an editable install, `src/few/data`).

Requested files are searched for in the download path, then the storage path, then the extra paths.

#### `file_extra_paths`

Extra paths are supplementary read-only directories where the file manager will search for requested files.
They are provided as a ";"-separated list of paths. The package data directory (`<few package>/data`) is always
included.

#### `file_disabled_tags`

Files of the registry carry tags (e.g. `testfile`, `kerr`, `high_memory`, `deprecated`). Accessing a file which
carries a disabled tag raises `FileManagerDisabledAccess`. This option is set by the `FEW_DISABLED_TAGS`
environment variable or by the `disable_file_tags()` method of the configuration setter (the test runner
`python -m few.tests --disable TAG` uses the latter).

#### `file_registry_path`

The *file registry* is a YAML file which lists known files which can be downloaded from FEW data repositories.
It defines these data repositories and then, for each file, lists the repositories they can be found in, their tags and their checksums.

A default `registry.yml` file is embedded with each version of `few` but one can provide a custom registry by setting its path
through the `file_registry_path` option.

#### `file_integrity_check`

By default, the first time a file is requested to the `FileManager`, its integrity is checked and cached so that future request to that
file are fast.

This behaviour can be tuned to disable entirely integrity checks, or to perform these checks each and every time a given file is requested.

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
| `enabled_backends` | `enabled-backends` | `FEW_ENABLED_BACKENDS` | `--enable-backend` | `cpu`, `cuda11x`, `cuda12x`, `cuda13x` | ";"-separated list of values. The CLI parameter can be used multiple time for each enabled backend. Currently has no effect (see [Backends management](#tunable-utilities)). |
| `file_registry_path` | `file-registry` | `FEW_FILE_REGISTRY` | `--file-registry` | Path to a `registry.yml` file |  |
| `file_storage_path` | `file-storage-dir` | `FEW_FILE_STORAGE_DIR` | `--storage-dir` | Absolute path, or relative to current working directory | Directory must already exist |
| `file_download_path` | `file-download-dir` | `FEW_FILE_DOWNLOAD_DIR` | `--download-dir` | Absolute path, or relative to the storage directory |  |
| `file_allow_download` | `file-allow-download` | `FEW_FILE_ALLOW_DOWNLOAD` | `--(no-)file-download` | Truthy (`yes`, `true`, `on`, `1`) or Falsy (`no`, `false`, `off`, `0`) value |  |
| `file_integrity_check` | `file-integrity-check` | `FEW_FILE_INTEGRITY_CHECK` | `--file-integrity-check` | `never`, `once`, `always`  |  |
| `file_extra_paths` | `file-extra-paths` | `FEW_FILE_EXTRA_PATHS` | `--extra-path` | ";"-separated list of directories | (1) |
| `file_disabled_tags` | | `FEW_DISABLED_TAGS` | | ";"-separated list of file tags | (1) |

(1) Extra paths (and disabled tags) are cumulated between option sources. *i.e.* the final options contains all paths from the config file, the environment variable and the CLI option.

In addition, command-line arguments accept `-v` / `--verbose` (repeatable; each occurrence lowers the log level by 10, down to `DEBUG`)
and `-Q` / `--quiet` (log level `CRITICAL`).

The `few_files` command-line utility accepts all the options above, and its `--help` message lists them together with
its own options.

One may also disable reading options from the config file either by setting the `FEW_IGNORE_CFG_FILE` to a *Truthy* value (yes/on/true/1) or by using the `--ignore-config-file` argument.
Similarly, one may disable options from the environment using the `--ignore-env` CLI option (in command line contexts only).

In your program, you may access the values of these configuration option by accessing the global configuration:

```{eval-rst}
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
.. testcode:: cfg-setter

    # import few
    # Access the setter
    setter = few.get_config_setter()

    # Use a single method
    setter.disable_file_download()

    # Chain methods
    setter.set_log_level(
        "debug"
    ).add_file_extra_paths("/tmp/few_data")
```

Options defined in the [*configuration setter*](few.utils.globals.ConfigurationSetter) can be applied
immediately by calling the `finalize()` method like:

```{eval-rst}
.. testcode:: cfg-explicit-finalize

    import few
    # Access the setter, set an option and finalize options
    few.get_config_setter().set_log_level("info").finalize()
```

But if `finalize()` is not explicitly called, the options set will still be taken into account
automatically at a latter stage:

```{eval-rst}
.. testcode:: cfg-implicit-finalized

    import few
    # Set the log level
    few.get_config_setter().set_log_level("debug")

    # Build a FEW object
    amp = few.amplitude.romannet.RomanAmplitude()

.. testoutput:: cfg-implicit-finalized

    ...
    ConfigInitialization: final configuration entries are
    ...
     log_level=10 (from: ConfigSource.SETTER)
    ...

```
