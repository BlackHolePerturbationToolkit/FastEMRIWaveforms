# Configuration options

The `few` package can be tuned through the use of optional configuration options which can be defined by either:

- A `few.ini` configuration file
- Environment variables
- Command-line arguments when using a compatible command-line utility

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

### Backend selection

Many classes in the `few` package can benefit from GPU acceleration. This is managed through the `use_gpu` boolean flag that
can be passed to the object constructor.

When `use_gpu==False`, the object will use methods that are provided by the `few.cutils.cpu` module. When `use_gpu==True`, the
object will rather use methods provided by the `few.cutils.fast` module.

Note however that `few.cutils.fast` is simply an alias for either the `cpu` backend, the `cuda11x` backend or the `cuda12x` backend.
By default, `few` will detect automatically the best available backend and silently revert back to the `cpu` backend
if CUDA backends are uninstalled, or unusable due to missing software or hardware.

The `fast_backend` option can change the `fast` backend selection mode:

- `cpu`: force the unnacelerated CPU backend (cannot fail)
- `cuda11x`: force the CUDA 11.x backend, fail if it has missing sotware or hardware dependencies
- `cuda12x`: force the CUDA 12.x backend, fail if it has missing sotware or hardware dependencies
- `lazy` (default): try to load the `cuda12x` backend, then the `cuda11x` in case of error, then the `cpu` if case of further error (cannot fail)
- `best`: detect the available CUDA version (if any) then act the same way as the `cpu`, `cuda11x` or `cuda12x` values corresponding to the detected version

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
| `fast_backend` | `fast-backend` | `FEW_FAST_BACKEND` | `--fast-backend` | `cpu`, `cuda11x`, `cuda12x`, `lazy`, `best` |  |
| `log_level` | `log-level` | `FEW_LOG_LEVEL` | `--log-level` | `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL` |  |
| `log_format` | `log-format` | `FEW_LOG_FORMAT` | `--log-format` | Any format string supported by [`logging.Formatter`](https://docs.python.org/3/library/logging.html#logging.Formatter) |  |
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
>>> import few.utils.globals
>>> cfg = few.utils.globals.get_config()
>>> cfg.log_level
30
>>> cfg.fast_backend
<BackendSelectionMode.LAZY: 'lazy'>
```

Unless stated otherwise, command-line arguments take precedence over environment variable which in turn take precedence over configuration file entries.
