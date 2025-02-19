# Adding features

## Add a new module to CPU and CUDA backends

Currently, CPU and CUDA backends share the same set of compiled modules: `pyAAK`, `pyinterp`, `pyAmpInterp2D` and `pymatmul`.

To add a new module to these backends, first define the core code of that module in a CUDA source file in `src/few/cutils/newmodule.cu` by making sure that
the file can be compiled in both C++ and CUDA modes, and define a Cython file wrapping the main functions of that module `module.pyx`.
Actual file naming is not important here.

The compiled module will have to be declared in `src/few/cutils/CMakeLists.txt` by adding a new call to the function `few_add_lib_to_cuda_cpu_backends`
at the end of the file, similar to the 4 entries for the already existing modules.

That function takes the following arguments:

- `NAME`: name of the module, this is the import name of the module (e.g. if set to `foo`, then it will be possible to do `from few_backend_cpu.foo import bar`)
- `PYX_SOURCES`: list of Cython files that must be included in the module (usually a single file defining the function wrappings)
- `CXX_SOURCES`: list of C++ source files that must be compiled into both CPU and GPU backends
- `CU_SOURCES`: list of CUDA source files that will be compiled as C++ files in the CPU backend, and as CUDA files in a GPU backend. If multiple files are provided, device linking is automatically enabled.
- `LINK`: List of targets to link both CPU and GPU backends against, with a CMake `PUBLIC`/`PRIVATE` prefix to determine whether the linking should be transitive. For example, to link against `libfftw.so`, define `LINK PUBLIC fftw`
- `CPU_LINK`: Similar to `LINK` but for linking libraries only to the CPU backend
- `INCLUDE`: List of directories which must be added as include paths to both CPU and GPU backends
- `HEADERS`: List of local header files that are required to compile the backend

Only the `NAME` parameter and at least one of `PYX_SOURCES`, `CXX_SOURCES` and `CU_SOURCES` must be provided, though not providing `PYX_SOURCES` will result in an unimportable module, and not providing `CU_SOURCES` will result in the exact same compiled code for both CPU and GPU backends (so without actual GPU acceleration).

See [here](deploy.md#understanding-the-cmake-compilation-mechanism) for details about the resulting compilations.

## Add a *Citable* class

The FastEMRIWaveforms package offers a *citation* framework so that you can specify which articles should
be cited by a user of a class you implemented.

The first step is to declare your article in [CITATION.cff](../CITATION) in the `references:` section.
You can then implement your class as deriving from [Citable](few.utils.citations.Citable) and implement the
class method `module_references()`. To handle the case where one parent class also derivates from
[Citable](few.utils.citations.Citable), it is best practice to always add your references to the parent classes
references:

```py3
from few.utils.citations import Citable, REFERENCE

MyOwnClass(Citable):

    @classmethod
    def module_references(cls):
        return ["my_ref_abbreviation"] + super().module_references()
```

You may also add your reference abbreviation in the [REFERENCE](few.utils.citations.REFERENCE) enum and then
return `[REFERENCE.REF_NAME] + super().module_references()` to alias your article with shortcut easier to remember
than the abbreviation used in the [CITATION.cff](../CITATION) file.

The citations related to your class can then be queried by running in a terminal:

```bash
$ few_citations few.amplitude.ampinterp2d.AmpInterp2D  # replace by your class module and name
@article{Chua:2018woh,
  author        = "Chua, Alvin J. K. and Galley, Chad R. and Vallisneri, Michele",
  title         = "{Reduced-Order Modeling with Artificial Neurons for Gravitational-Wave Inference}",
  journal       = "Physical Review Letters",
  year          = "2019",
  month         = "5",
  number        = "21",
  publisher     = "American Physical Society",
  pages         = "211101--211108",
  issn          = "1079-7114",
  doi           = "10.1103/physrevlett.122.211101",
  archivePrefix = "arXiv",
  eprint        = "1811.05491",
  primaryClass  = "astro-ph.im"
}
...
@software{FastEMRIWaveforms,
  author     = "Katz, Michael and Speri, Lorenzo and Chapman-Bird, Christian and Chua, Alvin J. K. and Warburton, Niels and Hughes, Scott",
  title      = "{FastEMRIWaveforms}",
  license    = "GPL-3.0",
  url        = "https://bhptoolkit.org/FastEMRIWaveforms/html/index.html",
  repository = "https://zenodo.org/records/8190418",
  doi        = "10.5281/zenodo.3969004"
}
```

You may also query the citations directly on an instance of your class:

```py3
>>> foo = MyOwnClass()
>>> print(foo.citation())
...
@software{FastEMRIWaveforms,
  author     = "Katz, Michael and Speri, Lorenzo and Chapman-Bird, Christian and Chua, Alvin J. K. and Warburton, Niels and Hughes, Scott",
  title      = "{FastEMRIWaveforms}",
  license    = "GPL-3.0",
  url        = "https://bhptoolkit.org/FastEMRIWaveforms/html/index.html",
  repository = "https://zenodo.org/records/8190418",
  doi        = "10.5281/zenodo.3969004"
}
```

## Implement access to a file

The FastEMRIWaveforms contain a [File Manager](few.files.FileManager) utility to simplify access to
files in a way configurable through [configuration options](../user/cfg.md#file-manager).

This file manager should be used to:

- Read a downloadable file
- Obtain the path to a write-only file (which should be located in the [storage directory](../user/cfg.md#file-storage-path))

### Declare a new downloadable file

If you implement a class which requires access to a large file, the recommended approach is to

1. Upload that file to a publicly available storage repository
2. Declare the file in the [FileRegistry](few.files.FileRegistry) by editing `src/few/files/registry.yml``
  - If the public repository you are using is not declared yet, add it to the `repositories` section
  - Declare your file in the `files` section by defining:
    - its name (will be used to build the file URL from the `url_pattern` declared with the repository, and to access the file from the file manager)
    - its repository(-ies)
    - its SHA256 checksum (use the command `sha256sum /path/to/file` to generate its checksum)

Once this is done, the file can be accessed automatically using its path:

```py3
from few import get_file_manager()
file_path = get_file_manager().get_file("filename")

with open(file_path, "r") as fp:
    # Do anything with the file
```

or directly open it through the FileManager open method:

```py3
from few import get_file_manager()

with get_file_manager().open("filename", "r") as fp:
    # Do anything with the file
```

The approach to use dopends on whether you need the file path itself (to open it through `h5py` for example), or if you'll directly open it.

If the file is not locally present in one of the file manager search paths, it will be first downloaded automatically.

### Obtain path to an output file

If you need to write a result file, multiple options are possible:

- Use an absolute path:

```py3
with open("/my/absolute/path/filename", "w") as fp:
  ...
```

- Use a path relative to current working directory (not advised):

```py3
with open("relative_path/filename", "w") as fp:
  ...
```

- Use a relative path, or only filename, relative to the file manager storage path:

```py3
with few.get_file_manager().open("filename", "w") as fp:
  ...
```

## Add a configuration option

FEW offers a [centralized configuration management](../user/cfg.md#configuration-options) which is meant to be highly customizable.

To add a new configuration option, one simply needs to declare it by adding
a new [Entry](few.utils.config.ConfigEntry) in the [`config_entries`](few.utils.config.Configuration.config_entries) method of the [Configuration](few.utils.config.Configuration) class in `src/few/utils/config.py`.

Each configuration entry is defined by:

- `label`: attribute name of that entry associated to the configuration (final value will be accessible as `few.get_config().label`)
- `description`: short description, can be used in CLI help message for example
- `type`: python datatype of this entry
- `default`: default value, must be of the type defined in `type``
- `cfg_entry` (optional): name of the configuration entry in the FEW config file, if not provided, this entry is not affected by the configuration file
- `env_var` (optional): environment variable whose value, if defined, will affect this configuration entry. Note that the variable name will be automatically prefixed with `FEW_`, so set `MYCFG` to have the variable `FEW_MYCFG` affect your entry.
- `cli_flags` (optional): list of command-line parameter flags (e.g. `-x`, `--my-option`, ...) associated to the configuration entry
- `cli_kwargs` (optional, ignored if `cli_flags` not set): dict of keyword arguments accepted by [argparse.ArgumentParser.add_argument](argparse.ArgumentParser.add_argument) to customize the way CLI parameters are parsed for this entry
- `convert` (optional): function that can take a [str](str) (and optionally other types) as input, and convert it to a value of type given by the `type`
- `validate` (optional): function that can take a value of type `type` and return `True` if the value if a valid one for this entry, and `False` otherwise

It is also strongly advised to declare the entry as a class attribute
in the header of the [Configuration](few.utils.config.Configuration) class.

Finally, configuration options can be modified via the [ConfigurationSetter](few.utils.globals.ConfigurationSetter) as [detailed here](../user/cfg.md#configuration-setter).
It is therefore advised to also add a new method to that class (defined in `src/few/utils/globals.py`) to make your configuration option
tunable via this method.

The list of action to take to add a new configuration option is thus:

- [ ] Add the new entry to the [`config_entries`](few.utils.config.Configuration.config_entries) method of the [Configuration](few.utils.config.Configuration) class in `src/few/utils/config.py`.
- [ ] Add the new option as a class attribute in the header of the [Configuration](few.utils.config.Configuration) class
- [ ] Add a method to tune the option to the [ConfigurationSetter](few.utils.globals.ConfigurationSetter)
- [ ] Add documentation for that option in `docs/source/user/cfg.md`


## Adding log messages

The recommended way to print messages in FEW is to use the package [logger](logging.Logger).
[pre-commit](./ide.md#pre-commit-apply-common-guidelines-to-your-code) will, by default, complain about the use of [`print`](print) statements which should be replaced by calls to the logger methods:

- `few_logger.debug(message)`: should contain detailed information about the innerworking of a piece of code to guide a user or developper during debugging phases
- `few_logger.info(message)`: should replace most print statements directed to the user
- `few_logger.warning(message)`: should warn the user about unexpected states which are recoverable
- `few_logger.error(message)`: should warn the user about unexpected state which are unrecoverable

The FEW logger is accessible by:

```py3
import few

few_logger = few.get_logger()

def myfunction():
  few_logger.debug("Now executing myfunction()")
```

The logger is defined to output `debug` and `info` messages to the standard output `stdout`, while `warning` and `error` messages are sent to the standard error `stderr`.
You may customize this behavior by clearing the logger `handlers` and adding your own. See the [logging](logging) package documentation for references.
