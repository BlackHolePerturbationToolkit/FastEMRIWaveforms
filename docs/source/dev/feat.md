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
