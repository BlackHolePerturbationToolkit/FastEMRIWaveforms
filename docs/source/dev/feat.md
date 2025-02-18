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
