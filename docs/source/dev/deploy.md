# Packaging

This section explains how the build process works and how to manually build wheels.
It should give insights into how package is deployed on [PyPI](https://pypi.org/project/fastemriwaveforms/) and
how the plugin system is setup.

## Default build process

The FastEMRIWaveforms package relies on [*scikit-build-core*](https://scikit-build-core.readthedocs.io) to handle
the build process. This tool makes use of [CMake](https://cmake.org) to handle the compilation of C/C++/CUDA backends
while being compatible with modern python packaging practices (*i.e.* by using a `pyproject.toml` file instead of the
legacy `setup.py` approach).

When installing `few` from sources, or simply building a wheel package, the main steps executed are:

- Create a temporary isolated environment
- Install in that environment the build system dependencies listed in `pyproject.toml`: `scikit-build-core`, `numpy`, `cython`, `setuptools_scm`, ...
- Let `scikit-build-core` orchestrate the next steps:
  - Read the main `CMakeLists.txt` file to detect the required CMake version
  - Detect whether a corresponding `cmake` command is available, otherwise automatically install it in the current isolated environment
  - Call `setuptools_scm` to detect the current project version from git tags and write it to `src/few/_version.py`
  - Call `CMake` to handle backends compilation
  - Package the resulting compiled modules and the project Python sources into a wheel

## Building core and plugin packages

By default, CMake will always compile at least the `CPU` backend, and will try to also compile the `GPU` backend if required dependencies are available. This results in a single wheel which contains (1) the pure-python core package, (2) the compiled `CPU` backend and optionally (3) a compiled `GPU` CUDA backend.

This is ideal for local development and installation from source, but different from how FEW is released on PyPI where 3 packages are deployed:

- `fastemriwaveforms`: contains the python code and the CPU backend
- `fastemriwaveforms-cuda11x`: contains only the `cuda11x` GPU backend
- `fastemriwaveforms-cuda12x`: contains only the `cuda12x` GPU backend

The process for building these differenciated wheels is defined in `.github/workflows/publish.yml` which handles this build process and package deployment to PyPI.
The core logic is handled by tweaking slightly the `pyproject.toml` file before building each category of wheels.

### Common steps

Some steps are performed for all built wheels:

```sh
# Change the version scheme to force a clean version like "1.5.2" instead of 1.5.2.post1.dev51+gfe23bf1.d20250218
sed -i 's|version_scheme = "no-guess-dev"|version_scheme = "only-version"|g' pyproject.toml
sed -i 's|local_scheme = "node-and-date"|local_scheme = "no-local-version"|g' pyproject.toml

```

### Building the core package

To build the core package wheel, the following command is executed after the common steps for python 3.9 to 3.13:

```sh
pip wheel ./ --no-deps -w ./dist \
  --config-settings=cmake.define.FEW_WITH_GPU=OFF
```

The wheels built on Linux are, by default, distribution specific and must be made into `manylinux` wheels to improve their compatibility with many distributions. This is done by *repairing* the wheels using [auditwheel](https://github.com/pypa/auditwheel):

```sh
for whl in ./dist/*.whl; do
    auditwheel repair "${whl}" -w ./wheelhouse/ --plat manylinux_2_27_x86_64
done
```

The `manylinux` wheels will be put into `./wheelhouse`.

### Building the GPU plugin packages

To build GPU plugin packages, multiple modifications to `pyproject.toml` must be applied:

```sh
# Change the project name to add the `-cuda11x` or `-cuda12x` suffix
sed -i 's|" #@NAMESUFFIX@|-cuda12x"|g' pyproject.toml

# Add `cupy-cuda11x` or `cupy-cuda12x` to the project dependencies
sed -i 's|#@DEPS_CUPYCUDA@|"cupy-cuda12x"|g' pyproject.toml

# Add a dependency of the project core package
sed -i 's|#@DEPS_FEWCORE@|"fastemriwaveforms"|g' pyproject.toml

# Delete the line containing the falg @SKIP_PLUGIN@ from pyproject.toml
# that line instruct scikit-build-core to add the directory src/few to the wheel
# so deleting it removes all python sources from the generated wheel
sed -i '/@SKIP_PLUGIN@/d' pyproject.toml
```

The wheels are then built for python 3.9 to 3.13 with the command:

```sh
pip wheel ./ --no-deps -w ./dist \
  --config-settings=cmake.define.FEW_WITH_GPU=ONLY
```

The option `FEW_WITH_GPU=ONLY` instructs CMake to build a GPU backend and to skip the CPU one. therefore, in the end, the wheel contains only the compiled modules for the GPU backend.

Just like for the core package, the wheels must be *repaired* to become `manylinux` wheels. Since they have a dependencies on NVIDIA dynamic libraries, they are not strictly-speaking `manylinux` compatible but mechanisms are in place on the core package Python code to detect issues with these dependencies and advise the user about required steps.
`auditwheel` must be instructed to ignore those external dependencies like so:

```sh
for whl in ./dist/*.whl; do
    auditwheel repair "${whl}" -w /wheelhouse/ \
        --plat manylinux_2_27_x86_64 \
        --exclude "libcudart.so.12" \
        --exclude "libcusparse.so.12" \
        --exclude "libcublas.so.12" \
        --exclude "libnvJitLink.so.12" \
        --exclude "libcublasLt.so.12"
done
# Replace the .so.12 extension by .so.11 if you are building the cuda11x plugin
```

## Understanding the CMake compilation mechanism

`CMake` is a powerful scripting language used to manage the compilation steps of
complex projects. One of its main advantage is its cross-platform compatibility:
it provides abstraction layers to make the compilation independent from the current
operating system and thus makes the building steps of FEW working on both Linux, macOS,
and Windows (though that last OS has not been tested thoroughly).

CMake is unfortunately also known having convoluted syntaxes and for having many
examples that make use of outdated/legacy syntaxes. This is why, for FEW, most of
CMake complexity is "hidden" from developers.

When [declaring a compiled module with CMake](feat.md#add-a-new-module-to-cpu-and-cuda-backends) with a declaration like

```cmake
few_add_lib_to_cuda_cpu_backends(
  NAME pyinterp
  PYX_SOURCES pyinterp.pyx
  CU_SOURCES interpolate.cu CPU_LINK PUBLIC "lapacke;lapack"
  HEADERS cuda_complex.hpp global.h matmul.hh
  INCLUDE PRIVATE ${Python_NumPy_INCLUDE_DIR})
```

The following steps are executed:

- CMake calls the following command to process the Cython file `pyinterp.pyx` into a C++ file:

```bash
$ python -m cython pyinterp.pyx \
    --output-file pyinterp.cpp \
    -3 \ # Select Python 3 syntax
    -+ \  # Build C++ output instead of C
    --module-name pyinterp \
    -I ./  # Search for header files in local directory
```

- If the CPU backend needs to be built, declare that a dynamic library `few_backend_cpu/pyinterp.cpython-3X-${arch}.so` must be built with:
  - C++ source files (or interpreted as such): `pyinterp.cpp` and `interpolate.cu`
  - Header files: `cuda_complex.hpp`, `global.h` and `matmul.hh`
  - Linking to the libraries referenced by the CMake targets `lapacke` and `lapack` (equivalent to the compilation flags `-L/path/to/lapack/lib -llapacke -llapack`)

- If a GPU backend needs to be build, declare that a dynamic library `few_backend_cuda12x/pyinterp.cpython-3X-${arch}.so` must be built with:
  - C++ source files: `pyinterp.cpp``
  - CUDA source file (compiled with nvcc): `interpolate.cu`
  - The same headers files that for the CPU backend
  - Link to CUDA libraries (`-lcudart -lcublas -lcusparse`)
  - Specific compilation flags related to device linking, CUDA architecture, ...

All that machinery is implemented in `cmake/FEW.cmake` and actually makes uses of filename manipulation,
temporary directories, file copying to differentiate compiling the same file in C++ mode and in CUDA
mode, etc... But these are implementation details, only elements declared in the call to `few_add_lib_to_cuda_cpu_backends`
should be of importance as detailed in the [dedicated section](feat.md#add-a-new-module-to-cpu-and-cuda-backends).
