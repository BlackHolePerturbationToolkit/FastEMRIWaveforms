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
- `fastemriwaveforms-cuda12x`: contains only the `cuda12x` GPU backend
- `fastemriwaveforms-cuda13x`: contains only the `cuda13x` GPU backend

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

To build the core package wheel, the following command is executed after the common steps for python 3.12 to 3.14:

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

To build GPU plugin packages, multiple modifications to `pyproject.toml` must be applied.
The example below builds the `cuda12x` plugin; replace `12` by `13` throughout to build the `cuda13x` one.

```sh
# Change the project name to add the `-cuda12x` suffix
sed -i 's|" #@NAMESUFFIX@|-cuda12x"|g' pyproject.toml

# Add `cupy-cuda12x` to the project dependencies
sed -i 's|#@DEPS_CUPYCUDA@|"cupy-cuda12x"|g' pyproject.toml

# Add a dependency of the project core package
sed -i 's|#@DEPS_FEWCORE@|"fastemriwaveforms"|g' pyproject.toml

# Delete the line containing the falg @SKIP_PLUGIN@ from pyproject.toml
# that line instruct scikit-build-core to add the directory src/few to the wheel
# so deleting it removes all python sources from the generated wheel
sed -i '/@SKIP_PLUGIN@/d' pyproject.toml
```

The build requires the CUDA Toolkit (`nvcc`, cuRAND, cuBLAS, cuSPARSE) and a host compiler supported by it.
In CI, this is provided by the official [manylinux CUDA container images](https://cibuildwheel.pypa.io/en/stable/faq/#building-wheels-with-cuda-on-linux)
which *cibuildwheel* is pointed at with:

```yaml
# CUDA 12 plugin (use ..._cuda13_1:latest for the CUDA 13 plugin)
CIBW_MANYLINUX_X86_64_IMAGE: quay.io/manylinux_cuda/manylinux_2_28_x86_64_cuda12_9:latest
CIBW_MANYLINUX_AARCH64_IMAGE: quay.io/manylinux_cuda/manylinux_2_28_aarch64_cuda12_9:latest
```

The `cuda12x` plugin is built with the CUDA 12.9 images and the `cuda13x` plugin with the CUDA 13.1 images.
CMake derives the backend name from the toolkit version found in the image, so `few_backend_cuda12x` and
`few_backend_cuda13x` come out of the same sources with no CMake change.

These images already expose `nvcc` on the `PATH`, set `LD_LIBRARY_PATH` to the CUDA libraries and ship a
compatible host compiler, so no CUDA toolchain installation is needed in the workflow. They do not include
cuSPARSE, which FEW requires, so it is installed from the (already configured) NVIDIA repository before building:

```sh
dnf install -y --setopt=obsoletes=0 libcusparse-devel-12-9  # or libcusparse-devel-13-1
```

The wheels are then built for python 3.12 to 3.14 with the command:

```sh
pip wheel ./ --no-deps -w ./dist \
  --config-settings=cmake.define.FEW_WITH_GPU=ONLY
```

The option `FEW_WITH_GPU=ONLY` instructs CMake to build a GPU backend and to skip the CPU one. therefore, in the end, the wheel contains only the compiled modules for the GPU backend.

Just like for the core package, the wheels must be *repaired* to become `manylinux` wheels. Since they depend on NVIDIA dynamic libraries, they are not strictly-speaking `manylinux` compatible, but mechanisms are in place on the core package Python code to detect issues with these dependencies and advise the user about required steps.
`auditwheel` must be instructed to ignore those external dependencies like so:

```sh
for whl in ./dist/*.whl; do
    auditwheel repair "${whl}" -w /wheelhouse/ \
        --plat manylinux_2_27_x86_64 \
        --exclude "libcudart.so.12" \
        --exclude "libcublas.so.12" \
        --exclude "libcublasLt.so.12" \
        --exclude "libnvJitLink.so.12" \
        --exclude "libcusparse.so.12"
done
```

:::{warning}
Those soversions are the ones of a CUDA 12 build, and they do not all follow the CUDA major version: a CUDA 13
build links `libcudart.so.13` and `libcublas.so.13`, but cuSPARSE kept its soversion and remains
`libcusparse.so.12`. An `--exclude` that matches nothing is not an error: auditwheel simply vendors the library
into the wheel (at great expense in terms of distribution size), which is why this should not be written out by hand.
:::

The CI does not hard-code this list. `.github/scripts/repair-cuda-wheel.sh` reads the `NEEDED`
entries of the freshly built extension modules with `patchelf --print-needed`, keeps the ones whose soname stem
is an NVIDIA library that must stay external (`libcudart`, `libcublas`, `libcublasLt`, `libcusparse`,
`libnvJitLink`), and passes exactly those to `auditwheel repair`. The excludes are therefore always in sync with
what was actually linked, whatever CUDA version the container provides, and the script fails loudly if it finds
no NVIDIA library at all rather than producing a wheel with the toolkit baked in.

## Understanding the CMake compilation mechanism

`CMake` is a powerful scripting language used to manage the compilation steps of
complex projects. One of its main advantage is its cross-platform compatibility:
it provides abstraction layers to make the compilation independent from the current
operating system and thus makes the building steps of FEW working on both Linux, macOS,
and Windows (though that last OS has not been tested thoroughly).

When [declaring a compiled module with CMake](feat.md#add-a-new-module-to-cpu-and-cuda-backends)
with a declaration like

```cmake
# ----------------
# --- pymatmul ---
# ----------------

# I. Process pymatmul.pyx into a C++ file
add_custom_command(
  OUTPUT "pymatmul.cxx"
  COMMENT "Cythonize pymatmul.pyx into pymatmul.cxx"
  COMMAND
    Python::Interpreter -m cython "${CMAKE_CURRENT_SOURCE_DIR}/pymatmul.pyx"
    --output-file "${CMAKE_CURRENT_BINARY_DIR}/pymatmul.cxx" -3 -+ --module-name
    "pymatmul" -I "${CMAKE_CURRENT_SOURCE_DIR}"
  DEPENDS "pymatmul.pyx"
  VERBATIM)

# II. Declare the CPU backend
if(FEW_WITH_CPU)
  add_custom_command(
    OUTPUT "matmul.cxx"
    COMMENT "Copy matmul.cu to matmul.cxx"
    COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/matmul.cu"
            "${CMAKE_CURRENT_BINARY_DIR}/matmul.cxx"
    DEPENDS "matmul.cu"
    VERBATIM)

  python_add_library(few_cpu_pymatmul MODULE WITH_SOABI pymatmul.cxx matmul.cxx)
  apply_cpu_backend_common_options(pymatmul)

  target_sources(few_cpu_pymatmul PUBLIC FILE_SET HEADERS FILES
                                         cuda_complex.hpp global.h matmul.hh)
endif()

# III. Declare the GPU backend
if(FEW_WITH_GPU)
  python_add_library(few_gpu_pymatmul MODULE WITH_SOABI pymatmul.cxx matmul.cu)
  apply_gpu_backend_common_options(pymatmul)
  target_sources(few_gpu_pymatmul PUBLIC FILE_SET HEADERS FILES
                                         cuda_complex.hpp global.h matmul.hh)
endif()
```

The steps are clearly decomposed into three partsexecuted:

1. CMake calls the equivalent of following command to process the Cython file
  `pymatmul.pyx` into a C++ file:

```bash
$ python -m cython pymatmul.pyx \
    --output-file pymatmul.cxx \
    -3 \ # Select Python 3 syntax
    -+ \  # Build C++ output instead of C
    --module-name pymatmul \
    -I ./  # Search for header files in local directory
```

- If the CPU backend needs to be built, declare that a dynamic library `few_backend_cpu/pymatmul.cpython-3X-${arch}.so` must be built with:
  - C++ source files: `pymatmul.cxx` and a copy of `matmul.cu` named `matmul.cxx`
  - Header files: `cuda_complex.hpp`, `global.h` and `matmul.hh`
  - The call to `apply_cpu_backend_common_options(pymatmul)` mainly ensures
  that the library will be installed in the right directory and can properly include NumPy header files

- If a GPU backend needs to be build, declare that a dynamic library `few_backend_cuda12x/pymatmul.cpython-3X-${arch}.so` must be built with:
  - C++ source files: `pymatmul.cxx``
  - CUDA source file (compiled with nvcc): `matmul.cu`
  - The same headers files that for the CPU backend
  - The call to `apply_gpu_backend_common_options(pymatmul)` also make sure that the library is installed in the right directory.
  It also applies compilations flags specific to the CUDA compiler.
