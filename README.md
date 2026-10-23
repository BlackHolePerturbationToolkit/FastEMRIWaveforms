# few: FastEMRIWaveforms

[![Documentation Status](https://app.readthedocs.org/projects/fastemriwaveforms/badge/?version=stable)](https://fastemriwaveforms.readthedocs.io/en/stable/)
[![DOI](https://zenodo.org/badge/223486766.svg)](https://doi.org/10.5281/zenodo.3969004)

This package contains a highly modular framework for the rapid generation of accurate extreme-mass-ratio inspiral (EMRI) waveforms. FEW combines a variety of separately accessible modules to construct EMRI waveform models for both CPUs and GPUs.

* Generally, the modules fall into four categories: trajectory, amplitudes, summation, and utilities. Please see the [documentation](https://fastemriwaveforms.readthedocs.io/en/stable) for further information on these modules.
* The code can be found on Github [here](https://github.com/BlackHolePerturbationToolkit/FastEMRIWaveforms).
* The data necessary for various modules in this package will automatically download the first time it is needed. If you would like to view the data, it can be found on [Zenodo](https://zenodo.org/records/15624459) (the record listed in the package file registry, `src/few/files/registry.yml`).
* The current and all past code release zip files can also be found on Zenodo [here](https://zenodo.org/records/3969004).

**Please see the [citation](#citation) section below for information on citing FEW.** This package is part of the [Black Hole Perturbation Toolkit](https://bhptoolkit.org/).

## Getting started

Detailed installation instructions can be found in the [documentation](https://fastemriwaveforms.readthedocs.io/en/stable/user/install.html).
Below is a quick set of instructions to install the FastEMRIWaveform package on CPUs and GPUs.

To install the latest version of `fastemriwaveforms` using `pip`, simply run:

```sh
# For CPU-only version
pip install fastemriwaveforms

# For GPU-enabled versions with CUDA 11.Y.Z
pip install fastemriwaveforms-cuda11x

# For GPU-enabled versions with CUDA 12.Y.Z
pip install fastemriwaveforms-cuda12x
```

There is no `fastemriwaveforms-cuda13x` package: the CUDA 13.x backend (`cuda13x`) is available from a source build
with a CUDA 13 toolkit (see [Installing from sources](#installing-from-sources)).

To know your CUDA version, run the tool `nvidia-smi` in a terminal a check the CUDA version reported in the table header:

```sh
$ nvidia-smi
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 550.54.15              Driver Version: 550.54.15      CUDA Version: 12.4     |
|-----------------------------------------+------------------------+----------------------+
...
```

You may also install `fastemriwaveforms` directly using conda (including on Windows)
as well as its CUDA 12.x plugin (only on Linux). It is strongly advised to:

1. Ensure that your conda environment makes sole use of the `conda-forge` channel
2. Install `fastemriwaveforms` directly when building your conda environment, not afterwards

```sh
# For CPU-only version, on either Linux, macOS or Windows:
conda create --name few_cpu -c conda-forge --override-channels python=3.12 fastemriwaveforms
conda activate few_cpu

# For CUDA 12.x version, only on Linux
conda create --name few_cuda -c conda-forge --override-channels python=3.12 fastemriwaveforms-cuda12x
conda activate few_cuda
```

Now, in a python file or notebook:

```py3
import few
```

FEW's backends are registered with the [`gpubackendtools`](https://github.com/lisa-analysis-tools/GPUBackendTools)
backend registry under the names `few_cpu`, `few_cuda11x`, `few_cuda12x` and `few_cuda13x`.
You may check the currently available backends:

```py3
>>> import few
>>> from gpubackendtools import has_backend
>>> for flavor in ["cpu", "cuda11x", "cuda12x", "cuda13x"]:
...     status = "available" if has_backend(f"few_{flavor}") else "unavailable"
...     print(f" - Backend 'few_{flavor}': {status}")
 - Backend 'few_cpu': available
 - Backend 'few_cuda11x': unavailable
 - Backend 'few_cuda12x': unavailable
 - Backend 'few_cuda13x': unavailable
```

`gpubackendtools.get_backend("few_cuda")` (or `"few_gpu"`) returns the first available of `few_cuda13x`, `few_cuda12x`
and `few_cuda11x`. FEW classes choose their backend at construction: by default the first available one they
support (GPU first for GPU-recommended classes), or the one given as `force_backend=` (e.g. `force_backend="cpu"`).

If you expected a backend to be available but it is not, request it with `get_backend` to obtain an error
message which can guide you to fix this issue: the final `BackendAccessException` is chained to the cause
(plugin package not installed, CuPy built for another CUDA version, unsupported driver, ...):

```py3
>>> from gpubackendtools import get_backend
>>> get_backend("few_cuda12x")
...
The above exception was the direct cause of the following exception:
...
gpubackendtools.gpubackendtools.BackendAccessException: Backend 'few_cuda12x' is unavailable. See previous error messages.
```

When the plugin package is missing, install `fastemriwaveforms-cuda12x` (the hint printed in the chained error names
a `gpubackendtools-cuda12x` package).

Once FEW is working and the expected backends are selected, check out the [examples notebooks](https://github.com/BlackHolePerturbationToolkit/FastEMRIWaveforms/tree/master/examples/)
on how to start with this software.

## Installing from sources

### Prerequisites

To install this software from source, you will need:

- A C++ compiler (g++, clang++, ...)
- Python >= 3.9 (`requires-python` in `pyproject.toml`)
- The [`gpubackendtools`](https://github.com/lisa-analysis-tools/GPUBackendTools) package: `few` imports it
  (backend registry), but it is not yet listed in the `pyproject.toml` dependencies, so install it first

Some installation steps require the external library `LAPACK` along with its C-bindings provided by `LAPACKE`.
If these libraries and their header files (in particular `lapacke.h`) are available on your system, they will be detected
and used automatically. If they are available on a non-standard location, see below for some options to help detecting them.
Note that by default, if `LAPACKE` is not available on your system, the installation step will attempt to download its sources
and add them to the compilation tree. This makes the installation a bit longer but a lot easier.

If you want to enable GPU support in FEW, you will also need the NVIDIA CUDA Compiler `nvcc` in your path as well as
the [CUDA toolkit](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html) (with, in particular, the
libraries `CUDA Runtime Library`, `cuBLAS` and `cuSPARSE`).

There are a set of files required for total use of this package. They will download automatically the first time they are needed. Most of the files used by the stock models are under about 13MB. However, there is a 100MB file (`Teuk_amps_a0.0_lmax_10_nmax_30_new.h5`) needed for the slow waveform and the bicubic amplitude interpolation, and the Kerr eccentric equatorial amplitude file (`ZNAmps_l10_m10_n55_DS2Outer.h5`, used by `FastKerrEccentricEquatorialFlux`) is about 5GB. These larger files will only download if you run the modules that need them (or prefetch them with `few_files fetch`). The files are hosted on the [Black Hole Perturbation Toolkit Download Server](https://download.bhptoolkit.org/few/data) and on Zenodo.

### Installation instructions using conda

We recommend to install FEW using conda in order to have the compilers all within an environment.
First clone the repo

```
git clone https://github.com/BlackHolePerturbationToolkit/FastEMRIWaveforms.git
cd FastEMRIWaveforms
```

Now create an environment (these instructions work for all platforms but some
adjustements can be needed, refer to the
[detailed installation documentation](https://fastemriwaveforms.readthedocs.io/en/stable/user/install.html) for more information):

```
conda create -n few_env -y -c conda-forge --override-channels |
    cxx-compiler pkgconfig conda-forge/label/lapack_rc::liblapacke
```

activate the environment

```
conda activate few_env
```

Then we can install locally for development:
```
pip install -e '.[doc, testing]'
```

### Installation instructions using conda on GPUs and linux
Below is a quick set of instructions to install the Fast EMRI Waveform package on GPUs and linux.

```sh
conda create -n few_env -c conda-forge fastemriwaveforms-cuda12x python=3.12
conda activate few_env
```

Test the installation device by running python
```python
import few
from gpubackendtools import get_backend
get_backend("few_cuda12x")
```

### Running the installation

To start the from-source installation, ensure the pre-requisite are met, clone
the repository, and then simply run a `pip install` command:

```sh
# Clone the repository
git clone https://github.com/BlackHolePerturbationToolkit/FastEMRIWaveforms.git
cd FastEMRIWaveforms

# Run the install
pip install .
```

If the installation does not work, first check the [detailed installation
documentation](https://fastemriwaveforms.readthedocs.io/en/stable/user/install.html). If
it still does not work, please open an issue on the
[GitHub repository](https://github.com/BlackHolePerturbationToolkit/FastEMRIWaveforms/issues)
or contact the developers through other means.



### Running the Tests

The tests require a few dependencies which are not installed by default. To install them, add the `[testing]` label to FEW package
name when installing it. E.g:

```sh
# For CPU-only version with testing enabled
pip install fastemriwaveforms[testing]

# For GPU version with CUDA 12.Y and testing enabled
pip install fastemriwaveforms-cuda12x[testing]

# For from-source install with testing enabled
git clone https://github.com/BlackHolePerturbationToolkit/FastEMRIWaveforms.git
cd FastEMRIWaveforms
pip install '.[testing]'
```

To run the tests, open a terminal in a directory containing the sources of FEW and then run the `unittest` module in `discover` mode:

```sh
$ git clone https://github.com/BlackHolePerturbationToolkit/FastEMRIWaveforms.git
$ cd FastEMRIWaveforms
$ python -m few.tests  # or "python -m unittest discover"
...
----------------------------------------------------------------------
Ran 20 tests in 71.514s
OK
```

## Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct, and the process for submitting pull requests to us.

If you want to develop FEW and produce documentation, install `few` from source with the `[doc]` label and in `editable` mode:

```
$ git clone https://github.com/BlackHolePerturbationToolkit/FastEMRIWaveforms.git
$ cd FastEMRIWaveforms
pip install -e '.[doc, testing]'
```

This will install necessary packages for building the documentation (`sphinx`, `myst-parser`, `pypandoc`, `sphinx-rtd-theme`, `nbsphinx`, ...) and to run the tests.

The documentation source files are in `docs/source`. To compile the documentation locally, run `sphinx-build -b html docs/source docs/build` from the repository root (see the developer guide page "Write documentation").

## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/BlackHolePerturbationToolkit/FastEMRIWaveforms/tags).

## Contributors

A (non-exhaustive) list of contributors to the FEW code can be found in [CONTRIBUTORS.md](CONTRIBUTORS.md).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

Please make sure to cite FEW papers and the FEW software on [Zenodo](https://zenodo.org/records/3969004).
We provide a set of prepared references in [PAPERS.bib](PAPERS.bib). There are other papers that require citation based on the classes used. For most classes this applies to, you can find these by checking the `citation` attribute for that class.  All references are detailed in the [CITATION.cff](CITATION.cff) file.

## Acknowledgments

* This research resulting in this code was supported by National Science Foundation under grant DGE-0948017 and the Chateaubriand Fellowship from the Office for Science \& Technology of the Embassy of France in the United States.
* It was also supported in part through the computational resources and staff contributions provided for the Quest/Grail high performance computing facility at Northwestern University.
