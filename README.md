# few: Fast EMRI Waveforms

This package contains the highly modular framework for fast and accurate extreme mass ratio inspiral (EMRI) waveforms from [arxiv.org/2104.04582](https://arxiv.org/abs/2104.04582) and [arxiv.org/2008.06071](https://arxiv.org/abs/2008.06071). The waveforms in this package combine a variety of separately accessible modules to form EMRI waveforms on both CPUs and GPUs. Generally, the modules fall into four categories: trajectory, amplitudes, summation, and utilities. Please see the [documentation](https://bhptoolkit.org/FastEMRIWaveforms/) for further information on these modules. The code can be found on Github [here](https://github.com/BlackHolePerturbationToolkit/FastEMRIWaveforms). The data necessary for various modules in this package will automatically download the first time it is needed. If you would like to view the data, it can be found on [Zenodo](https://zenodo.org/record/3981654#.XzS_KRNKjlw). The current and all past code release zip files can also be found on Zenodo [here](https://zenodo.org/record/3969004). Please see the [citation](#citation) section below for information on citing FEW.

This package is a part of the [Black Hole Perturbation Toolkit](https://bhptoolkit.org/).

If you use all or any parts of this code, please cite [arxiv.org/2104.04582](https://arxiv.org/abs/2104.04582) and [arxiv.org/2008.06071](https://arxiv.org/abs/2008.06071). See the [documentation](https://bhptoolkit.org/FastEMRIWaveforms/) to properly cite specific modules.

## Getting started

To install the latest pre-compiled version of `fastemriwaveforms`, simply run:

```sh
# For CPU-only version
pip install fastemriwaveforms-cpu

# For GPU-enabled versions with CUDA 11.Y.Z
pip install fastemriwaveforms-cuda11x

# For GPU-enabled versions with CUDA 12.Y.Z
pip install fastemriwaveforms-cuda12x
```

To know your CUDA version, run the tool `nvidia-smi` in a terminal a check the CUDA version reported in the table header:

```sh
$ nvidia-smi
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 550.54.15              Driver Version: 550.54.15      CUDA Version: 12.4     |
|-----------------------------------------+------------------------+----------------------+
...
```

Now, in a python file or notebook:

```py3
import few
```

You may check the currently available backends:

```py3
>>> for backend in ["cpu", "cuda11x", "cuda12x", "cuda", "gpu"]: print(f" - Backend '{backend}': {"un" if not few.has_backend(backend) else ""}available")
 - Backend 'cpu': available
 - Backend 'cuda11x': unavailable
 - Backend 'cuda12x': unavailable
 - Backend 'cuda': unavailable
 - Backend 'gpu': unavailable
```

Note that the `cuda` backend is an alias for either `cuda11x` or `cuda12x`. If any is available, then the `cuda` backend is available.
Similarly, the `gpu` backend is (fpr now) an alias for `cuda`.

If you expected a backend to be available but it is not, run the following command to obtain an error
message which can guide you to fix this issue:

```py3
>>> import few
>>> few.get_backend("cuda12x")
ModuleNotFoundError: No module named 'few_backend_cuda12x'

The above exception was the direct cause of the following exception:
...

few.cutils.BackendNotInstalled: The 'cuda12x' backend is not installed.

The above exception was the direct cause of the following exception:
...

few.cutils.MissingDependencies: FastEMRIWaveforms CUDA plugin is missing.
    If you are using few in an environment managed using pip, run:
        $ pip install fastemriwaveforms-cuda12x

The above exception was the direct cause of the following exception:
...

few.cutils.BackendAccessException: Backend 'cuda12x' is unavailable. See previous error messages.
```

Once FEW is working and the expected backends are selected, you may see
[examples notebook](https://github.com/BlackHolePerturbationToolkit/FastEMRIWaveforms/blob/master/examples/FastEMRIWaveforms_tutorial.ipynb)
on how to start with this software.

### Installing from sources

#### Prerequisites

To install this software from source, you will need:

- A C++ compiler (g++, clang++, ...)
- A Python version supported by [scikit-build-core](https://github.com/scikit-build/scikit-build-core) (>=3.7 as of Jan. 2025)

Some installation steps require the external library `LAPACK` along with its C-bindings provided by `LAPACKE`.
If these libraries and their header files (in particular `lapacke.h`) are available on your system, they will be detected
and used automatically. If they are available on a non-standard location, see below for some options to help detecting them.
Note that by default, if `LAPACKE` is not available on your system, the installation step will attempt to download its sources
and add them to the compilation tree. This makes the installation a bit longer but a lot easier.

If you want to enable GPU support in FEW, you will also need the NVIDIA CUDA Compiler `nvcc` in your path as well as
the [CUDA toolkit](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html) (with, in particular, the
libraries `CUDA Runtime Library`, `cuBLAS` and `cuSPARSE`).

There are a set of files required for total use of this package. They will download automatically the first time they are needed. Files are generally under 10MB. However, there is a 100MB file needed for the slow waveform and the bicubic amplitude interpolation. This larger file will only download if you run either of those two modules. The files are hosted on the [Black Hole Perturbation Toolkit Download Server](https://download.bhptoolkit.org/few/data).

#### Running the installation

To start the from-source installation, ensure the pre-requisite are met, clone the repository, and then simply run a `pip install` command:

```sh
# Clone the repository
git clone https://github.com/BlackHolePerturbationToolkit/FastEMRIWaveforms.git
cd FastEMRIWaveforms

# Run the install
pip install .
```

Many options are available to change the installation behaviour. These can be set by adding `--config-settings=cmake.define.OPTION_NAME=OPTION_VALUE` to the `pip` command. Available options are:

- `FEW_LAPACKE_FETCH=ON|OFF|[AUTO]`: Whether `LAPACK` and `LAPACKE` should be automatically fetched from internet.
  - `ON`: ignore pre-installed `LAPACK(E)` and always fetch and compile their sources
  - `OFF`: disable `LAPACK(E)` fetching and only use pre-installed library and headers (install will fail if pre-installed lib and headers are not available)
  - `AUTO` (default): try to detect pre-installed `LAPACK(E)` and their headers. If found, use them, otherwise fetch `LAPACK(E)`.
- `FEW_LAPACKE_DETECT_WITH=[CMAKE]|PKGCONFIG`: How `LAPACK(E)` should be detected
  - `CMAKE`: `LAPACK(E)` will be detected using the cmake `FindPackage` command. If your `LAPACK(E)` install provides `lapacke-config.cmake` in a non-standard location, add its path to the `CMAKE_PREFIX_PATH` environment variable.
  - `PKGCONFIG`: `LAPACK(E)` will be detected using `pkg-config` by searching for the files `lapack.pc` and `lapacke.pc` . If these files are provided by your `LAPACK(E)` install in a non-standard location, add their path to the environment variable `PKG_CONFIG_PATH`
  - `AUTO` (default): attempt both CMake and PkgConfig approaches
- `FEW_WITH_GPU=ON|OFF|[AUTO]`: Whether GPU-support must be enabled
  - `ON`: Forcefully enable GPU support (install will fail if GPU prerequisites are not met)
  - `OFF`: Disable GPU support
  - `AUTO` (default): Check whether `nvcc` and the `CUDA Toolkit` are available in environment and enable/disable GPU support accordingly.
- `FEW_CUDA_ARCH`: List of CUDA architectures that will be targeted by the CUDA compiler using [CMake CUDA_ARCHITECTURES](https://cmake.org/cmake/help/latest/prop_tgt/CUDA_ARCHITECTURES.html) syntax. (Default = `all`).

Example of custom install with specific options to forcefully enable GPU support with support for the host's GPU only (`native` architecture) using LAPACK fetched from internet:

```sh
pip install . \
  --config-settings=cmake.define.FEW_WITH_GPU=ON \
  --config-settings=cmake.define.FEW_CUDA_ARCH="native" \
  --config-settings=cmake.define.FEW_LAPACKE_FETCH=ON
```

If you enabled `GPU` support (or it was automatically enabled by the `AUTO` mode), you will also need to install the `nvidia-cuda-runtime`
package corresponding to the CUDA version detected by `nvidia-smi` as explained in the *Getting Started* section above.
You will also need to manually install `cupy-cuda11x` or `cupy-cuda12x` according to your CUDA version.

Please contact the developers if the installation does not work.


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
$ python -m unittest discover
...
----------------------------------------------------------------------
Ran 20 tests in 71.514s
OK
```

## Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct, and the process for submitting pull requests to us.

If you want to develop FEW and produce documentation, install `few` from source with the `[dev]` label and in `editable` mode:

```
$ git clone https://github.com/BlackHolePerturbationToolkit/FastEMRIWaveforms.git
$ cd FastEMRIWaveforms
pip install -e '.[dev, testing]'
```

This will install necessary packages for building the documentation (`sphinx`, `pypandoc`, `sphinx_rtd_theme`, `nbsphinx`) and to run the tests.

The documentation source files are in `docs/source`. To compile the documentation, change to the `docs` directory and run `make html`.

## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/BlackHolePerturbationToolkit/FastEMRIWaveforms/tags).

## Authors/Developers

* **Michael Katz**
* Lorenzo Speri
* Christian Chapman-Bird
* Alvin J. K. Chua
* Niels Warburton
* Scott Hughes

### Contributors

* Philip Lynch
* Soichiro Isoyama
* Ryuichi Fujita
* Monica Rizzo

## License

This project is licensed under the GNU License - see the [LICENSE](LICENSE) file for details.

## Citation

Please make sure to cite FEW papers and the FEW software on [Zenodo](https://zenodo.org/record/3969004). There are other papers that require citation based on the classes used. For most classes this applies to, you can find these by checking the `citation` attribute for that class. All references are detailed in the [CITATION.cff](CITATION.cff) file.

## Acknowledgments

* This research resulting in this code was supported by National Science Foundation under grant DGE-0948017 and the Chateaubriand Fellowship from the Office for Science \& Technology of the Embassy of France in the United States.
* It was also supported in part through the computational resources and staff contributions provided for the Quest/Grail high performance computing facility at Northwestern University.
