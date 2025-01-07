# few: Fast EMRI Waveforms

This package contains the highly modular framework for fast and accurate extreme mass ratio inspiral (EMRI) waveforms from [arxiv.org/2104.04582](https://arxiv.org/abs/2104.04582) and [arxiv.org/2008.06071](https://arxiv.org/abs/2008.06071). The waveforms in this package combine a variety of separately accessible modules to form EMRI waveforms on both CPUs and GPUs. Generally, the modules fall into four categories: trajectory, amplitudes, summation, and utilities. Please see the [documentation](https://bhptoolkit.org/FastEMRIWaveforms/) for further information on these modules. The code can be found on Github [here](https://github.com/BlackHolePerturbationToolkit/FastEMRIWaveforms). The data necessary for various modules in this package will automatically download the first time it is needed. If you would like to view the data, it can be found on [Zenodo](https://zenodo.org/record/3981654#.XzS_KRNKjlw). The current and all past code release zip files can also be found on Zenodo [here](https://zenodo.org/record/8190418). Please see the [citation](#citation) section below for information on citing FEW.

This package is a part of the [Black Hole Perturbation Toolkit](https://bhptoolkit.org/).

If you use all or any parts of this code, please cite [arxiv.org/2104.04582](https://arxiv.org/abs/2104.04582) and [arxiv.org/2008.06071](https://arxiv.org/abs/2008.06071). See the [documentation](https://bhptoolkit.org/FastEMRIWaveforms/) to properly cite specific modules.

## Getting started

To install the latest pre-compiled version of `fastemriwaveforms`, simply run:

```sh
# For CPU-only version
pip install fastemriwaveforms-cpu

# For GPU-enabled versions with CUDA 11.Y.Z
pip install fastemriwaveforms-cuda11x 'nvidia-cuda-runtime-cu11==11.Y.*'

# For GPU-enabled versions with CUDA 12.Y.Z
pip install fastemriwaveforms-cuda12x 'nvidia-cuda-runtime-cu12==12.Y.*'
```

To know your CUDA version, run the tool `nvidia-smi` in a terminal a check the CUDA version reported in the table header:

```sh
$ nvidia-smi
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 550.54.15              Driver Version: 550.54.15      CUDA Version: 12.4     |
|-----------------------------------------+------------------------+----------------------+
...

# CUDA version is 12.4, so run the following command to properly install FEW with support for CUDA 12.4
$ pip install fastemriwaveforms-cuda12x 'nvidia-cuda-runtime-cu12==12.4.*'
```

Now, in a python file or notebook:

```py3
import few
```

See [examples notebook](https://github.com/BlackHolePerturbationToolkit/FastEMRIWaveforms/blob/master/examples/FastEMRIWaveforms_tutorial.ipynb).

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
  - `CMAKE` (default): `LAPACK(E)` will be detected using the cmake `FindPackage` command. If your `LAPACK(E)` install provides `lapacke-config.cmake` in a non-standard location, add its path to the `CMAKE_PREFIX_PATH` environment variable.
  - `PKGCONFIG`: `LAPACK(E)` will be detected using `pkg-config` by searching for the files `lapack.pc` and `lapacke.pc` . If these files are provided by your `LAPACK(E)` install in a non-standard location, add their path to the environment variable `PKG_CONFIG_PATH`
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
pip install .[testing]
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

### Contibutors

* Philip Lynch
* Soichiro Isoyama
* Ryuichi Fujita
* Monica Rizzo

## License

This project is licensed under the GNU License - see the [LICENSE.md](LICENSE.md) file for details.

## Citation

Please make sure to cite FEW papers and the FEW software on [Zenodo](https://zenodo.org/record/8190418). There are other papers that require citation based on the classes used. For most classes this applies to, you can find these by checking the `citation` attribute for that class. Below is a list of citable papers that have lead to the development of FEW.

```
@article{Chua:2020stf,
    author = "Chua, Alvin J. K. and Katz, Michael L. and Warburton, Niels and Hughes, Scott A.",
    title = "{Rapid generation of fully relativistic extreme-mass-ratio-inspiral waveform templates for LISA data analysis}",
    eprint = "2008.06071",
    archivePrefix = "arXiv",
    primaryClass = "gr-qc",
    doi = "10.1103/PhysRevLett.126.051102",
    journal = "Phys. Rev. Lett.",
    volume = "126",
    number = "5",
    pages = "051102",
    year = "2021"
}

@article{Katz:2021yft,
    author = "Katz, Michael L. and Chua, Alvin J. K. and Speri, Lorenzo and Warburton, Niels and Hughes, Scott A.",
    title = "{FastEMRIWaveforms: New tools for millihertz gravitational-wave data analysis}",
    eprint = "2104.04582",
    archivePrefix = "arXiv",
    primaryClass = "gr-qc",
    month = "4",
    year = "2021"
}

@software{michael_l_katz_2023_8190418,
  author       = {Michael L. Katz and
                  Lorenzo Speri and
                  Alvin J. K. Chua and
                  Christian E. A. Chapman-Bird and
                  Niels Warburton and
                  Scott A. Hughes},
  title        = {{BlackHolePerturbationToolkit/FastEMRIWaveforms:
                   Frequency Domain Waveform Added!}},
  month        = jul,
  year         = 2023,
  publisher    = {Zenodo},
  version      = {v1.5.1},
  doi          = {10.5281/zenodo.8190418},
  url          = {https://doi.org/10.5281/zenodo.8190418}
}

@article{Chua:2018woh,
    author = "Chua, Alvin J.K. and Galley, Chad R. and Vallisneri, Michele",
    title = "{Reduced-order modeling with artificial neurons for gravitational-wave inference}",
    eprint = "1811.05491",
    archivePrefix = "arXiv",
    primaryClass = "astro-ph.IM",
    doi = "10.1103/PhysRevLett.122.211101",
    journal = "Phys. Rev. Lett.",
    volume = "122",
    number = "21",
    pages = "211101",
    year = "2019"
}

@article{Fujita:2020zxe,
    author = "Fujita, Ryuichi and Shibata, Masaru",
    title = "{Extreme mass ratio inspirals on the equatorial plane in the adiabatic order}",
    eprint = "2008.13554",
    archivePrefix = "arXiv",
    primaryClass = "gr-qc",
    doi = "10.1103/PhysRevD.102.064005",
    journal = "Phys. Rev. D",
    volume = "102",
    number = "6",
    pages = "064005",
    year = "2020"
}

@article{Stein:2019buj,
    author = "Stein, Leo C. and Warburton, Niels",
    title = "{Location of the last stable orbit in Kerr spacetime}",
    eprint = "1912.07609",
    archivePrefix = "arXiv",
    primaryClass = "gr-qc",
    doi = "10.1103/PhysRevD.101.064007",
    journal = "Phys. Rev. D",
    volume = "101",
    number = "6",
    pages = "064007",
    year = "2020"
}

@article{Chua:2015mua,
    author = "Chua, Alvin J.K. and Gair, Jonathan R.",
    title = "{Improved analytic extreme-mass-ratio inspiral model for scoping out eLISA data analysis}",
    eprint = "1510.06245",
    archivePrefix = "arXiv",
    primaryClass = "gr-qc",
    doi = "10.1088/0264-9381/32/23/232002",
    journal = "Class. Quant. Grav.",
    volume = "32",
    pages = "232002",
    year = "2015"
}

@article{Chua:2017ujo,
    author = "Chua, Alvin J.K. and Moore, Christopher J. and Gair, Jonathan R.",
    title = "{Augmented kludge waveforms for detecting extreme-mass-ratio inspirals}",
    eprint = "1705.04259",
    archivePrefix = "arXiv",
    primaryClass = "gr-qc",
    doi = "10.1103/PhysRevD.96.044005",
    journal = "Phys. Rev. D",
    volume = "96",
    number = "4",
    pages = "044005",
    year = "2017"
}

@article{Barack:2003fp,
    author = "Barack, Leor and Cutler, Curt",
    title = "{LISA capture sources: Approximate waveforms, signal-to-noise ratios, and parameter estimation accuracy}",
    eprint = "gr-qc/0310125",
    archivePrefix = "arXiv",
    doi = "10.1103/PhysRevD.69.082005",
    journal = "Phys. Rev. D",
    volume = "69",
    pages = "082005",
    year = "2004"
}

@article{Speri:2023jte,
    author = "Speri, Lorenzo and Katz, Michael L. and Chua, Alvin J. K. and Hughes, Scott A. and Warburton, Niels and Thompson, Jonathan E. and Chapman-Bird, Christian E. A. and Gair, Jonathan R.",
    title = "{Fast and Fourier: Extreme Mass Ratio Inspiral Waveforms in the Frequency Domain}",
    eprint = "2307.12585",
    archivePrefix = "arXiv",
    primaryClass = "gr-qc",
    month = "7",
    year = "2023"
}
```

## Acknowledgments

* This research resulting in this code was supported by National Science Foundation under grant DGE-0948017 and the Chateaubriand Fellowship from the Office for Science \& Technology of the Embassy of France in the United States.
* It was also supported in part through the computational resources and staff contributions provided for the Quest/Grail high performance computing facility at Northwestern University.
