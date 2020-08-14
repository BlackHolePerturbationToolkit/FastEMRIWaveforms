# few: Fast EMRI Waveforms

This package contains the highly modular framework for fast and accurate extreme mass ratio inspiral (EMRI) waveforms from (TODO: add arXiv). The waveforms in this package combine a variety of separately accessible modules to form EMRI waveforms on both CPUs and GPUs. Generally, the modules fall into four categories: trajectory, amplitudes, summation, and utilities. Please see the [documentation](https://bhptoolkit.org/FastEMRIWaveforms/) for further information on these modules. The code can be found on Github [here](https://github.com/BlackHolePerturbationToolkit/FastEMRIWaveforms). The data necessary for various modules in this package will automatically download the first time it is needed. If you would like to view the data, it can be found on [Zenodo](https://zenodo.org/record/3981654#.XzS_KRNKjlw).

This package is a part of the [Black Hole Perturbation Toolkit](https://bhptoolkit.org/).

If you use all or any parts of this code, please cite (TODO: add papers to cite. Do we want this to be per module or general use.). See the [documentation](https://bhptoolkit.org/FastEMRIWaveforms/) to properly cite specific modules.

## Getting Started

Below is a quick set of instructions to get you started with `few`.

0) [Install Anaconda](https://bhptoolkit.org/FastEMRIWaveforms/) if you do not have it.

1) Create a virtual environment.

```
conda create -n few_env -c conda-forge gcc_linux-64 gxx_linux-64 gsl lapack hdf5 numpy Cython scipy tqdm jupyter ipython h5py requests matplotlib python=3.7
conda activate few_env
```

    If on MACOSX, substitute `gcc_linux-64` and `gxx_linus-64` with `clang_osx-64` and `clangxx_osx-64`.
    If on Windows, substitute `gcc_linux-64` and `gxx_linus-64` with (TODO: Fill in) and (TODO: Fill in).

2) Clone the repository.

```
git clone https://github.com/BlackHolePerturbationToolkit/FastEMRIWaveforms.git
cd FastEMRIWaveforms
```

3) Run install. Make sure CUDA is on your PATH.

```
python setup.py install
```

4) To import few:

```
from few.waveform import FastSchwarzschildEccentricFlux
```

See [examples notebook](examples/SchwarzschildEccentricWaveform_intro.ipynb).


### Prerequisites

To install this software for CPU usage, you need [gsl >2.0](https://www.gnu.org/software/gsl/) , [lapack](https://www.netlib.org/lapack/lug/node14.html), Python >3.4, and NumPy. To run the examples, you will also need jupyter and matplotlib. We generally recommend installing everything, including gcc and g++ compilers, in the conda environment as is shown in the examples here. This generally helps avoid compilation and linking issues. If you use your own chosen compiler, you will need to make sure all necessary information is passed to the setup command (see below). You also may need to add information to the `setup.py` file.

To install this software for use with NVIDIA GPUs (compute capability >2.0), you need the [CUDA toolkit](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html) and [CuPy](https://cupy.chainer.org/). The CUDA toolkit must have cuda version >8.0. Be sure to properly install CuPy within the correct CUDA toolkit version. Make sure the nvcc binary is on `$PATH` or set it as the `CUDAHOME` environment variable.

There are a set of files required for total use of this package. They will download automatically when the first time they are needed. Files are generally under 10MB. However, there is a 100MB file needed for the slow waveform and the bicubic amplitude interpolation. This larger file will only download if you run either of those two modules. The files are hosted on [Zenodo](https://zenodo.org/record/3981654#.XzS_KRNKjlw).

### Installing


0) [Install Anaconda](https://docs.anaconda.com/anaconda/install/) if you do not have it.

1) Create a virtual environment.

```
conda create -n few_env -c conda-forge gcc_linux-64 gxx_linux-64 gsl lapack hdf5 numpy Cython scipy tqdm jupyter ipython h5py requests matplotlib python=3.7
conda activate few_env
```

    If on MACOSX, substitute `gcc_linux-64` and `gxx_linus-64` with `clang_osx-64` and `clangxx_osx-64`.
    If on Windows, substitute `gcc_linux-64` and `gxx_linus-64` with (TODO: Fill in) and (TODO: Fill in).

    If you want a faster install, you can install the python packages (numpy, Cython, scipy, tqdm, jupyter, ipython, h5py, requests, matplotlib) with pip

2) Clone the repository.

```
git clone https://github.com/BlackHolePerturbationToolkit/FastEMRIWaveforms.git
cd FastEMRIWaveforms
```

2) If using GPUs, use pip to [install cupy](https://docs-cupy.chainer.org/en/stable/install.html). If you have cuda version 9.2, for example:

```
pip install cupy-cuda92
```

3) Clone the repository.

```
git clone https://github.com/mikekatz04/FastEMRIWaveforms.git
```

4) Run install. Make sure CUDA is on your PATH.

```
python setup.py install
```


When installing lapack and gsl, the setup file will default to assuming lib and include for both are in `/usr/local/opt/lapack/` and `/usr/local/opt/gsl/`. To provide other lib and include directories you can provide command line options when installing. You can also remove usage of OpenMP.

```
python setup.py --help
usage: setup.py [-h] [--no_omp] [--lapack_lib LAPACK_LIB]
                [--lapack_include LAPACK_INCLUDE] [--lapack LAPACK]
                [--gsl_lib GSL_LIB] [--gsl_include GSL_INCLUDE] [--gsl GSL]

optional arguments:
  -h, --help            show this help message and exit
  --no_omp              If provided, install without OpenMP.
  --lapack_lib LAPACK_LIB
                        Directory of the lapack lib.
  --lapack_include LAPACK_INCLUDE
                        Directory of the lapack include.
  --lapack LAPACK       Directory of both lapack lib and include. '/include'
                        and '/lib' will be added to the end of this string.
  --gsl_lib GSL_LIB     Directory of the gsl lib.
  --gsl_include GSL_INCLUDE
                        Directory of the gsl include.
  --gsl GSL             Directory of both gsl lib and include. '/include' and
                        '/lib' will be added to the end of this string.
```



## Running the Tests

In the main directory of the package run in the terminal:
```
python -m unittest discover
```


## Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct, and the process for submitting pull requests to us.

## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/BlackHolePerturbationToolkit/FastEMRIWaveforms/tags).

Current Version: 1.0.0

## Authors

* **Michael Katz**
* Alvin J. K. Chua
* Niels Warburton

### Contibutors

TODO: add people

## License

This project is licensed under the GNU License - see the [LICENSE.md](LICENSE.md) file for details.

## Acknowledgments

* This research resulting in this code was supported by National Science Foundation under grant DGE-0948017 and the Chateaubriand Fellowship from the Office for Science \& Technology of the Embassy of France in the United States.
* It was also supported in part through the computational resources and staff contributions provided for the Quest/Grail high performance computing facility at Northwestern University.
