# few: Fast EMRI Waveforms

This package contains the highly modular framework for fast and accurate extreme mass ratio inspiral (EMRI) waveforms from (TODO: add arXiv). The waveforms in this package combine a variety of separately accessible modules to form EMRI waveforms on both CPUs and GPUs. Generally, the modules fall into four categories: trajectory, amplitudes, summation, and utilities. Please see the [documentation](https://mikekatz04.github.io/FastEMRIWaveforms/) for further information on these modules.

If you use all or any parts of this code, please cite (TODO: add papers to cite. Do we want this to be per module or general use.).

## Getting Started

Below is a quick set of instructions to get you started with `few`.

0) [Install Anaconda](https://docs.anaconda.com/anaconda/install/) if you do not have it.

1) Create a virtual environment.

```
conda create -n few_env numpy Cython scipy tqdm jupyter ipython python=3.8
conda activate few_env
```

2) Clone the repository.

```
git clone https://github.com/mikekatz04/FastEMRIWaveforms.git
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

To install this software for CPU usage, you need [gsl >2.0](https://www.gnu.org/software/gsl/) , [lapack](https://www.netlib.org/lapack/lug/node14.html), Python >3.4, and NumPy. To run the examples, you will also need jupyter and matplotlib. For Python packages, we generally recommend installing within a conda environment. For gsl and lapack, it may be better to use brew or apt-get. If you want to run with OpenMP, make sure that is installed.

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

To install this software for use with NVIDIA GPUs (compute capability >2.0), you need the [CUDA toolkit](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html) and [CuPy](https://cupy.chainer.org/). The CUDA toolkit must have cuda version >8.0. Be sure to properly install cupy within the correct CUDA toolkit version. Make sure the nvcc binary is on `$PATH` or set it as the `CUDAHOME` environment variable.

### Installing


0) [Install Anaconda](https://docs.anaconda.com/anaconda/install/) if you do not have it.

1) Create a virtual environment.

```
conda create -n few_env numpy Cython scipy tqdm jupyter ipython python=3.8
conda activate few_env
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

5) Get data extra data if desired. If you want to run the slow waveforms ([few.waveform.SlowSchwarzschildEccentricFlux](https://mikekatz04.github.io/FastEMRIWaveforms/html/index.html?highlight=slow#few.waveform.SlowSchwarzschildEccentricFlux)) or bicubic amplitude determination ([few.amplitudes.interp2dcubicspline.Interp2DAmplitude](https://mikekatz04.github.io/FastEMRIWaveforms/html/index.html?highlight=interp2d#few.amplitude.interp2dcubicspline.Interp2DAmplitude)), you will need to attain an hdf5 file (Teuk_amps_a0.0_lmax_10_nmax_30_new.h5) and put it in the `/Path/to/Installation/few/files/` directory. This is available at Zenodo (TODO: fill in).


## Running the Tests

When performing tests, you must have the hdf5 file (Teuk_amps_a0.0_lmax_10_nmax_30_new.h5) for ([few.waveform.SlowSchwarzschildEccentricFlux](https://mikekatz04.github.io/FastEMRIWaveforms/html/index.html?highlight=slow#few.waveform.SlowSchwarzschildEccentricFlux)) and bicubic amplitude determination ([few.amplitudes.interp2dcubicspline.Interp2DAmplitude](https://mikekatz04.github.io/FastEMRIWaveforms/html/index.html?highlight=interp2d#few.amplitude.interp2dcubicspline.Interp2DAmplitude)). See above in **Installing**.

In the main directory of the package run in the terminal:
```
python -m unittest discover
```


## Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct, and the process for submitting pull requests to us.

## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/mikekatz04/gce/tags).

Current Version: 0.1.0

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
