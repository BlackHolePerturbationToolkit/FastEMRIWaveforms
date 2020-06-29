# few: Fast EMRI Waveforms

This package contains the highly modular framework for fast and accurate extreme mass ratio inspiral (EMRI) waveforms from (TODO: add arXiv). The waveforms in this package combine a variety of separately accessible modules to form EMRI waveforms on both CPUs and GPUs. Generally, the modules fall into four categories: trajectory, amplitudes, summation, and utilities. Please see the documentation (TODO: add documentation site) for further information on these modules.

If you use all or any parts of this code, please cite (TODO: add papers to cite. Do we want this to be per module or general use.).

## Getting Started

0) [Install Anaconda](https://docs.anaconda.com/anaconda/install/) if you do not have it.

1) Create a virtual environment.

```
conda create -n few_env numpy Cython scipy tqdm python=3.8
conda activate few_env
```

2) Use pip to [install cupy](https://docs-cupy.chainer.org/en/stable/install.html). If you have cuda version 9.2, for example:

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

5) To import few:

```
TODO: import of high level waveform
```

6) Perform test:

```
python tests/test.py
```

### Prerequisites

To install this software, you need the [CUDA toolkit](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html), [gsl] (https://www.gnu.org/software/gsl/), [lapack](https://www.netlib.org/lapack/lug/node14.html) Python >3.4, numpy, and [cupy](https://cupy.chainer.org/). The CUDA toolkit must have cuda version >8.0. To run this software you need an NVIDIA GPU of compute capability >2.0.

### Installing


1) Create a virtual environment.

```
conda create -n few_env numpy Cython scipy tqdm python=3.8
conda activate few_env
```

2) Use pip to [install cupy](https://docs-cupy.chainer.org/en/stable/install.html). If you have cuda version 9.2, for example:

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

## Running the Tests

TODO: fill in


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
