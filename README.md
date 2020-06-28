# FastEMRIWaveforms

# gce: GPU-Accelerated Condtional Entropy

gce implements the Conditional Entropy (CE, [arXiv:1306.6664](https://arxiv.org/abs/1306.6664)) search technique for periodic objects in electromagnetic surveys. This implementation is specifically for use with graphics processing units (GPU). The user inputs light curves as well as grid parameters used for each CE calculation. Basic statistics are calculated for these CE grids, which are then used to assign a significance to the minimum CE value. The minimum CE value indicates the most likely set of parameters. Documentation for this package can be found [here](https://mikekatz04.github.io/gce/).

If you use all or any parts of this code, please cite "GPU-Accelerated Periodic Source Identification in Large-Scale Surveys: Measuring P and Pdot" ([arXiv:2006.06866](https://arxiv.org/abs/2006.06866)) and the original Conditional Entropy paper ([arXiv:1306.6664](https://arxiv.org/abs/1306.6664)).

## Getting Started

0) [Install Anaconda](https://docs.anaconda.com/anaconda/install/) if you do not have it.

1) Create a virtual environment.

```
conda create -n gce_env numpy Cython scipy tqdm python=3.8
conda activate gce_env
```

2) Use pip to [install cupy](https://docs-cupy.chainer.org/en/stable/install.html). If you have cuda version 9.2, for example:

```
pip install cupy-cuda92
```

3) Clone the repository.

```
git clone https://github.com/mikekatz04/gce.git
```

4) Run install. Make sure CUDA is on your PATH.

```
python setup.py install
```

5) To import gce:

```
from gcex.gce import ConditionalEntropy
```

6) Perform test:

```
python tests/test.py
```

### Prerequisites

To install this software, you need the [CUDA toolkit](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html), Python >3.4, numpy, and [cupy](https://cupy.chainer.org/). The CUDA toolkit must have cuda version >8.0. To run this software you need an NVIDIA GPU of compute capability >2.0.

### Installing


1) Clone the repository.

```
git clone https://github.com/mikekatz04/gce.git
```

2) Run install. Make sure CUDA is on your PATH.

```
python setup.py install
```

## Running on CPU vs. GPU

This code is designed to run on a GPU. However, it does have a CPU version that is very slow (implemented in python). This capability was added to test code locally before running on the GPUs. Therefore, if you run the test file (see above) or any other files locally on a CPU, it should run but very slowly. However, if it runs on the CPU locally, and the GPU version is installed correctly, a direct transition from the local version to the GPU version will work without making any changes to the gce interface.


## Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct, and the process for submitting pull requests to us.

## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/mikekatz04/gce/tags).

Current Version: 0.1.0

## Authors

* **Michael Katz**

### Contibutors

* Michael Coughlin
* Olivia Cooper
* Kevin Burdge

## License

This project is licensed under the GNU License - see the [LICENSE.md](LICENSE.md) file for details.

## Acknowledgments

* Katie Breivik
* Shane Larson
* This research resulting in this code was supported by National Science Foundation under grant DGE-0948017 and the Chateaubriand Fellowship from the Office for Science \& Technology of the Embassy of France in the United States.
* It was also supported in part through the computational resources and staff contributions provided for the Quest/Grail high performance computing facility at Northwestern University.
