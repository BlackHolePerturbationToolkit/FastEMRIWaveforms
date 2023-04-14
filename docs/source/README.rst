few: Fast EMRI Waveforms
========================

This package contains the highly modular framework for fast and accurate
extreme mass ratio inspiral (EMRI) waveforms from
`arxiv.org/2104.04582 <https://arxiv.org/abs/2104.04582>`__ and
`arxiv.org/2008.06071 <https://arxiv.org/abs/2008.06071>`__. The
waveforms in this package combine a variety of separately accessible
modules to form EMRI waveforms on both CPUs and GPUs. Generally, the
modules fall into four categories: trajectory, amplitudes, summation,
and utilities. Please see the
`documentation <https://bhptoolkit.org/FastEMRIWaveforms/>`__ for
further information on these modules. The code can be found on Github
`here <https://github.com/BlackHolePerturbationToolkit/FastEMRIWaveforms>`__.
The data necessary for various modules in this package will
automatically download the first time it is needed. If you would like to
view the data, it can be found on
`Zenodo <https://zenodo.org/record/3981654#.XzS_KRNKjlw>`__. The current
and all past code release zip files can also be found on Zenodo
`here <https://zenodo.org/record/4005001>`__.

This package is a part of the `Black Hole Perturbation
Toolkit <https://bhptoolkit.org/>`__.

If you use all or any parts of this code, please cite
`arxiv.org/2104.04582 <https://arxiv.org/abs/2104.04582>`__ and
`arxiv.org/2008.06071 <https://arxiv.org/abs/2008.06071>`__. See the
`documentation <https://bhptoolkit.org/FastEMRIWaveforms/>`__ to
properly cite specific modules.

Getting Started
---------------

Below is a quick set of instructions to get you started with ``few``.

0) `Install Anaconda <https://docs.anaconda.com/anaconda/install/>`__ if
   you do not have it.

1) Create a virtual environment. **Note**: There is no available
   ``conda`` compiler for Windows. If you want to install for Windows,
   you will probably need to add libraries and include paths to the
   ``setup.py`` file.

If on linux:

::

   conda create -n few_env -c conda-forge gcc_linux-64 gxx_linux-64 wget gsl lapack=3.6.1 hdf5 numpy Cython scipy tqdm jupyter ipython h5py requests matplotlib python=3.7
   conda activate few_env

If on MACOSX, substitute ``gcc_linux-64`` and ``gxx_linus-64`` with
``clang_osx-64`` and ``clangxx_osx-64`` as follows:

::

   conda create -n few_env -c conda-forge clangxx_osx-64 clang_osx-64 wget gsl lapack=3.6.1 hdf5 numpy Cython scipy tqdm jupyter ipython h5py requests matplotlib python=3.7
   conda activate few_env

If on M1 chip use the following command:

::

   conda create -n few_env -c conda-forge wget gsl hdf5 numpy Cython scipy tqdm jupyter ipython h5py requests matplotlib python=3.9 openblas lapack liblapacke
   conda activate few_env

2) Clone the repository.

::

   git clone https://github.com/BlackHolePerturbationToolkit/FastEMRIWaveforms.git
   cd FastEMRIWaveforms

3) If on MACOSX or linux run install:

::

   python setup.py install

If on M1 chip use the following command:

::

   python setup.py install --no_omp --ccbin /usr/bin/

4) To test few run:

::

   python -m unittest discover

See `examples
notebook <https://github.com/BlackHolePerturbationToolkit/FastEMRIWaveforms/blob/master/examples/FastEMRIWaveforms_tutorial.ipynb>`__.

Prerequisites
~~~~~~~~~~~~~

To install this software for CPU usage, you need `gsl
>2.0 <https://www.gnu.org/software/gsl/>`__ , `lapack
(3.6.1) <https://www.netlib.org/lapack/lug/node14.html>`__, Python >3.4,
wget, and NumPy. If you install lapack with conda, the new version (3.9)
seems to not install the correct header files. Therefore, the lapack
version must be 3.6.1. To run the examples, you will also need jupyter
and matplotlib. We generally recommend installing everything, including
gcc and g++ compilers, in the conda environment as is shown in the
examples here. This generally helps avoid compilation and linking
issues. If you use your own chosen compiler, you will need to make sure
all necessary information is passed to the setup command (see below).
You also may need to add information to the ``setup.py`` file.

To install this software for use with NVIDIA GPUs (compute capability
>2.0), you need the `CUDA
toolkit <https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html>`__
and `CuPy <https://cupy.chainer.org/>`__. The CUDA toolkit must have
cuda version >8.0. Be sure to properly install CuPy within the correct
CUDA toolkit version. Make sure the nvcc binary is on ``$PATH`` or set
it as the ``CUDAHOME`` environment variable.

There are a set of files required for total use of this package. They
will download automatically when the first time they are needed. Files
are generally under 10MB. However, there is a 100MB file needed for the
slow waveform and the bicubic amplitude interpolation. This larger file
will only download if you run either of those two modules. The files are
hosted on `Zenodo <https://zenodo.org/record/3981654#.XzS_KRNKjlw>`__.

Installing
~~~~~~~~~~

0) `Install Anaconda <https://docs.anaconda.com/anaconda/install/>`__ if
   you do not have it.

1) Create a virtual environment.

::

   conda create -n few_env -c conda-forge gcc_linux-64 gxx_linux-64 wget gsl lapack=3.6.1 hdf5 numpy Cython scipy tqdm jupyter ipython h5py requests matplotlib python=3.7
   conda activate few_env

::

   If on MACOSX, substitute `gcc_linux-64` and `gxx_linus-64` with `clang_osx-64` and `clangxx_osx-64`.

   If you want a faster install, you can install the python packages (numpy, Cython, scipy, tqdm, jupyter, ipython, h5py, requests, matplotlib) with pip.

2) Clone the repository.

::

   git clone https://github.com/BlackHolePerturbationToolkit/FastEMRIWaveforms.git
   cd FastEMRIWaveforms

3) If using GPUs, use pip to `install
   cupy <https://docs-cupy.chainer.org/en/stable/install.html>`__. If
   you have cuda version 9.2, for example:

::

   pip install cupy-cuda92

4) Run install.

::

   python setup.py install

When installing lapack and gsl, the setup file will default to assuming
lib and include for both are in installed within the conda environment.
To provide other lib and include directories you can provide command
line options when installing. You can also remove usage of OpenMP.

::

   python setup.py --help
   usage: setup.py [-h] [--no_omp] [--lapack_lib LAPACK_LIB]
                   [--lapack_include LAPACK_INCLUDE] [--lapack LAPACK]
                   [--gsl_lib GSL_LIB] [--gsl_include GSL_INCLUDE] [--gsl GSL]
                   [--ccbin CCBIN]

   optional arguments:
     -h, --help            show this help message and exit
     --no_omp              If provided, install without OpenMP.
     --lapack_lib LAPACK_LIB
                           Directory of the lapack lib. If you add lapack lib,
                           must also add lapack include.
     --lapack_include LAPACK_INCLUDE
                           Directory of the lapack include. If you add lapack
                           includ, must also add lapack lib.
     --lapack LAPACK       Directory of both lapack lib and include. '/include'
                           and '/lib' will be added to the end of this string.
     --gsl_lib GSL_LIB     Directory of the gsl lib. If you add gsl lib, must
                           also add gsl include.
     --gsl_include GSL_INCLUDE
                           Directory of the gsl include. If you add gsl include,
                           must also add gsl lib.
     --gsl GSL             Directory of both gsl lib and include. '/include' and
                           '/lib' will be added to the end of this string.
     --ccbin CCBIN         path/to/compiler to link with nvcc when installing
                           with CUDA.

When installing the package with ``python setup.py install``, the setup
file uses the C compiler present in your ``PATH``. However, it might
happen that the setup file incorrectly uses another compiler present on
your path. To solve this issue you can directly specify the C compiler
using the flag ``--ccbin`` as in the following example:

::

   python setup.py install --ccbin /path/to/anaconda3/envs/few_env/bin/x86_64-conda-linux-gnu-gcc

or if on MACOSX:

::

   python setup.py install --ccbin /path/to/anaconda3/envs/few_env/bin/x86_64-apple-darwin13.4.0-clang

Please contact the developers if the installation does not work.

Running the Tests
-----------------

In the main directory of the package run in the terminal:

::

   python -m unittest discover

Contributing
------------

Please read `CONTRIBUTING.md <CONTRIBUTING.md>`__ for details on our
code of conduct, and the process for submitting pull requests to us.

Versioning
----------

We use `SemVer <http://semver.org/>`__ for versioning. For the versions
available, see the `tags on this
repository <https://github.com/BlackHolePerturbationToolkit/FastEMRIWaveforms/tags>`__.

Current Version: 1.4.11

Authors
-------

-  **Michael Katz**
-  Alvin J. K. Chua
-  Niels Warburton
-  Lorenzo Speri
-  Scott Hughes

Contibutors
~~~~~~~~~~~

-  Philip Lynch
-  Christian Chapman-Bird
-  Soichiro Isoyama
-  Ryuichi Fujita
-  Monica Rizzo

License
-------

This project is licensed under the GNU License - see the
`LICENSE.md <LICENSE.md>`__ file for details.

Acknowledgments
---------------

-  This research resulting in this code was supported by National
   Science Foundation under grant DGE-0948017 and the Chateaubriand
   Fellowship from the Office for Science & Technology of the Embassy of
   France in the United States.
-  It was also supported in part through the computational resources and
   staff contributions provided for the Quest/Grail high performance
   computing facility at Northwestern University.
