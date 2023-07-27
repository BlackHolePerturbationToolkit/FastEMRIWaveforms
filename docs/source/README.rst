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
`here <https://zenodo.org/record/8190418>`__. Please see the
`citation <#citation>`__ section below for information on citing FEW.

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

1) Clone the repository.

::

   git clone https://github.com/BlackHolePerturbationToolkit/FastEMRIWaveforms.git
   cd FastEMRIWaveforms

2) Install FEW into a new conda environment called “few_env”. (**Note**:
   If you already have performed installation and you are updating FEW
   after a ``git pull``, then run ``pip install .`` rather than the
   following command.)

::

   bash install.sh

3) Load the environment:

::

   conda activate few_env

4) In a python file or notebook:

::

   import few

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
it as the ``CUDA_HOME`` environment variable.

There are a set of files required for total use of this package. They
will download automatically the first time they are needed. Files are
generally under 10MB. However, there is a 100MB file needed for the slow
waveform and the bicubic amplitude interpolation. This larger file will
only download if you run either of those two modules. The files are
hosted on `Zenodo <https://zenodo.org/record/3981654#.XzS_KRNKjlw>`__.

Installing
~~~~~~~~~~

0) `Install Anaconda <https://docs.anaconda.com/anaconda/install/>`__ if
   you do not have it.

1) Clone the repository.

::

   git clone https://github.com/BlackHolePerturbationToolkit/FastEMRIWaveforms.git
   cd FastEMRIWaveforms

2) Installation is made easy through `install.sh <install.sh>`__. This
   is a bash script that will create a conda environment, install FEW,
   run tests, and install any additional packages needed for sampling or
   development. It will look for an ``nvcc`` binary, the ``CUDA_HOME``
   variable, or the ``CUDAHOME`` variable. If it finds that information,
   it will install for CUDA as well (including installing the proper
   version of ``cupy``). **Note**: If you already have performed
   installation and you are updating FEW after a ``git pull``, then run
   ``pip install .`` rather than the following command.

::

   bash install.sh

Options for installation can be applied by running
``bash install.sh key=value``. These can be found with
``bash install.sh -h``:

::

   keyword argument options (given as key=value):
     env_name:  Name of generated conda environment. Default is 'few_env'.
     install_type:  Type of install. 'basic', 'development', or 'sampling'. 
         'development' adds packages needed for development and documentation.
         'sampling' adds packages for sampling like eryn, lisatools, corner, chainconsumer.
         Default is 'basic'. 
     run_tests: Either true or false. Whether to run tests after install. Default is true.

3) Load the environment (change “few_env” to the correct environment
   name is specified in previous step):

::

   conda activate few_env

Please contact the developers if the installation does not work.

More Customized Installation (legacy)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
   usage: setup.py [-h] [--lapack_lib LAPACK_LIB]
                   [--lapack_include LAPACK_INCLUDE] [--lapack LAPACK]
                   [--gsl_lib GSL_LIB] [--gsl_include GSL_INCLUDE] [--gsl GSL]
                   [--ccbin CCBIN]

   optional arguments:
     -h, --help            show this help message and exit
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

Running the Tests
-----------------

In the main directory of the package run in the terminal (if you run
`install.sh <install.sh>`__ with defaults, the tests will be performed):

::

   python -m unittest discover

Contributing
------------

Please read `CONTRIBUTING.md <CONTRIBUTING.md>`__ for details on our
code of conduct, and the process for submitting pull requests to us.

If you want to develop FEW and produce documentation, install ``few``
with

::

   bash install.sh install_type=development

This will install necessary packages for building the documentation
(``sphinx``, ``pypandoc``, ``sphinx_rtd_theme``, ``nbsphinx``). The
documentation source files are in ``docs/source``. To compile the
documentation, change to the ``docs`` directory and run ``make html``.

Versioning
----------

We use `SemVer <http://semver.org/>`__ for versioning. For the versions
available, see the `tags on this
repository <https://github.com/BlackHolePerturbationToolkit/FastEMRIWaveforms/tags>`__.

Current Version: 1.5.1

Authors/Developers
------------------

-  **Michael Katz**
-  Lorenzo Speri
-  Christian Chapman-Bird
-  Alvin J. K. Chua
-  Niels Warburton
-  Scott Hughes

Contibutors
~~~~~~~~~~~

-  Philip Lynch
-  Soichiro Isoyama
-  Ryuichi Fujita
-  Monica Rizzo

License
-------

This project is licensed under the GNU License - see the
`LICENSE.md <LICENSE.md>`__ file for details.

Citation
--------

Please make sure to cite FEW papers and the FEW software on
`Zenodo <https://zenodo.org/record/8190418>`__. There are other papers
that require citation based on the classes used. For most classes this
applies to, you can find these by checking the ``citation`` attribute
for that class. Below is a list of citable papers that have lead to the
development of FEW.

::

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

Acknowledgments
---------------

-  This research resulting in this code was supported by National
   Science Foundation under grant DGE-0948017 and the Chateaubriand
   Fellowship from the Office for Science & Technology of the Embassy of
   France in the United States.
-  It was also supported in part through the computational resources and
   staff contributions provided for the Quest/Grail high performance
   computing facility at Northwestern University.
