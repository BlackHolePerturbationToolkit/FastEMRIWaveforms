few: Fast EMRI Waveforms
========================

This package contains the highly modular framework for fast and accurate
extreme mass ratio inspiral (EMRI) waveforms from (TODO: add arXiv). The
waveforms in this package combine a variety of separately accessible
modules to form EMRI waveforms on both CPUs and GPUs. Generally, the
modules fall into four categories: trajectory, amplitudes, summation,
and utilities. Please see the
`documentation <https://mikekatz04.github.io/FastEMRIWaveforms/>`__ for
further information on these modules.

If you use all or any parts of this code, please cite (TODO: add papers
to cite. Do we want this to be per module or general use.).

Getting Started
---------------

Below is a quick set of instructions to get you started with ``few``.

0) `Install Anaconda <https://docs.anaconda.com/anaconda/install/>`__ if
   you do not have it.

1) Create a virtual environment.

::

   conda create -n few_env numpy Cython scipy tqdm jupyter ipython python=3.8
   conda activate few_env

2) Clone the repository.

::

   git clone https://github.com/mikekatz04/FastEMRIWaveforms.git

3) Run install. Make sure CUDA is on your PATH.

::

   python setup.py install

4) To import few:

::

   from few.waveforms import FastSchwarzschildEccentricFlux

See `examples
notebook <examples/SchwarzschildEccentricWaveform_intro.ipynb>`__.

Prerequisites
~~~~~~~~~~~~~

To install this software for CPU usage, you need [gsl >2.0]
(https://www.gnu.org/software/gsl/) ,
`lapack <https://www.netlib.org/lapack/lug/node14.html>`__, Python >3.4,
and NumPy. To run the examples, you will also need jupyter and
matplotlib.

To install this software for use with NVIDIA GPUs (compute capability
>2.0), you need the `CUDA
toolkit <https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html>`__
and `CuPy <https://cupy.chainer.org/>`__. The CUDA toolkit must have
cuda version >8.0. Be sure to properly install cupy within the correct
CUDA toolkit version.

Installing
~~~~~~~~~~

0) `Install Anaconda <https://docs.anaconda.com/anaconda/install/>`__ if
   you do not have it.

1) Create a virtual environment.

::

   conda create -n few_env numpy Cython scipy tqdm jupyter ipython python=3.8
   conda activate few_env

2) If using GPUs, use pip to `install
   cupy <https://docs-cupy.chainer.org/en/stable/install.html>`__. If
   you have cuda version 9.2, for example:

::

   pip install cupy-cuda92

3) Clone the repository.

::

   git clone https://github.com/mikekatz04/FastEMRIWaveforms.git

4) Run install. Make sure CUDA is on your PATH.

::

   python setup.py install

5) Get data extra data if desired. If you want to run the slow waveforms
   (`few.waveform.SlowSchwarzschildEccentricFlux <https://mikekatz04.github.io/FastEMRIWaveforms/html/index.html?highlight=slow#few.waveform.SlowSchwarzschildEccentricFlux>`__)
   or bicubic amplitude determination
   (`few.amplitudes.interp2dcubicspline.Interp2DAmplitude <https://mikekatz04.github.io/FastEMRIWaveforms/html/index.html?highlight=interp2d#few.amplitude.interp2dcubicspline.Interp2DAmplitude>`__),
   you will need to attain an hdf5 file
   (Teuk_amps_a0.0_lmax_10_nmax_30_new.h5) and put it in the
   ``/Path/to/Installation/few/files/`` directory. This is available at
   Zenodo (TODO: fill in).

Running the Tests
-----------------

When performing tests, you must have the hdf5 file
(Teuk_amps_a0.0_lmax_10_nmax_30_new.h5) for
(`few.waveform.SlowSchwarzschildEccentricFlux <https://mikekatz04.github.io/FastEMRIWaveforms/html/index.html?highlight=slow#few.waveform.SlowSchwarzschildEccentricFlux>`__)
and bicubic amplitude determination
(`few.amplitudes.interp2dcubicspline.Interp2DAmplitude <https://mikekatz04.github.io/FastEMRIWaveforms/html/index.html?highlight=interp2d#few.amplitude.interp2dcubicspline.Interp2DAmplitude>`__).
See above in **Installing**.

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
repository <https://github.com/mikekatz04/gce/tags>`__.

Current Version: 0.1.0

Authors
-------

-  **Michael Katz**
-  Alvin J. K. Chua
-  Niels Warburton

Contibutors
~~~~~~~~~~~

TODO: add people

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
