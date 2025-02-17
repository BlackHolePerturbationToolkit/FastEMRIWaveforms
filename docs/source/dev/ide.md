# Development Environment

Developping on FEW requires the following:

- A Python 3.9+ interpreter (ideally in a dedicated virtualenv, or conda environment)
- A C++ compiler to build the CPU backend
- An IDE you are confortable with

If you have a local GPU available and want to also develop for the GPU backend, you will also need:

- CUDA drivers properly installed (the command `nvidia-smi` should detect a CUDA version >=11.2 and a GPU)
- The NVIDIA CUDA compiler `nvcc``
- The CUDA HPC Toolkit

To develop into FEW, the project must be installed from sources in editable mode.

- If you will only develop Python code, it is advised to use [standard editable](#install-the-project-in-standard-editable-mode) installation.
- If tou will also develop C++/CUDA code, backends will need to be recompiled after each change. This can be done automatically
  using the *inplace editable* installation

## Installation steps

### Clone the repository

To clone the repository, run the following command:

```bash
$ git clone https://github.com/BlackHolePerturbationToolkit/FastEMRIWaveforms.git
FastEMRIWaveforms$ cd FastEMRIWaveforms
```

### Create a dedicated virtualenv

To create a dedicated virtualenv, run the following command:

```bash
FastEMRIWaveforms$ python3 -m venv build/few-venv
FastEMRIWaveforms$ source build/venv/bin/activate
(few-venv) FastEMRIWaveforms$
```

### Install the project in standard editable mode

The following command will install the project in editable mode so that any change
made to Python code in `src/` will directly be reflected when the package is imported
in a fresh python environment:

```bash
(few-venv) FastEMRIWaveforms$ pip install -e '.[testing, doc]'
Obtaining file:///workspaces/FastEMRIWaveforms
...
Successfully installed ... fastemriwaveforms-1.5.1.post1.dev47+g5a3a237.d20250217 ...
```

Note that this command also installd the dependencies required to run the tests and to build the documentation.

Whenever you want to check tests are correctly running, simply run:

```bash
# To execute unit tests
(few-venv) FastEMRIWaveforms$ python -m unittest discover
...
----------------------------------------------------------------------
Ran 27 tests in 156.359s

OK

# To execute doc tests
(few-venv) FastEMRIWaveforms$ sphinx-build -M doctest docs/source docs/build
...
Doctest summary
===============
    7 tests
    0 failures in tests
    0 failures in setup code
    0 failures in cleanup code
build succeeded, 4 warnings.
```

### Install the project with backend recompilation (inplace editable mode)

If you need backends to be automatically rebuilt on changes, you can install the project using the `inplace`
editable install provided by
[scikit-build-core **in experimental mode**](https://scikit-build-core.readthedocs.io/en/latest/configuration.html#editable-installs).

Note that this mode requires to pre-install all of the project build dependencies before-hand.
**inplace editable mode**:

```bash
# 1. First, install the project build dependencies
(few-venv) FastEMRIWaveforms$ pip install cython numpy scikit-build-core ninja cmake setuptools_scm

# 2. Also, make sure that LAPACK(E) is properly installed on your system, for example on Ubuntu 24.04:
(few-venv) FastEMRIWaveforms$ sudo apt-get install liblapacke-dev

# 3. Install the project in inplace editable mode
# Make the install verbose since this mode is more likely to fail since it is still experimental
# Note that the building steps will be performed in ./build/editable
#  Do not delete that directory, it is used by the build system
(few-venv) FastEMRIWaveforms$ pip install --no-build-isolation -Ceditable.mode=redirect -Ceditable.rebuild=true  -Cbuild-dir=./build/editable -Ccmake.verbose=true -Clogging.level=INFO -Ccmake.define.FEW_LAPACKE_FETCH=OFF -v -e '.[testing, doc]'
...
  *** Making editable...
  *** Created fastemriwaveforms-1.6.3.post1.dev47+g5a3a237.d20250217-cp312-cp312-linux_aarch64.whl
  Building editable for fastemriwaveforms (pyproject.toml) ... done
...
Successfully installed fastemriwaveforms-1.5.1.post1.dev47+g5a3a237.d20250217
```

Now, when opening a Python terminal and importing the `few` package, recompilation will take place when needed (when
first making use of any compiled backend that needs recompiling). By default, this recompilation step is quiet and you
will only notice a longer-than-usual latency when `few` backends are first loaded.
You can make the recompilation verbose by setting the environment variable `SKBUILD_EDITABLE_VERBOSE=1`:

```bash
(few-venv) FastEMRIWaveforms$ export SKBUILD_EDITABLE_VERBOSE=1
(few-venv) FastEMRIWaveforms$ python
>>> import few
>>> few.utils.globals.get_backend("cpu")  # Force loading the CPU backend
...
Running cmake --build & --install in /workspaces/FastEMRIWaveforms/build/editable
Change Dir: '/workspaces/FastEMRIWaveforms/build/editable'

Run Build Command(s): /home/few/.local/few-venv/bin/ninja -v
ninja: no work to do.
...
<few.cutils.CpuBackend object at 0xffffacae3890>
>>>
```
