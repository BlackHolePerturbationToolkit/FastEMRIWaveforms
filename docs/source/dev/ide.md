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
  using the [*rebuild editable* install](#install-the-project-with-backend-recompilation-rebuild-editable-mode)

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

### Install the project with backend recompilation (rebuild editable mode)

If you need backends to be automatically rebuilt on changes, you can install the project using the `rebuild`
editable install provided by
[scikit-build-core **in experimental mode**](https://scikit-build-core.readthedocs.io/en/latest/configuration.html#editable-installs).

Note that this mode requires to pre-install all of the project build dependencies before-hand.


```bash
# 1. First, install the project build dependencies
(few-venv) FastEMRIWaveforms$ pip install cython numpy scikit-build-core ninja cmake setuptools_scm

# 2. Also, make sure that LAPACK(E) is properly installed on your system, for example on Ubuntu 24.04:
(few-venv) FastEMRIWaveforms$ sudo apt-get install liblapacke-dev

# 3. Install the project in rebuild editable mode
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

## pre-commit: apply common guidelines to your code

The *FastEMRIWaveforms* package ships with a [`pre-commit`](https://pre-commit.com/) configuration file.
It is highly recommended to install that tool in your development environment so that each *commit*
you add is checked by pre-commit. This can be done by running:

```bash
(few-venv) FastEMRIWaveforms$ pip install pre-commit
(few-venv) FastEMRIWaveforms$ pre-commit install
pre-commit installed at .git/hooks/pre-commit
```

Now, whenever you add a new commit to the git history, pre-commit will automatically run the
configured tools to check the content of that commit, and suggest/apply changes if necessary:

```bash
(few-venv) FastEMRIWaveforms$ git commit -m "Add a new feature"
check for added large files..............................................Passed
check python ast.....................................(no files to check)Skipped
check for case conflicts.................................................Passed
check json...........................................(no files to check)Skipped
check for merge conflicts................................................Passed
check for broken symlinks............................(no files to check)Skipped
check xml............................................(no files to check)Skipped
check yaml...........................................(no files to check)Skipped
fix end of files.........................................................Passed
mixed line ending........................................................Passed
python tests naming..................................(no files to check)Skipped
pretty format json...................................(no files to check)Skipped
fix requirements.txt.................................(no files to check)Skipped
trim trailing whitespace.................................................Passed
yamlfmt..............................................(no files to check)Skipped
cmake-format.........................................(no files to check)Skipped
pyproject-fmt........................................(no files to check)Skipped
ruff.................................................(no files to check)Skipped
ruff-format..........................................(no files to check)Skipped
```

If all tests succeed, the commit created as expected.
If any tests fail, the commit is cancelled. Most of the time, the failing test
will automatically modify your source file (for example to enforce code style
guidelines). You simply need to check the proposed modification, stage the file
(`git add` it) and retry the commit.

You can also run the pre-commit tests manually using `pre-commit run`.
By default, `pre-commit` will only run on the files that have been staged (`git add`).
You can also run it on all files using `pre-commit run --all-files`.

If you need to commit your developments while ignoring the pre-commit tests, you can use the `--no-verify` option:

```bash
$ git commit -m "wip: unfinished incredible feature" --no-verify
```

## Standard development environment using devcontainers

If you are using [VSCode](https://code.visualstudio.com/) and have [Docker](https://www.docker.com/) installed, you can use the [devcontainers](https://code.visualstudio.com/docs/remote/containers) extension to quickly set up a development environment.
Follow the [devcontainer installation steps](https://code.visualstudio.com/docs/devcontainers/containers#_installation) to install the devcontainer extension.

You can then use the VSCode action `Clone Repository in Container` to clone the repository in a devcontainer
or you can simply [click here](vscode://ms-vscode-remote.remote-containers/cloneInVolume?url=https%3A%2F%2Fgithub.com%2FBlackHolePerturbationToolkit%2FFastEMRIWaveforms.git).

The *FastEMRIWaveforms* devcontainer is configured to use an Ubuntu 24.04 image with GNU compilers 14, system LAPACKE library, pre-configured pre-commit and a pre-loaded virtualenv based on Python 3.12.

To test the compilation of CUDA backends, you can also install the CUDA Toolkit within the devcontainer by
running the following command:

```bash
$ CUDA_VERSION=12.6.3-1 ./.devcontainer/install_cuda_toolkit.sh
# To know the list of available CUDA versions, run the script without
# the CUDA_VERSION variable like so:
$ ./.devcontainer/install_cuda_toolkit.sh
...
Please run the script as follow:
  $ CUDA_VERSION=12.6.3-1 ./.devcontainer/install_cuda_toolkit.sh
 with a version selected from this list:
cuda-toolkit |   12.8.0-1 | https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/sbsa  Packages
cuda-toolkit |   12.6.3-1 | https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/sbsa  Packages
cuda-toolkit |   12.6.2-1 | https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/sbsa  Packages
cuda-toolkit |   12.6.1-1 | https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/sbsa  Packages
cuda-toolkit |   12.6.0-1 | https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/sbsa  Packages
cuda-toolkit |   12.5.1-1 | https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/sbsa  Packages
```

Actual GPU support in devcontainers has not been tested yet, please reach out by opening a GitHub issue
if you have a Linux workstation with VScode, Docker and a NVIDIA GPU with CUDA >=11.2.
