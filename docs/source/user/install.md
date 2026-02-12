# Installation guide

This page is a guide for installing and executing FEW tests on most platforms and
some clusters available to members of the user community.

:::{important}
Last updated in July 2025, after the release of `FastEMRIWaveforms v2.0.0` and the
publication of a conda package on `conda-forge`.
If you read this page at a significantly latter date, note that these instructions might be outdated.
:::


## How to use this installation guide

To quickly find the right installation instructions for your needs, follow these steps:

1. **Identify your user type:**
  - If you want to modify the source code, follow the **from source** installation instructions.
  - If you just want to use the package, follow the **pip** or **conda** installation instructions.

2. **Select your platform:**
  - **Linux** (including Ubuntu and most clusters)
  - **Mac OS**
  - **Windows**
  - **HPC/Cluster** (e.g., CNES, CC-IN2P3)

3. **Decide if you need GPU support:**
  - Look for instructions mentioning CUDA or GPU backends if you require GPU
    acceleration.
  - Otherwise, follow the standard CPU installation steps.

4. **Choose your environment manager:**
  - Use the **conda** instructions if you prefer conda/mamba/micromamba.
  - Use the **pip** instructions if you prefer pip/venv.

5. **Jupyter Hub integration:**
  - If you need to use FEW in Jupyter Hub on a cluster, refer to the dedicated Jupyter Hub section in the relevant cluster instructions.

---

**Quick navigation table:**

| User Type / Platform      | Linux / Cluster | Mac OS | Windows | CNES Cluster | CC-IN2P3 Cluster |
|--------------------------|-----------------|--------|---------|--------------|------------------|
| **Install via pip (user)**      | [Generic pip](#using-pip) | [Generic pip](#using-pip) | Not available | [Generic pip](#using-pip) | [CC-IN2P3 pip](#on-the-cc-in2p3-cluster-with-gpu-support) |
| **Install via conda (user)**    | [Generic conda](#using-conda) | [Generic conda](#using-conda) | [Windows conda](#on-windows) | [Generic conda](#using-conda) | [CC-IN2P3 conda](#on-the-cc-in2p3-cluster-with-gpu-support) |
| **From source (developer)**          | [Generic source](#from-source) | [Mac source](#on-mac-os-from-sources) | [Windows source](#on-windows) | [CNES source](#on-cnes-cluster-with-gpu-and-jupyter-hub-supports) | [CC-IN2P3 source](#on-the-cc-in2p3-cluster-with-gpu-support) |
| **GPU support**          | [See Generic](#generic-installation-instructions) | Not available | Not available | [CNES GPU](#on-cnes-cluster-with-gpu-and-jupyter-hub-supports) | [CC-IN2P3 GPU](#on-the-cc-in2p3-cluster-with-gpu-support) |
| **Jupyter Hub**          | N/A | N/A | N/A | [CNES Jupyter](#make-the-conda-environment-available-as-a-jupyter-hub-kernel) | [CC-IN2P3 Jupyter](#enable-the-jupyter-hub-kernel) |

:::{tip}
Use the navigation table above to jump to the section that matches your needs.
Each section provides both package and from-source installation instructions, as
well as notes on GPU support and Jupyter Hub integration where relevant.
:::

## Generic installation instructions

### Using pip

You may install FEW in your environment with

```sh
pip install fastemriwaveforms
```

If NVIDIA GPUs are available in your environment, you may install the GPU
backend with

```sh
pip install fastemriwaveforms-cuda12x  # For CUDA 12.x support
# or
pip install fastemriwaveforms-cuda11x  # For CUDA 11.x support
```

:::{attention}
Be sure to select the right version of the package according to your CUDA driver
version. Use `nvidia-smi` to check the version of your CUDA driver. Nothing will
happen if you install the wrong version, but you will not be able to use the GPU
backend.
:::

### Using conda

When possible, we recommend using a `conda` distribution which defaults to the
`conda-forge` channel, like `mamba` or `micromamba`.

:::{important}
In following instructions,
you can replace `mamba create` by `conda create -c conda-forge --override-channels`
if you only have the base `conda` distribution available.
:::

To install the CPU version of FEW in a new conda environment, run:

```sh
mamba create -n few_env python=3.12 fastemriwaveforms  # You may use any python version >=3.10,<3.14
```

If your environment supports CUDA 12.x, you can install the GPU version with:

```sh
mamba create -n few_env python=3.12 fastemriwaveforms-cuda12x
```

:::{caution}
There is no CUDA 11.x support for conda-based installations.
:::

### From source

To install FEW from source in any environment, follow these steps:

1. **Clone the repository:**
  ```sh
  git clone https://github.com/BlackHolePerturbationToolkit/FastEMRIWaveforms.git
  cd FastEMRIWaveforms
  ```

2. **(Optional) Checkout a specific release:**
  ```sh
  git checkout v{{ few_short_version }}
  ```

3. **Create and activate a specific environment:**
  - If you use `conda`, create a new environment named `few_env` with the required
    compilers and libraries:
    ```sh
    conda create -n few_env python=3.12 cxx-compiler pkgconfig conda-forge/label/lapack_rc::liblapacke
    conda activate few_env
    ```
  - If you prefer `virtualenv`, create a virtual environment, your environment will
    need a C++ compiler, and either a Fortran compiler or `liblapacke` library:
    ```sh
    python3 -m venv few_env
    source few_env/bin/activate
    ```

4. **Install FEW in editable mode:**
  ```sh
  pip install -e '.[testing]'
  ```

  - To enable GPU support (if available), add the CMake option and install manually
    the required GPU dependencies:
    ```sh
    pip install -e '.[testing]' --config-settings=cmake.define.FEW_WITH_GPU=ON
    pip install cupy-cuda12x 'nvidia-cuda-runtime-cu12==12.4.*' # Replace 12.4 by the CUDA driver version returned by nvidia-smi, keep the '.*' at the end
    ```

  - Many options can be passed to the `pip install` command, see [below](#advanced-installation-options)
    for more details.

5. **(Optional) For advanced debugging, in case of compilation errors, add:**
  ```sh
  pip install -e . -v -Cbuild.verbose=true -Clogging.level=INFO
  ```

This will build and install FEW from source, allowing you to modify the code and
have changes reflected immediately in your environment.

### Configure file storage

To configure where FEW stores its required data files, you need to create a
configuration file specifying the storage directory. The location and content of
this file depend on your operating system:

- **Linux:** `~/.config/few.ini`
- **Mac OS:** `~/Library/Application Support/few.ini`
- **Windows:** `%APPDATA%\few.ini`

Example configuration for Linux:

```ini
[few]
file-storage-dir=/home/${USER}/few
file-extra-paths=/home/${USER}/few;/home/${USER}/few/download
```

:::{seealso}
Available options and their descriptions can be found [here](cfg.md#summary-of-configuration-options).
:::

Replace the paths as needed for your environment or storage preferences.

After creating the configuration file, you can pre-download all required files
for testing with:

```sh
few_files fetch --tag testfile
```

This ensures all necessary files are available before running tests or using FEW.

### Test FEW is working correctly

To verify your FEW installation, run the test suite provided with the package.
This ensures that all dependencies are correctly installed and the code is
functioning as expected.

Simply execute, in the environment where FEW is installed:

```sh
python3 -m few.tests
```

You should see output indicating the backend in use (e.g., `cpu` or `cuda12x`)
and a summary of the tests run. For example:

```
AAKWaveform test is running with backend 'cpu'
[...]
.......
----------------------------------------------------------------------
Ran 27 tests in 87.305s

OK
```

If you want to restrict FEW to use only the CPU backend, set the environment
variable before running the tests:

```sh
FEW_ENABLED_BACKENDS="cpu" python -m few.tests
```

To force the use of the GPU backend (and thus fail the test if it is not available),
set the environment variable accordingly:

```sh
FEW_ENABLED_BACKENDS="cpu;cuda12x" python -m few.tests
```

If any test fails, review the error messages for missing dependencies or
configuration issues. Refer to the relevant installation section above for
troubleshooting tips.

## Platform-specific installation instructions

### On Mac OS, from sources

The recommended way to install `FastEMRIWaveforms` on a Mac OSX workstation
from-sources is by using a `conda` environment to obtain the necessary
compilers and dependencies.

We recommend using the `micromamba` conda distribution which can be installed
using `brew`:

```sh
brew install micromamba
```

Then, create a new conda environment `few_env` with the required compilers and
a specific version of `liblapacke`:

```sh
micromamba create -n few_env python=3.12 cxx-compiler pkgconfig conda-forge/label/lapack_rc::liblapacke
```

Then activate this environment and proceed with the installation of FEW
as described [above](#from-source).


## On Windows

```{attention}
For now, only from-source and conda installations are supported on Windows.
The PyPI package is not available for Windows users.
```

To install FEW from sources, ensure you have a recent
[Microsoft Visual Studio](https://visualstudio.microsoft.com/fr/downloads/)
release installed locally. Tests were performed with *Visual Studio 2022 Community Edition*.
Install the required dependencies in a new conda environment:

```sh
$ conda create -n few_env -c conda-forge --override-channels \
    python=3.12 cxx-compiler pkgconfig conda-forge/label/lapack_rc::liblapacke
```

Then proceed with the installation of FEW as described [above](#from-source).


You may also directly install the package from conda with:

```sh
$ conda install -n few_env -c conda-forge --override-channels fastemriwaveforms
```

:::{attention}
Only the CPU backend is available on Windows using conda packages.
If you manage to compile the GPU backend using from-source install, please let
us know so we can update this documentation.
:::


## On CNES cluster, with GPU and jupyter hub supports

To install FEW on the CNES cluster, you need to use a GPU node in an interactive
session. Following instructions assume you have access to the `lisa` project but
can be easily adapted to any other project you have access to.


First, log into the TREX cluster and request an interactive session on a GPU node:

```sh
# Here with the "lisa" project and a session of 1h
$ sinter -A lisa -p gpu_std -q gpu_all --gpus 1 --time=01:00:00 --pty bash
```

On the GPU node, load the `conda` module and create a new conda environment named `few_env`,
then activate it:

```sh
$ module load conda/24.3.0
$ conda create -n few_env -c conda-forge --override-channels python=3.12
$ conda activate few_env
```

Load the `nvhpc` modules access the CUDA compiler, as well as the `cuda` module:
 ```sh
(few_env) $ module load cuda/12.4.1
(few_env) $ module load nvhpc/22.9
```

Then follow the instructions for from-source installation as [described above](#from-source)
but replace the `pip install` command with:

```sh
(few_env) $ CXX=g++ CC=gcc pip install -e '.[testing]' --config-settings=cmake.define.FEW_WITH_GPU=ON
```

### Configure file storage

As [explained previously](#configure-file-storage), it is advised
to create a configuration file to specify where the files should be downloaded.
Use a high-volumetry storage space for that purpose,
like [project shared-spaces on `/work/`](https://hpc.pages.cnes.fr/wiki-hpc-sphinx/page-stockage-work.html)
If you have access to the LISA project, you can for example use the following:

```sh
$ mkdir -p /work/LISA/${USER}/few_files
# Write FEW configuration into ~/.config/few.ini
$ cat << EOC > ~/.config/few.ini
[few]
file-storage-dir=/work/LISA/${USER}/few_files
EOC
```

### Make the conda environment available as a Jupyter Hub kernel
After ensuring that `few` is working as expected, enable support for [Jupyter Hub](https://jupyterhub.cnes.fr/).
First install `ipykernel` and declare a new kernel named `few_env`:

```sh
(few_env) $ conda install ipykernel
(few_env) $ python -m ipykernel install --user --name few_env
Installed kernelspec few_env in ~/.local/share/jupyter/kernels/few_env
```

Next, create a python wrapper to preload the modules in the Python context:

```sh
(few_env) $ python_wrapper_path=$(dirname $(which python))/wpython
(few_env)$ cat << EOC > ${python_wrapper_path}
#!/bin/bash

# Load the necessary modules
module load cuda/12.4.1
module load nvhpc/22.9

# Run the python command
$(which python) \$@
EOC

$ chmod +x ${python_wrapper_path}
```

And change the kernel start command to use this wrapper: edit the file `~/.local/share/jupyter/kernels/few_env/kernel.json` and
replace the first item in the `argv` field by replacing `.../bin/python` with `.../bin/wpython`.

Now, when connected to the [CNES Jupyter Hub](https://jupyterhub.cnes.fr/), you
should have access to the `few_env` kernel and FEW should work correctly in it.

## On the CC-IN2P3 cluster with GPU support

First log into the CC-IN2P3 cluster and request an interactive session on a GPU node:

```sh
# Here a 2h session with 64GB of RAM
$ srun -p gpu_interactive -t 0-02:00 --mem 64G --gres=gpu:v100:1 --pty bash -i
```

All installation methods described [above](#generic-installation-instructions)
are available on the CC-IN2P3 cluster but need little adjustments.

- For a **conda installation**, you can use the `anaconda` module available on the cluster.
  Load it and create a new environment named `few_env` with `fastemriwaveforms-cuda12x`:

  ```sh
  $ module load Programming_Languages/anaconda/3.11
  $ conda create -n few_env -c conda-forge --override-channels \
      python=3.13 fastemriwaveforms-cuda12x
  $ conda activate few_env
  ```

- From a **pip installation**, you can use the `Programming_Languages/python/3.12.2` module
  available on the cluster. Load it and create a new virtual environment named `few_env`:

  ```sh
  $ module load Programming_Languages/python/3.12.2
  $ python -m venv few_env
  $ source ./few_env/bin/activate
  $ pip install fastemriwaveforms-cuda12x
  ```

- For a **from-source installation**, follow the instructions from [above](#from-source)
  but first load the `nvhpc` module to get access to the CUDA compilers and use
  the following `pip install` command:

  ```sh
  # Create and activate a conda or venv environment named `few_env`
  (few_env) $ module load HPC_GPU/nvhpc/24.5  # Load the NVHPC module
  # ... Clone sources
  (few_env) $ CXX=g++ CC=gcc FC=gfortran pip install -e '.[testing]' \
                  --config-settings=cmake.define.FEW_WITH_GPU=ON \
                  --config-settings=cmake.define.FEW_LAPACKE_FETCH=ON
  ```

On this cluster, it is recommended to configure the file storage directory
to a large storage volume you have access to, such as `/sps/lisaf/${USER}/few_files`.
(you may use any large storage volume
[you have access to](https://doc.cc.in2p3.fr/fr/Data-storage/storage-areas.html)).

You can do this by creating a configuration file in `~/.config/few.ini`
[as explained before](#configure-file-storage):

```sh
(few_env) $ cat << EOC > ~/.config/few.ini
[few]
file-storage-dir=/sps/lisaf/${USER}/few_files
file-download-dir=/sps/lisaf/${USER}/few_files
EOC
(few_env) $ mkdir /sps/lisaf/${USER}/few_files
```

### Enable the Jupyter Hub kernel

If you installed FEW in a conda environment, you may enable support for
[Jupyter Hub](https://notebook.cc.in2p3.fr/) by first adding the `ipykernel` package
and then declaring a new kernel named `few_env`:

```sh
(few_env) $ conda install -c conda-forge --override-channels ipykernel
(few_env) $ python -m ipykernel install --user --name few_env
Installed kernelspec few_env in ~/.local/share/jupyter/kernels/few_env
```

Next, create a python wrapper to preload the modules in the Python context:

```sh
(few_env) $ python_wrapper_path=$(dirname $(which python))/wpython
(few_env) $ cat << EOC > ${python_wrapper_path}
#!/bin/bash
# Load the necessary modules
source /usr/share/Modules/init/bash
module load Programming_Languages/anaconda/3.11

conda activate few_env

unset PYTHONPATH

# Run the python command
exec $(which python) \$@
EOC
$ chmod +x ${python_wrapper_path}
```

And change the kernel start command to use this wrapper: edit the file
`~/.local/share/jupyter/kernels/few_env/kernel.json` and replace the first item
in the `argv` field by replacing `.../bin/python` with `.../bin/wpython`.

The kernel should be available in the Jupyter Hub interface and FEW should
work correctly in this environment.

## Knowledge base

### Advanced installation options

Many options are available to change the installation behaviour. These can be set by adding `--config-settings=cmake.define.OPTION_NAME=OPTION_VALUE` to the `pip` command. Available options are:

- `FEW_LAPACKE_FETCH=ON|OFF|[AUTO]`: Whether `LAPACK` and `LAPACKE` should be automatically fetched from internet.
  - `ON`: ignore pre-installed `LAPACK(E)` and always fetch and compile their sources
  - `OFF`: disable `LAPACK(E)` fetching and only use pre-installed library and headers (install will fail if pre-installed lib and headers are not available)
  - `AUTO` (default): try to detect pre-installed `LAPACK(E)` and their headers. If found, use them, otherwise fetch `LAPACK(E)`.
- `FEW_LAPACKE_DETECT_WITH=[CMAKE]|PKGCONFIG`: How `LAPACK(E)` should be detected
  - `CMAKE`: `LAPACK(E)` will be detected using the cmake `FindPackage` command. If your `LAPACK(E)` install provides `lapacke-config.cmake` in a non-standard location, add its path to the `CMAKE_PREFIX_PATH` environment variable.
  - `PKGCONFIG`: `LAPACK(E)` will be detected using `pkg-config` by searching for the files `lapack.pc` and `lapacke.pc` . If these files are provided by your `LAPACK(E)` install in a non-standard location, add their path to the environment variable `PKG_CONFIG_PATH`
  - `AUTO` (default): attempt both CMake and PkgConfig approaches
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

### conda vs pip installation

Both `conda` and `pip+venv` installations are supported, but they have different
advantages and disadvantages:

- **conda**:
  - Pros:
    - Automatically manages compilers and dependencies which makes it easier to
      define a build-from-source environment.
    - Provides a consistent environment across different platforms.
    - On some platforms, `conda` will automatically detect the capabilities of
      your CPU and will install a package optimized for it (for x86_64 architectures on linux/macOS).
  - Cons:
    - May not always have the latest version of FEW available immediately.
    - Requires conda/mamba/micromamba to be installed.
- **pip+venv**:
  - Pros:
    - More flexible in terms of package versions and dependencies.
    - Can be used in any Python environment without requiring conda.
    - Allows for more control over the installation process.
  - Cons:
    - Requires manual installation of compilers and dependencies.
    - May install non-optimized builds of the package to allow for broader
      hardware compatibility.
