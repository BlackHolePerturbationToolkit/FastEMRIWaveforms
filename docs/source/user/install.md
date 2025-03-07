# Installation guide

This page is a guide for installing and executing FEW tests on most platforms and some clusters available to members of the user community.

Last updated in March 2025 just after the release of `FastEMRIWaveforms v2.0.0rc1`.
If you read this page at a significantly latter date, note that these instructions might be outdated.

Note that most instructions are common for both package installations (using the wheels available on
[PyPI](https://pypi.org/project/fastemriwaveforms/2.0.0rc1/)) and from-source installations (using a
local copy of sources). In following instructions, from-source installations are always made in *editable mode*
where you can modify sources on the fly and these changes are taken into account in the python environment.
To disable this *editable mode* (and thus be able to delete the source directory whilst keeping access to FEW
in python), simply remove the `-e` option from the `pip install` commands in the from-source installation instructions.

If the from-source installation instructions do not work in your environment, retry by adding the options
`-v -Cbuild.verbose=true -Clogging.level=INFO` to the `pip install -e .` command to obtain detailed
information about the cause of failure.

## On Mac OS with Apple Silicon CPU (M1 to M4)

The preferred way to install `FastEMRIWaveforms` on a Mac OSX workstation is by using a `conda` environment.
Following instructions were tested with a MacBook Pro with a Apple M4 Pro CPU on macOS Sequoia 15.3.
`conda` was installed using `brew install miniconda`.

### Preliminary steps

Create a new conda environment `few2.0rc1` and activate it:

```sh
$ conda create -n few2.0rc1 python=3.12 --channel conda-forge --override-channels
$ conda activate few2.0rc1
(few2.0rc1) $ conda config --env --add channels conda-forge
```

Note that, to ensure having up-to-date compilers, we force the use of the `conda-forge` channel.

### Perform a from-source install

Clone FEW and checkout the `v2.0.0rc1` tag:

```sh
(few2.0rc1) $ git clone https://github.com/BlackHolePerturbationToolkit/FastEMRIWaveforms.git
(few2.0rc1) $ cd FastEMRIWaveforms
(few2.0rc1) $ git checkout v2.0.0rc1
```

#### With automatic LAPACK(E) fetching

This first method will automatically download the sources of LAPACK and LAPACKE, and compile
them along with FEW. For that, you will need a Fortran compiler as well as a C++ compiler:

```sh
(few2.0rc1) $ conda install cxx-compiler fortran-compiler
```

By default, the FEW installation process will try to detect LAPACK in your environment using
the CMake `FindLapack` feature which, on macOS, can link to the XCode *Accelerated* framework.
This results in a non-working FEW installation. To prevent that, disable completely LAPACK
detection:

```sh
(few2.0rc1) $ pip install -e '.[testing]' -Ccmake.define.FEW_LAPACKE_DETECT_WITH=NONE
Successfully installed [...] fastemriwaveforms-2.0.0rc1 [...]
```

#### With conda-installed LAPACK

As of March 2025, the default `conda-forge::lapack` package has version `v3.9.0` and does not
install the development files required for FEW compilation.
We will instead use the `lapack_rc` package, in version `v3.11.0` which does work as expected.

First, install a C++ compiler and the LAPACK package:

```sh
(few2.0rc1) $ conda install cxx-compiler pkgconfig conda-forge/label/lapack_rc::liblapacke
```

Then run FEW installation from source:

```sh
(few2.0rc1) $ pip install -e '.[testing]'
Successfully installed [...] fastemriwaveforms-2.0.0rc1 [...]
```


### Perform a package install

To install the package directly from PyPI, simply run:

```sh
(few2.0rc1) $ pip install --pre fastemriwaveforms==2.0.0rc1
...
Successfully installed [...] fastemriwaveforms-2.0.0rc1 [...]
```

### Configure file storage

Execute the following command to write a configuration file in `~/Library/Application Support/few.ini` which specifies that FEW files must be downloaded in `~/few`:

```sh
(few2.0rc1) $ cat << EOC > ~/Library/Application\ Support/few.ini
[few]
file-storage-dir=/Users/${USER}/few
file-extra-paths=/Users/${USER}/few;/Users/${USER}/few/download
EOC
```

Now, you may run the folowwing command to pre-download all files required for test execution.
If you don't run this command, files will still be downloaded during test execution:

```sh
(few2.0rc1) $ few_files fetch --tag unittest
Downloading all missing files tagged 'unittest' into '/Users/my_user_name/few/download'
Downloading 'AmplitudeVectorNorm.dat'... ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:00
Downloading 'FluxNewMinusPNScaled_fixed_y_order.dat'... ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:00
...
Downloading 'ZNAmps_l10_m10_n55_DS2Outer.h5'... ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:00
```

### Run tests

Execute the package tests suite to ensure that everything works as expected:

```sh
(few2.0rc1) $ python -m few.tests
AAKWaveform test is running with backend 'cpu'
[...]
.......
----------------------------------------------------------------------
Ran 27 tests in 87.305s

OK
```

## On Windows

```{attention}
For now, only from-source installations are possible on Microsoft Windows.
The compilation of backends on Windows is less stable than on other platforms. Try at your own risk!
```

### Preliminary steps

Ensure you have a recent [Microsoft Visual Studio](https://visualstudio.microsoft.com/fr/downloads/) release installed locally.
Tests were performed with *Visual Studio 2022 Community Edition*.

Create a new conda environment `few2.0rc1` and activate it:

```sh
$ conda create -n few2.0rc1 python=3.12 --channel conda-forge --override-channels
$ conda activate few2.0rc1
(few2.0rc1) $ conda config --env --add channels conda-forge
```

Like for [Mac OS](#on-mac-os-with-apple-silicon-cpu-m1-to-m4), we force here the use of the `conda-forge` channel.

Install the required dependencies in the conda environment:

```sh
(few2.0rc1) $ conda install cxx-compiler conda-forge/label/lapack_rc::liblapacke pkgconfig
```

### Perform a from-source install

Clone FEW and checkout the `Kerr_Equatorial_Eccentric` branch (Windows installation using MSVC was not working at the
time `v2.0.0rc1` was released):

```sh
(few2.0rc1) $ git clone https://github.com/BlackHolePerturbationToolkit/FastEMRIWaveforms.git
(few2.0rc1) $ cd FastEMRIWaveforms
(few2.0rc1) $ git checkout Kerr_Equatorial_Eccentric
```

Then run FEW installation from source:

```sh
(few2.0rc1) $ pip install -e '.[testing]'
Successfully installed [...] fastemriwaveforms-2.0.0rc1.post1.dev1+ge0c124b.d20250304 [...]
```

### Run tests

Execute the package tests suite to ensure that everything works as expected:

```sh
(few2.0rc1) $ python -m few.tests
AAKWaveform test is running with backend 'cpu'
[...]
.......
----------------------------------------------------------------------
Ran 27 tests in 87.305s

OK
```

## On CNES cluster, with GPU and jupyter hub supports

### Preliminary steps

First, log into the TREX cluster and request an interactive session on a GPU node:

```sh
# Here with the "lisa" project and a session of 1h
$ sinter -A lisa -p gpu_std -q gpu_all --gpus 1 --time=01:00:00 --pty bash
```

On the GPU node, load the `conda` module and create a new conda environment named `few2.0rc1`, then activate it:

```sh
$ module load conda/24.3.0
$ conda create -n few2.0rc1 python=3.12
$ conda activate few2.0rc1
```

### Perform a from-source install

Clone FEW and checkout the `v2.0.0rc1` tag:

```sh
(few2.0rc1) $ git clone https://github.com/BlackHolePerturbationToolkit/FastEMRIWaveforms.git
(few2.0rc1) $ cd FastEMRIWaveforms
(few2.0rc1) $ git checkout v2.0.0rc1
```

Load the `nvhpc` modules access the CUDA compiler, as well as the `cuda` module:

```sh
(few2.0rc1) $ module load cuda/12.4.1
(few2.0rc1) $ module load nvhpc/22.9
```

Install FEW and force the CUDA backend compilation with the option `FEW_WITH_GPU=ON`:

```sh
(few2.0rc1) $ CXX=g++ CC=gcc pip install -e '.[testing]' --config-settings=cmake.define.FEW_WITH_GPU=ON
...
Successfully installed fastemriwaveforms-2.0.0rc1
(few2.0rc1) $ pip install cupy-cuda12x nvidia-cuda-runtime-cu12==12.4.* # Must be installed manually when installed from source
```

### Perform a package install

To install the package directly from PyPI, simply run:

```sh
(few2.0rc1) $ pip install --pre fastemriwaveforms-cuda12x==2.0.0rc1
(few2.0rc1) $ pip install nvidia-cuda-runtime-cu12==12.4.*
```

### Test FEW is working correctly

Check that the code is properly installed and working by running the tests.
Note that the tests will automatically download required files. It is then advised
create a configuration file to specify where the files should be downloaded.
It is strongly advised to use a high-volumetry storage space for that purpose,
like [project shared-spaces on `/work/`](https://hpc.pages.cnes.fr/wiki-hpc-sphinx/page-stockage-work.html)
If you have access to the LISA project, you can for example use the following:

```sh
$ mkdir /work/LISA/${USER}/few_files
# Write FEW configuration into ~/.config/few.ini
$ cat << EOC > ~/.config/few.ini
[few]
file-storage-dir=/work/LISA/${USER}/few_files
EOC
```

Then actually run the tests:

```sh
(few2.0rc1) $ python -m few.tests
AAKWaveform test is running with backend 'cuda12x'
DetectorWave test is running with backend 'cuda12x'
...
----------------------------------------------------------------------
Ran 20 tests in 198.041s
```

The messages should indicate that they use the `cuda12x` backend. If that's not the case,
but instead use the `cpu` backend. Run the following command to get the reason why the
`cuda12x` backend cannot be loaded:

```sh
$ python
>>> import few
>>> few.get_backend("cuda12x")
# Example of possible output:
Traceback (most recent call last):
...
cupy_backends.cuda.libs.nvrtc.NVRTCError: NVRTC_ERROR_COMPILATION (6)

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
...
cupy.cuda.compiler.CompileException: /work/scratch/.../few2.0rc1/lib/python3.12/site-packages/cupy/_core/include/cupy/_cuda/cuda-12.4/cuda_fp16.h(129): catastrophic error: cannot open source file "vector_types.h"
  #include "vector_types.h"
                           ^

1 catastrophic error detected in the compilation of "/tmp/slurm-33342263/tmpq7dhwrdi/359445603dea8ee27f67d9bc57b875940bbb321b.cubin.cu".
Compilation terminated.


The above exception was the direct cause of the following exception:

Traceback (most recent call last):
                                              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
...
few.cutils.MissingDependencies: CuPy fails to run due to missing CUDA Runtime.
    If you are using few in an environment managed using pip, run:
        $ pip install nvidia-cuda-runtime-cu12
```

Note that you can also pre-download the files before running the tests with:

```sh
# Predownload the files requires by tests (not necessary, they will be pulled during tests in all cases)
$ few_files fetch --tag unittest
Downloading all missing files tagged 'unittest' into '/work/LISA/my_user_name/few_files/download'

Downloading 'AmplitudeVectorNorm.dat'... ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:00
Downloading 'FluxNewMinusPNScaled_fixed_y_order.dat'... ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:00
Downloading 'SchwarzschildEccentricInput.hdf5'... ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:03
Downloading 'Teuk_amps_a0.0_lmax_10_nmax_30_new.h5'... ━━━━━━╸━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  17% 0:00:24
...
```

You may also restrict FEW to the `cpu` backend with:

```sh
(few2.0rc1) $ FEW_ENABLED_BACKENDS="cpu" python -m few.tests
AAKWaveform test is running with backend 'cpu'
DetectorWave test is running with backend 'cpu'
...
----------------------------------------------------------------------
Ran 27 tests in 405.23s
```

### Make the conda environment available as a Jupyter Hub kernel

Now that `few` is working as expected, let's enable support for [Jupyter Hub](https://jupyterhub.cnes.fr/).
First install `ipykernel` and declare a new kernel named `few2.0rc1`:

```sh
(few2.0rc1) $ conda install ipykernel
(few2.0rc1) $ python -m ipykernel install --user --name few2.0rc1
Installed kernelspec few2.0rc1 in ~/.local/share/jupyter/kernels/few2.0rc1
```

Next, we need to create a python wrapper to preload the modules in the Python context:

```sh
(few2.0rc1) $ python_wrapper_path=$(dirname $(which python))/wpython
(few2.0rc1)$ cat << EOC > ${python_wrapper_path}
#!/bin/bash

# Load the necessary modules
module load cuda/12.4.1
module load nvhpc/22.9

# Run the python command
$(which python) \$@
EOC

$ chmod +x ${python_wrapper_path}
```

And change the kernel start command to use this wrapper: edit the file `~/.local/share/jupyter/kernels/few2.0rc1/kernel.json` and
replace the first item in the `argv` field by replacing `.../bin/python` with `.../bin/wpython`.

Now, when connected to the [CNES Jupyter Hub](https://jupyterhub.cnes.fr/), you should have access to the `few2.0rc1` kernel and FEW
should work correctly in it.

## On the CC-IN2P3 cluster with GPU support

### Preliminary steps

First log into the CC-IN2P3 cluster and request an interactive session on a GPU node:

```sh
# Here a 2h session with 64GB of RAM
$ srun -p gpu_interactive -t 0-02:00 --mem 64G --gres=gpu:v100:1 --pty bash -i
```

On the GPU node, load the python module and create a virtual environment named `few2.0rc1`, then activate it:

```sh
$ module load Programming_Languages/python/3.12.2
$ python -m venv few2.0rc1
$ source ./few2.0rc1/bin/activate
```

### Perform a from-source install

Clone FEW and checkout the `v2.0.0rc1` tag:

```sh
(few2.0rc1) $ git clone https://github.com/BlackHolePerturbationToolkit/FastEMRIWaveforms.git
(few2.0rc1) $ cd FastEMRIWaveforms
(few2.0rc1) $ git checkout v2.0.0rc1
```

Load the `nvhpc` module to get access to CUDA compilers:

```sh
(few2.0rc1) $ module load HPC_GPU/nvhpc/24.5
```

Install FEW and force the CUDA backend compilation with the option `FEW_WITH_GPU=ON`.
We also force fetching and compiling LAPACK(E), otherwise CMake will detect the system
LAPACK library (`/lib64/liblapack.so.3`) which makes somes tests fails with a segmentation fault error.
The installation command line is thus:

```sh
(few2.0rc1) $ CXX=g++ CC=gcc FC=gfortran pip install -e '.[testing]' \
                --config-settings=cmake.define.FEW_WITH_GPU=ON \
                --config-settings=cmake.define.FEW_LAPACKE_FETCH=ON \
                --config-settings=cmake.define.FEW_LAPACKE_EXTRA_LIBS=gfortran
...
Successfully installed fastemriwaveforms-2.0.0rc1
(few2.0rc1) $ pip install cupy-cuda12x nvidia-cuda-runtime-cu12==12.4.* # Must be installed manually when installed from source
```

### Perform a package install

To install the package directly from PyPI, simply run:

```sh
(few2.0rc1) $ pip install --pre fastemriwaveforms-cuda12x==2.0.0rc1
```

### Configure file storage

Execute the following command to write a configuration file in `~/.config/few.ini` which specifies that FEW files must be downloaded in `/sps/lisaf/${USER}}/few_files` (adapt to any large storage volume
[you have access to](https://doc.cc.in2p3.fr/fr/Data-storage/storage-areas.html)):

```sh
(few2.0rc1) $ cat << EOC > ~/.config/few.ini
[few]
file-storage-dir=/sps/lisaf/${USER}/few_files
file-download-dir=/sps/lisaf/${USER}/few_files
EOC
(few2.0rc1) $ mkdir /sps/lisaf/${USER}/few_files
```

Now, you may run the folowwing command to pre-download all files required for test execution.
If you don't run this command, files will still be downloaded during test execution:

```sh
(few2.0rc1) $ few_files fetch --tag unittest
Downloading all missing files tagged 'unittest' into '/sps/lisaf/your_username/few_files'
Downloading 'AmplitudeVectorNorm.dat'... ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:00
Downloading 'FluxNewMinusPNScaled_fixed_y_order.dat'... ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:00
...
Downloading 'ZNAmps_l10_m10_n55_DS2Outer.h5'... ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:00
```

### Run tests

Execute the package tests suite to ensure that everything works as expected:

```sh
(few2.0rc1) $ python -m few.tests
AAKWaveform test is running with backend 'cuda12x'
[...]
.......
----------------------------------------------------------------------
Ran 27 tests in 231.381s

OK
```

You may also force FEW to run only on the `cpu` backend with:

```sh
(few2.0rc1) $ FEW_ENABLED_BACKENDS="cpu" python -m few.tests
AAKWaveform test is running with backend 'cpu'
[...]
.......
----------------------------------------------------------------------
Ran 27 tests in 512.101s

OK
```
