# Working with FEW 2.0

This page is a guide for installing and executing FEW tests on most platforms and some clusters available to members of the user community.

It is written in Feb. 2025 just after the release of `FastEMRIWaveforms v2.0.0rc0`.
If you read this page at a significantly latter date, note that these instructions might be outdated.

## On Mac OS with Apple Silicon CPU (M1 to M4)

### Installing from packages


### Installing from sources


## At CC-IN2P3, with GPU support

### Installing from packages

### Installing from sources


## On CNES cluster, with GPU and jupyter hub supports

### Installing from packages

### Installing from sources

First, log into the TREX cluster and request an interactive session on a GPU node:

```sh
# Here with the "lisa" project and a session of 1h
$ sinter -A lisa -p gpu_std -q gpu_all --gpus 1 --time=01:00:00 --pty bash
```

Clone FEW and checkout the `v2.0.0rc0` tag:

```sh
$ git clone https://github.com/BlackHolePerturbationToolkit/FastEMRIWaveforms.git
$ cd FastEMRIWaveforms
$ git checkout v2.0.0rc0
```

Load the modules to have access to `conda` and to the CUDA ToolKit:

```sh
module load conda/24.3.0
module load cuda/12.4.1
module load nvhpc/22.9
```

Create a new conda environment named `few2.0rc0` and activate it:

```sh
$ conda create -n few2.0rc0 python=3.12
$ conda activate few2.0rc0
```

Install the project, as well as `ipykernel` (for jupyer hub support):

```sh
(few2.0rc0) $ conda install ipykernel
(few2.0rc0) $ CXX=g++ CC=gcc pip install -e '.[testing]' --config-settings=cmake.define.FEW_WITH_GPU=ON
...
Successfully installed fastemriwaveforms-2.0.0rc0
(few2.0rc0) $ pip install cupy-cuda12x nvidia-cuda-runtime-cu12==12.4.* # Must be installed manually when installed from source
```

Check that the code is properly installed and working by running the tests.
Note that the tests will automatically download required files. It is then advised
create a configuration file to specify where the files should be downloaded.
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
(few2.0rc0) $ python -m unittest discover
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
cupy.cuda.compiler.CompileException: /work/scratch/.../few2.0rc0/lib/python3.12/site-packages/cupy/_core/include/cupy/_cuda/cuda-12.4/cuda_fp16.h(129): catastrophic error: cannot open source file "vector_types.h"
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

Now that `few` is working as expected, let's enable support for [Jupyter Hub](https://jupyterhub.cnes.fr/).
First declare a new kernel named `few2.0rc0`:

```sh
$ python -m ipykernel install --user --name few2.0rc0
Installed kernelspec few2.0rc0 in ~/.local/share/jupyter/kernels/few2.0rc0
```

Next, we need to create a python wrapper for loading the correct modules in the Python context:

```sh
(few2.0rc0) $ python_wrapper_path=$(dirname $(which python))/wpython
(few2.0rc0)$ cat << EOC > ${python_wrapper_path}
#!/bin/bash

# Load the necessary modules
module load cuda/12.4.1
module load nvhpc/22.9

# Run the python command
$(which python) \$@
EOC

$ chmod +x ${python_wrapper_path}
```

And change the kernel start command to use this wrapper: edit the file `~/.local/share/jupyter/kernels/few2.0rc0/kernel.json` and
replace the first item in the `argv` field by replacing `.../bin/python` with `.../wpython`.

Now, when connected to the [CNES Jupyer Hub](https://jupyterhub.cnes.fr/), you should have access to the `few2.0rc0` kernel and FEW
should work correctly in it.
