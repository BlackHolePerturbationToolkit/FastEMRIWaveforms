#!/bin/bash

usage="$(basename "$0") [-h] -- program to install FastEMRIWaveforms

where:
    -h  show this help text

keyword argument options (given as key=value):
    env_name:  Name of generated conda environment. Default is 'few_env'.
    install_type:  Type of install. 'basic', 'development', or 'sampling'. 
        'development' adds packages needed for development and documentation.
        'sampling' adds packages for sampling like eryn, lisatools, corner, chainconsumer.
        Default is 'basic'. 
    run_tests: Either true or false. Whether to run tests after install. Default is true.
    
"

while getopts 'h' option; do
  case "$option" in
    h) echo "$usage"
       exit 3
       ;;
  esac
done

for ARGUMENT in "$@"
do
   KEY=$(echo $ARGUMENT | cut -f1 -d=)

   KEY_LENGTH=${#KEY}
   VALUE="${ARGUMENT:$KEY_LENGTH+1}"

   export "$KEY"="$VALUE"
done

if [ -z ${env_name+x} ]; then env_name="few_env"; fi
if [ -z ${install_type+x} ]; then install_type="basic"; fi
if [ -z ${run_tests+x} ]; then run_tests="true"; fi

if [[ "$run_tests" != "true" ]] && [[ "$run_tests" != "false" ]]; 
    then echo "run_tests variable must be 'true' or 'false'.";
    exit 2;
fi

nvcc=$(which nvcc)
if [[ "$nvcc" == "" ]];
        then nvcc="$CUDAHOME";
        if [[ "$nvcc" != "" ]];
                then nvcc="$nvcc/bin/nvcc";
        fi
fi

if [[ "$nvcc" == "" ]];
        then nvcc="$CUDA_HOME";
        if [[ "$nvcc" != "" ]];
                then nvcc=""$nvcc"bin/nvcc";
        fi
fi

if [[ "$nvcc" == "" ]];
        then use_gpu=false;
else
        echo "found nvcc: $nvcc";
        use_gpu=true;
fi
echo "use_gpu: "$use_gpu"";

if [[ "$use_gpu" == true ]];
    then export CUDA_HOME="${nvcc:0:-8}";
    tmp=$("$nvcc" --version | grep -i "Build cuda_");
    cuda_version="${tmp:11:4}";
    echo "CUDA_HOME: $CUDA_HOME";
    echo "cuda version: $cuda_version";
fi

echo "Now installing into conda environment named $env_name."

if [[ "$install_type" == "basic" ]]; 
    then echo "Installing basic setup."; 
elif [[ "$install_type" == "sampling" ]];
    then echo "Installing sampling setup."; 
elif [[ "$install_type" == "development" ]];
    then echo "Installing development setup."; 
else
    echo "If providing install_type variable, must be 'basic', 'sampling', or 'development'";
    exit 1;
fi

if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "This system is macOS."

    machine=$(uname -m)
    if [[ "$machine" == "arm64" ]]; then
        echo "This is an M1 Mac."
        conda create -n "$env_name" -c conda-forge -y wget gsl hdf5 numpy Cython scipy tqdm jupyter ipython h5py requests matplotlib python=3.9 openblas lapack liblapacke
    else
        echo "This is not an M1 Mac."
        conda create -n "$env_name" -c conda-forge -y clangxx_osx-64 clang_osx-64 wget gsl lapack=3.6.1 hdf5 numpy Cython scipy tqdm jupyter ipython h5py requests matplotlib python=3.10
    fi
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    echo "This system is Unix/Linux."
    conda create -n "$env_name" -c conda-forge -y gcc_linux-64 gxx_linux-64 wget gsl lapack=3.6.1 hdf5 numpy Cython scipy tqdm jupyter ipython h5py requests matplotlib python=3.10
else
    echo "Unsupported operating system."
fi

# get conda base information
IN="$(conda info | grep -i 'base environment')"

tmp=$(echo $IN | tr " " "\n")
i=0
out=()
for tmp_i in $tmp
do
    out[i]=$tmp_i
    let "i = $i + 1"
done

conda_base="${out[3]}"
conda_init_sh="$conda_base/etc/profile.d/conda.sh"
echo "$conda_init_sh"
source $conda_init_sh

conda activate "$env_name"
echo "should be done"
conda_check="$(conda info | grep -i 'active environment')"
echo "conda env: $conda_check"
echo "Environment created. Installing additional packages before FEW install."

python_here=""$conda_base"/envs/"$env_name"/bin/python"
pip_here=""$conda_base"/envs/"$env_name"/bin/pip"

echo "python: $python_here" 
echo "pip: $pip_here"
if [[ "$install_type" == "sampling" ]]; then
    "$pip_here" install corner eryn chainconsumer;
    "$pip_here" install git+https://github.com/mikekatz04/LISAanalysistools.git@dev;
elif [[ "$install_type" == "development" ]]; then
    conda install sphinx sphinx_rtd_theme pypandoc --yes;
    "$pip_here" install nbsphinx;
fi

if [[ "$use_gpu" == true ]];
    then pip install "cupy-cuda"${cuda_version:0:2}""${cuda_version:3:1}"";
fi

machine=$(uname -m)

if [[ "$machine" == "arm64" ]]; then
    "$pip_here" install . --ccbin /usr/bin/
else
    "$pip_here" install .
fi

if [[ "$run_tests" == "true" ]]; 
 then echo "Running tests...";
 "$python_here" -m unittest discover;
fi