# Installation instructions for workshop development
Installation instrucitions for broken Mac OS arm
```
git clone https://github.com/BlackHolePerturbationToolkit/FastEMRIWaveforms.git
cd FastEMRIWaveforms
git checkout Kerr_Equatorial_Eccentric
conda create -n few -y python=3.12
conda activate few
conda install -c conda-forge -y clangxx_osx-arm64 clangxx_osx-64 h5py wget gsl liblapacke lapack openblas fortran-compiler
export CXXFLAGS="-march=native"
export CFLAGS="-march=native"
pip install .
python -m unittest discover
```
