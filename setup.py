# from future.utils import iteritems
import os
import sys
from os.path import join as pjoin
from setuptools import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy
import shutil
import argparse


def find_in_path(name, path):
    """Find a file in a search path"""

    # Adapted fom http://code.activestate.com/recipes/52224
    for dir in path.split(os.pathsep):
        binpath = pjoin(dir, name)
        if os.path.exists(binpath):
            return os.path.abspath(binpath)
    return None


def locate_cuda():
    """Locate the CUDA environment on the system

    Returns a dict with keys 'home', 'nvcc', 'include', and 'lib64'
    and values giving the absolute path to each directory.

    Starts by looking for the CUDAHOME env variable. If not found,
    everything is based on finding 'nvcc' in the PATH.
    """

    # First check if the CUDAHOME env variable is in use
    if "CUDAHOME" in os.environ:
        home = os.environ["CUDAHOME"]
        nvcc = pjoin(home, "bin", "nvcc")
    else:
        # Otherwise, search the PATH for NVCC
        nvcc = find_in_path("nvcc", os.environ["PATH"])
        if nvcc is None:
            raise EnvironmentError(
                "The nvcc binary could not be "
                "located in your $PATH. Either add it to your path, "
                "or set $CUDAHOME"
            )
        home = os.path.dirname(os.path.dirname(nvcc))

    cudaconfig = {
        "home": home,
        "nvcc": nvcc,
        "include": pjoin(home, "include"),
        "lib64": pjoin(home, "lib64"),
    }
    for k, v in iter(cudaconfig.items()):
        if not os.path.exists(v):
            raise EnvironmentError(
                "The CUDA %s path could not be " "located in %s" % (k, v)
            )

    return cudaconfig


def customize_compiler_for_nvcc(self):
    """Inject deep into distutils to customize how the dispatch
    to gcc/nvcc works.

    If you subclass UnixCCompiler, it's not trivial to get your subclass
    injected in, and still have the right customizations (i.e.
    distutils.sysconfig.customize_compiler) run on it. So instead of going
    the OO route, I have this. Note, it's kindof like a wierd functional
    subclassing going on.
    """

    # Tell the compiler it can processes .cu
    self.src_extensions.append(".cu")

    # Save references to the default compiler_so and _comple methods
    default_compiler_so = self.compiler_so
    super = self._compile

    # Now redefine the _compile method. This gets executed for each
    # object but distutils doesn't have the ability to change compilers
    # based on source extension: we add it.
    def _compile(obj, src, ext, cc_args, extra_postargs, pp_opts):
        if os.path.splitext(src)[1] == ".cu":
            # use the cuda for .cu files
            self.set_executable("compiler_so", CUDA["nvcc"])
            # use only a subset of the extra_postargs, which are 1-1
            # translated from the extra_compile_args in the Extension class
            postargs = extra_postargs["nvcc"]
        else:
            postargs = extra_postargs["gcc"]

        super(obj, src, ext, cc_args, postargs, pp_opts)
        # Reset the default compiler_so, which we might have changed for cuda
        self.compiler_so = default_compiler_so

    # Inject our redefined _compile method into the class
    self._compile = _compile


# Run the customize_compiler
class custom_build_ext(build_ext):
    def build_extensions(self):
        customize_compiler_for_nvcc(self.compiler)
        build_ext.build_extensions(self)


try:
    CUDA = locate_cuda()
    run_cuda_install = True
except OSError:
    run_cuda_install = False

parser = argparse.ArgumentParser()

parser.add_argument(
    "--no_omp",
    help="If provided, install without OpenMP.",
    action="store_true",
    default=False,
)

parser.add_argument(
    "--lapack_lib",
    help="Directory of the lapack lib.",
    default="/usr/local/opt/lapack/lib",
)

parser.add_argument(
    "--lapack_include",
    help="Directory of the lapack include.",
    default="/usr/local/opt/lapack/include",
)

parser.add_argument(
    "--lapack",
    help="Directory of both lapack lib and include. '/include' and '/lib' will be added to the end of this string.",
)

parser.add_argument(
    "--gsl_lib", help="Directory of the gsl lib.", default="/usr/local/opt/gsl/lib"
)

parser.add_argument(
    "--gsl_include",
    help="Directory of the gsl include.",
    default="/usr/local/opt/gsl/include",
)

parser.add_argument(
    "--gsl",
    help="Directory of both gsl lib and include. '/include' and '/lib' will be added to the end of this string.",
)

args, unknown = parser.parse_known_args()

for key in [
    args.gsl_include,
    args.gsl_lib,
    args.gsl,
    "--gsl",
    "--gsl_include",
    "--gsl_lib",
    args.lapack_include,
    args.lapack_lib,
    args.lapack,
    "--lapack",
    "--lapack_lib",
    "--lapack_include",
]:
    try:
        sys.argv.remove(key)
    except ValueError:
        pass

use_omp = not args.no_omp


# Obtain the numpy include directory. This logic works across numpy versions.
try:
    numpy_include = numpy.get_include()
except AttributeError:
    numpy_include = numpy.get_numpy_include()

if args.lapack is None:
    lapack_include = [args.lapack_include]
    lapack_lib = [args.lapack_lib]

else:
    lapack_include = [args.lapack + "/include"]
    lapack_lib = [args.lapack + "/lib"]

if args.gsl is None:
    gsl_include = [args.gsl_include]
    gsl_lib = [args.gsl_lib]

else:
    gsl_include = [args.gsl + "/include"]
    gsl_lib = [args.gsl + "/lib"]


if "--no_omp" in sys.argv:
    use_omp = False
    sys.argv.remove("--no_omp")

else:
    use_omp = True

fp_out_name = "few/utils/constants.py"
fp_in_name = "include/global.h"

# develop few.utils.constants.py
with open(fp_out_name, "w") as fp_out:
    with open(fp_in_name, "r") as fp_in:
        lines = fp_in.readlines()
        for line in lines:
            if len(line.split()) == 3:
                if line.split()[0] == "#define":
                    try:
                        _ = float(line.split()[2])
                        string_out = line.split()[1] + " = " + line.split()[2] + "\n"
                        fp_out.write(string_out)

                    except (ValueError) as e:
                        continue


# if installing for CUDA, build Cython extensions for gpu modules
if run_cuda_install:

    gpu_extension = dict(
        libraries=["cudart", "cublas", "cusparse", "gsl", "gslcblas", "gomp"],
        library_dirs=[CUDA["lib64"]] + gsl_lib,
        runtime_library_dirs=[CUDA["lib64"]],
        language="c++",
        # This syntax is specific to this build system
        # we're only going to use certain compiler args with nvcc
        # and not with gcc the implementation of this trick is in
        # customize_compiler()
        extra_compile_args={
            "gcc": ["-std=c++11", "-fopenmp", "-D__USE_OMP__"],  # '-g'],
            "nvcc": [
                "-arch=sm_70",
                # "-gencode=arch=compute_30,code=sm_30",
                # "-gencode=arch=compute_50,code=sm_50",
                # "-gencode=arch=compute_52,code=sm_52",
                # "-gencode=arch=compute_60,code=sm_60",
                # "-gencode=arch=compute_61,code=sm_61",
                "-gencode=arch=compute_70,code=sm_70",
                #'-gencode=arch=compute_75,code=sm_75',
                #'-gencode=arch=compute_75,code=compute_75',
                "-std=c++11",
                "--default-stream=per-thread",
                "--ptxas-options=-v",
                "-c",
                "--compiler-options",
                "'-fPIC'",
                "-Xcompiler",
                "-fopenmp",
                "-D__USE_OMP__",
                # "-G",
                # "-g",
                # "-O0",
                # "-lineinfo",
            ],  # for debugging
        },
        include_dirs=[numpy_include, CUDA["include"], "include"] + gsl_include,
    )

    if use_omp is False:
        gpu_extension["extra_compile_args"]["nvcc"].remove("-fopenmp")
        gpu_extension["extra_compile_args"]["gcc"].remove("-fopenmp")
        gpu_extension["extra_compile_args"]["nvcc"].remove("-D__USE_OMP__")
        gpu_extension["extra_compile_args"]["gcc"].remove("-D__USE_OMP__")

    matmul_ext = Extension(
        "pymatmul", sources=["src/matmul.cu", "src/pymatmul.pyx"], **gpu_extension
    )

    interp_ext = Extension(
        "pyinterp", sources=["src/interpolate.cu", "src/pyinterp.pyx"], **gpu_extension
    )

# build all cpu modules
cpu_extension = dict(
    libraries=["gsl", "gslcblas", "lapack", "gomp", "hdf5", "hdf5_hl"],
    language="c++",
    runtime_library_dirs=[],
    extra_compile_args={
        "gcc": ["-std=c++11", "-fopenmp", "-fPIC", "-D__USE_OMP__"]
    },  # '-g'
    include_dirs=[numpy_include, "include"] + lapack_include + gsl_include,
    library_dirs=lapack_lib + gsl_lib,
    # library_dirs=["/home/ajchua/lib/"],
)

if use_omp is False:
    cpu_extension["extra_compile_args"]["gcc"].remove("-fopenmp")
    cpu_extension["extra_compile_args"]["gcc"].remove("-D__USE_OMP__")

FLUX_ext = Extension(
    "pyFLUX",
    sources=["src/Interpolant.cc", "src/FluxInspiral.cc", "src/FLUX.pyx"],
    **cpu_extension,
)

# Install cpu versions of gpu modules

# need to copy cuda files to cpp for this special compiler we are using
# also copy pyx files to cpu version
src = "src/"

cp_cu_files = ["matmul", "interpolate"]
cp_pyx_files = ["pymatmul", "pyinterp"]

for fp in cp_cu_files:
    shutil.copy(src + fp + ".cu", src + fp + ".cpp")

for fp in cp_pyx_files:
    shutil.copy(src + fp + ".pyx", src + fp + "_cpu.pyx")

matmul_cpu_ext = Extension(
    "pymatmul_cpu", sources=["src/matmul.cpp", "src/pymatmul_cpu.pyx"], **cpu_extension
)

shutil.copy("src/interpolate.cu", "src/interpolate.cpp")
shutil.copy("src/pyinterp.pyx", "src/pyinterp_cpu.pyx")

interp_cpu_ext = Extension(
    "pyinterp_cpu",
    sources=["src/interpolate.cpp", "src/pyinterp_cpu.pyx"],
    **cpu_extension,
)


spher_harm_ext = Extension(
    "pySpinWeightedSpherHarm",
    sources=["src/SWSH.cc", "src/pySWSH.pyx"],
    **cpu_extension,
)

Interp2DAmplitude_ext = Extension(
    "pyInterp2DAmplitude",
    sources=["src/Interpolant.cc", "src/Amplitude.cc", "src/pyinterp2damp.pyx"],
    **cpu_extension,
)

cpu_extensions = [
    matmul_cpu_ext,
    FLUX_ext,
    interp_cpu_ext,
    spher_harm_ext,
    Interp2DAmplitude_ext,
]

if run_cuda_install:
    gpu_extensions = [matmul_ext, interp_ext]
    extensions = gpu_extensions + cpu_extensions
else:
    # extensions = [FLUX_ext, SlowFlux_ext, spher_harm_ext, Interp2DAmplitude_ext]
    extensions = cpu_extensions
    # extensions = [interp_cpu_ext]

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="few",
    author="Michael Katz",
    author_email="mikekatz04@gmail.com",
    description="Fast and accurate EMRI Waveforms.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    version="0.1.0",
    url="https://github.com/mikekatz04/FastEMRIWaveforms",
    ext_modules=extensions,
    packages=["few", "few.utils", "few.trajectory", "few.amplitude", "few.summation"],
    py_modules=[
        "few.trajectory.flux",
        "few.waveform",
        "few.amplitude.romannet",
        "few.amplitude.interp2dcubicspline",
        "few.utils.modeselector",
        "few.summation.directmodesum",
        "few.utils.ylm",
        "few.summation.interpolatedmodesum",
        "few.utils.constants",
        "few.utils.getfiles",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License (GPL)",
        "Environment :: GPU :: NVIDIA CUDA",
        "Natural Language :: English",
        "Programming Language :: C++",
        "Programming Language :: Cython",
        "Programming Language :: Python :: 3.7",
    ],
    # Inject our custom trigger
    cmdclass={"build_ext": custom_build_ext},
    # Since the package has c code, the egg cannot be zipped
    zip_safe=False,
    python_requires=">=3.6",
)


# remove src files created in this setup (cpp, pyx cpu files for gpu modules)
for fp in cp_cu_files:
    os.remove(src + fp + ".cpp")

for fp in cp_pyx_files:
    os.remove(src + fp + "_cpu.pyx")
