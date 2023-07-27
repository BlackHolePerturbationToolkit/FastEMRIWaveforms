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
    if "CUDAHOME" in os.environ or "CUDA_HOME" in os.environ:
        try:
            home = os.environ["CUDAHOME"]
        except KeyError:
            home = os.environ["CUDA_HOME"]

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
    "--lapack_lib",
    help="Directory of the lapack lib. If you add lapack lib, must also add lapack include.",
)

parser.add_argument(
    "--lapack_include",
    help="Directory of the lapack include. If you add lapack includ, must also add lapack lib.",
)

parser.add_argument(
    "--lapack",
    help="Directory of both lapack lib and include. '/include' and '/lib' will be added to the end of this string.",
)

parser.add_argument(
    "--gsl_lib",
    help="Directory of the gsl lib. If you add gsl lib, must also add gsl include.",
)

parser.add_argument(
    "--gsl_include",
    help="Directory of the gsl include. If you add gsl include, must also add gsl lib.",
)

parser.add_argument(
    "--gsl",
    help="Directory of both gsl lib and include. '/include' and '/lib' will be added to the end of this string.",
)

parser.add_argument(
    "--ccbin", help="path/to/compiler to link with nvcc when installing with CUDA."
)

args, unknown = parser.parse_known_args()

for key1, key2 in [
    ["--gsl_include", args.gsl_include],
    ["--gsl_lib", args.gsl_lib],
    ["--gsl", args.gsl],
    ["--lapack_include", args.lapack_include],
    ["--lapack_lib", args.lapack_lib],
    ["--lapack", args.lapack],
    ["--ccbin", args.ccbin],
]:
    keys = [key1, key2]
    if key2 is not None:
        keys.append(key1 + "=" + key2)
    for key in keys:
        try:
            sys.argv.remove(key)
        except ValueError:
            pass

# Obtain the numpy include directory. This logic works across numpy versions.
try:
    numpy_include = numpy.get_include()
except AttributeError:
    numpy_include = numpy.get_numpy_include()


if args.lapack is not None:
    add_lapack = True
    lapack_include = [args.lapack + "/include"]
    lapack_lib = [args.lapack + "/lib"]

elif args.lapack_lib is not None or args.lapack_include is not None:
    if None in [args.lapack_lib, args.lapack_include]:
        raise ValueError("If you add the lapack lib or include, you must add both.")

    add_lapack = True
    lapack_include = [args.lapack_include]
    lapack_lib = [args.lapack_lib]

else:
    add_lapack = False

if args.gsl is not None:
    add_gsl = True
    gsl_include = [args.gsl + "/include"]
    gsl_lib = [args.gsl + "/lib"]

elif args.gsl_lib is not None or args.gsl_include is not None:
    if None in [args.gsl_lib, args.gsl_include]:
        raise ValueError("If you add the gsl lib or include, you must add both.")

    add_gsl = True
    gsl_include = [args.gsl_include]
    gsl_lib = [args.gsl_lib]

else:
    add_gsl = False

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

                    except ValueError as e:
                        continue


# if installing for CUDA, build Cython extensions for gpu modules
if run_cuda_install:
    gpu_extension = dict(
        libraries=["gsl", "gslcblas", "cudart", "cublas", "cusparse"],
        library_dirs=[CUDA["lib64"]],
        runtime_library_dirs=[CUDA["lib64"]],
        language="c++",
        # This syntax is specific to this build system
        # we're only going to use certain compiler args with nvcc
        # and not with gcc the implementation of this trick is in
        # customize_compiler()
        extra_compile_args={
            "gcc": ["-std=c++11", "-fopenmp"],  # '-g'],
            "nvcc": [
                "-arch=sm_70",
                "-gencode=arch=compute_50,code=sm_50",
                "-gencode=arch=compute_52,code=sm_52",
                "-gencode=arch=compute_60,code=sm_60",
                "-gencode=arch=compute_61,code=sm_61",
                "-gencode=arch=compute_70,code=sm_70",
                "-gencode=arch=compute_75,code=sm_75",
                "-gencode=arch=compute_80,code=sm_80",
                "-std=c++11",
                "--default-stream=per-thread",
                "--ptxas-options=-v",
                "-c",
                "--compiler-options",
                "'-fPIC'",
                # "-G",
                # "-g",
                # "-O0",
                # "-lineinfo",
            ],  # for debugging
        },
        include_dirs=[numpy_include, CUDA["include"], "include"],
    )

    if args.ccbin is not None:
        gpu_extension["extra_compile_args"]["nvcc"].insert(
            0, "-ccbin={0}".format(args.ccbin)
        )

    matmul_ext = Extension(
        "pymatmul", sources=["src/matmul.cu", "src/pymatmul.pyx"], **gpu_extension
    )

    interp_ext = Extension(
        "pyinterp",
        sources=["src/Utility.cc", "src/interpolate.cu", "src/pyinterp.pyx"],
        **gpu_extension,
    )

    gpuAAK_ext = Extension(
        "pygpuAAK",
        sources=["src/Utility.cc", "src/gpuAAK.cu", "src/gpuAAKWrap.pyx"],
        **gpu_extension,
    )

# build all cpu modules
cpu_extension = dict(
    libraries=["gsl", "gslcblas", "lapack", "lapacke", "hdf5", "hdf5_hl"],
    language="c++",
    runtime_library_dirs=[],
    extra_compile_args={"gcc": ["-std=c++11", "-fopenmp", "-fPIC"]},  # '-g'
    include_dirs=[numpy_include, "include"],
    library_dirs=None,
    # library_dirs=["/home/ajchua/lib/"],
)

if add_lapack:
    cpu_extension["library_dirs"] = (
        lapack_lib
        if cpu_extension["library_dirs"] is None
        else cpu_extension["library_dirs"] + lapack_lib
    )
    cpu_extension["include_dirs"] += lapack_include

if add_gsl:
    cpu_extension["library_dirs"] = (
        gsl_lib
        if cpu_extension["library_dirs"] is None
        else cpu_extension["library_dirs"] + gsl_lib
    )
    cpu_extension["include_dirs"] += gsl_include

Interp2DAmplitude_ext = Extension(
    "pyInterp2DAmplitude",
    sources=["src/Interpolant.cc", "src/Amplitude.cc", "src/pyinterp2damp.pyx"],
    **cpu_extension,
)

inspiral_ext = Extension(
    "pyInspiral",
    sources=[
        "src/Utility.cc",
        "src/Interpolant.cc",
        "src/dIdt8H_5PNe10.cc",
        "src/ode.cc",
        "src/Inspiral.cc",
        "src/inspiralwrap.pyx",
    ],
    **cpu_extension,
)

par_map_ext = Extension(
    "pyParameterMap",
    sources=["src/ParameterMapAAK.cc", "src/ParMap.pyx"],
    **cpu_extension,
)

fund_freqs_ext = Extension(
    "pyUtility",
    sources=["src/Utility.cc", "src/utility_functions.pyx"],
    **cpu_extension,
)

# Install cpu versions of gpu modules

# need to copy cuda files to cpp for this special compiler we are using
# also copy pyx files to cpu version
src = "src/"

cp_cu_files = ["matmul", "interpolate", "gpuAAK"]
cp_pyx_files = ["pymatmul", "pyinterp", "gpuAAKWrap"]

for fp in cp_cu_files:
    shutil.copy(src + fp + ".cu", src + fp + ".cpp")

for fp in cp_pyx_files:
    shutil.copy(src + fp + ".pyx", src + fp + "_cpu.pyx")

matmul_cpu_ext = Extension(
    "pymatmul_cpu", sources=["src/matmul.cpp", "src/pymatmul_cpu.pyx"], **cpu_extension
)

interp_cpu_ext = Extension(
    "pyinterp_cpu",
    sources=["src/Utility.cc", "src/interpolate.cpp", "src/pyinterp_cpu.pyx"],
    **cpu_extension,
)

AAK_cpu_ext = Extension(
    "pycpuAAK",
    sources=["src/Utility.cc", "src/gpuAAK.cpp", "src/gpuAAKWrap_cpu.pyx"],
    **cpu_extension,
)


spher_harm_ext = Extension(
    "pySpinWeightedSpherHarm",
    sources=["src/SWSH.cc", "src/pySWSH.pyx"],
    **cpu_extension,
)

cpu_extensions = [
    matmul_cpu_ext,
    inspiral_ext,
    par_map_ext,
    interp_cpu_ext,
    spher_harm_ext,
    Interp2DAmplitude_ext,
    fund_freqs_ext,
    AAK_cpu_ext,
]

if run_cuda_install:
    gpu_extensions = [matmul_ext, interp_ext, gpuAAK_ext]
    extensions = gpu_extensions + cpu_extensions
else:
    extensions = cpu_extensions

with open("README.md", "r") as fh:
    long_description = fh.read()

# setup version file
with open("README.md", "r") as fh:
    lines = fh.readlines()

for line in lines:
    if line.startswith("Current Version"):
        version_string = line.split("Current Version: ")[1].split("\n")[0]

with open("few/_version.py", "w") as f:
    f.write("__version__ = '{}'".format(version_string))

# prepare the ode files
from few.utils.odeprepare import ode_prepare

ode_prepare()

setup(
    name="few",
    author="Michael Katz",
    author_email="mikekatz04@gmail.com",
    description="Fast and accurate EMRI Waveforms.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    version=version_string,
    url="https://github.com/mikekatz04/FastEMRIWaveforms",
    ext_modules=extensions,
    packages=["few", "few.utils", "few.trajectory", "few.amplitude", "few.summation"],
    py_modules=["few.waveform"],
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
