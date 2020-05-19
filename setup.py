# from future.utils import iteritems
import os
from os.path import join as pjoin
from setuptools import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy


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

# Obtain the numpy include directory. This logic works across numpy versions.
try:
    numpy_include = numpy.get_include()
except AttributeError:
    numpy_include = numpy.get_numpy_include()

# lib_gsl_dir = "/opt/local/lib"
# include_gsl_dir = "/opt/local/include"

if run_cuda_install:
    ext = Extension(
        "FEW",
        sources=[
            "inspiral/src/Interpolant.cc",
            "inspiral/src/FluxInspiral.cc",
            "few/src/ylm.cpp",
            "few/src/elliptic.cu",
            "few/src/interpolate.cu",
            "few/src/kernel.cu",
            "few/src/manager.cu",
            "few/FastEMRIWaveforms.pyx",
        ],
        library_dirs=[CUDA["lib64"]],
        libraries=["cudart", "cublas", "cusparse", "gsl", "gslcblas"],
        language="c++",
        runtime_library_dirs=[CUDA["lib64"]],
        # This syntax is specific to this build system
        # we're only going to use certain compiler args with nvcc
        # and not with gcc the implementation of this trick is in
        # customize_compiler()
        extra_compile_args={
            "gcc": ["-std=c++11"],  # '-g'],
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
                # "-G",
                # "-g",
                # "-O0",
                # "-lineinfo",
            ],  # for debugging
        },
        include_dirs=[
            numpy_include,
            CUDA["include"],
            "few/src",
            "inspiral/include",
            "/home/mlk667/.conda/envs/few_env/include/",
        ],
    )

    test_EXT = Extension(
        "testmatmul",
        sources=["few/src/test.cu", "few/src/testmatmul.pyx"],
        library_dirs=[CUDA["lib64"]],
        libraries=["cudart", "cublas", "cusparse", "gsl", "gslcblas"],
        language="c++",
        runtime_library_dirs=[CUDA["lib64"]],
        # This syntax is specific to this build system
        # we're only going to use certain compiler args with nvcc
        # and not with gcc the implementation of this trick is in
        # customize_compiler()
        extra_compile_args={
            "gcc": ["-std=c++11"],  # '-g'],
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
                # "-fmad=true",
                # "-ftz=true",
                # "-prec-div=false",
                # "-prec-sqrt=false",
                # "-G",
                # "-g",
                # "-O0",
                # "-lineinfo",
            ],  # for debugging
        },
        include_dirs=[
            numpy_include,
            CUDA["include"],
            "few/src",
            "inspiral/include",
            "/home/mlk667/.conda/envs/few_env/include/",
        ],
    )

NIT_ext = Extension(
    "pyNIT",
    sources=[
        "inspiral/src/Interpolant.cc",
        "inspiral/src/FluxInspiral.cc",
        "inspiral/NIT.pyx",
    ],
    library_dirs=["/home/ajchua/lib/"],
    libraries=["gsl", "gslcblas"],
    language="c++",
    runtime_library_dirs=[],
    # This syntax is specific to this build system
    # we're only going to use certain compiler args with nvcc
    # and not with gcc the implementation of this trick is in
    # customize_compiler()
    extra_compile_args={"gcc": ["-std=c++11"]},  # '-g'],
    include_dirs=[
        numpy_include,
        "few/src",
        "inspiral/include",
        "/home/mlk667/.conda/envs/few_env/include/",
    ],
)


if run_cuda_install:
    extensions = [ext, NIT_ext, test_EXT]
else:
    extensions = [NIT_ext]

extensions = [test_EXT]
setup(
    name="few",
    # Random metadata. there's more you can supply
    author="Michael Katz",
    version="0.1",
    ext_modules=extensions,
    packages=["few"],
    py_modules=["few.nit"],
    # Inject our custom trigger
    cmdclass={"build_ext": custom_build_ext},
    # Since the package has c code, the egg cannot be zipped
    zip_safe=False,
)
