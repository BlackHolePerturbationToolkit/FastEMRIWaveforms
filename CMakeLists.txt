# ====================================
# == FEW project base configuration ==
# ====================================

# ---- CMake related definitions ----
cmake_minimum_required(VERSION 3.23...3.31)

# ---- Main project definition ----
project(fastemriwaveforms VERSION ${SKBUILD_PROJECT_VERSION} LANGUAGES CXX)

# ---- Find required dependencies ----
find_package(Python COMPONENTS Interpreter Development.Module NumPy REQUIRED)

# ---- Import project-specific CMake functions ----
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# ---- Define a list of FEW specific options ----
set_property(GLOBAL PROPERTY FEW_OPTION_NAMES)

# ---- Define project specific options ----

# FEW_WITH_GPU controls the activation of both GPU and CPU backend compilations.
# Its allowed values are:
#
# * AUTO: enable CPU backend, and enable GPU backend if CUDA toolchain is found
#   in environment, otherwise disable it
# * ON: enable CPU backend, and enable GPU backend, fail if CUDA toolchain is
#   not available
# * OFF: enable CPU backend and disable GPU backend
# * ONLY: disable CPU backend and enable GPU backend (used in plugin wheel build
#   process)
# * BARE: disable both CPU and GPU backends
set(FEW_WITH_GPU "AUTO" CACHE STRING "Whether to compile GPU backend")
set_property(CACHE FEW_WITH_GPU PROPERTY STRINGS "AUTO" "ON" "OFF" "ONLY"
                                         "BARE")
set_property(GLOBAL APPEND PROPERTY FEW_OPTION_NAMES FEW_WITH_GPU)

# FEW_CUDA_ARCH will be passed as the CUDA_ARCHITECTURES property for the CUDA
# backend if it is compiled. See the documentation of CUDA_ARCHITECTURES:
# https://cmake.org/cmake/help/latest/prop_tgt/CUDA_ARCHITECTURES.html
set(FEW_CUDA_ARCH "native"
    CACHE STRING "CUDA Architecture targetted for FEW compilation (see doc of \
          CMAKE_CUDA_ARCHITECTURES).")
set_property(GLOBAL APPEND PROPERTY FEW_OPTION_NAMES FEW_CUDA_ARCH)

set(FEW_MARCH "native"
    CACHE STRING "Value of the -march compiler option if supported by compiler")
set_property(GLOBAL APPEND PROPERTY FEW_OPTION_NAMES FEW_MARCH)

# FEW_LAPACKE_DETECT_WITH sets the tool used to try to detect LAPACKE locally.
# Possible values are:
#
# * AUTO: attempts PKGCONFIG and, in case of failure, CMAKE
# * CMAKE: use CMake find_package() mechanism (try to find
#   'lapacke-config.cmake' in CMAKE_PREFIX_PATH or in standard system locations)
# * PKGCONFIG: use pkgconfig (which must be installed), try to find 'lapacke.pc'
#   and its dependencies in standard system locations or in PKG_CONFIG_PATH
# * DISABLE (or any other value): disable detection of local lapacke
#   installation
set(FEW_LAPACKE_DETECT_WITH "AUTO" CACHE STRING "Tool used to locate LAPACKE.")
set_property(CACHE FEW_LAPACKE_DETECT_WITH PROPERTY STRINGS "AUTO" "CMAKE"
                                                    "PKGCONFIG" "DISABLE")
set_property(GLOBAL APPEND PROPERTY FEW_OPTION_NAMES FEW_LAPACKE_DETECT_WITH)

# FEW_LAPACKE_FETCH controls whether LAPACKE sources will be downloaded and
# compiled with FEW backends. Possible values are:
#
# * AUTO: will try to locate LAPACKE locally and, in case of failure, will
#   enable the fetch mechanism
# * ON: will always fetch LAPACKE sources and compile them, even if already
#   present locally
# * OFF: will disable LAPACKE fetching mechanism
set(FEW_LAPACKE_FETCH "AUTO" CACHE STRING
                                   "Whether to download and compile LAPACK(E)")
set_property(CACHE FEW_LAPACKE_FETCH PROPERTY STRINGS "AUTO" "ON" "OFF")
set_property(GLOBAL APPEND PROPERTY FEW_OPTION_NAMES FEW_LAPACKE_FETCH)

# FEW_LAPACKE_EXTRA_LIBS sets extra libraries that must be linked when linking
# LAPACKE. For example, if importing the CPU backend fails and complains about
# missing "gfortran" symbols, set FEW_LAPACKE_EXTRA_LIBS=gfortran. Note that if
# this option is unset, it will be automatically set to "gfortran" if that
# library in linkable.
include(CheckLinkerFlag)
check_linker_flag(CXX "-lgfortran" GFORTRAN_AVAILABLE)
if(GFORTRAN_AVAILABLE)
  set(DEFAULT_EXTRA_LIBS "gfortran")
else()
  set(DEFAULT_EXTRA_LIBS "")
endif()
set(FEW_LAPACKE_EXTRA_LIBS "${DEFAULT_EXTRA_LIBS}"
    CACHE STRING "Extra libs to link to when\
linking against LAPACKE.")
unset(GFORTRAN_AVAILABLE)
unset(DEFAULT_EXTRA_LIBS)

# ---- Phony target for project specific properties ----
add_library(fastemriwaveforms INTERFACE)

# ---- Enable building the CPU version of backends by default ----
set_target_properties(fastemriwaveforms PROPERTIES WITH_CPU ON)

# ---- Test whether the FEW_MARCH option is supported by CXX compiler ----
include(CheckCXXCompilerFlag)
set(FEW_MARCH_CXX_OPT "-march=${FEW_MARCH}")
check_cxx_compiler_flag("${FEW_MARCH_CXX_OPT}" CXX_COMPILER_SUPPORTS_FEW_MARCH)
if(CXX_COMPILER_SUPPORTS_FEW_MARCH)
  set_property(TARGET fastemriwaveforms PROPERTY CXX_MARCH
                                                 "${FEW_MARCH_CXX_OPT}")
  message(STATUS "The CXX compiler supports option '${FEW_MARCH_CXX_OPT}'.")
else()
  message(
    WARNING "The CXX compiler does not support option '${FEW_MARCH_CXX_OPT}'. \
      It will be ignored.")
endif()

# ---- Optionnally check if GPU is supported ----
if(FEW_WITH_GPU STREQUAL "AUTO")
  if(DEFINED ENV{READTHEDOCS})
    message(
      STATUS
        "FEW GPU backend generation is disabled in ReadTheDocs build context.")
    set(FEW_WITH_GPU "OFF")
  endif()
endif()
if(FEW_WITH_GPU STREQUAL "AUTO")
  include(CheckLanguage)
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    find_package(CUDAToolkit)
  endif()
  if(CMAKE_CUDA_COMPILER AND CUDAToolkit_FOUND)
    message(
      STATUS
        "FEW GPU support was set to AUTO and will be turned ON as CUDA and \
        CUDA Toolkit are available with CUDA version \
        ${CUDAToolkit_VERSION_MAJOR}.")
    set_target_properties(fastemriwaveforms PROPERTIES WITH_GPU ON)
  else()
    message(
      STATUS
        "FEW GPU support was set to AUTO and will be turned OFF as CUDA and \
        CUDA Toolkit are not found (CMAKE_CUDA_COMPILER:${CMAKE_CUDA_COMPILER} \
        CUDAToolkit_FOUND:${CUDAToolkit_FOUND}).")
    set_target_properties(fastemriwaveforms PROPERTIES WITH_GPU OFF)
  endif()
elseif(FEW_WITH_GPU STREQUAL "ONLY")
  message(
    STATUS
      "FEW GPU support is set to ON and CPU support is set to OFF (use only to \
      build a standalone plugin).")
  set_target_properties(fastemriwaveforms PROPERTIES WITH_GPU ON)
  set_target_properties(fastemriwaveforms PROPERTIES WITH_CPU OFF)
elseif(FEW_WITH_GPU STREQUAL "BARE")
  message(STATUS "FEW GPU and CPU support are disabled (use only to build a \
      non-functional pure-python release).")
  set_target_properties(fastemriwaveforms PROPERTIES WITH_GPU OFF)
  set_target_properties(fastemriwaveforms PROPERTIES WITH_CPU OFF)
else()
  message(STATUS "FEW GPU support is set to ${FEW_WITH_GPU}.")
  set_target_properties(fastemriwaveforms PROPERTIES WITH_GPU ${FEW_WITH_GPU})
endif()

# ---- Handle GPU support ----
get_target_property(_FEW_WITH_GPU fastemriwaveforms WITH_GPU)
if(_FEW_WITH_GPU)
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
endif()

# ---- Include sources ----
add_subdirectory(src)

# ---- Let CMake handle copying CITATION.cff into wheel ----
if(SKBUILD_STATE STREQUAL "sdist" OR SKBUILD_STATE STREQUAL "wheel")
  if(_FEW_WITH_CPU)
    # A bare install does not need to embed CITATION.cff
    install(FILES CITATION.cff DESTINATION few)
  endif()
endif()

# ---- Print the list of options used for this build ----
get_property(ALL_FEW_OPTION_NAMES GLOBAL PROPERTY FEW_OPTION_NAMES)
if(ALL_FEW_OPTION_NAMES)
  message(STATUS "FEW options used for this build:")
  foreach(few_option_name ${ALL_FEW_OPTION_NAMES})
    if(DEFINED CACHE{${few_option_name}})
      get_property(FEW_OPTION_VALUE CACHE ${few_option_name} PROPERTY VALUE)
      if(DEFINED FEW_OPTION_VALUE)
        message(STATUS "  ${few_option_name}: ${FEW_OPTION_VALUE}")
      else()
        message(STATUS "  ${few_option_name}: not set")
      endif()
    else()
      message(AUTHOR_WARNING "  ${few_option_name}: not defined in cache")
    endif()
  endforeach()
else()
  message(STATUS "No FEW options were set for this build.")
endif()
