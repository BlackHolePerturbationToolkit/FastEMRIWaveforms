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

# ---- Define project specific options ----
set(FEW_WITH_GPU AUTO CACHE STRING "Enable GPU support")
set_property(
  CACHE FEW_WITH_GPU
  PROPERTY STRINGS
           AUTO
           ON
           OFF
           ONLY
           HELPSTRING
           "Whether to enable GPU support"
           TYPE
           "STRING"
           VALUE
           "AUTO")
set(FEW_CUDA_ARCH
    CACHE STRING "CUDA Architecture targetted for FEW compilation (see doc of \
          CMAKE_CUDA_ARCHITECTURE).")

set(FEW_LAPACKE_FETCH AUTO CACHE STRING
                                 "Download and compile LAPACK (with LAPACKE)")
set_property(
  CACHE FEW_LAPACKE_FETCH
  PROPERTY STRINGS
           AUTO
           ON
           OFF
           HELPSTRING
           "Whether to download and compile LAPACK(E)"
           TYPE
           "STRING"
           VALUE
           "AUTO")

set(FEW_LAPACKE_LIBS "" CACHE STRING "Libs to link to build against LAPACKE.")
set_property(
  CACHE FEW_LAPACKE_LIBS
  PROPERTY HELPSTRING "Libs to link to build against LAPACKE." TYPE "STRING")

set(FEW_LAPACKE_DETECT_WITH "AUTO" CACHE STRING "Tool used to locate LAPACKE.")
set_property(
  CACHE FEW_LAPACKE_DETECT_WITH
  PROPERTY STRINGS
           AUTO
           CMAKE
           PKGCONFIG
           HELPSTRING
           "Tool used to locate LAPACKE."
           TYPE
           "STRING"
           VALUE
           "CMAKE")

# ---- Phony target for project specific properties ----
add_library(fastemriwaveforms INTERFACE)

# ---- Enable building the CPU version of backends by default ----
set_target_properties(fastemriwaveforms PROPERTIES WITH_CPU ON)

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
unset(_FEW_WITH_GPU)

# ---- Include sources ----
add_subdirectory(src)

# ---- Let CMake handle copying CITATION.cff into wheel ----
if(SKBUILD_STATE STREQUAL "sdist" OR SKBUILD_STATE STREQUAL "wheel")
  install(FILES CITATION.cff DESTINATION few)
endif()
