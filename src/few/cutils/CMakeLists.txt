# Define the compilation of the Cython part of this project

# ---- Handle LAPACK/LAPACKE access ----
if(FEW_LAPACKE_FETCH STREQUAL "AUTO")
  # Try to find both LAPACK and LAPACKE
  if(FEW_LAPACKE_DETECT_WITH STREQUAL "CMAKE")
    find_package(lapacke)
  elseif(FEW_LAPACKE_DETECT_WITH STREQUAL "PKGCONFIG")
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(LAPACKE IMPORTED_TARGET lapacke)
    pkg_check_modules(LAPACK IMPORTED_TARGET lapack)
  endif()
  if(lapacke_FOUND AND lapack_FOUND)
    set(_FEW_ACTUALLY_FETCH_LAPACK OFF)
    message(STATUS "Found LAPACKE ${lapacke_VERSION}.")
    if(NOT FEW_LAPACKE_LIBS)
      if(FEW_LAPACKE_DETECT_WITH STREQUAL "CMAKE")
        set(FEW_LAPACKE_LIBS "lapacke")
      elseif(FEW_LAPACKE_DETECT_WITH STREQUAL "PKGCONFIG")
        set(FEW_LAPACKE_LIBS "PkgConfig::LAPACKE;PkgConfig::LAPACK")
      endif()
    endif()
  else()
    set(_FEW_ACTUALLY_FETCH_LAPACK ON)
  endif()

elseif(FEW_LAPACKE_FETCH STREQUAL "ON")
  set(_FEW_ACTUALLY_FETCH_LAPACK ON)

elseif(FEW_LAPACKE_FETCH STREQUAL "OFF")
  if(FEW_LAPACKE_DETECT_WITH STREQUAL "CMAKE")
    find_package(lapacke REQUIRED)
    if(NOT FEW_LAPACKE_LIBS)
      set(FEW_LAPACKE_LIBS "lapacke")
    endif()
  elseif(FEW_LAPACKE_DETECT_WITH STREQUAL "PKGCONFIG")
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(LAPACKE REQUIRED IMPORTED_TARGET lapacke)
    pkg_check_modules(LAPACK REQUIRED IMPORTED_TARGET lapack)
    if(NOT FEW_LAPACKE_LIBS)
      set(FEW_LAPACKE_LIBS "PkgConfig::LAPACKE;PkgConfig::LAPACK")
    endif()
  endif()
  set(_FEW_ACTUALLY_FETCH_LAPACK OFF)

  message(STATUS "Found LAPACKE ${lapacke_VERSION}.")

else()
  message(
    FATAL_ERROR
      "Unexpected value for FEW_FETCH_LAPACK (=${FEW_FETCH_LAPACK}, expected [ON|OFF|AUTO])."
  )

endif()

# Add message to detail how AUTO is handled
if(FEW_FETCH_LAPACK STREQUAL "AUTO")
  # Test whether LAPACK(E) is accessible in environment
  if(NOT _FEW_ACTUALLY_FETCH_LAPACK)
    message(
      STATUS
        "FEW_FETCH_LAPACK was set to AUTO and will be turned OFF as LAPACK(E) is already available."
    )
  else()
    message(
      STATUS
        "FEW_FETCH_LAPACK was set to AUTO and will be turned ON as LAPACK(E) is not available."
    )
  endif()
endif()

# ---- Install LAPACK using FetchContent ----
if(_FEW_ACTUALLY_FETCH_LAPACK)
  include(CPM)
  enable_language(Fortran)
  CPMAddPackage(
    NAME lapack
    GITHUB_REPOSITORY Reference-LAPACK/lapack
    GIT_TAG 04b044e020a3560ccfa9988c8a80a1fb7083fc2e # v3.12.0
    OPTIONS "LAPACKE"
            "ON"
            "CMAKE_POSITION_INDEPENDENT_CODE"
            "ON"
            "CMAKE_UNITY_BUILD"
            "ON"
            "CMAKE_UNITY_BUILD_BATCH_SIZE"
            64)
  if(NOT FEW_LAPACKE_LIBS)
    set(FEW_LAPACKE_LIBS "lapacke;lapack")
  endif()
endif()

set(_COMMON_ARGS INCLUDE PRIVATE ${Python_NumPy_INCLUDE_DIR} WITH_CPU_VERSION)

few_add_cython_library(
  NAME pymatmul DESTINATION "few/cutils"
  PYX_SOURCES pymatmul.pyx
  CU_SOURCES matmul.cu
  HEADERS cuda_complex.hpp global.h matmul.hh ${_COMMON_ARGS})

few_add_cython_library(
  NAME pyinterp DESTINATION "few/cutils"
  PYX_SOURCES pyinterp.pyx
  CU_SOURCES interpolate.cu
  LINK PUBLIC ${FEW_LAPACKE_LIBS}
  HEADERS cuda_complex.hpp global.h interpolate.hh ${_COMMON_ARGS})

few_add_cython_library(
  NAME pyAAK DESTINATION "few/cutils"
  PYX_SOURCES gpuAAKWrap.pyx
  CU_SOURCES gpuAAK.cu
  HEADERS cuda_complex.hpp global.h gpuAAK.hh ${_COMMON_ARGS})

few_add_cython_library(
  NAME pyAmpInterp2D DESTINATION "few/cutils"
  PYX_SOURCES pyampinterp2D.pyx
  CU_SOURCES AmpInterp2D.cu
  HEADERS cuda_complex.hpp global.h AmpInterp2D.hh ${_COMMON_ARGS})
