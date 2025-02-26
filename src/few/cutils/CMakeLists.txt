# Define the compilation of the Cython part of this project

# ---- Handle LAPACK/LAPACKE access ----
get_target_property(_FEW_WITH_CPU fastemriwaveforms WITH_CPU)

if(_FEW_WITH_CPU)
  if(FEW_LAPACKE_FETCH STREQUAL "AUTO")
    message(STATUS "FEW_LAPACKE_FETCH set to AUTO...")
    # Try to find both LAPACK and LAPACKE
    if(FEW_LAPACKE_DETECT_WITH STREQUAL "AUTO")
      message(STATUS "  ... trying to detect LAPACK(E) with any method")
      message(STATUS "    - attempting with CMake...")

      find_package(lapack)
      find_package(lapacke)
      if(lapack_FOUND AND lapacke_FOUND)
        message(STATUS "      ... LAPACK(E) found with CMake!")
        set(FEW_LAPACKE_LIBS "lapacke")
      else()
        message(STATUS "      ... LAPACK(E) not found with CMake")
        message(STATUS "    - attempting with PkgConfig...")

        find_package(PkgConfig)

        if(PkgConfig_FOUND)
          pkg_check_modules(lapacke IMPORTED_TARGET lapacke)
          pkg_check_modules(lapack IMPORTED_TARGET lapack)
          if(lapacke_FOUND AND lapack_FOUND)
            message(STATUS "      ... LAPACK(E) found with PkgConfig!")
            set(FEW_LAPACKE_LIBS "PkgConfig::lapacke;PkgConfig::lapack")
          else()
            message(STATUS "      ... LAPACK(E) not found with PkgConfig")
          endif()
        else()
          message(STATUS "      ... PkgConfig is not available.")
        endif()
      endif()
    elseif(FEW_LAPACKE_DETECT_WITH STREQUAL "CMAKE")
      find_package(lapack)
      find_package(lapacke)
    elseif(FEW_LAPACKE_DETECT_WITH STREQUAL "PKGCONFIG")
      find_package(PkgConfig REQUIRED)
      pkg_check_modules(lapacke IMPORTED_TARGET lapacke)
      pkg_check_modules(lapack IMPORTED_TARGET lapack)
    endif()
    if(lapacke_FOUND AND lapack_FOUND)
      set(_FEW_ACTUALLY_FETCH_LAPACK OFF)
      message(STATUS "Found LAPACKE ${lapacke_VERSION}.")
      if(NOT FEW_LAPACKE_LIBS)
        if(FEW_LAPACKE_DETECT_WITH STREQUAL "CMAKE")
          set(FEW_LAPACKE_LIBS "lapacke")
        elseif(FEW_LAPACKE_DETECT_WITH STREQUAL "PKGCONFIG")
          set(FEW_LAPACKE_LIBS "PkgConfig::lapacke;PkgConfig::lapack")
        endif()
      endif()
    else()
      message(STATUS "Could not find LAPACKE. Turning fetching ON.")
      set(_FEW_ACTUALLY_FETCH_LAPACK ON)
    endif()

  elseif(FEW_LAPACKE_FETCH STREQUAL "ON")
    message(STATUS "FEW_LAPACKE_FETCH set to ON...")
    set(_FEW_ACTUALLY_FETCH_LAPACK ON)

  elseif(FEW_LAPACKE_FETCH STREQUAL "OFF")
    message(STATUS "FEW_LAPACKE_FETCH set to OFF...")
    if(FEW_LAPACKE_DETECT_WITH STREQUAL "AUTO")
      message(STATUS "  ... trying to detect LAPACK(E) with any method")
      message(STATUS "    - attempting with CMake...")
      # Try with CMake
      find_package(lapack)
      find_package(lapacke)
      if(lapacke_FOUND AND lapack_FOUND)
        message(STATUS "      ... LAPACK(E) found with CMake!")
        set(FEW_LAPACKE_LIBS "lapacke")
      else()
        message(STATUS "      ... LAPACK(E) not found with CMake")
        message(STATUS "    - attempting with PkgConfig...")
        find_package(PkgConfig)
        if(PkgConfig_FOUND)
          pkg_check_modules(lapacke IMPORTED_TARGET lapacke)
          pkg_check_modules(lapack IMPORTED_TARGET lapack)
          if(lapacke_FOUND AND lapack_FOUND)
            message(STATUS "      ... LAPACK(E) found with PkgConfig!")
            set(FEW_LAPACKE_LIBS "PkgConfig::lapacke;PkgConfig::lapack")
          else()
            message(STATUS "      ... LAPACK(E) not found with PkgConfig")
          endif()
        else()
          message(STATUS "      ... PkgConfig is not available.")
        endif()
      endif()
      if(NOT (lapacke_FOUND AND lapack_FOUND))
        message(
          FATAL_ERROR
            "LAPACKE fetching is disabled but no method managed to locate it.
Ensure you have a local install of lapack with lapacke support.
If needed, add the directory containing 'lapacke.pc' to PKG_CONFIG_PATH
or the one containing lapacke-config.cmake to CMAKE_PREFIX_PATH and ensure pkgconf is installed."
        )
      endif()
    elseif(FEW_LAPACKE_DETECT_WITH STREQUAL "CMAKE")
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
      GIT_TAG 6ec7f2bc4ecf4c4a93496aa2fa519575bc0e39ca # v3.12.1
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
endif()

set(_COMMON_ARGS INCLUDE PRIVATE ${Python_NumPy_INCLUDE_DIR})

few_add_lib_to_cuda_cpu_backends(
  NAME pymatmul
  PYX_SOURCES pymatmul.pyx
  CU_SOURCES matmul.cu
  HEADERS cuda_complex.hpp global.h matmul.hh ${_COMMON_ARGS})

few_add_lib_to_cuda_cpu_backends(
  NAME pyinterp
  PYX_SOURCES pyinterp.pyx
  CU_SOURCES interpolate.cu CPU_LINK PUBLIC ${FEW_LAPACKE_LIBS}
  HEADERS cuda_complex.hpp global.h matmul.hh ${_COMMON_ARGS})

few_add_lib_to_cuda_cpu_backends(
  NAME pyAAK
  PYX_SOURCES gpuAAKWrap.pyx
  CU_SOURCES gpuAAK.cu
  HEADERS cuda_complex.hpp global.h gpuAAK.hh ${_COMMON_ARGS})

few_add_lib_to_cuda_cpu_backends(
  NAME pyAmpInterp2D
  PYX_SOURCES pyampinterp2D.pyx
  CU_SOURCES AmpInterp2D.cu
  HEADERS cuda_complex.hpp global.h AmpInterp2D.hh ${_COMMON_ARGS})
