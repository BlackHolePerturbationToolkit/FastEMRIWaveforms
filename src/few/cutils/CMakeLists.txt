# =============================
# ==== BACKEND COMPILATION ====
# =============================

# * * * * * * * * * * * * * * * * * * * * *
# * * Helper functions to detect LAPACK * *
# * * * * * * * * * * * * * * * * * * * * *

# few_try_get_lapacke_with_cmake: This method will try to locate LAPACKE using
# CMake find_package() mechanism
#
# This functions sets the following variables in the parent scope:
#
# * In case of success:
#
#   * LAPACKE_WITH_CMAKE_SUCCESS to ON
#   * LAPACKE_WITH_CMAKE_LIBS to targets to link against lapack
#
# * In case of failure:
#
#   * LAPACKE_WITH_CMAKE_SUCCESS to OFF
function(few_try_get_lapacke_with_cmake)

  message(CHECK_START "Trying with find_package()")
  find_package(LAPACKE)
  if(LAPACKE_FOUND)
    set(LAPACKE_WITH_CMAKE_SUCCESS ON PARENT_SCOPE)
    set(LAPACKE_WITH_CMAKE_LIBS "LAPACK::LAPACK" PARENT_SCOPE)
    message(CHECK_PASS "success! Found LAPACKE ${LAPACKE_VERSION}")
  else()
    set(LAPACKE_WITH_CMAKE_SUCCESS OFF PARENT_SCOPE)
    message(CHECK_FAIL "not found")
  endif()
endfunction()

# This method will try to locate LAPACKE using PkgConfig
#
# This functions sets the following variables in the parent scope:
#
# * In case of success:
#
#   * LAPACKE_WITH_PKGCONFIG_SUCCESS to ON
#   * LAPACKE_WITH_PKGCONFIG_LIBS to targets to link against lapack
#
# * In case of failure:
#
#   * LAPACKE_WITH_PKGCONFIG_SUCCESS to OFF
#   * LAPACKE_WITH_PKGCONFIG_REASON to
#
#     * "MISSING_PKGCONFIG" if PKGCONFIG is not available
#     * "MISSING_LAPACKE" if lapacke.pc is not found
function(few_try_get_lapacke_with_pkgconfig)
  message(CHECK_START "Trying with PkgConfig")
  find_package(PkgConfig)
  if(NOT PkgConfig_FOUND)
    message(CHECK_FAIL "PkgConfig not available")
    set(LAPACKE_WITH_PKGCONFIG_SUCCESS OFF PARENT_SCOPE)
    set(LAPACKE_WITH_PKGCONFIG_REASON "MISSING_PKGCONFIG" PARENT_SCOPE)
    return()
  endif()

  pkg_check_modules(lapacke IMPORTED_TARGET lapacke)

  if(lapacke_FOUND)
    message(CHECK_PASS "success! Found LAPACKE ${lapacke_VERSION}")
    set(LAPACKE_WITH_PKGCONFIG_SUCCESS ON PARENT_SCOPE)
    set(LAPACKE_WITH_PKGCONFIG_LIBS "PkgConfig::lapacke" PARENT_SCOPE)
  else()
    message(CHECK_FAIL "not found")
    set(LAPACKE_WITH_PKGCONFIG_SUCCESS OFF PARENT_SCOPE)
    set(LAPACKE_WITH_PKGCONFIG_REASON "MISSING_LAPACKE" PARENT_SCOPE)
  endif()
endfunction()

# This method will use CPM (CMake Package Manager) to fetch LAPACK sources and
# and them to the build tree with LAPACKE support.
#
# This function sets to following variables in the parent scope:
#
# * LAPACKE_WITH_CPM_SUCCESS to ON
# * LAPACKE_WITH_CPM_LIBS to the link targets
function(few_try_get_lapacke_with_cpm)
  include(CPM)
  message(CHECK_START "Trying with automatic fetching of Reference LAPACK")
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
  set(LAPACKE_WITH_CPM_SUCCESS ON PARENT_SCOPE)
  set(LAPACKE_WITH_CPM_LIBS "lapacke" PARENT_SCOPE)
  message(CHECK_PASS "done")
endfunction()

# This method will try to make LAPACKE available using the following strategies:
#
# * PKGCONFIG: use the function "few_try_get_lapacke_with_pkgconfig"
# * CMAKE: use the function "few_try_get_lapacke_with_cmake"
# * FETCH: use the function "few_try_get_lapacke_with_cpm"
#
# The PKGCONFIG strategy is only attempted if FEW_LAPACKE_DETECT_WITH is "AUTO"
# or "PKGCONFIG". The CMAKE strategy is only attempted if
# FEW_LAPACKE_DETECT_WITH is "AUTO" or "CMAKE". The FETCH strategy is only
# attempted if FEW_LAPACKE_FETCH is "AUTO" or "ON".
#
# If no strategy is attempted, the function raises a warning and does not set
# any variable in parent scope. In this case, it is the user responsibility to
# set FEW_LAPACKE_LIBS to any relevant value.
#
# If at least one strategy is attempted and if no strategy succeeds, the method
# fails with a fatal error explaining how to help attempted strategies to work.
#
# If one strategy succeeds, the following variables are set in parent scope:
#
# * FEW_LAPACKE_LIBS: list of libraries to link against to use LAPACKE
# * FEW_LAPACKE_GET_SUCCESS: ON
# * FEW_LAPACKE_GET_STRATEGY: strategy that succeeded (PKGCONFIG|CMAKE|FETCH )
function(few_get_lapacke)
  # cmake-lint: disable=R0912,R0915
  message(CHECK_START "Locating LAPACKE")

  # I. Detect enabled strategies
  if(FEW_LAPACKE_DETECT_WITH STREQUAL "AUTO" OR FEW_LAPACKE_DETECT_WITH
                                                STREQUAL "PKGCONFIG")
    set(pkgconfig_strategy_enabled ON)
  else()
    set(pkgconfig_strategy_enabled OFF)
  endif()
  if(FEW_LAPACKE_DETECT_WITH STREQUAL "AUTO" OR FEW_LAPACKE_DETECT_WITH
                                                STREQUAL "CMAKE")
    set(cmake_strategy_enabled ON)
  else()
    set(cmake_strategy_enabled OFF)
  endif()
  if(FEW_LAPACKE_FETCH STREQUAL "AUTO" OR FEW_LAPACKE_FETCH STREQUAL "ON")
    set(fetch_strategy_enabled ON)
  else()
    set(fetch_strategy_enabled OFF)
  endif()
  if(pkgconfig_strategy_enabled OR cmake_strategy_enabled
     OR fetch_strategy_enabled)
    set(any_strategy_enabled ON)
  else()
    set(any_strategy_enabled OFF)
  endif()

  # II. Apply the PkgConfig strategy if enabled
  if(pkgconfig_strategy_enabled)
    few_try_get_lapacke_with_pkgconfig()
    if(LAPACKE_WITH_PKGCONFIG_SUCCESS)
      set(FEW_LAPACKE_LIBS "${LAPACKE_WITH_PKGCONFIG_LIBS}" PARENT_SCOPE)
      set(FEW_LAPACKE_GET_SUCCESS ON PARENT_SCOPE)
      set(FEW_LAPACKE_GET_STRATEGY "PKGCONFIG" PARENT_SCOPE)
      message(CHECK_PASS "found with pkgconfig")
      return()
    endif()
  endif()

  # III. Apply the CMake strategy if enabled
  if(cmake_strategy_enabled)
    few_try_get_lapacke_with_cmake()
    if(LAPACKE_WITH_CMAKE_SUCCESS)
      set(FEW_LAPACKE_LIBS "${LAPACKE_WITH_CMAKE_LIBS}" PARENT_SCOPE)
      set(FEW_LAPACKE_GET_SUCCESS ON PARENT_SCOPE)
      set(FEW_LAPACKE_GET_STRATEGY "CMAKE" PARENT_SCOPE)
      message(CHECK_PASS "found with find_package()")
      return()
    endif()
  endif()

  # IV. Apply the Fetch strategy if enabled
  if(fetch_strategy_enabled)
    few_try_get_lapacke_with_cpm()
    if(LAPACKE_WITH_CPM_SUCCESS)
      set(FEW_LAPACKE_LIBS "${LAPACKE_WITH_CPM_LIBS}" PARENT_SCOPE)
      set(FEW_LAPACKE_GET_SUCCESS ON PARENT_SCOPE)
      set(FEW_LAPACKE_GET_STRATEGY "FETCH" PARENT_SCOPE)
      message(CHECK_PASS "added to build tree with automatic fetching")
      return()
    endif()
  endif()

  # V. Fail if any strategy was applied
  if(any_strategy_enabled)
    message(CHECK_FAIL "not found")
    if(pkgconfig_strategy_enabled)
      message(WARNING "LAPACKE could not be located with PKGCONFIG.")
      if(LAPACKE_WITH_PKGCONFIG_REASON STREQUAL "MISSING_PKGCONFIG")
        message(
          WARNING "Make sure that pkg-config executable is "
                  "installed in your environment.\n\n"
                  "On Ubuntu, it can be installed with\n"
                  "  $ sudo apt install pkg-config\n\n"
                  "On mac OS, it can be installed with Homebrew with\n"
                  "  $ brew install pkgconf\n\n"
                  "In conda environment, it can be installed with\n"
                  "  $ conda install pkgconfig")
      elseif(LAPACKE_WITH_PKGCONFIG_REASON STREQUAL "MISSING_LAPACKE")
        message(
          WARNING
            "PkgConfig could not locate the file 'lapacke.pc'.\n"
            "If your LAPACK installation provides it, add its directory to "
            "the PKG_CONFIG_PATH environment variable.\n"
            "It is usually located in the library install path, in the "
            "'lib/pkgconfig' subdirectory.")
      endif()
    endif()
    if(cmake_strategy_enabled)
      message(
        WARNING
          "LAPACKE could not be located with CMake find_package() mechanism.\n"
          "If your LAPACK installation provides a 'lapacke-config.cmake' file "
          "(or similar installed target file), add its path to the "
          "CMAKE_PREFIX_PATH environment variable.\n"
          "It is usually located in the library install path, in the "
          "'lib/cmake' subdirectory.")
    endif()
    if(fetch_strategy_enabled)
      message(
        WARNING "LAPACKE automatic fetching was enabled but somehow "
                "failed?...\n"
                "CMake processing should have stop much sooner with a detailed "
                "explanation of the failure.\n" "See previous error messages.")
    endif()
    message(
      FATAL_ERROR
        "LAPACKE support is required but could not be satisfied." "\n\n"
        "READ CAREFULLY PREVIOUS WARNINGS - "
        "THEY SHOULD HELP YOU TO FIX THE ISSUE.")
  endif()

  # VI. Add message
  message(CHECK_PASS "ignored "
          "(FEW_LAPACKE_DETECT_WITH=${FEW_LAPACKE_DETECT_WITH}"
          " and FEW_LAPACKE_FETCH=${FEW_LAPACKE_FETCH})")
  message(
    WARNING
      "LAPACKE detection strategies were disabled.\n"
      "Manually define FEW_LAPACKE_LIBS using pip "
      "--config-settings=cmake.define.FEW_LAPACKE_LIBS=value\n\n"
      "Make sure that the compiler can locate lapacke libraries (usually by "
      "adding their directory to "
      "the LIBRARY_PATH environment variable) and the header 'lapacke.h' "
      "(usually done by adding its directory to the CPATH environment "
      "variable).")
endfunction()

# ---- Handle LAPACK/LAPACKE access ----
get_target_property(_FEW_WITH_CPU fastemriwaveforms WITH_CPU)

if(_FEW_WITH_CPU)
  # This method will locate or download LAPACKE and set ${FEW_LAPACKE_LIBS}. In
  # case of failure, it will stop the processing execution.
  few_get_lapacke()
endif()

set(COMMON_ARGS INCLUDE PRIVATE ${Python_NumPy_INCLUDE_DIR})

few_add_lib_to_cuda_cpu_backends(
  NAME pymatmul
  PYX_SOURCES pymatmul.pyx
  CU_SOURCES matmul.cu
  HEADERS cuda_complex.hpp global.h matmul.hh ${COMMON_ARGS})

few_add_lib_to_cuda_cpu_backends(
  NAME pyinterp
  PYX_SOURCES pyinterp.pyx
  CU_SOURCES interpolate.cu CPU_LINK PUBLIC ${FEW_LAPACKE_LIBS}
  HEADERS cuda_complex.hpp global.h matmul.hh ${COMMON_ARGS})

few_add_lib_to_cuda_cpu_backends(
  NAME pyAAK
  PYX_SOURCES gpuAAKWrap.pyx
  CU_SOURCES gpuAAK.cu
  HEADERS cuda_complex.hpp global.h gpuAAK.hh ${COMMON_ARGS})

few_add_lib_to_cuda_cpu_backends(
  NAME pyAmpInterp2D
  PYX_SOURCES pyampinterp2D.pyx
  CU_SOURCES AmpInterp2D.cu
  HEADERS cuda_complex.hpp global.h AmpInterp2D.hh ${COMMON_ARGS})
