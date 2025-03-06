# =============================
# ==== BACKEND COMPILATION ====
# =============================

# * * * * * * * * * * * * * * * * * * * * *
# * * Helper functions to detect LAPACK * *
# * * * * * * * * * * * * * * * * * * * * *

# few_try_get_lapacke_with_cmake
# ------------------------------
#
# This method will try to locate LAPACKE using CMake find_package() mechanism.
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

# few_try_get_lapacke_with_pkgconfig
# ----------------------------------
#
# This method will try to locate LAPACKE using PkgConfig.
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

# few_try_get_lapacke_with_cpm
# ----------------------------
#
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
  set(LAPACKE_WITH_CPM_LIBS "lapacke;lapack" PARENT_SCOPE)
  message(CHECK_PASS "done")
endfunction()

# few_get_lapacke
# ---------------
#
# This method will try to make LAPACKE available using the following strategies:
#
# * PKGCONFIG: use the function "few_try_get_lapacke_with_pkgconfig"
# * CMAKE: use the function "few_try_get_lapacke_with_cmake"
# * FETCH: use the function "few_try_get_lapacke_with_cpm"
#
# The PKGCONFIG strategy is only attempted if FEW_LAPACKE_DETECT_WITH is "AUTO"
# or "PKGCONFIG". The CMAKE strategy is only attempted if
# FEW_LAPACKE_DETECT_WITH is "AUTO" or "CMAKE". The FETCH strategy is only
# attempted if FEW_LAPACKE_FETCH is "AUTO" or "ON". The CMAKE and PKGCONFIG
# strategies are forcefully disabled if FEW_LAPACKE_FETCH is "ON"
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
  if(FEW_LAPACKE_FETCH STREQUAL "ON")
    set(pkgconfig_strategy_enabled OFF)
    set(cmake_strategy_enabled OFF)
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
                "explanation of the failure.\nSee previous error messages.")
    endif()
    message(
      FATAL_ERROR
        "LAPACKE support is required but could not be satisfied." "\n\n"
        "READ CAREFULLY PREVIOUS WARNINGS - "
        "THEY SHOULD HELP YOU TO FIX THE ISSUE.")
  endif()

  # VI. Add message if lapacke detection is ignored
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

get_target_property(FEW_WITH_CPU fastemriwaveforms WITH_CPU)
get_target_property(FEW_WITH_GPU fastemriwaveforms WITH_GPU)

if(FEW_WITH_CPU)
  # This method will locate or download LAPACKE and set ${FEW_LAPACKE_LIBS}. In
  # case of failure, it will stop the processing execution.
  few_get_lapacke()
  # Now, the libraries that need access to liblapacke and 'liapacke.h' must link
  # against ${FEW_LAPACKE_LIBS}
endif()

if(FEW_WITH_GPU)
  set(FEW_GPU_BACKEND_NAME "few_backend_cuda${CUDAToolkit_VERSION_MAJOR}")
endif()

if(SKBUILD_STATE STREQUAL "editable")
  set(BACKEND_BASE_OUTPUT_DIRECTORY "${fastemriwaveforms_BINARY_DIR}/src")
else()
  set(BACKEND_BASE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")
endif()

set(COMMON_ARGS INCLUDE PRIVATE ${Python_NumPy_INCLUDE_DIR})

# ----------------
# --- pymatmul ---
# ----------------

# I. Process pymatmul.pyx into a C++ file
add_custom_command(
  OUTPUT "pymatmul.cxx"
  COMMENT "Cythonize pymatmul.pyx into pymatmul.cxx"
  COMMAND
    Python::Interpreter -m cython "${CMAKE_CURRENT_SOURCE_DIR}/pymatmul.pyx"
    --output-file "${CMAKE_CURRENT_BINARY_DIR}/pymatmul.cxx" -3 -+ --module-name
    "pymatmul" -I "${CMAKE_CURRENT_SOURCE_DIR}"
  DEPENDS "pymatmul.pyx"
  VERBATIM)

set_source_files_properties(
  "${CMAKE_CURRENT_BINARY_DIR}/pymatmul.cxx"
  PROPERTIES COMPILE_DEFINITIONS "NPY_NO_DEPRECATED_API=NPY_1_9_API_VERSION")

# II. Declare the CPU backend
if(FEW_WITH_CPU)
  add_custom_command(
    OUTPUT "matmul.cxx"
    COMMENT "Copy matmul.cu to matmul.cxx"
    COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/matmul.cu"
            "${CMAKE_CURRENT_BINARY_DIR}/matmul.cxx"
    DEPENDS "matmul.cu"
    VERBATIM)

  python_add_library(few_cpu_pymatmul MODULE WITH_SOABI pymatmul.cxx matmul.cxx)
  target_include_directories(few_cpu_pymatmul
                             PRIVATE ${Python_NumPy_INCLUDE_DIR})
  target_sources(
    few_cpu_pymatmul
    PUBLIC FILE_SET
           HEADERS
           FILES
           cuda_complex.hpp
           global.h
           matmul.hh)

  install(TARGETS few_cpu_pymatmul DESTINATION few_backend_cpu)
  set_property(TARGET few_cpu_pymatmul PROPERTY OUTPUT_NAME pymatmul)
  set_property(
    TARGET few_cpu_pymatmul
    PROPERTY LIBRARY_OUTPUT_DIRECTORY
             "${BACKEND_BASE_OUTPUT_DIRECTORY}/few_backend_cpu")

endif()

# III. Declare the GPU backend
if(FEW_WITH_GPU)
  python_add_library(few_gpu_pymatmul MODULE WITH_SOABI pymatmul.cxx matmul.cu)
  target_include_directories(few_gpu_pymatmul
                             PRIVATE ${Python_NumPy_INCLUDE_DIR})
  target_sources(
    few_gpu_pymatmul
    PUBLIC FILE_SET
           HEADERS
           FILES
           cuda_complex.hpp
           global.h
           matmul.hh)

  install(TARGETS few_gpu_pymatmul DESTINATION ${FEW_GPU_BACKEND_NAME})
  set_property(TARGET few_gpu_pymatmul PROPERTY OUTPUT_NAME pymatmul)
  set_property(
    TARGET few_gpu_pymatmul
    PROPERTY LIBRARY_OUTPUT_DIRECTORY
             "${BACKEND_BASE_OUTPUT_DIRECTORY}/${FEW_GPU_BACKEND_NAME}")
endif()

# ----------------
# --- pyinterp ---
# ----------------

# I. Process pyinterp.pyx into a C++ file
add_custom_command(
  OUTPUT "pyinterp.cxx"
  COMMENT "Cythonize pyinterp.pyx into pyinterp.cxx"
  COMMAND
    Python::Interpreter -m cython "${CMAKE_CURRENT_SOURCE_DIR}/pyinterp.pyx"
    --output-file "${CMAKE_CURRENT_BINARY_DIR}/pyinterp.cxx" -3 -+ --module-name
    "pyinterp" -I "${CMAKE_CURRENT_SOURCE_DIR}"
  DEPENDS "pyinterp.pyx"
  VERBATIM)

set_source_files_properties(
  "${CMAKE_CURRENT_BINARY_DIR}/pyinterp.cxx"
  PROPERTIES COMPILE_DEFINITIONS "NPY_NO_DEPRECATED_API=NPY_1_9_API_VERSION")

# II. Declare the CPU backend
if(FEW_WITH_CPU)
  add_custom_command(
    OUTPUT "interpolate.cxx"
    COMMENT "Copy interpolate.cu to interpolate.cxx"
    COMMAND
      ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/interpolate.cu"
      "${CMAKE_CURRENT_BINARY_DIR}/interpolate.cxx"
    DEPENDS "interpolate.cu"
    VERBATIM)

  python_add_library(few_cpu_pyinterp MODULE WITH_SOABI pyinterp.cxx
                     interpolate.cxx)
  target_include_directories(few_cpu_pyinterp
                             PRIVATE ${Python_NumPy_INCLUDE_DIR})
  target_link_libraries(few_cpu_pyinterp PUBLIC ${FEW_LAPACKE_LIBS})

  target_sources(
    few_cpu_pyinterp
    PUBLIC FILE_SET
           HEADERS
           FILES
           cuda_complex.hpp
           global.h
           interpolate.hh)

  install(TARGETS few_cpu_pyinterp DESTINATION few_backend_cpu)
  set_property(TARGET few_cpu_pyinterp PROPERTY OUTPUT_NAME pyinterp)
  set_property(
    TARGET few_cpu_pyinterp
    PROPERTY LIBRARY_OUTPUT_DIRECTORY
             "${BACKEND_BASE_OUTPUT_DIRECTORY}/few_backend_cpu")

endif()

# III. Declare the GPU backend
if(FEW_WITH_GPU)
  python_add_library(few_gpu_pyinterp MODULE WITH_SOABI pyinterp.cxx
                     interpolate.cu)
  target_include_directories(few_gpu_pyinterp
                             PRIVATE ${Python_NumPy_INCLUDE_DIR})
  target_sources(
    few_gpu_pyinterp
    PUBLIC FILE_SET
           HEADERS
           FILES
           cuda_complex.hpp
           global.h
           interpolate.hh)

  install(TARGETS few_gpu_pyinterp DESTINATION ${FEW_GPU_BACKEND_NAME})
  set_property(TARGET few_gpu_pyinterp PROPERTY OUTPUT_NAME pyinterp)
  set_property(
    TARGET few_gpu_pyinterp
    PROPERTY LIBRARY_OUTPUT_DIRECTORY
             "${BACKEND_BASE_OUTPUT_DIRECTORY}/${FEW_GPU_BACKEND_NAME}")
endif()

# -------------
# --- pyAAK ---
# -------------

# I. Process pymatmul.pyx into a C++ file
add_custom_command(
  OUTPUT "gpuAAKWrap.cxx"
  COMMENT "Cythonize gpuAAKWrap.pyx into gpuAAKWrap.cxx"
  COMMAND
    Python::Interpreter -m cython "${CMAKE_CURRENT_SOURCE_DIR}/gpuAAKWrap.pyx"
    --output-file "${CMAKE_CURRENT_BINARY_DIR}/gpuAAKWrap.cxx" -3 -+
    --module-name "pyAAK" -I "${CMAKE_CURRENT_SOURCE_DIR}"
  DEPENDS "gpuAAKWrap.pyx"
  VERBATIM)

set_source_files_properties(
  "${CMAKE_CURRENT_BINARY_DIR}/gpuAAKWrap.cxx"
  PROPERTIES COMPILE_DEFINITIONS "NPY_NO_DEPRECATED_API=NPY_1_9_API_VERSION")

# II. Declare the CPU backend
if(FEW_WITH_CPU)
  add_custom_command(
    OUTPUT "gpuAAK.cxx"
    COMMENT "Copy gpuAAK.cu to gpuAAK.cxx"
    COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/gpuAAK.cu"
            "${CMAKE_CURRENT_BINARY_DIR}/gpuAAK.cxx"
    DEPENDS "gpuAAK.cu"
    VERBATIM)

  python_add_library(few_cpu_pyAAK MODULE WITH_SOABI gpuAAKWrap.cxx gpuAAK.cxx)
  target_include_directories(few_cpu_pyAAK PRIVATE ${Python_NumPy_INCLUDE_DIR})
  target_sources(
    few_cpu_pyAAK
    PUBLIC FILE_SET
           HEADERS
           FILES
           cuda_complex.hpp
           global.h
           gpuAAK.hh)

  install(TARGETS few_cpu_pyAAK DESTINATION few_backend_cpu)
  set_property(TARGET few_cpu_pyAAK PROPERTY OUTPUT_NAME pyAAK)
  set_property(
    TARGET few_cpu_pyAAK
    PROPERTY LIBRARY_OUTPUT_DIRECTORY
             "${BACKEND_BASE_OUTPUT_DIRECTORY}/few_backend_cpu")

endif()

# III. Declare the GPU backend
if(FEW_WITH_GPU)
  python_add_library(few_gpu_pyAAK MODULE WITH_SOABI gpuAAKWrap.cxx gpuAAK.cu)
  target_include_directories(few_gpu_pyAAK PRIVATE ${Python_NumPy_INCLUDE_DIR})
  target_sources(
    few_gpu_pyAAK
    PUBLIC FILE_SET
           HEADERS
           FILES
           cuda_complex.hpp
           global.h
           gpuAAK.hh)

  install(TARGETS few_gpu_pyAAK DESTINATION ${FEW_GPU_BACKEND_NAME})
  set_property(TARGET few_gpu_pyAAK PROPERTY OUTPUT_NAME pyAAK)
  set_property(
    TARGET few_gpu_pyAAK
    PROPERTY LIBRARY_OUTPUT_DIRECTORY
             "${BACKEND_BASE_OUTPUT_DIRECTORY}/${FEW_GPU_BACKEND_NAME}")
endif()

# ---------------------
# --- pyAmpInterp2D ---
# ---------------------

# I. Process pyampinterp2D.pyx into a C++ file
add_custom_command(
  OUTPUT "pyampinterp2D.cxx"
  COMMENT "Cythonize pyampinterp2D.pyx into pyampinterp2D.cxx"
  COMMAND
    Python::Interpreter -m cython
    "${CMAKE_CURRENT_SOURCE_DIR}/pyampinterp2D.pyx" --output-file
    "${CMAKE_CURRENT_BINARY_DIR}/pyampinterp2D.cxx" -3 -+ --module-name
    "pyAmpInterp2D" -I "${CMAKE_CURRENT_SOURCE_DIR}"
  DEPENDS "pyampinterp2D.pyx"
  VERBATIM)

set_source_files_properties(
  "${CMAKE_CURRENT_BINARY_DIR}/pyampinterp2D.cxx"
  PROPERTIES COMPILE_DEFINITIONS "NPY_NO_DEPRECATED_API=NPY_1_9_API_VERSION")

# II. Declare the CPU backend
if(FEW_WITH_CPU)
  add_custom_command(
    OUTPUT "AmpInterp2D.cxx"
    COMMENT "Copy AmpInterp2D.cu to AmpInterp2D.cxx"
    COMMAND
      ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/AmpInterp2D.cu"
      "${CMAKE_CURRENT_BINARY_DIR}/AmpInterp2D.cxx"
    DEPENDS "AmpInterp2D.cu"
    VERBATIM)

  python_add_library(few_cpu_pyAmpInterp2D MODULE WITH_SOABI pyampinterp2D.cxx
                     AmpInterp2D.cxx)
  target_include_directories(few_cpu_pyAmpInterp2D
                             PRIVATE ${Python_NumPy_INCLUDE_DIR})
  target_sources(
    few_cpu_pyAmpInterp2D
    PUBLIC FILE_SET
           HEADERS
           FILES
           cuda_complex.hpp
           global.h
           AmpInterp2D.hh)

  install(TARGETS few_cpu_pyAmpInterp2D DESTINATION few_backend_cpu)
  set_property(TARGET few_cpu_pyAmpInterp2D PROPERTY OUTPUT_NAME pyAmpInterp2D)
  set_property(
    TARGET few_cpu_pyAmpInterp2D
    PROPERTY LIBRARY_OUTPUT_DIRECTORY
             "${BACKEND_BASE_OUTPUT_DIRECTORY}/few_backend_cpu")

endif()

# III. Declare the GPU backend
if(FEW_WITH_GPU)
  python_add_library(few_gpu_pyAmpInterp2D MODULE WITH_SOABI pyampinterp2D.cxx
                     AmpInterp2D.cu)
  target_include_directories(few_gpu_pyAmpInterp2D
                             PRIVATE ${Python_NumPy_INCLUDE_DIR})
  target_sources(
    few_gpu_pyAmpInterp2D
    PUBLIC FILE_SET
           HEADERS
           FILES
           cuda_complex.hpp
           global.h
           AmpInterp2D.hh)

  install(TARGETS few_gpu_pyAmpInterp2D DESTINATION ${FEW_GPU_BACKEND_NAME})
  set_property(TARGET few_gpu_pyAmpInterp2D PROPERTY OUTPUT_NAME pyAmpInterp2D)
  set_property(
    TARGET few_gpu_pyAmpInterp2D
    PROPERTY LIBRARY_OUTPUT_DIRECTORY
             "${BACKEND_BASE_OUTPUT_DIRECTORY}/${FEW_GPU_BACKEND_NAME}")
endif()
