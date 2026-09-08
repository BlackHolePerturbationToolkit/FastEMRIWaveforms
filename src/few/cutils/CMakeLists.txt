# =============================
# ==== BACKEND COMPILATION ====
# =============================

# * * * * * * * * * * * * * * * * * * * * * * * * *
# * * Helper functions to define backend option * *
# * * * * * * * * * * * * * * * * * * * * * * * * *

# In the project root CMakeLists.txt, we defined a "fastemriwaveforms" interface
# target with properties WITH_CPU and WITH_GPU defining whether the CPU and a
# GPU backend need to be compiled. Let's retrieve these information here:
get_target_property(FEW_WITH_CPU fastemriwaveforms WITH_CPU)
get_target_property(FEW_WITH_GPU fastemriwaveforms WITH_GPU)

# Adapter to let inplace editable install work: the compiled backend will be
# placed into the source-tree in the 'src' directory:
if(SKBUILD_STATE STREQUAL "editable")
  set(BACKEND_BASE_OUTPUT_DIRECTORY "${fastemriwaveforms_BINARY_DIR}/src")
else()
  set(BACKEND_BASE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")
endif()

# apply_cpu_backend_common_options
# --------------------------------
#
# This method applies some common directive to CPU backend targets. It:
#
# * Expects a single "libname" argument
# * Expects the target to be named "few_cpu_${libname}"
# * Defines the LIBRARY_OUTPUT_DIRECTORY property
# * Defines the OUTPUT_NAME property
# * Installs the target in the CPU backend directory
# * Ensures the target includes the NumPy header directory
# * Disable NumPy deprecated API
#
# Usage example: apply_cpu_backend_common_options(pymatmul)
function(apply_cpu_backend_common_options libname)
  set(target_name "few_cpu_${libname}")
  set_property(
    TARGET ${target_name}
    PROPERTY LIBRARY_OUTPUT_DIRECTORY
             "${BACKEND_BASE_OUTPUT_DIRECTORY}/few_backend_cpu")
  set_property(TARGET ${target_name} PROPERTY OUTPUT_NAME ${libname})

  install(TARGETS ${target_name} DESTINATION few_backend_cpu)

  get_target_property(FEW_CXX_MARCH_OPT fastemriwaveforms CXX_MARCH)
  if(FEW_CXX_MARCH_OPT)
    target_compile_options(${target_name} PRIVATE "${FEW_CXX_MARCH_OPT}")
  endif()

  target_include_directories(${target_name} PRIVATE ${Python_NumPy_INCLUDE_DIR})
  target_compile_definitions(${target_name}
                             PRIVATE NPY_NO_DEPRECATED_API=NPY_1_9_API_VERSION)
endfunction()

# apply_gpu_backend_common_options
# --------------------------------
#
# This method applies some common directive to GPU backend targets. It:
#
# * Expects a single "libname" argument
# * Expects the target to be named "few_gpu_${libname}"
# * Defines the LIBRARY_OUTPUT_DIRECTORY property
# * Defines the OUTPUT_NAME property
# * Installs the target in the GPU backend directory (e.g.
#   'few_backend_cuda12x')
# * Ensures the target includes the NumPy header directory
# * Disable NumPy deprecated API
# * Ensures the target links against CUDA libraries (cuBLAS, cuSPARSE, ...)
# * Defines the CUDA_ARCHITECTURE property
#
# Usage example: apply_gpu_backend_common_options(pymatmul)
function(apply_gpu_backend_common_options libname)
  set(target_name "few_gpu_${libname}")
  set(backend_name "few_backend_cuda${CUDAToolkit_VERSION_MAJOR}x")
  set_property(
    TARGET ${target_name}
    PROPERTY LIBRARY_OUTPUT_DIRECTORY
             "${BACKEND_BASE_OUTPUT_DIRECTORY}/${backend_name}")
  set_property(TARGET ${target_name} PROPERTY OUTPUT_NAME ${libname})

  install(TARGETS ${target_name} DESTINATION ${backend_name})

  target_include_directories(${target_name} PRIVATE ${Python_NumPy_INCLUDE_DIR})
  target_compile_definitions(${target_name}
                             PRIVATE NPY_NO_DEPRECATED_API=NPY_1_9_API_VERSION)
  target_link_libraries(${target_name} PUBLIC CUDA::cudart CUDA::cublas
                                              CUDA::cusparse)
  set_property(TARGET ${target_name} PROPERTY CUDA_ARCHITECTURES
                                              ${FEW_CUDA_ARCH})
endfunction()

# * * * * * * * * * * * * * * * * * * * *
# * * Definition of compiled backends * *
# * * * * * * * * * * * * * * * * * * * *

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
  apply_cpu_backend_common_options(pymatmul)

  target_sources(few_cpu_pymatmul PUBLIC FILE_SET HEADERS FILES
                                         cuda_complex.hpp global.h matmul.hh)
endif()

# III. Declare the GPU backend
if(FEW_WITH_GPU)
  python_add_library(few_gpu_pymatmul MODULE WITH_SOABI pymatmul.cxx matmul.cu)
  apply_gpu_backend_common_options(pymatmul)
  target_sources(few_gpu_pymatmul PUBLIC FILE_SET HEADERS FILES
                                         cuda_complex.hpp global.h matmul.hh)
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
  apply_cpu_backend_common_options(pyinterp)
  target_sources(
    few_cpu_pyinterp PUBLIC FILE_SET HEADERS FILES cuda_complex.hpp global.h
                            interpolate.hh)

endif()

# III. Declare the GPU backend
if(FEW_WITH_GPU)
  python_add_library(few_gpu_pyinterp MODULE WITH_SOABI pyinterp.cxx
                     interpolate.cu)
  apply_gpu_backend_common_options(pyinterp)
  target_sources(
    few_gpu_pyinterp PUBLIC FILE_SET HEADERS FILES cuda_complex.hpp global.h
                            interpolate.hh)
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
  apply_cpu_backend_common_options(pyAAK)
  target_sources(few_cpu_pyAAK PUBLIC FILE_SET HEADERS FILES cuda_complex.hpp
                                      global.h gpuAAK.hh)

endif()

# III. Declare the GPU backend
if(FEW_WITH_GPU)
  python_add_library(few_gpu_pyAAK MODULE WITH_SOABI gpuAAKWrap.cxx gpuAAK.cu)
  apply_gpu_backend_common_options(pyAAK)
  target_sources(few_gpu_pyAAK PUBLIC FILE_SET HEADERS FILES cuda_complex.hpp
                                      global.h gpuAAK.hh)

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
  apply_cpu_backend_common_options(pyAmpInterp2D)

  target_sources(
    few_cpu_pyAmpInterp2D PUBLIC FILE_SET HEADERS FILES cuda_complex.hpp
                                 global.h AmpInterp2D.hh)

endif()

# III. Declare the GPU backend
if(FEW_WITH_GPU)
  python_add_library(few_gpu_pyAmpInterp2D MODULE WITH_SOABI pyampinterp2D.cxx
                     AmpInterp2D.cu)
  apply_gpu_backend_common_options(pyAmpInterp2D)
  target_sources(
    few_gpu_pyAmpInterp2D PUBLIC FILE_SET HEADERS FILES cuda_complex.hpp
                                 global.h AmpInterp2D.hh)
endif()
