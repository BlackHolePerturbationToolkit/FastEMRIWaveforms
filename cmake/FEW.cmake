# Process a .pyx file into a .c file
function(few_process_cython IN_PYXFILE OUT_CXXFILE MODULE_NAME)
  add_custom_command(
    OUTPUT "${MODULE_NAME}/${OUT_CXXFILE}"
    COMMENT
      "Using Cython to build ${CMAKE_CURRENT_BINARY_DIR}/${MODULE_NAME}/${OUT_CXXFILE} from ${CMAKE_CURRENT_SOURCE_DIR}/${IN_PYXFILE}."
    COMMAND ${CMAKE_COMMAND} ARGS -E make_directory
            "${CMAKE_CURRENT_BINARY_DIR}/${MODULE_NAME}/"
    COMMAND
      Python::Interpreter ARGS -m cython
      "${CMAKE_CURRENT_SOURCE_DIR}/${IN_PYXFILE}" --output-file
      "${CMAKE_CURRENT_BINARY_DIR}/${MODULE_NAME}/${OUT_CXXFILE}" -3 -+
      --module-name "${MODULE_NAME}" -I "${CMAKE_CURRENT_SOURCE_DIR}"
    DEPENDS "${IN_PYXFILE}"
    VERBATIM)
endfunction()

# Local macro to compute name of C file corresponding to a PYX file
function(_few_convert_pyx_to_cxx PYX_FILE OUT_C_FILE)
  get_filename_component(NAME_FILE_WLE "${PYX_FILE}" NAME_WLE)
  set("${OUT_C_FILE}" "${NAME_FILE_WLE}.cpp" PARENT_SCOPE)
endfunction()

# Define a library into a backend
function(_few_add_lib_to_backend)
  # Process arguments
  set(options)
  set(oneValueArgs NAME BACKEND)
  set(multiValueArgs SOURCES LINK INCLUDE HEADERS PROPERTIES)

  cmake_parse_arguments(FEW_ADDLIB "${options}" "${oneValueArgs}"
                        "${multiValueArgs}" ${ARGN})

  # Define the library
  set(FEW_ADDLIB_TARGET_NAME
      "few_cutils_${FEW_ADDLIB_BACKEND}_${FEW_ADDLIB_NAME}")
  python_add_library(${FEW_ADDLIB_TARGET_NAME} MODULE ${FEW_ADDLIB_SOURCES}
                     WITH_SOABI)

  # Add links to other libraries
  if(FEW_ADDLIB_LINK)
    target_link_libraries(${FEW_ADDLIB_TARGET_NAME} ${FEW_ADDLIB_LINK})
  endif()

  # Add include directories
  if(FEW_ADDLIB_INCLUDE)
    target_include_directories(${FEW_ADDLIB_TARGET_NAME} ${FEW_ADDLIB_INCLUDE})
  endif()

  # Add headers to sources
  if(FEW_ADDLIB_HEADERS)
    target_sources(${FEW_ADDLIB_TARGET_NAME} PUBLIC FILE_SET HEADERS FILES
                                                    ${FEW_ADDLIB_HEADERS})
  endif()

  # Add properties
  if(FEW_ADDLIB_PROPERTIES)
    set_target_properties(${FEW_ADDLIB_TARGET_NAME}
                          PROPERTIES ${FEW_ADDLIB_PROPERTIES})
  endif()

  # Make target installed
  install(TARGETS ${FEW_ADDLIB_TARGET_NAME}
          DESTINATION "few/cutils/${FEW_ADDLIB_BACKEND}/")
  set_target_properties(
    ${FEW_ADDLIB_TARGET_NAME}
    PROPERTIES OUTPUT_NAME "${FEW_ADDLIB_NAME}"
               LIBRARY_OUTPUT_DIRECTORY
               "$CMAKE_CURRENT_BINARY_DIR}/few/cutils/${FEW_ADDLIB_BACKEND}/")

endfunction() # _few_add_lib_to_backend

# Define a library based on Cython and CUDA/CXX files. The CXX version of CUDA
# sources will be used for CPU backends. The CUDA version of CUDA sources will
# be used for GPU backends if enabled.
function(few_add_lib_to_cuda_cpu_backends)
  # Process arguments
  set(options)
  set(oneValueArgs NAME)
  set(multiValueArgs
      CU_SOURCES
      CXX_SOURCES
      PYX_SOURCES
      LINK
      INCLUDE
      HEADERS)

  cmake_parse_arguments(FEW_ADDLIB "${options}" "${oneValueArgs}"
                        "${multiValueArgs}" ${ARGN})

  # Process Cython files into CXX files
  set(FEW_ADDLIB_PYX_TO_CXX_SOURCES)
  foreach(FEW_PYX_SOURCE IN ITEMS ${FEW_ADDLIB_PYX_SOURCES})
    _few_convert_pyx_to_cxx("${FEW_PYX_SOURCE}" "FEW_CXX_SOURCE")
    list(APPEND FEW_ADDLIB_PYX_TO_CXX_SOURCES
         "${FEW_ADDLIB_NAME}/${FEW_CXX_SOURCE}")
    few_process_cython("${FEW_PYX_SOURCE}" "${FEW_CXX_SOURCE}"
                       "${FEW_ADDLIB_NAME}")
  endforeach()

  # Copy CUDA files into CXX files
  set(FEW_ADDLIB_CU_TO_CXX_SOURCES)
  foreach(CU_FILE IN ITEMS ${FEW_ADDLIB_CU_SOURCES})
    add_custom_command(
      OUTPUT "${FEW_ADDLIB_NAME}_cpu/${CU_FILE}.cxx"
      COMMENT
        "Copy ${CMAKE_CURRENT_SOURCE_DIR}/${CU_FILE} to ${CMAKE_CURRENT_BINARY_DIR}/${FEW_ADDLIB_NAME}_cpu/${CU_FILE}.cxx ."
      COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/${CU_FILE}
              ${CMAKE_CURRENT_BINARY_DIR}/${FEW_ADDLIB_NAME}_cpu/${CU_FILE}.cxx
      DEPENDS "${CU_FILE}"
      VERBATIM)
    list(APPEND FEW_ADDLIB_CU_TO_CXX_SOURCES
         "${CMAKE_CURRENT_BINARY_DIR}/${FEW_ADDLIB_NAME}_cpu/${CU_FILE}.cxx")
  endforeach()

  # Set the language of CXX source files
  set_source_files_properties(${FEW_ADDLIB_CXX_SOURCES} PROPERTIES LANGUAGE CXX)
  set_source_files_properties(${FEW_ADDLIB_PYX_TO_CXX_SOURCES}
                              PROPERTIES LANGUAGE CXX)
  set_source_files_properties(${FEW_ADDLIB_CU_TO_CXX_SOURCES}
                              PROPERTIES LANGUAGE CXX)

  # Define the CPU version
  _few_add_lib_to_backend(
    NAME "${FEW_ADDLIB_NAME}"
    BACKEND "cpu"
    SOURCES ${FEW_ADDLIB_CXX_SOURCES} ${FEW_ADDLIB_PYX_TO_CXX_SOURCES}
            ${FEW_ADDLIB_CU_TO_CXX_SOURCES}
    LINK ${FEW_ADDLIB_LINK}
    INCLUDE ${FEW_ADDLIB_INCLUDE}
    HEADERS ${FEW_ADDLIB_HEADERS})

  # Define the GPU version
  get_target_property(_FEW_WITH_GPU ${SKBUILD_PROJECT_NAME} WITH_GPU)
  if(_FEW_WITH_GPU)
    set_source_files_properties(${FEW_ADDLIB_CU_SOURCES} PROPERTIES LANGUAGE
                                                                    CUDA)
    _few_add_lib_to_backend(
      NAME "${FEW_ADDLIB_NAME}"
      BACKEND "cuda${CUDAToolkit_VERSION_MAJOR}x"
      SOURCES ${FEW_ADDLIB_CXX_SOURCES} ${FEW_ADDLIB_PYX_TO_CXX_SOURCES}
              ${FEW_ADDLIB_CU_SOURCES}
      LINK ${FEW_ADDLIB_LINK} PUBLIC CUDA::cudart CUDA::cublas CUDA::cusparse
      INCLUDE ${FEW_ADDLIB_INCLUDE}
      HEADERS ${FEW_ADDLIB_HEADERS}
      PROPERTIES CUDA_ARCHITECTURES ${FEW_CUDA_ARCH})
  endif()

endfunction() # few_add_lib_to_cuda_cpu_backends
