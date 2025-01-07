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

# Define a python library based on one or multiple Cython files
# few_add_cython_library(NAME libname SOURCES file0.pyx file1.pyx)
function(few_add_cython_library)
  # Process arguments
  set(options WITH_CPU_VERSION)
  set(oneValueArgs NAME DESTINATION)
  set(multiValueArgs
      PYX_SOURCES
      CU_SOURCES
      CXX_SOURCES
      LINK
      INCLUDE
      HEADERS)

  cmake_parse_arguments(FEW_ADDLIB "${options}" "${oneValueArgs}"
                        "${multiValueArgs}" ${ARGN})

  # Add processing command to each input cython file and build list of output C
  # files
  set(FEW_ADDLIB_PYX_TO_CXX_SOURCES)
  foreach(FEW_PYX_SOURCE IN ITEMS ${FEW_ADDLIB_PYX_SOURCES})
    _few_convert_pyx_to_cxx("${FEW_PYX_SOURCE}" "FEW_CXX_SOURCE")
    list(APPEND FEW_ADDLIB_PYX_TO_CXX_SOURCES
         "${FEW_ADDLIB_NAME}/${FEW_CXX_SOURCE}")
    few_process_cython("${FEW_PYX_SOURCE}" "${FEW_CXX_SOURCE}"
                       "${FEW_ADDLIB_NAME}")
  endforeach()

  # Define the cython-based python library
  python_add_library(
    ${FEW_ADDLIB_NAME}
    MODULE
    ${FEW_ADDLIB_PYX_TO_CXX_SOURCES}
    ${FEW_ADDLIB_CU_SOURCES}
    ${FEW_ADDLIB_CXX_SOURCES}
    WITH_SOABI)

  # Ensure CU files are interpreted as CXX or CUDA depending on GPU support
  get_target_property(_FEW_CU_LANGUAGE ${SKBUILD_PROJECT_NAME} CU_LANGUAGE)
  set_source_files_properties(${FEW_ADDLIB_CU_SOURCES}
                              PROPERTIES LANGUAGE ${_FEW_CU_LANGUAGE})
  unset(_FEW_CU_LANGUAGE)

  # If GPU support is enabled and at least one CU file is provided, then link
  # cudart, cublas and cusparse
  get_target_property(_FEW_WITH_GPU ${SKBUILD_PROJECT_NAME} WITH_GPU)
  if(_FEW_WITH_GPU)
    target_link_libraries(${FEW_ADDLIB_NAME} PUBLIC CUDA::cudart CUDA::cublas
                                                    CUDA::cusparse)
  endif()
  unset(_FEW_WITH_GPU)

  set_source_files_properties(${FEW_ADDLIB_CXX_SOURCES} PROPERTIES LANGUAGE CXX)

  # Add links to other libraries
  if(FEW_ADDLIB_LINK)
    target_link_libraries(${FEW_ADDLIB_NAME} ${FEW_ADDLIB_LINK})
  endif()

  # Add include directories
  if(FEW_ADDLIB_INCLUDE)
    target_include_directories(${FEW_ADDLIB_NAME} ${FEW_ADDLIB_INCLUDE})
  endif()

  # Add headers to sources
  if(FEW_ADDLIB_HEADERS)
    target_sources(${FEW_ADDLIB_NAME} PUBLIC FILE_SET HEADERS FILES
                                             ${FEW_ADDLIB_HEADERS})
  endif()

  # Define the CUDA architectures on target
  set_property(TARGET ${FEW_ADDLIB_NAME} PROPERTY CUDA_ARCHITECTURES
                                                  ${FEW_CUDA_ARCH})

  # Make target installed
  if(NOT FEW_ADDLIB_DESTINATION)
    set(FEW_ADDLIB_DESTINATION ".")
  endif()
  install(TARGETS ${FEW_ADDLIB_NAME} DESTINATION ${FEW_ADDLIB_DESTINATION})

  # Define the CPU version if requested
  if(FEW_ADDLIB_WITH_CPU_VERSION)
    foreach(CU_FILE IN ITEMS ${FEW_ADDLIB_CU_SOURCES})
      add_custom_command(
        OUTPUT "${FEW_ADDLIB_NAME}_cpu/${CU_FILE}.cxx"
        COMMENT
          "Copy ${CMAKE_CURRENT_SOURCE_DIR}/${CU_FILE} to ${CMAKE_CURRENT_BINARY_DIR}/${FEW_ADDLIB_NAME}_cpu/${CU_FILE}.cxx ."
        COMMAND
          ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/${CU_FILE}
          ${CMAKE_CURRENT_BINARY_DIR}/${FEW_ADDLIB_NAME}_cpu/${CU_FILE}.cxx
        DEPENDS "${CU_FILE}"
        VERBATIM)
      list(APPEND FEW_ADDLIB_CXX_SOURCES
           "${CMAKE_CURRENT_BINARY_DIR}/${FEW_ADDLIB_NAME}_cpu/${CU_FILE}.cxx")
    endforeach()

    few_add_cython_library(
      NAME "${FEW_ADDLIB_NAME}_cpu"
      PYX_SOURCES ${FEW_ADDLIB_PYX_SOURCES} CXX_SOURCES
                  ${FEW_ADDLIB_CXX_SOURCES}
      LINK ${FEW_ADDLIB_LINK}
      INCLUDE ${FEW_ADDLIB_INCLUDE}
      HEADERS ${FEW_ADDLIB_HEADERS} DESTINATION ${FEW_ADDLIB_DESTINATION})
  endif()
endfunction()
