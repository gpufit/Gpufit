
# Python

# Python package

set( build_directory "${CMAKE_BINARY_DIR}/${CMAKE_CFG_INTDIR}/pyGpufit" )
set( setup_files
  "${CMAKE_CURRENT_SOURCE_DIR}/README.txt"
  "${CMAKE_CURRENT_SOURCE_DIR}/setup.py"
  "${CMAKE_CURRENT_SOURCE_DIR}/setup.cfg"
)
set( module_directory "${build_directory}/pygpufit" )
set( module_files
  "${CMAKE_CURRENT_SOURCE_DIR}/pygpufit/__init__.py"
  "${CMAKE_CURRENT_SOURCE_DIR}/pygpufit/gpufit.py"
  "${CMAKE_CURRENT_SOURCE_DIR}/pygpufit/version.py"
)

if ( USE_CUBLAS AND DEFINED CUBLAS_DLL )
	set( binary $<TARGET_FILE:Gpufit> ${CUBLAS_DLL} )
else()
	set( binary $<TARGET_FILE:Gpufit> )
endif()

add_custom_target( PYTHON_PACKAGE
  COMMAND ${CMAKE_COMMAND} -E
    remove_directory ${build_directory}
  COMMAND ${CMAKE_COMMAND} -E
    make_directory ${build_directory}
  COMMAND ${CMAKE_COMMAND} -E
    copy_if_different ${setup_files} ${build_directory}
  COMMAND ${CMAKE_COMMAND} -E
    make_directory ${module_directory}
  COMMAND ${CMAKE_COMMAND} -E
    copy_if_different ${module_files} ${module_directory}
  COMMAND ${CMAKE_COMMAND} -E
    copy_if_different ${binary} ${module_directory}
)

set_property( TARGET PYTHON_PACKAGE PROPERTY FOLDER CMakePredefinedTargets )
add_dependencies( PYTHON_PACKAGE Gpufit )

if( NOT PYTHONINTERP_FOUND )
  message( STATUS "Python NOT found - skipping creation of Python wheel!" )
  return()
endif()

# Python wheel (output name is incorrect, requires plattform tag, see packaging)

add_custom_target( PYTHON_WHEEL ALL
  COMMAND ${CMAKE_COMMAND} -E
    chdir ${build_directory} "${PYTHON_EXECUTABLE}" setup.py clean --all
  COMMAND ${CMAKE_COMMAND} -E
    chdir ${build_directory} "${PYTHON_EXECUTABLE}" setup.py bdist_wheel
  COMMENT "Preparing Python Wheel"
)
set_property( TARGET PYTHON_WHEEL PROPERTY FOLDER CMakePredefinedTargets )
add_dependencies( PYTHON_WHEEL PYTHON_PACKAGE )

# add launcher to Python package
