
# Python

# Python package

set( build_directory "${CMAKE_BINARY_DIR}/${CMAKE_CFG_INTDIR}/pyCpufit" )
set( setup_files
  "${CMAKE_CURRENT_SOURCE_DIR}/README.txt"
  "${CMAKE_CURRENT_SOURCE_DIR}/setup.py"
  "${CMAKE_CURRENT_SOURCE_DIR}/setup.cfg"
)
set( module_directory "${build_directory}/pycpufit" )
set( module_files
  "${CMAKE_CURRENT_SOURCE_DIR}/pycpufit/__init__.py"
  "${CMAKE_CURRENT_SOURCE_DIR}/pycpufit/cpufit.py"
  "${CMAKE_CURRENT_SOURCE_DIR}/pycpufit/version.py"
)


set( binary $<TARGET_FILE:Cpufit> )


add_custom_target( PYTHON_PACKAGE_CPUFIT
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

set_property( TARGET PYTHON_PACKAGE_CPUFIT PROPERTY FOLDER CMakePredefinedTargets )
add_dependencies( PYTHON_PACKAGE_CPUFIT Cpufit )

if( NOT PYTHONINTERP_FOUND )
  message( STATUS "Python NOT found - skipping creation of Python wheel!" )
  return()
endif()

# Python wheel (output name is incorrect, requires plattform tag, see packaging)

add_custom_target( PYTHON_WHEEL_CPUFIT ALL
  COMMAND ${CMAKE_COMMAND} -E
    chdir ${build_directory} "${PYTHON_EXECUTABLE}" setup.py clean --all
  COMMAND ${CMAKE_COMMAND} -E
    chdir ${build_directory} "${PYTHON_EXECUTABLE}" setup.py bdist_wheel
  COMMENT "Preparing Python Wheel"
)
set_property( TARGET PYTHON_WHEEL_CPUFIT PROPERTY FOLDER CMakePredefinedTargets )
add_dependencies( PYTHON_WHEEL_CPUFIT PYTHON_PACKAGE_CPUFIT )

# add launcher to Python package
