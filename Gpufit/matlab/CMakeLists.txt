
# MATLAB Gpufit binding

find_package( Matlab COMPONENTS MX_LIBRARY )

if( NOT Matlab_FOUND )
  message( STATUS "Matlab and/or MX_Library NOT found - skipping Gpufit Matlab binding!" )
  return()
endif()

# MATLAB MEX FILE

set( Headers
  )

set( Sources
  mex/GpufitMex.cpp
  )

add_library( GpufitMex SHARED
  ${Headers}
  ${Sources}
  )

set_property( TARGET GpufitMex
  PROPERTY SUFFIX .${Matlab_MEX_EXTENSION} )

set_property( TARGET GpufitMex
  PROPERTY RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}" )

target_include_directories( GpufitMex PRIVATE ${Matlab_INCLUDE_DIRS} ${PROJECT_SOURCE_DIR} )

target_link_libraries( GpufitMex Gpufit ${Matlab_LIBRARIES} )

if( WIN32 )
  SET(CMAKE_SHARED_LINKER_FLAGS "/export:mexFunction")
endif()

add_matlab_launcher( GpufitMex "${CMAKE_CURRENT_SOURCE_DIR}" )

# MATLAB Gpufit PACKAGE

set( build_directory "${CMAKE_BINARY_DIR}/${CMAKE_CFG_INTDIR}/matlab" )
set( package_files
  "${CMAKE_CURRENT_SOURCE_DIR}/EstimatorID.m"
  "${CMAKE_CURRENT_SOURCE_DIR}/gpufit.m"
  "${CMAKE_CURRENT_SOURCE_DIR}/gpufit_version.m"
  "${CMAKE_CURRENT_SOURCE_DIR}/ModelID.m"
  "${CMAKE_CURRENT_SOURCE_DIR}/README.txt"
)
set( binary_gpufit $<TARGET_FILE:Gpufit> )
set( binary_mex $<TARGET_FILE:GpufitMex> )

add_custom_target( MATLAB_GPUFIT_PACKAGE
  COMMAND ${CMAKE_COMMAND} -E
    remove_directory ${build_directory}
  COMMAND ${CMAKE_COMMAND} -E
    make_directory ${build_directory}
  COMMAND ${CMAKE_COMMAND} -E
    copy_if_different ${package_files} ${build_directory}
  COMMAND ${CMAKE_COMMAND} -E
    copy_if_different ${binary_gpufit} ${build_directory}
  COMMAND ${CMAKE_COMMAND} -E
    copy_if_different ${binary_mex} ${build_directory}	
  COMMENT "Creating Gpufit Matlab package"
)
set_property( TARGET MATLAB_GPUFIT_PACKAGE PROPERTY FOLDER CMakePredefinedTargets )
add_dependencies( MATLAB_GPUFIT_PACKAGE Gpufit GpufitMex)

# add launcher
