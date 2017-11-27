# Levenberg Marquardt curve fitting in CUDA 
# https://github.com/gpufit/Gpufit
# see also CMake configuration in /docs/installation.rst

# CMake

cmake_minimum_required( VERSION 3.7 )
set_property( GLOBAL PROPERTY USE_FOLDERS ON )

if( NOT PROJECT_NAME )
  project( Gpufit VERSION 1.0.2 )
  include( CTest )
endif()

if( NOT CMAKE_CXX_STANDARD )
	set( CMAKE_CXX_STANDARD 14 )
endif()

if( MSVC ) # link runtime statically
  foreach( type ${CMAKE_CONFIGURATION_TYPES} ${CMAKE_BUILD_TYPE} )
    string( TOUPPER ${type} TYPE )
    foreach( flags CMAKE_C_FLAGS_${TYPE} CMAKE_CXX_FLAGS_${TYPE} )
      get_property( help CACHE ${flags} PROPERTY HELPSTRING )
      string( REPLACE "/MD" "/MT" ${flags} "${${flags}}" )
      set( ${flags} "${${flags}}" CACHE STRING "${help}" FORCE )
    endforeach()
  endforeach()
endif()

function( add_launcher target executable arguments working_directory )
  if( MSVC12 OR MSVC14 )
    file( WRITE ${CMAKE_CURRENT_BINARY_DIR}/${target}.vcxproj.user
"<?xml version=\"1.0\" encoding=\"utf-8\"?>\n"
"<Project ToolsVersion=\"14.0\" xmlns=\"http://schemas.microsoft.com/developer/msbuild/2003\">\n"
"  <PropertyGroup>\n"
"    <LocalDebuggerCommand>${executable}</LocalDebuggerCommand>\n"
"    <LocalDebuggerCommandArguments>${arguments}</LocalDebuggerCommandArguments>\n"
"    <LocalDebuggerWorkingDirectory>${working_directory}</LocalDebuggerWorkingDirectory>\n"
"  </PropertyGroup>\n"
"</Project>\n"
    )
  endif()
endfunction()

# Boost

find_package( Boost 1.58.0 )
if( Boost_FOUND )
  function( add_boost_test modules name )
    string( REPLACE ";" "_" prefix "${modules}" )
    set( target ${prefix}_Test_${name} )
    add_executable( ${target} ${name}.cpp
      ${PROJECT_SOURCE_DIR}/tests/utils.h
      ${PROJECT_SOURCE_DIR}/tests/utils.cpp
    )
    target_include_directories( ${target} PRIVATE ${PROJECT_SOURCE_DIR} )
    target_link_libraries( ${target} ${modules} Boost::boost )
    set_property( TARGET ${target}
      PROPERTY RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}" )
    set_property( TARGET ${target} PROPERTY FOLDER Tests )

    add_test( NAME ${target}
      COMMAND ${target} --build_info --log_level=all --report_level=detailed )
  endfunction()
else()
  set( BUILD_TESTING OFF )
  message( WARNING "Boost NOT found - skipping tests! (set BOOST_ROOT manually)" )
endif()

# MATLAB

find_package( Matlab )
if( Matlab_FOUND )
  find_program( Matlab_EXECUTABLE matlab
    PATHS "${Matlab_ROOT_DIR}/bin" PATH_SUFFIXES win32 win64 NO_DEFAULT_PATH )
  function( add_matlab_launcher target )
    set( paths "${CMAKE_BINARY_DIR}/$(Configuration)" ${ARGN} )
    list( GET paths -1 working_directory )
    string( REPLACE ";" "','" paths "${paths}" )
    set( arguments "-r addpath('${paths}');addpath(genpath(pwd))" )
    add_launcher( ${target} "${Matlab_EXECUTABLE}" "${arguments}" "${working_directory}" )
  endfunction()
endif()

# Python

find_package( PythonInterp )
if( PYTHONINTERP_FOUND )
  function( add_python_launcher target )
    set( paths "${CMAKE_BINARY_DIR}/$(Configuration)" ${ARGN} )
    list( GET paths -1 working_directory )
    string( REPLACE ";" "')\nsys.path.append('" paths "${paths}" )
    set( arguments "-i -c \"import sys\nsys.path.append('${paths}')\"" )
    add_launcher( ${target} "${PYTHON_EXECUTABLE}" "${arguments}" "${working_directory}" )
  endfunction()
endif()

# Cpufit

add_subdirectory( Cpufit )

# Gpufit

add_subdirectory( Gpufit )

# Examples using Gpufit and Cpufit

add_subdirectory( examples )

# Launcher
#
# Uses the following variables:
#
#   Matlab_WORKING_DIRECTORY (Default: user home directory)
#   -- Working directory for MATLAB applications using Cpufit and Gpufit.
#   Python_WORKING_DIRECTORY (Default: user home directory)
#   -- Working directory for Python applications using Gpufit.

if( WIN32 )
	file( TO_CMAKE_PATH "$ENV{HOMEPATH}" home )
else()
	file( TO_CMAKE_PATH "$ENV{HOME}" home )
endif()

if( Matlab_FOUND )
  set( Matlab_WORKING_DIRECTORY "${home}" CACHE PATH "MATLAB working directory" )
  if( Matlab_WORKING_DIRECTORY )
    add_custom_target( RUN_MATLAB )
    set_property( TARGET RUN_MATLAB PROPERTY FOLDER CMakePredefinedTargets )
    add_dependencies( RUN_MATLAB CpufitMex GpufitMex )
    add_matlab_launcher( RUN_MATLAB
      "${CMAKE_SOURCE_DIR}/Cpufit/matlab"
      "${CMAKE_SOURCE_DIR}/Gpufit/matlab"
      "${Matlab_WORKING_DIRECTORY}"
    )
  endif()
endif()

if( PYTHONINTERP_FOUND )
  set( Python_WORKING_DIRECTORY "${home}" CACHE PATH "Python working directory" )
  if( Python_WORKING_DIRECTORY )
    add_custom_target( RUN_PYTHON )
    set_property( TARGET RUN_PYTHON PROPERTY FOLDER CMakePredefinedTargets )
    add_dependencies( RUN_PYTHON Gpufit )
    add_python_launcher( RUN_PYTHON
      "${CMAKE_SOURCE_DIR}/Gpufit/python"
      "${Python_WORKING_DIRECTORY}"
    )
  endif()
endif()

# Tests

if( BUILD_TESTING )
  add_subdirectory( tests )
endif()

# Package

#set( CPACK_PACKAGE_VERSION ${PROJECT_VERSION} )
#set( CPACK_GENERATOR ZIP )

#include( CPack )
