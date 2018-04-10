# Java

# Find Java
find_package( Java 1.8 COMPONENTS Development )

# Set the current process environment variable JAVA_HOME to ensure that only one version 
# of Java is used
get_filename_component(JAVA_BIN_DIR ${Java_JAVAC_EXECUTABLE} PATH)
get_filename_component(JAVA_HOME ${JAVA_BIN_DIR} PATH)
set( ENV{JAVA_HOME} ${JAVA_HOME} )

if( NOT Java_FOUND )
  message( "JAVA JDK NOT found - skipping Gpufit Java binding!" )
  return()
endif()

# Find JNI
find_package( JNI )

if( NOT JNI_FOUND )
  message( "JAVA JNI NOT found - skipping Gpufit Java binding!" )
  return()
endif()

# Try running Java
exec_program( java ARGS -version RETURN_VALUE java_return_value )

if( java_return_value EQUAL -1 )
    message( "Java NOT found by environment - skipping Gpufit Java binding!" )
    return()
endif()

# Adapter library

add_subdirectory( adapter )

# Java package

set( build_directory "${CMAKE_BINARY_DIR}/${CMAKE_CFG_INTDIR}/java" )
set( java_directory "${CMAKE_CURRENT_SOURCE_DIR}/gpufit" )
file( GLOB java_jar "${java_directory}/build/libs/Gpufit*.jar" LIST_DIRECTORIES false )
set( binary_files
  "$<TARGET_FILE:Gpufit>"
  "$<TARGET_FILE:GpufitJNI>"
  "${java_jar}"
  "${CMAKE_CURRENT_SOURCE_DIR}/README.txt"
)

add_custom_target( JAVA_PACKAGE ALL
  COMMAND ${CMAKE_COMMAND} -E
    remove_directory ${build_directory}
  COMMAND ${CMAKE_COMMAND} -E
    make_directory ${build_directory}
  COMMAND ${CMAKE_COMMAND} -E
    chdir ${java_directory} gradlew.bat build "-Dorg.gradle.java.home=${JAVA_HOME}"
  COMMAND ${CMAKE_COMMAND} -E
    copy_if_different ${binary_files} ${build_directory}
  COMMAND ${CMAKE_COMMAND} -E
    copy_directory "${java_directory}/src/test/" ${build_directory}
)
set_property( TARGET JAVA_PACKAGE PROPERTY FOLDER CMakePredefinedTargets )
add_dependencies( JAVA_PACKAGE Gpufit GpufitJNI )