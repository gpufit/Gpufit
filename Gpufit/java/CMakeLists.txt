# Java

find_package( JNI )

if( NOT JNI_FOUND )
  message( STATUS "JAVA JNI NOT found - skipping Gpufit Java binding!" )
  return()
endif()

# Adapter library

add_subdirectory( adapter )

# Java package

set( build_directory "${CMAKE_BINARY_DIR}/${CMAKE_CFG_INTDIR}/java" )
set( java_directory "${CMAKE_CURRENT_SOURCE_DIR}/gpufit" )
set( binary_files
  "$<TARGET_FILE:Gpufit>"
  "$<TARGET_FILE:GpufitJNI>"
  "${java_directory}/build/libs/Gpufit-1.1.0.jar"
  "${CMAKE_CURRENT_SOURCE_DIR}/README.txt"
)

add_custom_target( JAVA_PACKAGE ALL
  COMMAND ${CMAKE_COMMAND} -E
    remove_directory ${build_directory}
  COMMAND ${CMAKE_COMMAND} -E
    make_directory ${build_directory}
  COMMAND ${CMAKE_COMMAND} -E
    chdir ${java_directory} gradlew.bat build
  COMMAND ${CMAKE_COMMAND} -E
    copy_if_different ${binary_files} ${build_directory}
  COMMAND ${CMAKE_COMMAND} -E
    copy_directory "${java_directory}/src/test/java/com/github/gpufit" ${build_directory}
)
set_property( TARGET JAVA_PACKAGE PROPERTY FOLDER CMakePredefinedTargets )
add_dependencies( JAVA_PACKAGE Gpufit GpufitJNI )