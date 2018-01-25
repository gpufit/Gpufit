# Java

find_package( JNI )

if( NOT JNI_FOUND )
  message( STATUS "JAVA JNI NOT found - skipping Gpufit Java binding!" )
  return()
endif()

# Adapter library

add_subdirectory( adapter )

# Java package

set( build_directory "${CMAKE_BINARY_DIR}/${CMAKE_CFG_INTDIR}/Java-Gpufit" )
set( binary $<TARGET_FILE:Gpufit> )

add_custom_target( JAVA_PACKAGE
  COMMAND ${CMAKE_COMMAND} -E
    remove_directory ${build_directory}
  COMMAND ${CMAKE_COMMAND} -E
    make_directory ${build_directory}
  COMMAND ${CMAKE_COMMAND} -E
    copy_if_different ${binary} ${build_directory}
)
set_property( TARGET JAVA_PACKAGE PROPERTY FOLDER CMakePredefinedTargets )
add_dependencies( JAVA_PACKAGE Gpufit )