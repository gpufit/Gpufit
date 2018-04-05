# Java

function( largest_common_prefix a b prefix )

# minimum of lengths of both strings
string( LENGTH ${a} len_a )
string( LENGTH ${a} len_b )

if( ${len_a} LESS ${len_b} )
    set( len ${len_a} )
else()
    set( len ${len_b} )
endif()

# iterate over the length
foreach( end RANGE 1 ${len} )
    # get substrings
    string( SUBSTRING ${a} 0 ${end} sub_a )
    string( SUBSTRING ${b} 0 ${end} sub_b )

    # if equal store, otherwise break
    if ( ${sub_a} STREQUAL ${sub_b} )
        set( ${prefix} ${sub_a} PARENT_SCOPE )
    else()
        break()
    endif()
endforeach()

endfunction()

find_package( Java 1.8 COMPONENTS Development )

if( NOT Java_FOUND )
  message( STATUS "JAVA JDK NOT found - skipping Gpufit Java binding!" )
  return()
endif()

find_package( JNI )

if( NOT JNI_FOUND )
  message( STATUS "JAVA JNI NOT found - skipping Gpufit Java binding!" )
  return()
endif()

# that may not work on linux but is there a better way to get to the JDK home?
largest_common_prefix( ${Java_JAVAC_EXECUTABLE} ${JAVA_JVM_LIBRARY} java_jdk_home )

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

add_custom_target( JAVA_PACKAGE
  COMMAND ${CMAKE_COMMAND} -E
    remove_directory ${build_directory}
  COMMAND ${CMAKE_COMMAND} -E
    make_directory ${build_directory}
  COMMAND ${CMAKE_COMMAND} -E
    chdir ${java_directory} gradlew.bat build "-Dorg.gradle.java.home=${java_jdk_home}"
  COMMAND ${CMAKE_COMMAND} -E
    copy_if_different ${binary_files} ${build_directory}
  COMMAND ${CMAKE_COMMAND} -E
    copy_directory "${java_directory}/src/test/" ${build_directory}
)
set_property( TARGET JAVA_PACKAGE PROPERTY FOLDER CMakePredefinedTargets )
add_dependencies( JAVA_PACKAGE Gpufit GpufitJNI )