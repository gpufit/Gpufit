
# Applications

if( USE_CUBLAS )
    add_definitions( -DUSE_CUBLAS )
endif()

function( add_example modules name )
  set( target ${name} )
  add_executable( ${target} ${name}.cpp
    ${PROJECT_SOURCE_DIR}/tests/utils.h
    ${PROJECT_SOURCE_DIR}/tests/utils.cpp
  )  
  target_include_directories( ${target} PRIVATE ${PROJECT_SOURCE_DIR} )
  target_link_libraries( ${target} ${modules} )
  set_property( TARGET ${target}
    PROPERTY RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}" )
  set_property( TARGET ${target} PROPERTY FOLDER GpufitCpufitExamples )
#  install( TARGETS ${target} RUNTIME DESTINATION bin )
endfunction()

add_example( "Cpufit;Gpufit" Gpufit_Cpufit_Performance_Comparison )

add_example( "Cpufit;Gpufit" Gpufit_Cpufit_Nvidia_Profiler_Test )
