# Java Gpufit Adapter

set( Headers
  com_github_gpufit_Gpufit.h
)

set( Sources
  com_github_gpufit_Gpufit.cpp
)

add_library( GpufitJNI SHARED
  ${Headers}
  ${Sources}
)

set_target_properties( GpufitJNI
  PROPERTIES
		RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}"
)

target_include_directories( GpufitJNI PRIVATE ${PROJECT_SOURCE_DIR} ${JNI_INCLUDE_DIRS} )

target_link_libraries( GpufitJNI Gpufit ${JNI_LIBRARIES} )