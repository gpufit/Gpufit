
# Cpufit

set( CpuHeaders
	cpufit.h
	../Gpufit/constants.h
	info.h
	lm_fit.h
	interface.h
)

set( CpuSources
	cpufit.cpp
	info.cpp
	lm_fit.cpp
	lm_fit_cpp.cpp
	interface.cpp
	Cpufit.def
)

add_library( Cpufit SHARED
	${CpuHeaders}
	${CpuSources}
)
set_property( TARGET Cpufit
	PROPERTY RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}" )

#install( TARGETS Cpufit RUNTIME DESTINATION bin )

add_subdirectory( matlab )
