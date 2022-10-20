
# CUDA
#
# Uses the CMAKE standard CUDA tools

cmake_minimum_required(VERSION 3.18)

if(NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()


message( STATUS "CMAKE_CUDA_COMPILER_VERSION=${CMAKE_CUDA_COMPILER_VERSION}")


# All does not include the latest PTX!
  set( CMAKE_CUDA_ARCHITECTURES_D "" )

  if( CMAKE_CUDA_COMPILER_VERSION VERSION_LESS "12.0" )
    list( INSERT CMAKE_CUDA_ARCHITECTURES_D 0 35 50 52 )
  endif()
  if( CMAKE_CUDA_COMPILER_VERSION VERSION_LESS "11.0" )
    list( INSERT CMAKE_CUDA_ARCHITECTURES_D 0 30 32)
  endif()
  if( CMAKE_CUDA_COMPILER_VERSION VERSION_LESS "9.0" )
    list( INSERT CMAKE_CUDA_ARCHITECTURES_D 0 20 21 )
  endif()
 
 
  if( CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER "6.5" )
      list( APPEND CMAKE_CUDA_ARCHITECTURES_D 53 )
  endif()
  
  if( CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER "7.5" )
    list( APPEND CMAKE_CUDA_ARCHITECTURES_D 60 61 )
  endif()
  
  if( CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER "8.0" )
    list( APPEND CMAKE_CUDA_ARCHITECTURES_D 70 72)
  endif()
  
  if( CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER "9.2" )
    list( APPEND CMAKE_CUDA_ARCHITECTURES_D 75 )
  endif()
  
  if( CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER "10.2" )
    list( APPEND CMAKE_CUDA_ARCHITECTURES_D 80 )
  endif()
  
  if( CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER "11.0" )
    list( APPEND CMAKE_CUDA_ARCHITECTURES_D 86 )
  endif()
#  string( APPEND CMAKE_CUDA_ARCHITECTURES "+PTX" ) 
  set( CMAKE_CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES_D}" CACHE STRING "List of architectures to compile, default is all")


# Gpufit
set( GpuHeaders
	gpufit.h
	constants.h
	definitions.h
	info.h
	lm_fit.h
	interface.h
)

set( GpuSources
	gpufit.cpp
	info.cpp
	lm_fit.cpp
	lm_fit_cuda.cpp
	interface.cpp
	Gpufit.def
)

set( GpuCudaHeaders
	cuda_kernels.cuh
	gpu_data.cuh
	cuda_gaussjordan.cuh
)

set( GpuCudaSources
	lm_fit_cuda.cu
	cuda_kernels.cu
	info.cu
	gpu_data.cu
	cuda_gaussjordan.cu
)

file( GLOB GpuCudaModels
	RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
	${CMAKE_CURRENT_SOURCE_DIR}/models/*.cuh
)

file( GLOB GpuCudaEstimators
	RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
	${CMAKE_CURRENT_SOURCE_DIR}/estimators/*.cuh
)

source_group("CUDA Header Files" FILES ${GpuCudaHeaders})
source_group("CUDA Source Files" FILES ${GpuCudaSources})
source_group("CUDA Model Files" FILES ${GpuCudaModels})
source_group("CUDA Estimator Files" FILES ${GpuCudaEstimators})

add_library( Gpufit SHARED
	${GpuHeaders}
	${GpuSources}
	${GpuCudaHeaders}
	${GpuCudaSources}
	${GpuCudaModels}
	${GpuCudaEstimators}
)

set_target_properties( Gpufit
	PROPERTIES
		RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}"
		CXX_VISIBILITY_PRESET hidden
)

target_include_directories( Gpufit SYSTEM PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

# USE_CUBLAS    
if( CMAKE_SIZEOF_VOID_P EQUAL 8 AND CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER "6.5")
	set( USE_CUBLAS ${DEFAULT_USE_CUBLAS} CACHE BOOL "ON | OFF")
	if( USE_CUBLAS )
            find_package(CUDAToolkit REQUIRED)
            set( STATIC_CUBLAS ON CACHE BOOL "ON | OFF")
            if ( STATIC_CUBLAS )
                target_link_libraries( Gpufit CUDA::cublas_static CUDA::cublasLt_static)
            else ()
                target_link_libraries( Gpufit CUDA::cublas CUDA::cublasLt)
            endif ()
            add_definitions( -DUSE_CUBLAS )
        endif() 
elseif( CMAKE_SIZEOF_VOID_P EQUAL 4 )
	message( STATUS "CUBLAS: 32-bit architecture detected; USE_CUBLAS flag ignored." )
elseif( CUDA_VERSION VERSION_LESS "7.0" )
	message( STATUS "CUBLAS: CUDA Version < 7.0 detected; USE_CUBLAS flag ignored." )
endif()

#install( TARGETS Gpufit RUNTIME DESTINATION bin )

# Examples using only Gpufit

add_subdirectory( ../examples/c++ "${CMAKE_CURRENT_BINARY_DIR}/examples" )

# Tests

if( BUILD_TESTING )
	add_subdirectory( tests )
endif()

# Bindings

add_subdirectory( matlab )
add_subdirectory( python )
add_subdirectory( java )

