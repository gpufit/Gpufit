
# CUDA
#
# Uses the following variables:
#
#   CUDA_ARCHITECTURES (Default All)
#   -- Argument passed to CUDA_SELECT_NVCC_ARCH_FLAGS(...)
#      resulting in code_generation_flags
#      (see http://cmake.org/cmake/help/v3.7/module/FindCUDA.html).
#      CUDA_ARCHITECTURES: Auto | Common | All | ARCH_AND_PTX ...
#      Auto: Detects local machine GPU architecture.
#      Common: Covers common subset of architectures.
#      All: Covers all known architectures.
#      ARCH_AND_PTX: NAME | NUM.NUM | NUM.NUM(NUM.NUM) | NUM.NUM+PTX
#      NAME: Fermi Kepler Maxwell Kepler+Tegra Kepler+Tesla Maxwell+Tegra Pascal
#      NUM: Any number.
#      Only those pairs are currently accepted by NVCC though:
#        2.0 2.1 3.0 3.2 3.5 3.7 5.0 5.2 5.3 6.0 6.2
#      Examples:
#        2.1(2.0) results in
#          -gencode;arch=compute_20,code=sm_21
#        Kepler+Tesla results in
#          -gencode;arch=compute_37,code=sm_37
#        6.2+PTX results in
#          -gencode;arch=compute_62,code=sm_62;-gencode;arch=compute_62,code=compute_62
#
#   CUDA_NVCC_FLAGS (Default ${code_generation_flags})
#   -- Additional NVCC command line arguments
#      (see http://cmake.org/cmake/help/v3.7/module/FindCUDA.html).
#      NOTE that multiple arguments must be semi-colon delimited
#      (e.g. --compiler-options;-Wall)
#
#   Multiple CUDA versions installed, specify which version to use
#      Set CUDA_BIN_PATH before running CMake or CUDA_TOOLKIT_ROOT_DIR after first configuration
#      to installation folder of desired CUDA version

find_package( CUDA 6.5 REQUIRED )

set( CUDA_ARCHITECTURES ${DEFAULT_CUDA_ARCH} CACHE STRING
  "Auto | Common | All | ... see CUDA_SELECT_NVCC_ARCH_FLAGS(...)" )

if( CUDA_ARCHITECTURES STREQUAL Auto )

  set( file ${PROJECT_BINARY_DIR}/detect_cuda_architectures.cpp )
  file( WRITE ${file} ""
    "#include <cuda_runtime.h>\n"
    "#include <cstdio>\n"
    "int main()\n"
    "{\n"
    "  int count = 0;\n"
    "  if (cudaSuccess != cudaGetDeviceCount(&count)) return -1;\n"
    "  if (count == 0) return -1;\n"
    "  for (int device = 0; device < count; ++device)\n"
    "  {\n"
    "    cudaDeviceProp prop;\n"
    "    if (cudaSuccess == cudaGetDeviceProperties(&prop, device))\n"
    "      std::printf(\"%d.%d \", prop.major, prop.minor);\n"
    "  }\n"
    "  return 0;\n"
    "}\n"
  )
  try_run( run_result compile_result ${PROJECT_BINARY_DIR} ${file}
    CMAKE_FLAGS "-DINCLUDE_DIRECTORIES=${CUDA_INCLUDE_DIRS}"
    LINK_LIBRARIES ${CUDA_LIBRARIES}
    RUN_OUTPUT_VARIABLE architectures
  )
  if( run_result EQUAL 0 )
    string( REPLACE "2.1" "2.1(2.0)" architectures "${architectures}" )
    if( CUDA_VERSION VERSION_LESS "7.0" )
      string( REGEX REPLACE "3\\.[27]|5\\.[23]|6\\.[01]" "5.2+PTX" architectures "${architectures}" )
    elseif( CUDA_VERSION VERSION_LESS "8.0" )
      string( REGEX REPLACE "5\\.3|6\\.[01]" "5.3+PTX" architectures "${architectures}" )
    endif()
    set( CUDA_ARCHITECTURES "${architectures}" )
  endif()
  
elseif( CUDA_ARCHITECTURES STREQUAL All )

# All does not include the latest PTX!
  set( CUDA_ARCHITECTURES "" )

  if( CUDA_VERSION VERSION_LESS "12.0" )
    list( INSERT CUDA_ARCHITECTURES 0 "3.5" "5.0" "5.2" )
  endif()
  if( CUDA_VERSION VERSION_LESS "11.0" )
    list( INSERT CUDA_ARCHITECTURES 0 "3.0" "3.2")
  endif()
  if( CUDA_VERSION VERSION_LESS "9.0" )
    list( INSERT CUDA_ARCHITECTURES 0 "2.0" "2.1(2.0)" )
  endif()
  
  if( CUDA_VERSION VERSION_GREATER "6.5" )
      list( APPEND CUDA_ARCHITECTURES "5.3" )
  endif()
  
  if( CUDA_VERSION VERSION_GREATER "7.5" )
    list( APPEND CUDA_ARCHITECTURES "6.0" "6.1" )
  endif()
  
  if( CUDA_VERSION VERSION_GREATER "8.0" )
    list( APPEND CUDA_ARCHITECTURES "7.0" "7.2")
  endif()
  
  if( CUDA_VERSION VERSION_GREATER "9.2" )
    list( APPEND CUDA_ARCHITECTURES "7.5" )
  endif()
  
  if( CUDA_VERSION VERSION_GREATER "10.2" )
    list( APPEND CUDA_ARCHITECTURES "8.0" )
  endif()
  
  if( CUDA_VERSION VERSION_GREATER "11.0" )
    list( APPEND CUDA_ARCHITECTURES "8.6" )
  endif()
  
  string( APPEND CUDA_ARCHITECTURES "+PTX" )
  
endif()

message( STATUS "CUDA_ARCHITECTURES=${CUDA_ARCHITECTURES}" )

CUDA_SELECT_NVCC_ARCH_FLAGS( code_generation_flags "${CUDA_ARCHITECTURES}" )
list( APPEND CUDA_NVCC_FLAGS ${code_generation_flags} )

message( STATUS "CUDA_NVCC_FLAGS=${code_generation_flags}" )

if( NOT WIN32 )
	list( APPEND CUDA_NVCC_FLAGS --std=c++11)
endif()

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

cuda_add_library( Gpufit SHARED
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

# USE_CUBLAS
if( CMAKE_SIZEOF_VOID_P EQUAL 8 AND CUDA_VERSION VERSION_GREATER "6.5")
	set( USE_CUBLAS ${DEFAULT_USE_CUBLAS} CACHE BOOL "ON | OFF")
	if( USE_CUBLAS )
        if ( WIN32 )
	    if( CMAKE_SIZEOF_VOID_P EQUAL 8 AND CUDA_VERSION VERSION_GREATER_EQUAL "10")
                set( CUBLAS_DLL "${CUDA_TOOLKIT_ROOT_DIR}/bin/cublas64_${CUDA_VERSION_MAJOR}.dll" )
	    else()
		set( CUBLAS_DLL "${CUDA_TOOLKIT_ROOT_DIR}/bin/cublas64_${CUDA_VERSION_MAJOR}${CUDA_VERSION_MINOR}.dll" )
	    endif()
            add_custom_command( TARGET Gpufit POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E
                    copy_if_different ${CUBLAS_DLL} $<TARGET_FILE_DIR:Gpufit> )
            target_link_libraries( Gpufit ${CUDA_CUBLAS_LIBRARIES} )
        else()
            set( STATIC_CUBLAS ON CACHE BOOL "ON | OFF")
            if ( STATIC_CUBLAS )
                find_cuda_helper_libs(cublas_static)
                find_cuda_helper_libs(cublasLt_static)
                find_cuda_helper_libs(culibos)

                set( CUDA_CUBLAS_LIBRARIES 
                    ${CUDA_cublas_static_LIBRARY}
                    ${CUDA_cublasLt_static_LIBRARY}
                    ${CUDA_cudart_static_LIBRARY}
                    ${CUDA_culibos_LIBRARY}
                    dl
                    pthread
                    rt )
                target_link_libraries( Gpufit ${CUDA_CUBLAS_LIBRARIES} )
            else ()
                # Dynamic Link can be done with a single function call
                CUDA_ADD_CUBLAS_TO_TARGET( Gpufit )
            endif ()
        endif() 
		add_definitions( -DUSE_CUBLAS )
	endif()
elseif( CMAKE_SIZEOF_VOID_P EQUAL 4 )
	message( STATUS "CUBLAS: 32-bit architecture detected; USE_CUBLAS flag ignored." )
elseif( CUDA_VERSION VERSION_LESS "7.0" )
	message( STATUS "CUBLAS: CUDA Version < 7.0 detected; USE_CUBLAS flag ignored." )
endif()

# Select Gpufit models, because the GPU resources can support only a certain few at once.
# USE_GAUSSIAN_2D
option(GPUFIT_USE_GAUSSIAN2D "Use the Gaussian 2D model, recommended for GPU vs CPU fitting comparison." ON)
if( GPUFIT_USE_GAUSSIAN2D )
	add_definitions( -DUSE_GAUSS2D )
endif()

# USE_BASE_MODELS
option( GPUFIT_USE_BASE_MODELS "Use the original Gpufit models: Linear 1D, Gaussian 1D, Gaussian 2D Rotated/Elliptic, Cauchy 2D Elliptic, Fletcher-Powell Helix, Brown-Dennis, plus an added exponential model." )
if ( GPUFIT_USE_BASE_MODELS )
	add_definitions( -DUSE_BASE_MODELS )
endif()

# Use added models (improve descriptions and defs)
option( GPUFIT_USE_DIXON_2 "Use the Dixon 2 parameter model. (Extremely high resource usage)" )
option( GPUFIT_USE_DIXON_3 "Use the Dixon 3 parameter model. (Extremely high resource usage)" )
option( GPUFIT_USE_DIXON_4 "Use the Dixon 4 parameter model. (Extremely high resource usage)" )
option( GPUFIT_USE_PATLAK "Use the Patlak model to fit." )
option( GPUFIT_USE_TOFTS "Use the Tofts model to fit." )
option( GPUFIT_USE_TOFTS_EXTENDED "Use the extended Tofts model to fit." )
option( GPUFIT_USE_TISSUE_UPTAKE "Use the tissue uptake model to fit." )
option( GPUFIT_USE_2CXM "Use the two-compartment exchange model to fit." )

if( GPUFIT_USE_DIXON_2 )
	add_definitions( -DUSE_DIXON_2 )
endif()

if( GPUFIT_USE_DIXON_3 )
	add_definitions( -DUSE_DIXON_3 )
endif()

if( GPUFIT_USE_DIXON_4 )
	add_definitions( -DUSE_DIXON_4 )
endif()

if ( GPUFIT_USE_PATLAK )
	add_definitions( -DUSE_PATLAK )
endif()

if ( GPUFIT_USE_TOFTS )
	add_definitions( -DUSE_TOFTS )
endif()

if ( GPUFIT_USE_TOFTS_EXTENDED )
	add_definitions( -DUSE_TOFTS_EXTENDED )
endif()

if ( GPUFIT_USE_TISSUE_UPTAKE )
	add_definitions( -DUSE_TISSUE_UPTAKE )
endif()

if ( GPUFIT_USE_2CXM )
	add_definitions( -DUSE_2CXM )
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

