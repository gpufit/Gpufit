
function( add_example module name )
	add_executable( ${name} ${name}.cpp )
	target_link_libraries( ${name} ${module} )
	set_property( TARGET ${name}
		PROPERTY RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}" )
	set_property( TARGET ${name} PROPERTY FOLDER GpufitExamples )
endfunction()

function( add_cuda_example module name )
	if( USE_HIP )
		# Compile the .cu device-API example as a HIP TU. The compat header lives
		# in the Gpufit module dir; force-include it and add the hip_compat shims.
		set( gpufit_src_dir ${CMAKE_CURRENT_SOURCE_DIR}/../../Gpufit )
		set_source_files_properties( ${name}.cu PROPERTIES LANGUAGE HIP )
		add_executable( ${name} ${name}.cu )
		# project(... LANGUAGES CXX HIP) in the top-level CMakeLists already honors an
		# explicit -DCMAKE_HIP_ARCHITECTURES, otherwise auto-detects the host GPU(s)
		# and errors on a no-GPU host, so no arch default is pinned here.
		set_target_properties( ${name} PROPERTIES HIP_ARCHITECTURES "${CMAKE_HIP_ARCHITECTURES}" )
		target_compile_definitions( ${name} PRIVATE USE_HIP )
		# clang-cl (MSVC frontend) ignores GNU-style -include; use /FI instead.
		target_compile_options( ${name} PRIVATE
			$<$<COMPILE_LANGUAGE:HIP>:$<IF:$<STREQUAL:${CMAKE_HIP_COMPILER_FRONTEND_VARIANT},MSVC>,/FI${gpufit_src_dir}/cuda_to_hip.h,-include${gpufit_src_dir}/cuda_to_hip.h>> )
		target_include_directories( ${name} PRIVATE ${gpufit_src_dir}/hip_compat )
	else()
		cuda_add_executable( ${name} ${name}.cu )
	endif()
	target_link_libraries( ${name} ${module} )
	set_property( TARGET ${name}
		PROPERTY RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}" )
	set_property( TARGET ${name} PROPERTY FOLDER GpufitExamples )
endfunction()

# Examples

add_example( Gpufit Simple_Example )
add_example( Gpufit Linear_Regression_Example )
add_example( Gpufit Gauss_Fit_2D_Example )

add_cuda_example( Gpufit CUDA_Interface_Example )
