
function( add_example module name )
	add_executable( ${name} ${name}.cpp )
	target_link_libraries( ${name} ${module} )
	set_property( TARGET ${name}
		PROPERTY RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}" )
	set_property( TARGET ${name} PROPERTY FOLDER GpufitExamples )
endfunction()

function( add_cuda_example module name )
	cuda_add_executable( ${name} ${name}.cu )
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
