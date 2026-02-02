
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


# ---- External dependency: splines (Gpuspline) ----
set(SPLINES_DLL_DIR "" CACHE PATH "Path to directory containing splines.dll")


# Examples

add_example( Gpufit Simple_Example )
add_example( Gpufit Linear_Regression_Example )
add_example( Gpufit Gauss_Fit_2D_Example )
add_example( Gpufit Natural_BSpline_1D_Example )
add_example( Gpufit Natural_BSpline_ND_Example_2D_Gauss )

add_cuda_example( Gpufit CUDA_Interface_Example )


if (WIN32 AND NOT SPLINES_DLL_DIR STREQUAL "")
    add_custom_command(TARGET Natural_BSpline_1D_Example POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${SPLINES_DLL_DIR}/splines.dll"
            $<TARGET_FILE_DIR:Natural_BSpline_1D_Example>
    )
endif()

