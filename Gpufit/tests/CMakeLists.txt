
# Tests

add_boost_test( Gpufit Error_Handling )
add_boost_test( Gpufit Linear_Fit_1D )
add_boost_test( Gpufit Gauss_Fit_1D )
add_boost_test( Gpufit Gauss_Fit_2D )
add_boost_test( Gpufit Gauss_Fit_2D_Elliptic )
add_boost_test( Gpufit Gauss_Fit_2D_Rotated )
add_boost_test( Gpufit Cauchy_Fit_2D_Elliptic )
add_boost_test( Gpufit Fletcher_Powell_Helix_Fit )
add_boost_test( Gpufit Brown_Dennis_Fit )

if( UNIX )
  # Copy over run tests script
  file(COPY "${CMAKE_SOURCE_DIR}/Gpufit/tests/test_all.sh" 
    DESTINATION "${CMAKE_BINARY_DIR}" 
    FILE_PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE
  ) 
endif()
