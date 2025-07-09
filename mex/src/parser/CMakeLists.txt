matlab_add_mex(
	NAME parse_GMSH_output
	SRC 
		parse_GMSH_output.cpp
	LINK_TO
		Eigen3::Eigen
		MyFEM
	R2017b
)

set_target_properties(parse_GMSH_output PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO
	LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/../../build/compiled"
	RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/../../build/compiled"
)
get_target_property(INTERFACE_OUTPUT_DIR parse_GMSH_output LIBRARY_OUTPUT_DIRECTORY)

add_custom_command(TARGET parse_GMSH_output 
	POST_BUILD
		COMMAND ${CMAKE_COMMAND} -E copy ${INTERFACE_OUTPUT_DIR}/parse_GMSH_output.${Matlab_MEX_EXTENSION} ${PROJECT_SOURCE_DIR}/parse_GMSH_output.${Matlab_MEX_EXTENSION}
)