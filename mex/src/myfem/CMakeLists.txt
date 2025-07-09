add_library(MyFEM STATIC
	include/myfem.h 
	src/myfem.cpp 
)
target_include_directories(MyFEM
	PUBLIC
        ${Matlab_INCLUDE_DIRS}
        include
)
target_link_libraries(MyFEM 
    PUBLIC
        Eigen3::Eigen
        ${Matlab_MEX_LIBRARY}
) 
set_target_properties(MyFEM PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)