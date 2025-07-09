set(MEX_INTERFACES_TO_COMPILE 
    build_grad_hess_TLE2d
    build_grad_hess_stab_TLE2d
    build_grad_hess_TLE3d
    build_grad_hess_TLF2d
    build_grad_hess_TLF3d
    build_inclusions_2d
    build_inclusions_3d
    build_kinematic_averages
    build_mass_matrix_2d
    build_phi_ho_2d
    build_phi_ho_3d
    micromorphic_closed
    micromorphic_computational
    micromorphic_computational_now
    my_linsolve
    sample_material_law_TLE2d
    sample_material_law_TLE3d
    sample_material_law_TLF2d
    sample_material_law_TLF3d
    sample_UES_2d
    sample_UES_3d
    sample_UFP_2d
    sample_UFP_3d
    test_constitutive_law2d
    test_constitutive_law3d
)

foreach(INTERFACE_SRC IN LISTS MEX_INTERFACES_TO_COMPILE)

    matlab_add_mex(
        NAME ${INTERFACE_SRC}
        SRC 
            ${INTERFACE_SRC}.cpp
        LINK_TO
            Eigen3::Eigen
            MyFEM
        R2017b
    )

    set_target_properties(${INTERFACE_SRC} PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/../../build/compiled"
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/../../build/compiled"
        RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CMAKE_CURRENT_SOURCE_DIR}/../../build/compiled"
        RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_CURRENT_SOURCE_DIR}/../../build/compiled"
    )
    get_target_property(INTERFACE_OUTPUT_DIR ${INTERFACE_SRC} RUNTIME_OUTPUT_DIRECTORY)
    add_custom_command(TARGET ${INTERFACE_SRC} 
        POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy ${INTERFACE_OUTPUT_DIR}/${INTERFACE_SRC}.${Matlab_MEX_EXTENSION} ${PROJECT_SOURCE_DIR}/${INTERFACE_SRC}.${Matlab_MEX_EXTENSION}
    )

endforeach()
