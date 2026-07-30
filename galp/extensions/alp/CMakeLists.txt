add_library(galp_alp_support STATIC
        alp_bindings.cu
)
set_target_properties(galp_alp_support PROPERTIES
        CUDA_RESOLVE_DEVICE_SYMBOLS ON)
target_include_directories(galp_alp_support PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_include_directories(galp_alp_support PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../../src
        ${CUDAToolkit_INCLUDE_DIRS})
target_link_libraries(galp_alp_support PUBLIC
        galp_engine)
apply_warnings(galp_alp_support)
