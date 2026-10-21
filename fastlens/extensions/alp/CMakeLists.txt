add_library(fastlens_alp_support STATIC
        alp_bindings.cu
)
set_target_properties(fastlens_alp_support PROPERTIES
        CUDA_RESOLVE_DEVICE_SYMBOLS ON)
target_include_directories(fastlens_alp_support PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_include_directories(fastlens_alp_support PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../../src
        ${CUDAToolkit_INCLUDE_DIRS})
target_link_libraries(fastlens_alp_support PUBLIC
        fastlens_engine)
apply_warnings(fastlens_alp_support)
