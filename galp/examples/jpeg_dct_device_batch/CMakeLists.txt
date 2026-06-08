add_executable(galp_jpeg_dct_device_batch
        jpeg_dct_device_batch.cpp
)
target_compile_features(galp_jpeg_dct_device_batch PRIVATE cxx_std_20)
target_link_libraries(galp_jpeg_dct_device_batch PRIVATE Galp::core)
