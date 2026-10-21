add_executable(fastlens_jpeg_dct_device_batch
        jpeg_dct_device_batch.cpp
)
target_compile_features(fastlens_jpeg_dct_device_batch PRIVATE cxx_std_20)
target_link_libraries(fastlens_jpeg_dct_device_batch PRIVATE FastLens::jpeg_dct_device)
