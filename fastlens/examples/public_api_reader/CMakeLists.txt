add_executable(fastlens_public_api_reader
        public_api_reader.cpp
)
target_compile_features(fastlens_public_api_reader PRIVATE cxx_std_20)
target_link_libraries(fastlens_public_api_reader PRIVATE FastLens::engine)
