cmake_minimum_required(VERSION 3.22)
project(GalpCMakeConsumerExample LANGUAGES CXX CUDA)

find_package(Galp CONFIG REQUIRED)

add_executable(galp_public_api_reader ../public_api_reader/public_api_reader.cpp)
target_compile_features(galp_public_api_reader PRIVATE cxx_std_20)
target_link_libraries(galp_public_api_reader PRIVATE Galp::core)
