# C++ Example : --------------------------------------------------------------------------------------------------------

find_path(JPEG_INCLUDE_DIR jpeglib.h PATHS /usr/include /usr/local/include)
find_library(JPEG_LIBRARY jpeg PATHS /lib /usr/lib /usr/lib/x86_64-linux-gnu)



add_executable(cpp_example ${CMAKE_CURRENT_SOURCE_DIR}/cpp_example.cpp)
add_executable(jpeg_loader_example ${CMAKE_CURRENT_SOURCE_DIR}/jpeg_loader_example.cpp)
target_link_libraries(cpp_example PUBLIC FastLanes)
target_include_directories(jpeg_loader_example PRIVATE ${JPEG_INCLUDE_DIR})
target_link_libraries(jpeg_loader_example PUBLIC FastLanes ${JPEG_LIBRARY})


add_executable(jpeg_loader_example_simple ${CMAKE_CURRENT_SOURCE_DIR}/jpeg_loader_example_simple.cpp)
target_include_directories(jpeg_loader_example_simple PRIVATE ${JPEG_INCLUDE_DIR})
target_link_libraries(jpeg_loader_example_simple PUBLIC FastLanes ${JPEG_LIBRARY})

if (WIN32 AND FLS_BUILD_SHARED_LIBS)
    add_custom_command(
            TARGET cpp_example
            POST_BUILD
            COMMAND ${CMAKE_COMMAND}
            -E copy_if_different
            $<TARGET_RUNTIME_DLLS:cpp_example>
            $<TARGET_FILE_DIR:cpp_example>
    )
endif ()
