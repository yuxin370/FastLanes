add_library(jpeg_loader
        OBJECT
        #[[]]
        jpeg_loader.cpp)

if (FLS_ENABLE_IWYU)
    set_target_properties(jpeg_loader PROPERTIES CXX_INCLUDE_WHAT_YOU_USE ${iwyu_path})
endif ()

set(FASTLANES_OBJECT_FILES
        ${FASTLANES_OBJECT_FILES} $<TARGET_OBJECTS:jpeg_loader>
        PARENT_SCOPE)

target_link_libraries(jpeg_loader
        PUBLIC
        FastLanes::headers
)

target_include_directories(jpeg_loader
    PRIVATE
        ${GPUJPEG_INCLUDE_DIRS}     # 你定义的 GPUJPEG 头文件路径
)

target_compile_options(jpeg_loader PRIVATE
    -Wno-gnu-anonymous-struct
    -Wno-nested-anon-types
)