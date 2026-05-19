set(GALP_GENERATED_INCLUDE_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated")
configure_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/../cmake/config.hpp.in
        ${GALP_GENERATED_INCLUDE_DIR}/galp/config.hpp
        @ONLY
)

set(GALP_CORE_SOURCES
        api/table.cu
        core/enums.cu
        core/expression.cu
        cuda/memory/device_arena.cu
        engine/operators/column.cu
        engine/operators/rowgroup.cu
        engine/table/table.cu
        io/csv_writer.cu
        engine/materialization/metadata.cu
        engine/materialization/pinned_d2h.cu
        engine/table/pipeline_state.cu
        engine/workset/streams.cu
        engine/workset/upload.cu
        format/reader.cu
        format/rowgroup_io.cu
        format/schema_plan.cu
        engine/materialization/zero_copy_materializer.cu
        jpeg/jpeg_dct_order.cpp
)
if (GALP_WITH_JPEG_DCT)
        list(APPEND GALP_CORE_SOURCES
                jpeg/jpeg_dct.cpp
        )
endif ()
add_library(galp_core STATIC ${GALP_CORE_SOURCES})
add_library(Galp::core ALIAS galp_core)
set_target_properties(galp_core PROPERTIES
        EXPORT_NAME core
        CUDA_RESOLVE_DEVICE_SYMBOLS ON)
target_include_directories(galp_core PUBLIC
        $<BUILD_INTERFACE:${GALP_GENERATED_INCLUDE_DIR}>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_include_directories(galp_core PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CUDAToolkit_INCLUDE_DIRS})
target_link_libraries(galp_core PUBLIC
        CUDA::cudart
        Thrust::Thrust
        $<BUILD_INTERFACE:FastLanes::core>
        $<INSTALL_INTERFACE:FLS::FastLanes>)
if (GALP_WITH_JPEG_DCT)
        target_link_libraries(galp_core PRIVATE JPEG::JPEG)
        target_compile_definitions(galp_core PRIVATE GALP_WITH_JPEG_DCT=1)
endif ()
apply_warnings(galp_core)

if (GALP_ENABLE_INSTALL)
        install(
                TARGETS galp_core
                EXPORT GalpTargets
                ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
                LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
                RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
                INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
        )

        install(
                DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../include/
                DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
                FILES_MATCHING
                PATTERN "*.h"
                PATTERN "*.hpp"
                PATTERN "*.cuh"
        )
        install(
                FILES ${GALP_GENERATED_INCLUDE_DIR}/galp/config.hpp
                DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/galp
        )

        install(
                EXPORT GalpTargets
                FILE GalpTargets.cmake
                NAMESPACE Galp::
                DESTINATION ${CMAKE_INSTALL_LIBDIR}/Galp
        )

        export(
                EXPORT GalpTargets
                FILE "${CMAKE_BINARY_DIR}/GalpTargets.cmake"
                NAMESPACE Galp::
        )
        export(
                EXPORT GalpTargets
                FILE "${CMAKE_CURRENT_BINARY_DIR}/GalpTargets.cmake"
                NAMESPACE Galp::
        )

        configure_package_config_file(
                ${CMAKE_CURRENT_SOURCE_DIR}/../cmake/GalpConfig.cmake.in
                ${CMAKE_CURRENT_BINARY_DIR}/GalpConfig.cmake
                INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/Galp
        )
        write_basic_package_version_file(
                ${CMAKE_CURRENT_BINARY_DIR}/GalpConfigVersion.cmake
                VERSION 0.1.0
                COMPATIBILITY SameMajorVersion
        )
        install(
                FILES
                        ${CMAKE_CURRENT_BINARY_DIR}/GalpConfig.cmake
                        ${CMAKE_CURRENT_BINARY_DIR}/GalpConfigVersion.cmake
                DESTINATION ${CMAKE_INSTALL_LIBDIR}/Galp
        )
endif ()
