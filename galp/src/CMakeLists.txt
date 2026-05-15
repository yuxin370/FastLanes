set(GALP_CORE_SOURCES
        api/table.cu
        alp/alp-bindings.cu
        memory/device_arena.cu
        engine/enums.cu
        engine/format/reader.cu
        engine/format/rowgroup_io.cu
        engine/format/schema_plan.cu
        engine/format/zero_copy_materializer.cu
        engine/execution/column.cu
        engine/runtime/materialize/metadata.cu
        engine/runtime/materialize/pinned_d2h.cu
        engine/runtime/table/helpers.cu
        engine/runtime/workset/streams.cu
        engine/runtime/workset/upload.cu
        engine/execution/rowgroup.cu
        engine/execution/table.cu
        engine/io/to_csv.cu
        engine/expression.cu
)
add_library(galp_core STATIC ${GALP_CORE_SOURCES})
add_library(Galp::core ALIAS galp_core)
set_target_properties(galp_core PROPERTIES
        EXPORT_NAME core
        CUDA_RESOLVE_DEVICE_SYMBOLS ON)
target_include_directories(galp_core PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_include_directories(galp_core PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CUDAToolkit_INCLUDE_DIRS})
target_link_libraries(galp_core PUBLIC
        CUDA::cudart
        Thrust::Thrust
        $<BUILD_INTERFACE:FastLanes::core>
        $<INSTALL_INTERFACE:FLS::FastLanes>)
apply_warnings(galp_core)

add_library(galp_benchmark_support STATIC
        engine/benchmark/table.cu
)
set_target_properties(galp_benchmark_support PROPERTIES
        CUDA_RESOLVE_DEVICE_SYMBOLS ON)
target_include_directories(galp_benchmark_support PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_include_directories(galp_benchmark_support PRIVATE
        ${CUDAToolkit_INCLUDE_DIRS})
target_link_libraries(galp_benchmark_support PUBLIC
        galp_core)
apply_warnings(galp_benchmark_support)

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
