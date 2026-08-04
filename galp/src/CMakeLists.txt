set(GALP_GENERATED_INCLUDE_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated")
configure_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/../cmake/config.hpp.in
        ${GALP_GENERATED_INCLUDE_DIR}/galp/config.hpp
        @ONLY
)

function(galp_configure_library target)
        target_include_directories(${target} PUBLIC
                $<BUILD_INTERFACE:${GALP_GENERATED_INCLUDE_DIR}>
                $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
                $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
        target_include_directories(${target} PRIVATE
                ${CMAKE_CURRENT_SOURCE_DIR}
                ${CUDAToolkit_INCLUDE_DIRS})
        apply_warnings(${target})
endfunction()

add_library(galp_compact_format STATIC
        format/compact_descriptor_v3.cpp
        format/compact_read_plan.cpp
        format/compact_shard_reader.cpp
        format/range_reader.cpp)
add_library(Galp::compact_format ALIAS galp_compact_format)
set_target_properties(galp_compact_format PROPERTIES EXPORT_NAME compact_format)
galp_configure_library(galp_compact_format)
target_link_libraries(galp_compact_format PUBLIC
        $<BUILD_INTERFACE:FastLanes::core>
        $<INSTALL_INTERFACE:FLS::FastLanes>)

set(GALP_ENGINE_SOURCES
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
)

add_library(galp_engine STATIC ${GALP_ENGINE_SOURCES})
add_library(Galp::engine ALIAS galp_engine)
set_target_properties(galp_engine PROPERTIES
        EXPORT_NAME engine
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON)
galp_configure_library(galp_engine)
target_link_libraries(galp_engine PUBLIC
        galp_compact_format
        CUDA::cudart
        Thrust::Thrust
        $<BUILD_INTERFACE:FastLanes::core>
        $<INSTALL_INTERFACE:FLS::FastLanes>)

if (GALP_WITH_JPEG_DCT)
	set_source_files_properties(engine/materialization/selected_vector_compactor.cpp PROPERTIES LANGUAGE CUDA)
		add_library(galp_jpeg_dct_storage STATIC
				jpeg/jpeg_dct_block_major_access.cpp
				jpeg/jpeg_dct_block_major_plan.cpp
				jpeg/jpeg_dct_decode.cpp
				jpeg/jpeg_dct_expression_validation.cpp
                jpeg/jpeg_dct_metadata.cpp
                jpeg/jpeg_dct_order.cpp
                jpeg/jpeg_dct_policy.cpp
                jpeg/jpeg_dct_shard_reader.cpp
                jpeg/jpeg_dct_shard_writer.cpp)
        add_library(Galp::jpeg_dct_storage ALIAS galp_jpeg_dct_storage)
        set_target_properties(galp_jpeg_dct_storage PROPERTIES EXPORT_NAME jpeg_dct_storage)
        galp_configure_library(galp_jpeg_dct_storage)
        target_compile_definitions(galp_jpeg_dct_storage PRIVATE GALP_WITH_JPEG_DCT=1)
        target_link_libraries(galp_jpeg_dct_storage PUBLIC
                galp_compact_format
                JPEG::JPEG
                $<BUILD_INTERFACE:FastLanes::core>
                $<INSTALL_INTERFACE:FLS::FastLanes>)

        add_library(galp_jpeg_dct_device STATIC
                api/direct_dct.cpp
                engine/materialization/selected_vector_compactor.cpp
                jpeg/jpeg_dct_device_bridge.cpp
                jpeg/jpeg_dct_exact_verifier.cpp
                jpeg/jpeg_dct_prepared_plan.cpp
                jpeg/jpeg_dct_planner.cpp
                jpeg/jpeg_dct_device.cu
                jpeg/jpeg_dct_gather_kernels.cu
                jpeg/jpeg_dct_transform_kernels.cu)
        add_library(Galp::jpeg_dct_device ALIAS galp_jpeg_dct_device)
        set_target_properties(galp_jpeg_dct_device PROPERTIES
                EXPORT_NAME jpeg_dct_device
                CUDA_SEPARABLE_COMPILATION ON
                CUDA_RESOLVE_DEVICE_SYMBOLS ON)
        galp_configure_library(galp_jpeg_dct_device)
        target_compile_definitions(galp_jpeg_dct_device PRIVATE GALP_WITH_JPEG_DCT=1)
        target_link_libraries(galp_jpeg_dct_device PUBLIC
                galp_engine
                galp_jpeg_dct_storage)
endif ()

# Compatibility umbrella: existing consumers of Galp::core retain the complete
# feature set, while internal non-JPEG targets can link Galp::engine directly
# and avoid the JPEG CUDA archive and its device-link requirements.
add_library(galp_core STATIC core/compatibility.cpp)
add_library(Galp::core ALIAS galp_core)
set_target_properties(galp_core PROPERTIES EXPORT_NAME core)
galp_configure_library(galp_core)
target_link_libraries(galp_core PUBLIC galp_engine)
if (GALP_WITH_JPEG_DCT)
        target_link_libraries(galp_core PUBLIC galp_jpeg_dct_device)
endif ()

if (GALP_ENABLE_INSTALL)
        set(GALP_INSTALL_TARGETS galp_core galp_engine galp_compact_format)
        if (GALP_WITH_JPEG_DCT)
                list(APPEND GALP_INSTALL_TARGETS galp_jpeg_dct_storage galp_jpeg_dct_device)
        endif ()
        install(
                TARGETS ${GALP_INSTALL_TARGETS}
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
