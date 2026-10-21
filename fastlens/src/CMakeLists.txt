set(FASTLENS_GENERATED_INCLUDE_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated")
configure_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/../cmake/config.hpp.in
        ${FASTLENS_GENERATED_INCLUDE_DIR}/fastlens/config.hpp
        @ONLY
)

function(fastlens_configure_library target)
        target_include_directories(${target} PUBLIC
                $<BUILD_INTERFACE:${FASTLENS_GENERATED_INCLUDE_DIR}>
                $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
                $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
        target_include_directories(${target} PRIVATE
                ${CMAKE_CURRENT_SOURCE_DIR}
                ${CUDAToolkit_INCLUDE_DIRS})
        apply_warnings(${target})
endfunction()

add_library(fastlens_compact_format STATIC
        format/compact_descriptor_v3.cpp
        format/compact_read_plan.cpp
        format/compact_shard_reader.cpp
        format/range_reader.cpp)
add_library(FastLens::compact_format ALIAS fastlens_compact_format)
set_target_properties(fastlens_compact_format PROPERTIES EXPORT_NAME compact_format)
fastlens_configure_library(fastlens_compact_format)
target_link_libraries(fastlens_compact_format PUBLIC
        $<BUILD_INTERFACE:FastLanes::core>
        $<INSTALL_INTERFACE:FLS::FastLanes>)

set(FASTLENS_ENGINE_SOURCES
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

add_library(fastlens_engine STATIC ${FASTLENS_ENGINE_SOURCES})
add_library(FastLens::engine ALIAS fastlens_engine)
set_target_properties(fastlens_engine PROPERTIES
        EXPORT_NAME engine
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON)
fastlens_configure_library(fastlens_engine)
target_link_libraries(fastlens_engine PUBLIC
        fastlens_compact_format
        CUDA::cudart
        Thrust::Thrust
        $<BUILD_INTERFACE:FastLanes::core>
        $<INSTALL_INTERFACE:FLS::FastLanes>)

if (FASTLENS_WITH_JPEG_DCT)
	set_source_files_properties(engine/materialization/selected_vector_compactor.cpp PROPERTIES LANGUAGE CUDA)
		add_library(fastlens_jpeg_dct_storage STATIC
				jpeg/jpeg_dct_block_major_access.cpp
				jpeg/jpeg_dct_canonical_plan.cpp
				jpeg/jpeg_dct_block_major_plan.cpp
				jpeg/jpeg_dct_decode.cpp
				jpeg/jpeg_dct_expression_validation.cpp
                jpeg/jpeg_dct_metadata.cpp
                jpeg/jpeg_dct_order.cpp
                jpeg/jpeg_dct_policy.cpp
                jpeg/jpeg_dct_shard_reader.cpp
                jpeg/jpeg_dct_shard_writer.cpp)
        add_library(FastLens::jpeg_dct_storage ALIAS fastlens_jpeg_dct_storage)
        set_target_properties(fastlens_jpeg_dct_storage PROPERTIES EXPORT_NAME jpeg_dct_storage)
        fastlens_configure_library(fastlens_jpeg_dct_storage)
        target_compile_definitions(fastlens_jpeg_dct_storage PRIVATE FASTLENS_WITH_JPEG_DCT=1)
        target_link_libraries(fastlens_jpeg_dct_storage PUBLIC
                fastlens_compact_format
                JPEG::JPEG
                $<BUILD_INTERFACE:FastLanes::core>
                $<INSTALL_INTERFACE:FLS::FastLanes>)

        add_library(fastlens_jpeg_dct_device STATIC
                api/direct_dct.cpp
                api/direct_dct_pls.cpp
                api/direct_dct_pls_postprocess.cu
                engine/materialization/selected_vector_compactor.cpp
                jpeg/jpeg_dct_device_bridge.cpp
				jpeg/jpeg_dct_active_output_schedule.cpp
                jpeg/jpeg_dct_exact_verifier.cpp
                jpeg/jpeg_dct_prepared_plan.cpp
                jpeg/jpeg_dct_planner.cpp
                jpeg/jpeg_dct_device.cu
                jpeg/jpeg_dct_gather_kernels.cu
                jpeg/jpeg_dct_transform_kernels.cu)
        add_library(FastLens::jpeg_dct_device ALIAS fastlens_jpeg_dct_device)
        set_target_properties(fastlens_jpeg_dct_device PROPERTIES
                EXPORT_NAME jpeg_dct_device
                CUDA_SEPARABLE_COMPILATION ON
                CUDA_RESOLVE_DEVICE_SYMBOLS ON)
        fastlens_configure_library(fastlens_jpeg_dct_device)
        target_compile_definitions(fastlens_jpeg_dct_device PRIVATE FASTLENS_WITH_JPEG_DCT=1)
        target_link_libraries(fastlens_jpeg_dct_device PUBLIC
                fastlens_engine
                fastlens_jpeg_dct_storage)
endif ()

# Compatibility umbrella: existing consumers of FastLens::core retain the complete
# feature set, while internal non-JPEG targets can link FastLens::engine directly
# and avoid the JPEG CUDA archive and its device-link requirements.
add_library(fastlens_core STATIC core/compatibility.cpp)
add_library(FastLens::core ALIAS fastlens_core)
set_target_properties(fastlens_core PROPERTIES EXPORT_NAME core)
fastlens_configure_library(fastlens_core)
target_link_libraries(fastlens_core PUBLIC fastlens_engine)
if (FASTLENS_WITH_JPEG_DCT)
        target_link_libraries(fastlens_core PUBLIC fastlens_jpeg_dct_device)
endif ()

if (FASTLENS_ENABLE_INSTALL)
        set(FASTLENS_INSTALL_TARGETS fastlens_core fastlens_engine fastlens_compact_format)
        if (FASTLENS_WITH_JPEG_DCT)
                list(APPEND FASTLENS_INSTALL_TARGETS fastlens_jpeg_dct_storage fastlens_jpeg_dct_device)
        endif ()
        install(
                TARGETS ${FASTLENS_INSTALL_TARGETS}
                EXPORT FastLensTargets
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
                FILES ${FASTLENS_GENERATED_INCLUDE_DIR}/fastlens/config.hpp
                DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/fastlens
        )

        install(
                EXPORT FastLensTargets
                FILE FastLensTargets.cmake
                NAMESPACE FastLens::
                DESTINATION ${CMAKE_INSTALL_LIBDIR}/FastLens
        )
        export(
                EXPORT FastLensTargets
                FILE "${CMAKE_BINARY_DIR}/FastLensTargets.cmake"
                NAMESPACE FastLens::
        )
        export(
                EXPORT FastLensTargets
                FILE "${CMAKE_CURRENT_BINARY_DIR}/FastLensTargets.cmake"
                NAMESPACE FastLens::
        )

        configure_package_config_file(
                ${CMAKE_CURRENT_SOURCE_DIR}/../cmake/FastLensConfig.cmake.in
                ${CMAKE_CURRENT_BINARY_DIR}/FastLensConfig.cmake
                INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/FastLens
        )
        write_basic_package_version_file(
                ${CMAKE_CURRENT_BINARY_DIR}/FastLensConfigVersion.cmake
                VERSION 0.1.0
                COMPATIBILITY SameMajorVersion
        )
        install(
                FILES
                        ${CMAKE_CURRENT_BINARY_DIR}/FastLensConfig.cmake
                        ${CMAKE_CURRENT_BINARY_DIR}/FastLensConfigVersion.cmake
                DESTINATION ${CMAKE_INSTALL_LIBDIR}/FastLens
        )
endif ()
