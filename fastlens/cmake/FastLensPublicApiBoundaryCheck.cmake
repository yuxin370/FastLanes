if (NOT DEFINED FASTLENS_ROOT)
	get_filename_component(FASTLENS_ROOT "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)
endif ()

set(FASTLENS_PUBLIC_INCLUDE_DIR "${FASTLENS_ROOT}/include")
# Public FASTLENS headers are installed from include/fastlens. Includes that reach into
# these project-private prefixes leak implementation layout to consumers.
set(FASTLENS_PUBLIC_HEADER_INCLUDE_PATTERN "#[ \t]*include[ \t]*[<\"](api|core|codecs|cuda|engine|format|io|compression|decompression|memory|benchmark|benchmarks|test|tests|tool|tools|extension|extensions|alp|nvcomp|generator|generated)/")
set(FASTLENS_HEADER_USING_NAMESPACE_PATTERN "(^|\n)[ \t]*using[ \t]+namespace[ \t]+[^;\n]+;")

set(FASTLENS_STABLE_UMBRELLA "${FASTLENS_PUBLIC_INCLUDE_DIR}/fastlens/stable.hpp")
set(FASTLENS_DIRECT_DCT_LEGACY_HEADER "${FASTLENS_PUBLIC_INCLUDE_DIR}/fastlens/direct_dct.hpp")
set(FASTLENS_DIRECT_DCT_ADVANCED_HEADER "${FASTLENS_PUBLIC_INCLUDE_DIR}/fastlens/advanced/direct_dct.hpp")
set(FASTLENS_DIRECT_DCT_DIAGNOSTICS_HEADER "${FASTLENS_PUBLIC_INCLUDE_DIR}/fastlens/diagnostics/direct_dct.hpp")

if (NOT EXISTS "${FASTLENS_PUBLIC_INCLUDE_DIR}")
	message(FATAL_ERROR "FASTLENS public include directory does not exist: ${FASTLENS_PUBLIC_INCLUDE_DIR}")
endif ()

file(GLOB_RECURSE FASTLENS_PUBLIC_HEADERS
	"${FASTLENS_PUBLIC_INCLUDE_DIR}/*.h"
	"${FASTLENS_PUBLIC_INCLUDE_DIR}/*.hpp"
	"${FASTLENS_PUBLIC_INCLUDE_DIR}/*.cuh")

foreach (header IN LISTS FASTLENS_PUBLIC_HEADERS)
	file(READ "${header}" contents)

	string(REGEX MATCH "${FASTLENS_PUBLIC_HEADER_INCLUDE_PATTERN}" leaked_include "${contents}")
	if (leaked_include)
		message(FATAL_ERROR
			"Public FASTLENS header includes a private implementation header: ${header}\n"
			"Matched: ${leaked_include}")
	endif ()

	string(REGEX MATCH "${FASTLENS_HEADER_USING_NAMESPACE_PATTERN}" header_using_namespace "${contents}")
	if (header_using_namespace)
		message(FATAL_ERROR
			"Public FASTLENS header contains a using namespace directive: ${header}\n"
			"Matched: ${header_using_namespace}\n"
			"Use local aliases or fully qualified names instead.")
	endif ()
endforeach ()

# The canonical stable umbrella must not make raw CUDA/device/physical
# Direct-DCT types visible accidentally. fastlens/fastlens.hpp remains the legacy
# umbrella for source compatibility; new consumers use fastlens/stable.hpp.
file(READ "${FASTLENS_STABLE_UMBRELLA}" stable_umbrella_contents)
string(REGEX MATCH
	"#[ \t]*include[ \t]*[<\"]fastlens/(advanced/)?direct_dct[^>\"]*[>\"]|#[ \t]*include[ \t]*[<\"]fastlens/jpeg_dct[^>\"]*[>\"]"
	stable_direct_dct_leak
	"${stable_umbrella_contents}")
if (stable_direct_dct_leak)
	message(FATAL_ERROR
		"Stable FASTLENS umbrella exposes advanced Direct-DCT/device types: ${stable_direct_dct_leak}")
endif ()

foreach(required_boundary IN ITEMS
		"${FASTLENS_DIRECT_DCT_LEGACY_HEADER}"
		"${FASTLENS_DIRECT_DCT_ADVANCED_HEADER}"
		"${FASTLENS_DIRECT_DCT_DIAGNOSTICS_HEADER}")
	if (NOT EXISTS "${required_boundary}")
		message(FATAL_ERROR "Missing Direct-DCT API boundary header: ${required_boundary}")
	endif ()
endforeach ()

file(READ "${FASTLENS_DIRECT_DCT_LEGACY_HEADER}" legacy_direct_dct_contents)
string(FIND "${legacy_direct_dct_contents}" "fastlens/advanced/direct_dct.hpp" legacy_advanced_include)
string(FIND "${legacy_direct_dct_contents}" "class DirectDctBatch" legacy_duplicate_definition)
if (legacy_advanced_include EQUAL -1 OR NOT legacy_duplicate_definition EQUAL -1)
	message(FATAL_ERROR
		"fastlens/direct_dct.hpp must remain a thin compatibility adapter to the advanced boundary")
endif ()


message(STATUS "FASTLENS public API boundary checks passed")
