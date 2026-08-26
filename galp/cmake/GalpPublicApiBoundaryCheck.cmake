if (NOT DEFINED GALP_ROOT)
	get_filename_component(GALP_ROOT "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)
endif ()

set(GALP_PUBLIC_INCLUDE_DIR "${GALP_ROOT}/include")
# Public GALP headers are installed from include/galp. Includes that reach into
# these project-private prefixes leak implementation layout to consumers.
set(GALP_PUBLIC_HEADER_INCLUDE_PATTERN "#[ \t]*include[ \t]*[<\"](api|core|codecs|cuda|engine|format|io|compression|decompression|memory|benchmark|benchmarks|test|tests|tool|tools|extension|extensions|alp|nvcomp|generator|generated)/")
set(GALP_HEADER_USING_NAMESPACE_PATTERN "(^|\n)[ \t]*using[ \t]+namespace[ \t]+[^;\n]+;")

set(GALP_STABLE_UMBRELLA "${GALP_PUBLIC_INCLUDE_DIR}/galp/stable.hpp")
set(GALP_DIRECT_DCT_LEGACY_HEADER "${GALP_PUBLIC_INCLUDE_DIR}/galp/direct_dct.hpp")
set(GALP_DIRECT_DCT_ADVANCED_HEADER "${GALP_PUBLIC_INCLUDE_DIR}/galp/advanced/direct_dct.hpp")
set(GALP_DIRECT_DCT_DIAGNOSTICS_HEADER "${GALP_PUBLIC_INCLUDE_DIR}/galp/diagnostics/direct_dct.hpp")

if (NOT EXISTS "${GALP_PUBLIC_INCLUDE_DIR}")
	message(FATAL_ERROR "GALP public include directory does not exist: ${GALP_PUBLIC_INCLUDE_DIR}")
endif ()

file(GLOB_RECURSE GALP_PUBLIC_HEADERS
	"${GALP_PUBLIC_INCLUDE_DIR}/*.h"
	"${GALP_PUBLIC_INCLUDE_DIR}/*.hpp"
	"${GALP_PUBLIC_INCLUDE_DIR}/*.cuh")

foreach (header IN LISTS GALP_PUBLIC_HEADERS)
	file(READ "${header}" contents)

	string(REGEX MATCH "${GALP_PUBLIC_HEADER_INCLUDE_PATTERN}" leaked_include "${contents}")
	if (leaked_include)
		message(FATAL_ERROR
			"Public GALP header includes a private implementation header: ${header}\n"
			"Matched: ${leaked_include}")
	endif ()

	string(REGEX MATCH "${GALP_HEADER_USING_NAMESPACE_PATTERN}" header_using_namespace "${contents}")
	if (header_using_namespace)
		message(FATAL_ERROR
			"Public GALP header contains a using namespace directive: ${header}\n"
			"Matched: ${header_using_namespace}\n"
			"Use local aliases or fully qualified names instead.")
	endif ()
endforeach ()

# The canonical stable umbrella must not make raw CUDA/device/physical
# Direct-DCT types visible accidentally. galp/galp.hpp remains the legacy
# umbrella for source compatibility; new consumers use galp/stable.hpp.
file(READ "${GALP_STABLE_UMBRELLA}" stable_umbrella_contents)
string(REGEX MATCH
	"#[ \t]*include[ \t]*[<\"]galp/(advanced/)?direct_dct[^>\"]*[>\"]|#[ \t]*include[ \t]*[<\"]galp/jpeg_dct[^>\"]*[>\"]"
	stable_direct_dct_leak
	"${stable_umbrella_contents}")
if (stable_direct_dct_leak)
	message(FATAL_ERROR
		"Stable GALP umbrella exposes advanced Direct-DCT/device types: ${stable_direct_dct_leak}")
endif ()

foreach(required_boundary IN ITEMS
		"${GALP_DIRECT_DCT_LEGACY_HEADER}"
		"${GALP_DIRECT_DCT_ADVANCED_HEADER}"
		"${GALP_DIRECT_DCT_DIAGNOSTICS_HEADER}")
	if (NOT EXISTS "${required_boundary}")
		message(FATAL_ERROR "Missing Direct-DCT API boundary header: ${required_boundary}")
	endif ()
endforeach ()

file(READ "${GALP_DIRECT_DCT_LEGACY_HEADER}" legacy_direct_dct_contents)
string(FIND "${legacy_direct_dct_contents}" "galp/advanced/direct_dct.hpp" legacy_advanced_include)
string(FIND "${legacy_direct_dct_contents}" "class DirectDctBatch" legacy_duplicate_definition)
if (legacy_advanced_include EQUAL -1 OR NOT legacy_duplicate_definition EQUAL -1)
	message(FATAL_ERROR
		"galp/direct_dct.hpp must remain a thin compatibility adapter to the advanced boundary")
endif ()


message(STATUS "GALP public API boundary checks passed")
