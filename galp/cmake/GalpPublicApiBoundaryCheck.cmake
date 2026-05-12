if (NOT DEFINED GALP_ROOT)
	get_filename_component(GALP_ROOT "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)
endif ()

set(GALP_PUBLIC_INCLUDE_DIR "${GALP_ROOT}/include")
# Public GALP headers are installed from include/galp. Includes that reach into
# these project-private prefixes leak implementation layout to consumers.
set(GALP_PUBLIC_HEADER_INCLUDE_PATTERN "#[ \t]*include[ \t]*[<\"](engine|flsgpu|benchmark|alp|nvcomp|generator)/")
set(GALP_HEADER_USING_NAMESPACE_PATTERN "(^|\n)[ \t]*using[ \t]+namespace[ \t]+[^;\n]+;")

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


message(STATUS "GALP public API boundary checks passed")
