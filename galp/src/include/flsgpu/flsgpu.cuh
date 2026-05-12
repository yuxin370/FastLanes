// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// galp/src/include/flsgpu/flsgpu.cuh
// ────────────────────────────────────────────────────────
// Barrel header for the flsgpu device-side API (column types, fls kernels, alp
// kernels, shared structs). Include this from consumers that want the full
// GPU-side surface; for finer-grained deps, include the sub-headers directly.
#ifndef FLSGPU_FLSGPU_CUH
#define FLSGPU_FLSGPU_CUH

#include "alp.cuh"
#include "fls-switch-case.cuh"
#include "fls.cuh"
#include "structs.cuh"

namespace galp::codec::alp {}

#endif // FLSGPU_FLSGPU_CUH
