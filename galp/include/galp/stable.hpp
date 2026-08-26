#ifndef GALP_STABLE_HPP
#define GALP_STABLE_HPP

// Canonical stable C++ umbrella. Direct-DCT's model-facing stable API is the
// galp.torch facade; raw native CUDA/device contracts require an explicit
// advanced or diagnostics include.
#define GALP_STABLE_API 1

#include "galp/config.hpp"
#include "galp/errors.hpp"
#include "galp/options.hpp"
#include "galp/reader.hpp"
#include "galp/table.hpp"

#endif // GALP_STABLE_HPP
