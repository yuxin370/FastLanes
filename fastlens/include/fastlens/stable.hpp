#ifndef FASTLENS_STABLE_HPP
#define FASTLENS_STABLE_HPP

// Canonical stable C++ umbrella. Direct-DCT's model-facing stable API is the
// fastlens.torch facade; raw native CUDA/device contracts require an explicit
// advanced or diagnostics include.
#define FASTLENS_STABLE_API 1

#include "fastlens/config.hpp"
#include "fastlens/errors.hpp"
#include "fastlens/options.hpp"
#include "fastlens/reader.hpp"
#include "fastlens/table.hpp"

#endif // FASTLENS_STABLE_HPP
