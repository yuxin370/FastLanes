#include "cuda/memory/device_arena.cuh"

#include <array>
#include <cstddef>
#include <gtest/gtest.h>

TEST(DeviceArenaBackingRegions, KeepsTouchingHalfOpenRangesSeparate) {
	alignas(256) std::array<std::byte, 64> backing {};
	galp::memory::DeviceArena arena(nullptr);

	arena.register_backing(backing.data(), 32U);
	arena.register_backing(backing.data() + 32U, 32U);
	arena.coalesce_backing_regions();

	// Separate 32-byte regions start at device offsets 0 and 256, and the
	// aggregate arena rounds the end to the next 256-byte boundary.
	EXPECT_EQ(arena.total_bytes(), 512U);
}

TEST(DeviceArenaBackingRegions, StillCoalescesTrueOverlap) {
	alignas(256) std::array<std::byte, 64> backing {};
	galp::memory::DeviceArena arena(nullptr);

	arena.register_backing(backing.data(), 48U);
	arena.register_backing(backing.data() + 32U, 32U);
	arena.coalesce_backing_regions();

	EXPECT_EQ(arena.total_bytes(), 256U);
}
