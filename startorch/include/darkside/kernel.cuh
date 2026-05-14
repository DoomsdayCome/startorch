#pragma once

#include "startorch/memory.hpp"

#include <cstdint>

namespace darkside {
template <typename T> void fillData(void *data, uint64_t size, T value, startorch::Arena *arena);
template <typename T> void fillIncreasedData(void *data, uint64_t size, T start, T step, startorch::Arena *arena);
template <typename T> void fillDecreasedData(void *data, uint64_t size, T start, T step, startorch::Arena *arena);
} // namespace darkside
