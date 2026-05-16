#pragma once

#include "startorch/memory.hpp"

#include <cstdint>

namespace darkside {
template <typename T> void fillData(T *data, uint64_t size, T value, startorch::Arena *arena);
template <typename T> void fillIncreasedData(T *data, uint64_t size, T start, T step, startorch::Arena *arena);
template <typename T> void fillDecreasedData(T *data, uint64_t size, T start, T step, startorch::Arena *arena);
} // namespace darkside
