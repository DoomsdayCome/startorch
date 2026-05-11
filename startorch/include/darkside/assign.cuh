#pragma once

#include "startorch/common.hpp"
#include "startorch/memory.hpp"

#include <cstdint>

namespace darkside {
template <typename T>
void fillData(void *data, uint64_t size, T value, startorch::Arena *arena);
template <typename T>
void fillIncreasedData(void *data, uint64_t size, T start, T step,
                      startorch::Arena *arena);
template <typename T>
void fillDecreasedData(void *data, uint64_t size, T start, T step,
                      startorch::Arena *arena);
template <typename T>
void fillOrderedData(void *data, uint64_t size, T start, T step,
                     startorch::OrderType order_type, startorch::Arena *arena);
template <typename T>
void fillStrides(void *strides, void *shape, void *order, uint64_t size,
                 startorch::Arena *arena);
} // namespace darkside
