#pragma once

#include "startorch/common.hpp"
#include "startorch/memory.hpp"

#include <cstdint>

namespace darkside {
template <typename T>
void fillData(void *data, uint64_t size, T value, startorch::Arena *arena);
template <typename T>
void fillIncreaseData(void *data, uint64_t size, startorch::Arena *arena);
template <typename T>
void fillDecreaseData(void *data, uint64_t size, startorch::Arena *arena);
template <typename T>
void fillOrderedData(void *data, uint64_t size, startorch::OrderType order_type,
                     startorch::Arena *arena);
} // namespace darkside
